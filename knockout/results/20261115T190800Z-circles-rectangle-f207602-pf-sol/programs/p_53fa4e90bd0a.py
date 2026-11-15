import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'explore_fraction': (0.72, 0.4, 0.85),
    'iterations': (220, 80, 500),
    'safety': (3e-14, 1e-14, 1e-10, 'log'),
    'restart_scale': (0.025, 0.001, 0.1)
}
I, J = np.triu_indices(21, 1)

class StopSearch(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return -np.inf
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    perimeter = (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r)) +
                 max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)))
    return float(np.sum(a[:, 2])) if perimeter <= 2 else -np.inf

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    exploration_end = start + 0.70*budget_s*params['explore_fraction']
    rng = np.random.default_rng(seed)
    eps = params['safety']
    best = None
    best_a = None
    score = -np.inf
    try:
        if init is not None:
            a = (np.asarray(init['circles'], float) if 'circles' in init else
                 np.asarray(init['state'], float).reshape(3,21).T)
            s = exact_score(a)
            if np.isfinite(s):
                best, best_a, score = init, a.copy(), s
    except (KeyError, ValueError, TypeError):
        pass

    def offer(a):
        nonlocal best, best_a, score
        a = np.asarray(a, float).copy()
        if a.shape != (21,3) or not np.isfinite(a).all() or np.min(a[:,2]) <= 0:
            return
        d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
        ratio = min(1.0, float(np.min(d/(a[I,2]+a[J,2]))))
        if ratio <= 0:
            return
        a[:,2] *= ratio*(1-eps)
        lo = np.min(a[:,:2]-a[:,2,None], axis=0)
        hi = np.max(a[:,:2]+a[:,2,None], axis=0)
        scale = (2-16*eps)/float(np.sum(hi-lo))
        a[:,:2] = (a[:,:2]-(hi+lo)/2)*scale
        a[:,2] *= scale
        if float(np.sum(a[:,2])) <= score:
            return
        s = exact_score(a)
        if s > score:
            score, best_a = s, a.copy()
            best = {'state': a.T.reshape(-1).tolist()}

    # Always have a certified fallback, independent of the supplied start.
    fallback = np.array([[0.2*(k%3), 0.2*(k//3), 0.099999999999] for k in range(21)])
    offer(fallback)

    def model(sym):
        if sym:
            n = 34
            ids = np.r_[np.arange(12), np.arange(3,12)]
            xi = ids
            yi = np.maximum(12+ids-3, 0)
            yi[:3] = 0
            ys = np.r_[np.zeros(3), np.ones(9), -np.ones(9)]
            ri = 21+ids
            pairs = [(i,j) for i in range(12) for j in range(i+1,12)]
            pairs += [(i,12+j-3) for i in range(3,12) for j in range(i,12)]
            pi, pj = np.array(pairs).T
            wall = np.arange(12)
        else:
            n = 64
            xi = np.arange(21)
            yi = xi+21
            ys = np.ones(21)
            ri = xi+42
            pi, pj = I, J
            wall = np.arange(21)
        h = len(wall)
        k = np.arange(h)
        b = np.zeros((4*h,n))
        b[k,xi[wall]] = 1
        b[k,ri[wall]] = -1
        b[h+k,xi[wall]] = -1
        b[h+k,ri[wall]] = -1
        b[h+k,-1] = 1
        b[2*h+k,yi[wall]] = -ys[wall]
        b[2*h+k,ri[wall]] -= 1
        b[2*h+k,-1] = -.5
        b[3*h+k,yi[wall]] = ys[wall]
        b[3*h+k,ri[wall]] -= 1
        b[3*h+k,-1] = -.5
        off = np.r_[np.zeros(2*h), np.ones(2*h)]
        obj = np.zeros(n)
        np.add.at(obj,ri,-1.)
        bounds = [(None,None)]*n
        for q in np.unique(ri):
            if time.monotonic() >= deadline:
                raise StopSearch()
            bounds[int(q)] = (1e-7,.65)
        bounds[-1] = (.15,1.85)
        return n,xi,yi,ys,ri,pi,pj,b,off,obj,bounds

    def run(z, mod, stop):
        n,xi,yi,ys,ri,pi,pj,b,off,obj,bounds = mod
        m = len(pi)
        rows = np.arange(m)
        weights = 1/np.maximum(2*(z[ri[pi]]+z[ri[pj]]), .08)
        cache = [None,None,None]
        last = [z.copy()]
        count = [0]
        def unpack(v):
            return np.column_stack((v[xi],ys*v[yi],v[ri]))
        def calc(v):
            if time.monotonic() >= stop:
                raise StopSearch()
            if cache[0] is not None and np.array_equal(v,cache[0]):
                return cache[1],cache[2]
            x,y,r = v[xi],ys*v[yi],v[ri]
            dx,dy,rr = x[pi]-x[pj],y[pi]-y[pj],r[pi]+r[pj]
            c = np.r_[(dx*dx+dy*dy-rr*rr)*weights,b@v+off]
            jac = np.zeros((m+len(b),n))
            tx,ty,tr = 2*weights*dx,2*weights*dy,-2*weights*rr
            np.add.at(jac,(rows,xi[pi]),tx)
            np.add.at(jac,(rows,xi[pj]),-tx)
            np.add.at(jac,(rows,yi[pi]),ty*ys[pi])
            np.add.at(jac,(rows,yi[pj]),-ty*ys[pj])
            np.add.at(jac,(rows,ri[pi]),tr)
            np.add.at(jac,(rows,ri[pj]),tr)
            jac[m:] = b
            cache[:] = [v.copy(),c,jac]
            return c,jac
        def objective(v):
            if time.monotonic() >= stop:
                raise StopSearch()
            return float(obj@v),obj
        def callback(v):
            if time.monotonic() >= stop:
                raise StopSearch()
            last[0] = v.copy()
            count[0] += 1
            if count[0]%10 == 0:
                offer(unpack(v))
        try:
            result = minimize(objective,z,jac=True,method='SLSQP',bounds=bounds,
                              constraints={'type':'ineq','fun':lambda v:calc(v)[0],
                                           'jac':lambda v:calc(v)[1]},callback=callback,
                              options={'maxiter':int(params['iterations']), 'ftol':5e-13,'disp':False})
            z = result.x
        except StopSearch:
            z = last[0]
        offer(unpack(z))
        return z

    def full_state(a):
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        return np.r_[a[:,0]-lo[0], a[:,1]-(hi[1]+lo[1])/2, a[:,2], hi[0]-lo[0]]

    def symmetric_seed(kind):
        if kind == 0:
            xx = np.r_[[-2.,0.,2.],[-3.,-1.,1.,3.],[-4.,-2.,0.,2.,4.]]
            yy = np.r_[np.full(4,np.sqrt(3)),np.full(5,2*np.sqrt(3))]
        elif kind == 1:
            xx = np.r_[[-2.,0.,2.],[-4.,-2.,0.,2.,4.],[-3.,-1.,1.,3.]]
            yy = np.r_[np.full(5,2.),np.full(4,2.+np.sqrt(3))]
        else:
            xx = np.r_[[-2.,0.,2.],[-1.,1.,3.],[-2.,0.,2.],[-1.,1.,3.]]
            yy = np.repeat(np.arange(1,4)*np.sqrt(3),3)
        rad = np.ones(12)
        xmin,xmax = np.min(xx-rad),np.max(xx+rad)
        height = 2*np.max(yy+1)
        scale = 2/(xmax-xmin+height)
        return np.r_[(xx-xmin)*scale,yy*scale,rad*scale,(xmax-xmin)*scale]

    try:
        sym = model(True)
        full = model(False)
        if score > 2.30:
            run(full_state(best_a),full,exploration_end)
        else:
            # Short quotient solves are independent of the arbitrary cold centres.
            for kind in range(3):
                now = time.monotonic()
                if now >= exploration_end-.025:
                    break
                stop = min(exploration_end, now + max(.15,(exploration_end-now)/(3-kind)))
                z = symmetric_seed(kind)
                z = run(z,sym,stop)
        if time.monotonic() < deadline-.04:
            run(full_state(best_a),full,deadline-.015)
        attempt = 0
        while time.monotonic() < deadline-.07 and attempt < 3:
            attempt += 1
            z = full_state(best_a)
            z[:42] += rng.normal(0,params['restart_scale'],42)
            run(z,full,deadline-.015)
    except StopSearch:
        pass
    return best
