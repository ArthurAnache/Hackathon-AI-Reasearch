import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'time_fraction': (0.70, 0.5, 0.72),
    'explore_fraction': (0.77, 0.5, 0.9),
    'iterations': (240, 100, 500),
    'ftol': (3e-13, 1e-14, 1e-9, 'log'),
    'safety': (2e-14, 1e-15, 1e-10, 'log')
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
    p = (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r)) +
         max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)))
    return float(np.sum(a[:, 2])) if p <= 2 else -np.inf

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + params['time_fraction']*budget_s
    explore_end = started + params['explore_fraction']*(deadline-started)
    eps = params['safety']
    best, best_a, score = None, None, -np.inf
    rng = np.random.default_rng(seed)
    if init is not None:
        try:
            a = (np.asarray(init['circles'], float) if 'circles' in init else
                 np.asarray(init['state'], float).reshape(3,21).T)
            s = exact_score(a)
            if np.isfinite(s):
                best, best_a, score = init, a.copy(), s
        except (KeyError, ValueError, TypeError):
            pass

    def offer(a):
        nonlocal best, best_a, score
        a = np.array(a, dtype=float, copy=True)
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
        ratio = min(1., float(np.min(d/(a[I,2]+a[J,2]))))
        if ratio <= 0:
            return
        a[:,2] *= ratio*(1-eps)
        lo = np.min(a[:,:2]-a[:,2,None], axis=0)
        hi = np.max(a[:,:2]+a[:,2,None], axis=0)
        scale = (2-12*eps)/float(np.sum(hi-lo))
        a[:,:2] = (a[:,:2]-(lo+hi)/2)*scale
        a[:,2] *= scale
        if np.sum(a[:,2]) <= score:
            return
        s = exact_score(a)
        if s > score:
            best_a, score = a.copy(), s
            best = {'state': a.T.reshape(-1).tolist()}

    offer([[.2*(k%5), .2*(k//5), .099999999999] for k in range(21)])

    def model(sym):
        if time.monotonic() >= deadline:
            raise StopSearch()
        if sym:
            n = 34
            ids = np.r_[np.arange(12),np.arange(3,12)]
            xi = ids
            yi = np.maximum(ids+9,0)
            yi[:3] = 0
            ys = np.r_[np.zeros(3),np.ones(9),-np.ones(9)]
            ri = 21+ids
            pairs = [(i,j) for i in range(12) for j in range(i+1,12)]
            pairs += [(i,12+j-3) for i in range(3,12) for j in range(i,12)]
            pi,pj = np.asarray(pairs).T
            wall = np.arange(12)
        else:
            n = 64
            xi = np.arange(21)
            yi, ri = xi+21, xi+42
            ys = np.ones(21)
            pi,pj = I,J
            wall = np.arange(21)
        h = len(wall)
        k = np.arange(h)
        b = np.zeros((4*h,n))
        b[k,xi[wall]] = 1
        b[h+k,xi[wall]] = -1
        b[:2*h,-1] = .5
        b[k,ri[wall]] = -1
        b[h+k,ri[wall]] = -1
        b[2*h+k,yi[wall]] = ys[wall]
        b[3*h+k,yi[wall]] = -ys[wall]
        b[2*h+k,ri[wall]] -= 1
        b[3*h+k,ri[wall]] -= 1
        b[2*h:,-1] = -.5
        off = np.r_[np.zeros(2*h),np.ones(2*h)]
        objective = np.zeros(n)
        np.add.at(objective,ri,-1.)
        bounds = [(None,None)]*n
        for q in np.unique(ri):
            if time.monotonic() >= deadline:
                raise StopSearch()
            bounds[int(q)] = (1e-7,.6)
        bounds[-1] = (.15,1.85)
        if sym:
            bounds[12:21] = [(0.,None)]*9
        rows = np.arange(len(pi))
        dxj = np.zeros((len(pi),n))
        dyj = np.zeros_like(dxj)
        rrj = np.zeros_like(dxj)
        np.add.at(dxj,(rows,xi[pi]),1.)
        np.add.at(dxj,(rows,xi[pj]),-1.)
        np.add.at(dyj,(rows,yi[pi]),ys[pi])
        np.add.at(dyj,(rows,yi[pj]),-ys[pj])
        np.add.at(rrj,(rows,ri[pi]),1.)
        np.add.at(rrj,(rows,ri[pj]),1.)
        return xi,yi,ys,ri,pi,pj,b,off,objective,bounds,dxj,dyj,rrj

    def run(z, mod, stop):
        xi,yi,ys,ri,pi,pj,b,off,obj,bounds,dxj,dyj,rrj = mod
        cached = [None,None,None]
        last = [z.copy()]
        count = [0]
        def unpack(v):
            return np.column_stack((v[xi],ys*v[yi],v[ri]))
        def calc(v):
            if time.monotonic() >= stop:
                raise StopSearch()
            if cached[0] is not None and np.array_equal(v,cached[0]):
                return cached[1],cached[2]
            x,y,r = v[xi],ys*v[yi],v[ri]
            dx,dy = x[pi]-x[pj],y[pi]-y[pj]
            d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-15)
            c = np.r_[d-r[pi]-r[pj],b@v+off]
            jac = np.vstack(((dx/d)[:,None]*dxj+(dy/d)[:,None]*dyj-rrj,b))
            cached[:] = [v.copy(),c,jac]
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
            if count[0]%12 == 0:
                offer(unpack(v))
        try:
            result = minimize(objective,z,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda v:calc(v)[0],
                             'jac':lambda v:calc(v)[1]},callback=callback,
                options={'maxiter':int(params['iterations']),'ftol':params['ftol'],'disp':False})
            z = result.x
        except StopSearch:
            z = last[0]
        offer(unpack(z))
        return z

    def full_state(a):
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        return np.r_[a[:,0]-(hi[0]+lo[0])/2,a[:,1]-(hi[1]+lo[1])/2,a[:,2],hi[0]-lo[0]]

    def lattice(kind):
        if kind == 0:
            xx = np.r_[[-2.,0.,2.],[-3.,-1.,1.,3.],[-4.,-2.,0.,2.,4.]]
            yy = np.r_[np.full(4,np.sqrt(3)),np.full(5,2*np.sqrt(3))]
        elif kind == 1:
            xx = np.r_[[-2.,0.,2.],[-4.,-2.,0.,2.,4.],[-3.,-1.,1.,3.]]
            yy = np.r_[np.full(5,2.),np.full(4,2.+np.sqrt(3))]
        else:
            xx = np.r_[[-2.,0.,2.],[-1.,1.,3.],[-2.,0.,2.],[-1.,1.,3.]]
            yy = np.repeat(np.arange(1,4)*np.sqrt(3),3)
        width = np.ptp(xx)+2
        scale = 2/(width+2*np.max(yy+1))
        xx = xx-(np.min(xx)+np.max(xx))/2
        return np.r_[xx*scale,yy*scale,np.full(12,scale),width*scale]

    try:
        sym, full = model(True), model(False)
        if score > 2.30:
            run(full_state(best_a),full,explore_end)
        else:
            for index,kind in enumerate((1,0,2)):
                now = time.monotonic()
                if now >= explore_end-.03:
                    break
                stop = min(explore_end,now+(explore_end-now)/(3-index))
                run(lattice(kind),sym,stop)
        if time.monotonic() < deadline-.035:
            run(full_state(best_a),full,deadline-.015)
        attempt = 0
        while time.monotonic() < deadline-.08 and attempt < 3:
            attempt += 1
            z = full_state(best_a)
            z[:42] += rng.normal(0,.012,42)
            run(z,full,deadline-.015)
    except StopSearch:
        pass
    return best
