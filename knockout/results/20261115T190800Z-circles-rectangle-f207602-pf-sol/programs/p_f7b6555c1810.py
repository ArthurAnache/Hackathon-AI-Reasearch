import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {
    'symmetric_fraction': (0.72, 0.4, 0.9),
    'iterations': (240, 80, 500),
    'safety': (2e-14, 5e-15, 1e-10, 'log'),
    'restart_size': (0.016, 0.001, 0.08)
}
I, J = np.triu_indices(21, 1)

class StopSearch(Exception):
    pass

def valid(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
        return False
    d = np.sqrt((a[I,0]-a[J,0])**2 + (a[I,1]-a[J,1])**2)
    if np.any(d < a[I,2]+a[J,2]):
        return False
    x = [Fraction(float(v)) for v in a[:,0]]
    y = [Fraction(float(v)) for v in a[:,1]]
    r = [Fraction(float(v)) for v in a[:,2]]
    return (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r)) +
            max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))) <= 2

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    eps = float(params['safety'])
    best = None
    best_a = None
    best_score = -np.inf
    original = None
    try:
        if init is not None:
            original = (np.asarray(init['circles'], dtype=float) if 'circles' in init else
                        np.asarray(init['state'], dtype=float).reshape(3,21).T)
            if valid(original):
                best, best_a = init, original.copy()
                best_score = float(np.sum(original[:,2]))
    except (ValueError, TypeError, KeyError):
        original = None

    def offer(a):
        nonlocal best, best_a, best_score
        a = np.asarray(a, dtype=float).copy()
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
        ratio = min(1.0, float(np.min(d/(a[I,2]+a[J,2]))))
        if ratio <= 0:
            return
        a[:,2] *= ratio*(1-eps)
        low = np.min(a[:,:2]-a[:,2,None], axis=0)
        high = np.max(a[:,:2]+a[:,2,None], axis=0)
        span = float(np.sum(high-low))
        if not span > 0:
            return
        scale = (2-12*eps)/span
        a[:,:2] = (a[:,:2]-(low+high)/2)*scale
        a[:,2] *= scale
        value = float(np.sum(a[:,2]))
        if value > best_score and valid(a):
            best_score = value
            best_a = a.copy()
            best = {'state': a.T.reshape(-1).tolist()}

    # Five staggered columns: precisely three axis circles and nine pairs.
    lattice = []
    for row, count in enumerate((3,4,5,4,5)):
        if time.monotonic() >= deadline:
            break
        for k in range(count):
            if time.monotonic() >= deadline:
                break
            lattice.append([row*np.sqrt(3.0), 2*k-(count-1), 1.0])
    if len(lattice) == 21:
        offer(np.array(lattice))
    if original is not None:
        offer(original)
    if best is None:
        a = np.array([[0.04*(k%3),0.04*(k//3),0.019] for k in range(21)])
        offer(a)
    if time.monotonic() >= deadline-.04:
        return best

    def make_model(sym):
        if sym:
            n = 34
            ids = np.r_[np.arange(12),np.arange(3,12)]
            xi = ids.copy()
            yi = np.where(ids >= 3, 12+ids-3, 0)
            sy = np.r_[np.zeros(3),np.ones(9),-np.ones(9)]
            ri = 21+ids
            pairs = [(i,j) for i in range(12) for j in range(i+1,12)]
            pairs += [(i,12+j-3) for i in range(3,12) for j in range(i,12)]
            pi,pj = np.array(pairs).T
            wall = np.arange(12)
        else:
            n = 64
            xi = np.arange(21)
            yi = 21+xi
            sy = np.ones(21)
            ri = 42+xi
            pi,pj = I,J
            wall = np.arange(21)
        h = len(wall)
        b = np.zeros((4*h,n))
        k = np.arange(h)
        b[k,xi[wall]] = 1
        b[k,ri[wall]] = -1
        b[h+k,xi[wall]] = -1
        b[h+k,ri[wall]] = -1
        b[h+k,-1] = 1
        b[2*h+k,yi[wall]] = -sy[wall]
        b[2*h+k,ri[wall]] -= 1
        b[2*h+k,-1] = -.5
        b[3*h+k,yi[wall]] = sy[wall]
        b[3*h+k,ri[wall]] -= 1
        b[3*h+k,-1] = -.5
        off = np.r_[np.zeros(2*h),np.ones(2*h)]
        obj = np.zeros(n)
        np.add.at(obj,ri,-1.)
        bounds = [(None,None)]*n
        for q in np.unique(ri):
            if time.monotonic() >= deadline:
                break
            bounds[int(q)] = (1e-7,.65)
        if sym:
            for q in range(12,21):
                if time.monotonic() >= deadline:
                    break
                bounds[q] = (0,None)
        bounds[-1] = (.15,1.85)
        return sym,n,xi,yi,sy,ri,pi,pj,b,off,obj,bounds

    def unpack(z,mod):
        _,_,xi,yi,sy,ri,*_ = mod
        return np.column_stack((z[xi],sy*z[yi],z[ri]))

    def optimize(z, mod, stop):
        sym,n,xi,yi,sy,ri,pi,pj,b,off,obj,bounds = mod
        rows = np.arange(len(pi))
        m = len(pi)
        weights = 1/np.maximum(2*(z[ri[pi]]+z[ri[pj]]),.06)
        cache = [None,None,None]
        last = [z.copy()]
        count = [0]
        def calc(v):
            if time.monotonic() >= stop:
                raise StopSearch
            if cache[0] is not None and np.array_equal(cache[0],v):
                return cache[1],cache[2]
            x,y,r = v[xi],sy*v[yi],v[ri]
            dx,dy,rr = x[pi]-x[pj],y[pi]-y[pj],r[pi]+r[pj]
            c = np.r_[(dx*dx+dy*dy-rr*rr)*weights,b@v+off]
            jac = np.zeros((m+len(b),n))
            tx,ty,tr = 2*weights*dx,2*weights*dy,-2*weights*rr
            jac[rows,xi[pi]] = tx
            if sym:
                np.add.at(jac,(rows,xi[pj]),-tx)
                np.add.at(jac,(rows,yi[pi]),ty*sy[pi])
                np.add.at(jac,(rows,yi[pj]),-ty*sy[pj])
                np.add.at(jac,(rows,ri[pi]),tr)
                np.add.at(jac,(rows,ri[pj]),tr)
            else:
                jac[rows,xi[pj]] = -tx
                jac[rows,yi[pi]] = ty
                jac[rows,yi[pj]] = -ty
                jac[rows,ri[pi]] = tr
                jac[rows,ri[pj]] = tr
            jac[m:] = b
            cache[:] = [v.copy(),c,jac]
            last[0] = v.copy()
            return c,jac
        def fun(v):
            if time.monotonic() >= stop:
                raise StopSearch
            return float(obj@v),obj
        def callback(v):
            if time.monotonic() >= stop:
                raise StopSearch
            count[0] += 1
            if count[0] % 12 == 0:
                offer(unpack(v,mod))
        try:
            res = minimize(fun,z,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda v:calc(v)[0], 'jac':lambda v:calc(v)[1]},
                callback=callback,options={'maxiter':int(params['iterations']),'ftol':5e-13,'disp':False})
            z = res.x
        except StopSearch:
            z = last[0]
        offer(unpack(z,mod))
        return z

    def full_state(a):
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        return np.r_[a[:,0]-lo[0],a[:,1]-(lo[1]+hi[1])/2,a[:,2],hi[0]-lo[0]]

    def symmetric_state(a):
        options = []
        for swap in (False,True):
            if time.monotonic() >= deadline-.025:
                break
            q = a.copy()
            if swap:
                q[:,:2] = q[:,[1,0]]
            lo = np.min(q[:,:2]-q[:,2,None],axis=0)
            hi = np.max(q[:,:2]+q[:,2,None],axis=0)
            q[:,0] -= lo[0]
            q[:,1] -= (lo[1]+hi[1])/2
            singles = np.argsort(abs(q[:,1]))[:3]
            rem = np.setdiff1d(np.arange(21),singles)
            order = rem[np.argsort(q[rem,1])]
            neg,pos = order[:9],order[9:]
            cost = ((q[pos,None,0]-q[None,neg,0])**2+
                    (q[pos,None,1]+q[None,neg,1])**2+
                    (q[pos,None,2]-q[None,neg,2])**2)
            rr,cc = linear_sum_assignment(cost)
            neg = neg[cc]
            error = float(cost[rr,cc].sum()+np.sum(q[singles,1]**2))
            z = np.r_[q[singles,0],.5*(q[pos,0]+q[neg,0]),
                      .5*(q[pos,1]-q[neg,1]),q[singles,2],
                      .5*(q[pos,2]+q[neg,2]),hi[0]-lo[0]]
            options.append((error,z))
        return min(options,key=lambda t:t[0])[1] if options else None

    sym = make_model(True)
    full = make_model(False)
    z = symmetric_state(best_a)
    split = start + .70*budget_s*float(params['symmetric_fraction'])
    if z is not None and time.monotonic() < split-.03:
        z = optimize(z,sym,split)
    if time.monotonic() < deadline-.04:
        optimize(full_state(best_a),full,deadline-.02)
    # If both solves finish early, try a small coordinated quotient restart.
    for attempt in range(3):
        now = time.monotonic()
        if now >= deadline-.075:
            break
        z = symmetric_state(best_a)
        if z is None:
            break
        z[:12] += rng.normal(0,float(params['restart_size']),12)
        z[12:21] = np.maximum(0,z[12:21]+rng.normal(0,float(params['restart_size']),9))
        optimize(z,sym,deadline-.02)
    return best
