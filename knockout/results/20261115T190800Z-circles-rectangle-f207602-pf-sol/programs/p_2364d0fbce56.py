import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {
    'full_fraction': (0.80, 0.45, 0.95),
    'iterations': (320, 100, 600),
    'safety': (1e-14, 4e-15, 1e-10, 'log'),
    'perturbation': (0.009, 0.001, 0.06)
}
I, J = np.triu_indices(21, 1)

class Finished(Exception):
    pass

def valid(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return False
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return False
    x = list(map(Fraction, map(float, a[:, 0])))
    y = list(map(Fraction, map(float, a[:, 1])))
    r = list(map(Fraction, map(float, a[:, 2])))
    return (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r)) +
            max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))) <= 2

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + .70*budget_s
    rng = np.random.default_rng(seed)
    eps = float(params['safety'])
    best = None
    score = -np.inf
    original = None
    try:
        if init is not None:
            original = np.asarray(init['circles'], dtype=float) if 'circles' in init else np.asarray(init['state'], dtype=float).reshape(3,21).T
            if valid(original):
                best = init
                score = float(np.sum(original[:,2]))
    except (ValueError, TypeError, KeyError):
        original = None

    def offer(a):
        nonlocal best, score
        a = np.asarray(a, dtype=float).copy()
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
        ratio = min(1., float(np.min(d/(a[I,2]+a[J,2]))))
        if ratio <= 0:
            return
        a[:,2] *= ratio*(1-eps)
        lo = np.min(a[:,:2]-a[:,2,None], axis=0)
        hi = np.max(a[:,:2]+a[:,2,None], axis=0)
        scale = (2-8*eps)/np.sum(hi-lo)
        a[:,:2] = (a[:,:2]-(lo+hi)/2)*scale
        a[:,2] *= scale
        s = float(np.sum(a[:,2]))
        if s > score and valid(a):
            best = {'state': a.T.reshape(-1).tolist()}
            score = s

    if original is not None:
        offer(original)
    lattice = np.array([[.2*(i%3), .2*(i//3), .099999999999] for i in range(21)])
    offer(lattice)
    if best is None:
        return {'circles': lattice.tolist()}

    def best_array():
        return np.asarray(best['circles'], dtype=float).copy() if 'circles' in best else np.asarray(best['state'], dtype=float).reshape(3,21).T.copy()

    def model(sym):
        if sym:
            n = 34
            ids = np.r_[np.arange(12), np.arange(3,12)]
            xi = ids.copy()
            yi = np.where(ids >= 3, 12+ids-3, 0)
            ys = np.r_[np.r_[np.zeros(3), np.ones(9)], -np.ones(9)]
            ri = 21+ids
            pairs = [(i,j) for i in range(12) for j in range(i+1,12)]
            pairs += [(i,12+j-3) for i in range(3,12) for j in range(i,12)]
            pi,pj = np.asarray(pairs).T
            wall = np.arange(12)
        else:
            n = 64
            xi = np.arange(21)
            yi = 21+xi
            ys = np.ones(21)
            ri = 42+xi
            pi,pj = I,J
            wall = np.arange(21)
        m = len(pi)
        rows = np.arange(m)
        # Constant wall Jacobian, and objective.
        b = np.zeros((4*len(wall),n))
        k = np.arange(len(wall))
        h = len(wall)
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
        off = np.r_[np.zeros(2*h),np.ones(2*h)]
        obj = np.zeros(n)
        np.add.at(obj,ri,-1.)
        bounds = [(None,None)]*n
        for q in np.unique(ri):
            if time.monotonic() >= deadline:
                break
            bounds[int(q)] = (1e-8,.6)
        bounds[-1] = (.1,1.9)
        return n,xi,yi,ys,ri,pi,pj,rows,b,off,obj,bounds

    def optimize(z, mod, stop, iterations):
        n,xi,yi,ys,ri,pi,pj,rows,b,off,obj,bounds = mod
        rs0 = z[ri[pi]]+z[ri[pj]]
        weights = 1/np.maximum(2*rs0,.08)
        cache = [None,None,None]
        last = [z.copy()]
        count = [0]
        def unpack(v):
            return np.column_stack((v[xi],ys*v[yi],v[ri]))
        def calc(v):
            if time.monotonic() >= stop:
                raise Finished()
            if cache[0] is not None and np.array_equal(cache[0],v):
                return cache[1],cache[2]
            x = v[xi]; y = ys*v[yi]; r = v[ri]
            dx = x[pi]-x[pj]; dy = y[pi]-y[pj]; rr = r[pi]+r[pj]
            c = np.r_[(dx*dx+dy*dy-rr*rr)*weights,b@v+off]
            jac = np.zeros((len(rows)+len(b),n))
            tx = 2*weights*dx; ty = 2*weights*dy; tr = -2*weights*rr
            jac[rows,xi[pi]] = tx
            np.add.at(jac,(rows,xi[pj]),-tx)
            np.add.at(jac,(rows,yi[pi]),ty*ys[pi])
            np.add.at(jac,(rows,yi[pj]),-ty*ys[pj])
            np.add.at(jac,(rows,ri[pi]),tr)
            np.add.at(jac,(rows,ri[pj]),tr)
            jac[len(rows):] = b
            cache[:] = [v.copy(),c,jac]
            last[0] = v.copy()
            return c,jac
        def fun(v):
            if time.monotonic() >= stop:
                raise Finished()
            return float(obj@v),obj
        def callback(v):
            if time.monotonic() >= stop:
                raise Finished()
            count[0] += 1
            if count[0]%16 == 0:
                offer(unpack(v))
        try:
            result = minimize(fun,z,jac=True,method='SLSQP',bounds=bounds,
                              constraints={'type':'ineq','fun':lambda v:calc(v)[0],'jac':lambda v:calc(v)[1]},
                              callback=callback,options={'maxiter':iterations,'ftol':8e-14,'disp':False})
            z = result.x
        except Finished:
            z = last[0]
        offer(unpack(z))
        return z

    a = original
    if a is None or a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
        a = best_array()
    lo = np.min(a[:,:2]-a[:,2,None],axis=0)
    hi = np.max(a[:,:2]+a[:,2,None],axis=0)
    p = a[:,:2]-lo
    p[:,1] -= .5*(hi[1]-lo[1])
    z = np.r_[p[:,0],p[:,1],a[:,2],hi[0]-lo[0]]
    full = model(False)
    optimize(z,full,min(deadline-.025,start+.70*budget_s*float(params['full_fraction'])),int(params['iterations']))
    if time.monotonic() >= deadline-.035:
        return best

    a = best_array()
    choices = []
    for swap in (False,True):
        if time.monotonic() >= deadline-.025:
            return best
        b = a.copy()
        if swap:
            b[:,:2] = b[:,[1,0]]
        lo = np.min(b[:,:2]-b[:,2,None],axis=0)
        hi = np.max(b[:,:2]+b[:,2,None],axis=0)
        b[:,0] -= lo[0]
        b[:,1] -= (lo[1]+hi[1])/2
        reflected = b.copy(); reflected[:,1] *= -1
        cost = np.sum((b[:,None,:]-reflected[None,:,:])**2,axis=2)
        rows,cols = linear_sum_assignment(cost)
        choices.append((float(cost[rows,cols].sum()),b,hi[0]-lo[0]))
    _,a,width = min(choices,key=lambda t:t[0])
    singles = np.argsort(np.abs(a[:,1]))[:3]
    remaining = np.setdiff1d(np.arange(21),singles)
    order = remaining[np.argsort(a[remaining,1])]
    negative,positive = order[:9],order[9:]
    cost = ((a[positive,None,0]-a[None,negative,0])**2 +
            (a[positive,None,1]+a[None,negative,1])**2 +
            (a[positive,None,2]-a[None,negative,2])**2)
    _,cols = linear_sum_assignment(cost)
    negative = negative[cols]
    base = np.r_[a[singles,0],.5*(a[positive,0]+a[negative,0]),
                 .5*(a[positive,1]-a[negative,1]),a[singles,2],
                 .5*(a[positive,2]+a[negative,2]),width]
    sym = model(True)
    base = optimize(base,sym,deadline-.018,int(params['iterations']))
    for attempt in range(4):
        now = time.monotonic()
        if now >= deadline-.04:
            break
        trial = base.copy()
        trial[:12] += rng.normal(0,float(params['perturbation']),12)
        trial[12:21] = np.abs(trial[12:21]+rng.normal(0,float(params['perturbation']),9))
        candidate = optimize(trial,sym,min(deadline-.012,now+.65*(deadline-now)),100)
        if sym[10]@candidate < sym[10]@base:
            base = candidate
    return best
