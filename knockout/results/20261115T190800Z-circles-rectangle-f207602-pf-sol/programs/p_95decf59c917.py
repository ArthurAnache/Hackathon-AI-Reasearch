import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {
    'search_fraction': (0.70, 0.60, 0.74),
    'restart_seconds': (0.24, 0.10, 0.45),
    'perturbation': (0.025, 0.002, 0.12, 'log'),
    'iterations': (90, 30, 180),
    'safety': (2e-15, 5e-16, 1e-12, 'log')
}
I, J = np.triu_indices(21, 1)

class Stop(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2+(a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return -np.inf
    xs = [Fraction(float(v)) for v in a[:, 0]]
    ys = [Fraction(float(v)) for v in a[:, 1]]
    rs = [Fraction(float(v)) for v in a[:, 2]]
    p = max(x+r for x,r in zip(xs,rs))-min(x-r for x,r in zip(xs,rs))
    p += max(y+r for y,r in zip(ys,rs))-min(y-r for y,r in zip(ys,rs))
    return float(np.sum(a[:, 2])) if p <= 2 else -np.inf

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + params['search_fraction']*budget_s
    rng = np.random.default_rng(seed)
    best = init
    bestscore = -np.inf
    best_a = None
    eps = params['safety']
    if init is not None:
        a = (np.asarray(init['circles'], dtype=float) if 'circles' in init else
             np.asarray(init['state'], dtype=float).reshape(3,21).T.copy())
        bestscore = exact_score(a)
        best_a = a.copy()

    def offer(a):
        nonlocal best, bestscore, best_a
        if time.monotonic() >= deadline:
            return
        a = np.array(a, dtype=float, copy=True)
        if not np.isfinite(a).all() or np.min(a[:,2]) <= 0:
            return
        ds = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
        q = min(1., float(np.min(ds/(a[I,2]+a[J,2]))))
        if q <= 0:
            return
        a[:,2] *= q*(1-eps)
        lo = np.min(a[:,:2]-a[:,2,None], axis=0)
        hi = np.max(a[:,:2]+a[:,2,None], axis=0)
        scale = (2-8*eps)/float(np.sum(hi-lo))
        a[:,:2] = (a[:,:2]-(hi+lo)/2)*scale
        a[:,2] *= scale
        score = exact_score(a)
        if score > bestscore:
            bestscore = score
            best_a = a.copy()
            best = {'state': a.T.reshape(-1).tolist()}

    if best_a is None:
        a = np.array([[2*(i%7),2*(i//7),1.] for i in range(21)], float)
        offer(a)
    original = best_a.copy()
    # Select the closer of the two possible rectangle reflection axes.
    choices = []
    for swap in (False, True):
        if time.monotonic() >= deadline:
            return best
        a = original.copy()
        if swap:
            a[:,[0,1]] = a[:,[1,0]]
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        a[:,:2] -= (lo+hi)/2
        axis = np.argsort(np.abs(a[:,1]))[:3]
        rest = np.setdiff1d(np.arange(21),axis)
        pos = rest[np.argsort(a[rest,1])[9:]]
        neg = rest[np.argsort(a[rest,1])[:9]]
        cost = ((a[pos,None,0]-a[None,neg,0])**2 +
                (a[pos,None,1]+a[None,neg,1])**2 +
                (a[pos,None,2]-a[None,neg,2])**2)
        ii,jj = linear_sum_assignment(cost)
        pos,neg = pos[ii],neg[jj]
        err = float(cost[ii,jj].sum()+np.sum(a[axis,1]**2))
        choices.append((err,a,axis,pos,neg,float(hi[0]-lo[0])))
    _,a,axis,pos,neg,width = min(choices,key=lambda t:t[0])
    order = np.r_[axis,pos,neg]
    a = a[order]
    a[:,0] += width/2
    # Reflection quotient: twelve x coordinates, nine positive y values,
    # twelve radii, and rectangle width.
    n = 34
    X = np.zeros((21,n)); Y = X.copy(); R = X.copy()
    ids = np.r_[np.arange(12),np.arange(3,12)]
    X[np.arange(21),ids] = 1
    k = np.arange(3,21)
    Y[k,12+ids[k]-3] = np.r_[np.ones(9),-np.ones(9)]
    R[np.arange(21),21+ids] = 1
    W = np.zeros(n); W[-1] = 1
    pairs = [(i,j) for i in range(12) for j in range(i+1,12)]
    pairs += [(i,12+j-3) for i in range(3,12) for j in range(i,12)]
    pi,pj = np.array(pairs).T
    DX,DY,RS = X[pi]-X[pj],Y[pi]-Y[pj],R[pi]+R[pj]
    B = np.concatenate((X[:12]-R[:12],W-X[:12]-R[:12],
                        -.5*W-Y[:12]-R[:12],-.5*W+Y[:12]-R[:12]))
    offset = np.r_[np.zeros(24),np.ones(24)]
    obj = -R.sum(axis=0)
    z = np.r_[np.r_[a[:3,0],(a[3:12,0]+a[12:,0])/2],
              (a[3:12,1]-a[12:,1])/2,
              np.r_[a[:3,2],(a[3:12,2]+a[12:,2])/2],width]
    weights = 1/np.maximum(2*(RS@z),.08)

    def unpack(v):
        return np.column_stack((X@v,Y@v,R@v))

    def calc(v):
        dx,dy,rs = DX@v,DY@v,RS@v
        c = np.r_[(dx*dx+dy*dy-rs*rs)*weights,B@v+offset]
        jac = np.concatenate((2*weights[:,None]*(dx[:,None]*DX+dy[:,None]*DY-rs[:,None]*RS),B))
        return c,jac

    def polish(v,stop):
        v = v.copy()
        c,jac = calc(v)
        active = np.flatnonzero(c < 3e-7)
        if len(active) < n:
            return v
        for _ in range(7):
            if time.monotonic() >= stop:
                break
            c,jac = calc(v)
            step = np.linalg.lstsq(jac[active],-c[active],rcond=1e-11)[0]
            if np.max(np.abs(step)) > .003:
                break
            v += step
            offer(unpack(v))
            if np.max(np.abs(step)) < 2e-15:
                break
        return v

    z = polish(z,min(deadline,start+.18*budget_s))
    offer(unpack(z))
    bounds = [(None,None)]*21+[(1e-6,.5)]*12+[(.2,1.8)]
    champion = z.copy()
    champval = -np.inf
    attempt = 0
    while time.monotonic() < deadline-.025:
        stop = min(deadline-.012,time.monotonic()+params['restart_seconds'])
        v = champion.copy()
        if attempt:
            amplitude = params['perturbation']*[.35,1.,2.,.65,3.][(attempt-1)%5]
            group = rng.choice(12, size=3, replace=False)
            for g in group:
                if time.monotonic() >= stop:
                    break
                v[g] += rng.normal()*amplitude
                v[21+g] *= np.exp(rng.normal()*.20)
                if g >= 3:
                    v[12+g-3] += rng.normal()*amplitude
            v[12:21] = np.maximum(v[12:21],.015)
            v[33] += rng.normal()*amplitude
            v[33] = np.clip(v[33],.3,1.7)
        accepted = [v.copy()]
        cache = [None,None,None]
        def cached(q):
            if time.monotonic() >= stop:
                raise Stop
            if cache[0] is None or not np.array_equal(cache[0],q):
                c,j = calc(q)
                cache[:] = [q.copy(),c,j]
            return cache[1],cache[2]
        def fun(q):
            if time.monotonic() >= stop:
                raise Stop
            return float(obj@q),obj
        def callback(q):
            if time.monotonic() >= stop:
                raise Stop
            accepted[0] = q.copy()
        try:
            out = minimize(fun,v,method='SLSQP',jac=True,bounds=bounds,
                constraints={'type':'ineq','fun':lambda q:cached(q)[0],
                             'jac':lambda q:cached(q)[1]},callback=callback,
                options={'ftol':2e-14,'maxiter':int(params['iterations']),'disp':False})
            v = out.x
        except Stop:
            v = accepted[0]
        if time.monotonic() < deadline-.01:
            offer(unpack(v))
            v = polish(v,min(deadline-.006,time.monotonic()+.025))
            c,_ = calc(v)
            value = float(-obj@v)
            if np.min(c) > -1e-8 and value > champval:
                champion,champval = v.copy(),value
        attempt += 1
    return best
