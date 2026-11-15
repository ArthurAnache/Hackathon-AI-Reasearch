import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'iterations': (90, 20, 180),
    'attempts': (8, 2, 20),
    'pole_clearance': (0.004, 0.0001, 0.05, 'log'),
}
N = 21
I, J = np.triu_indices(N, 1)

class StopSearch(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2] + a[J, 2]):
        return -np.inf
    x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
    wh = (max(a+b for a,b in zip(x,r))-min(a-b for a,b in zip(x,r))
          +max(a+b for a,b in zip(y,r))-min(a-b for a,b in zip(y,r)))
    return float(np.sum(a[:, 2])) if wh <= 2 else -np.inf

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.69*budget_s
    rng = np.random.default_rng(seed)
    best = init
    bestscore = -np.inf
    if init is not None:
        original = np.asarray(init['circles'], dtype=float)
        bestscore = exact_score(original)
    else:
        original = np.array([[(j-1)*0.28, (i-3)*0.23, 0.10] for i in range(7) for j in range(3)])
    a = original.copy()
    lo = np.min(a[:, :2]-a[:, 2,None], axis=0)
    hi = np.max(a[:, :2]+a[:, 2,None], axis=0)
    a[:, :2] -= (hi+lo)/2
    # Put the reflection axis on y=0 when a reflection is present.
    errors = []
    for axis in (0,1):
        refl = a.copy()
        refl[:,axis] *= -1
        errors.append(np.max(np.min(np.sum((refl[:,None,:]-a[None,:,:])**2,axis=2),axis=1)))
    if errors[0] < errors[1]:
        a[:, [0,1]] = a[:, [1,0]]
    symmetric = min(errors) < 1e-12
    groups = []
    remaining = set(range(N))
    while remaining:
        if time.monotonic() >= end:
            return best if best is not None else {'circles': original.tolist()}
        i = min(remaining)
        remaining.remove(i)
        if symmetric and abs(a[i,1]) > 1e-6:
            target = a[i].copy(); target[1] *= -1
            choices = sorted(remaining)
            if choices:
                j = min(choices, key=lambda k: float(np.sum((a[k]-target)**2)))
                remaining.remove(j)
                groups.append([i,j])
            else:
                groups.append([i])
        else:
            groups.append([i])
    # Linear map from independent symmetric variables to x,y,r,W.
    cols = []
    radius_columns = []
    for group in groups:
        c = np.zeros(64); c[group] = 1; cols.append(c)
        if len(group) == 2:
            c = np.zeros(64)
            c[21+group[0]] = 1
            c[21+group[1]] = -1
            cols.append(c)
        elif not symmetric:
            c = np.zeros(64); c[21+group[0]] = 1; cols.append(c)
        c = np.zeros(64); c[42+np.array(group)] = 1
        radius_columns.append(len(cols)); cols.append(c)
    c = np.zeros(64); c[63] = 1; cols.append(c)
    T = np.stack(cols,axis=1)
    norms = np.sum(T*T,axis=0)
    objective = -np.sum(T[42:63],axis=0)
    bounds = [(None,None)]*T.shape[1]
    for k in radius_columns:
        bounds[k] = (1e-7,0.7)
    bounds[-1] = (0.1,1.9)
    last_u = None
    cache_u = None
    cache = None

    def evaluate(u):
        nonlocal cache_u, cache
        if time.monotonic() >= end:
            raise StopSearch()
        if cache_u is not None and np.array_equal(u,cache_u):
            return cache
        z = T@u
        x,y,r,w = z[:21],z[21:42],z[42:63],z[63]
        dx,dy = x[I]-x[J],y[I]-y[J]
        d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-15)
        g = np.concatenate((d-r[I]-r[J], w/2-x-r,w/2+x-r,1-w/2-y-r,1-w/2+y-r))
        jac = np.zeros((len(g),64))
        rows = np.arange(len(I))
        jac[rows,I] = dx/d; jac[rows,J] = -dx/d
        jac[rows,21+I] = dy/d; jac[rows,21+J] = -dy/d
        jac[rows,42+I] = -1; jac[rows,42+J] = -1
        for k in range(4):
            rr = len(I)+21*k+np.arange(21)
            jac[rr,42+np.arange(21)] = -1
            jac[rr,(0 if k<2 else 21)+np.arange(21)] = -1 if k%2==0 else 1
            jac[rr,63] = 0.5 if k<2 else -0.5
        cache_u = u.copy(); cache = (g,jac@T)
        return cache

    def consider(u):
        nonlocal best,bestscore
        if time.monotonic() >= end:
            return
        z = T@u
        p = np.stack((z[:21],z[21:42]),axis=1)
        r = np.maximum(z[42:63],1e-12)
        d = np.sqrt(np.sum((p[I]-p[J])**2,axis=1))
        fac = min(1.0,float(np.min(d/(r[I]+r[J]))))
        if fac <= 0:
            return
        r = r*fac*(1-4e-15)
        lo = np.min(p-r[:,None],axis=0)
        hi = np.max(p+r[:,None],axis=0)
        scale = 2/float(np.sum(hi-lo))*(1-4e-15)
        out = np.column_stack(((p-(lo+hi)/2)*scale,r*scale))
        score = exact_score(out)
        if score > bestscore:
            bestscore = score
            best = {'circles':out.tolist()}

    # Axis poles in internal voids exchange the outer face; external poles
    # give gentler, finite conformal deformations.
    left = float(np.min(a[:,0]-a[:,2]))
    right = float(np.max(a[:,0]+a[:,2]))
    poles = []
    for k in range(100):
        if time.monotonic() >= end:
            break
        cx = float(rng.uniform(left-0.3,right+0.3))
        clearance = np.sqrt((a[:,0]-cx)**2+a[:,1]**2)-a[:,2]
        if np.min(clearance) > float(params['pole_clearance']):
            poles.append(cx)
    poles = sorted(poles,key=lambda x: (not(left<x<right),float(rng.random())))
    poles += [left-0.15,right+0.15,left-0.5,right+0.5]
    for attempt,cx in enumerate(poles[:int(params['attempts'])]):
        if time.monotonic() >= end-0.035:
            break
        delta = a[:,:2]-np.array([cx,0.0])
        denom = np.sum(delta*delta,axis=1)-a[:,2]**2
        if np.min(denom) <= 0:
            continue
        b = np.column_stack((delta/denom[:,None],a[:,2]/denom))
        lo = np.min(b[:,:2]-b[:,2,None],axis=0)
        hi = np.max(b[:,:2]+b[:,2,None],axis=0)
        scale = 2/float(np.sum(hi-lo))
        b[:,:2] = (b[:,:2]-(hi+lo)/2)*scale
        b[:,2] *= scale
        z = np.r_[b[:,0],b[:,1],b[:,2],(hi[0]-lo[0])*scale]
        u = (T.T@z)/norms
        for k in radius_columns:
            u[k] = max(1e-7,min(0.7,u[k]))
        u[-1] = np.clip(u[-1],0.1,1.9)
        last_u = u.copy()
        local_end = min(end,time.monotonic()+0.32*budget_s)
        def callback(v):
            nonlocal last_u
            last_u = v.copy()
            if time.monotonic() >= local_end:
                raise StopSearch()
        try:
            result = minimize(lambda v: float(objective@v),u,jac=lambda v: objective,
                bounds=bounds,method='SLSQP',
                constraints={'type':'ineq','fun':lambda v:evaluate(v)[0], 'jac':lambda v:evaluate(v)[1]},
                callback=callback,options={'maxiter':int(params['iterations']),'ftol':2e-12})
            last_u = result.x
        except StopSearch:
            pass
        consider(last_u)
    if best is None:
        p = original[:,:2].copy()
        r = np.full(21,1e-5)
        d = np.sqrt(np.sum((p[I]-p[J])**2,axis=1))
        r *= min(1.0,float(np.min(d))/(2e-5)*0.99)
        lo=np.min(p-r[:,None],axis=0); hi=np.max(p+r[:,None],axis=0)
        s=1.999999/np.sum(hi-lo)
        best={'circles':np.column_stack(((p-(hi+lo)/2)*s,r*s)).tolist()}
    return best
