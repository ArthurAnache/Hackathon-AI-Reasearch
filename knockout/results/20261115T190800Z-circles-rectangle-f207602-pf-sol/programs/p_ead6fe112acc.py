import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    "entropy": (0.65, 0.05, 2.0),
    "intermediate": (0.10, 0.0, 0.5),
    "iterations": (100, 20, 250),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class StopSearch(Exception):
    pass

def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -float('inf')
    d = np.sqrt((a[I, 0] - a[J, 0])**2 + (a[I, 1] - a[J, 1])**2)
    if np.any(d < a[I, 2] + a[J, 2]):
        return -float('inf')
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    p = (max(u+v for u,v in zip(x,r)) - min(u-v for u,v in zip(x,r))
         + max(u+v for u,v in zip(y,r)) - min(u-v for u,v in zip(y,r)))
    if p > 2:
        return -float('inf')
    return float(np.sum(a[:, 2]))

def make_candidate(z, deadline):
    p = np.column_stack((z[:N], z[N:2*N]))
    r = np.maximum(z[2*N:3*N], 1e-14).copy()
    if not np.isfinite(p).all() or not np.isfinite(r).all():
        return None
    d = np.sqrt(np.sum((p[I]-p[J])**2, axis=1))
    fac = min(1.0, float(np.min(d/(r[I]+r[J]))))
    if fac <= 0:
        return None
    r *= fac * (1.0-3e-14)
    lo = np.min(p-r[:,None], axis=0)
    hi = np.max(p+r[:,None], axis=0)
    s = 2.0 / np.sum(hi-lo)
    p = (p-lo)*s
    r *= s
    for k in range(8):
        if time.monotonic() >= deadline:
            return None
        a = np.column_stack((p,r))
        if exact_score(a) > -float('inf'):
            return a
        r *= 1.0-2e-13*(2**k)
    return None

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    stop = start + 0.70*budget_s
    search_end = start + 0.65*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -float('inf')
    if init is not None:
        a = np.asarray(init['circles'], dtype=float)
        best_score = exact_score(a)
    else:
        a = np.zeros((N,3))
        for k in range(N):
            if time.monotonic() >= stop:
                break
            a[k] = [(k%5+0.5)*0.2, (k//5+0.5)*0.2, 0.08]
    r = np.maximum(a[:,2], 1e-8)
    total = float(np.sum(r))
    lo = np.min(a[:,:2]-r[:,None],axis=0)
    hi = np.max(a[:,:2]+r[:,None],axis=0)
    p = (a[:,:2]-lo)/total
    r = r/total
    sides = (hi-lo)/total
    z = np.r_[p[:,0],p[:,1],r,sides]
    dim = 3*N+2
    base = np.zeros((M+4*N,dim))
    ix = np.arange(N)
    rows = np.arange(M)
    base[rows,2*N+I] = -1
    base[rows,2*N+J] = -1
    q = M+ix
    base[q,ix] = 1
    base[q,2*N+ix] = -1
    q = M+N+ix
    base[q,ix] = -1
    base[q,2*N+ix] = -1
    base[q,3*N] = 1
    q = M+2*N+ix
    base[q,N+ix] = 1
    base[q,2*N+ix] = -1
    q = M+3*N+ix
    base[q,N+ix] = -1
    base[q,2*N+ix] = -1
    base[q,3*N+1] = 1
    eqjac = np.zeros(dim)
    eqjac[2*N:3*N] = 1
    snapshots = [z.copy()]
    phase_end = search_end
    last = z.copy()

    def check():
        if time.monotonic() >= min(phase_end,search_end):
            raise StopSearch()

    def cons(v):
        check()
        x,y,r = v[:N],v[N:2*N],v[2*N:3*N]
        dx,dy = x[I]-x[J],y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        return np.r_[d-r[I]-r[J],x-r,v[3*N]-x-r,y-r,v[3*N+1]-y-r]

    def jac(v):
        check()
        dx = v[I]-v[J]
        dy = v[N+I]-v[N+J]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        out = base.copy()
        out[rows,I] = dx/d
        out[rows,J] = -dx/d
        out[rows,N+I] = dy/d
        out[rows,N+J] = -dy/d
        return out

    def callback(v):
        nonlocal last
        last = v.copy()
        check()

    constraints = [
        {'type':'ineq','fun':cons,'jac':jac},
        {'type':'eq','fun':lambda v: np.sum(v[2*N:3*N])-1.0,
         'jac':lambda v:eqjac},
    ]
    bounds = [(None,None)]*(2*N)+[(1e-5,0.5)]*N+[(0.03,2.0)]*2
    stages = [(float(params['entropy']),0.27),
              (float(params['intermediate']),0.44),(0.0,0.65)]
    for tau, fraction in stages:
        if time.monotonic() >= search_end:
            break
        phase_end = start + fraction*budget_s
        if time.monotonic() >= phase_end:
            continue
        def objective(v):
            check()
            rr = np.maximum(v[2*N:3*N],1e-15)
            f = v[-2]+v[-1]+tau*np.sum(rr*np.log(rr))
            g = np.zeros(dim)
            g[-2:] = 1
            g[2*N:3*N] = tau*(np.log(rr)+1)
            return f,g
        last = z.copy()
        try:
            result = minimize(objective,z,jac=True,method='SLSQP',bounds=bounds,
                              constraints=constraints,callback=callback,
                              options={'maxiter':int(params['iterations']),
                                       'ftol':2e-13,'disp':False})
            z = result.x.copy()
        except StopSearch:
            z = last.copy()
        snapshots.append(z.copy())
    for v in reversed(snapshots):
        if time.monotonic() >= stop:
            break
        candidate = make_candidate(v,stop)
        if candidate is not None:
            score = exact_score(candidate)
            if score > best_score:
                best_score = score
                best = {'circles':candidate.tolist()}
    if best is None:
        fallback = np.array([[(k+0.5)/21,0.5,0.01] for k in range(N)])
        best = {'circles':fallback.tolist()}
    return best
