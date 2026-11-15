import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'first_iterations': (150, 50, 300),
    'restart_iterations': (65, 20, 150),
    'restart_noise': (0.018, 0.001, 0.12, 'log'),
    'ftol': (2e-11, 1e-13, 1e-6, 'log')
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
K = np.arange(N)

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
    perimeter = (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
                 + max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)))
    if perimeter > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    best = None
    best_score = -np.inf
    best_z = None
    if init is not None:
        try:
            a = np.asarray(init['circles'], dtype=float).copy()
            sc = exact_score(a)
            if np.isfinite(sc):
                best, best_score = init, sc
        except (ValueError, KeyError, TypeError):
            a = None
    else:
        a = None
    if a is None or a.shape != (21,3) or not np.isfinite(a).all():
        a = np.array([[(i % 5 + .5)/5, (i//5 + .5)/5, .07] for i in range(N)])
    a[:,2] = np.maximum(np.abs(a[:,2]), 1e-5)
    lo = (a[:,:2]-a[:,2,None]).min(axis=0)
    sides = (a[:,:2]+a[:,2,None]).max(axis=0)-lo
    scale = 2/max(float(sides.sum()), 1e-8)
    p = (a[:,:2]-lo)*scale
    r = a[:,2]*scale
    z = np.r_[p[:,0], p[:,1], r, sides[0]*scale]

    def check_time():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def retain(v):
        nonlocal best, best_score, best_z
        if time.monotonic() >= deadline:
            return
        p = np.column_stack((v[:N],v[N:2*N]))
        r = np.maximum(v[2*N:3*N], 1e-10).copy()
        if not np.isfinite(p).all() or not np.isfinite(r).all():
            return
        d = np.sqrt(((p[I]-p[J])**2).sum(axis=1))
        factor = min(1.0, float(np.min(d/(r[I]+r[J]))))
        if factor <= 0:
            return
        r *= factor*(1-3e-14)
        low = (p-r[:,None]).min(axis=0)
        side = (p+r[:,None]).max(axis=0)-low
        s = (2-8e-14)/float(side.sum())
        rr = r*s
        estimate = float(rr.sum())
        if estimate <= best_score:
            return
        pp = (p-low)*s
        cand = np.column_stack((pp,rr))
        score = exact_score(cand)
        if score > best_score:
            best_score = score
            best = {'circles': cand.tolist()}
            best_z = np.r_[pp[:,0],pp[:,1],rr,side[0]*s]

    retain(z)
    gradient = np.r_[np.zeros(2*N),-np.ones(N),0.]
    wall_jac = np.zeros((4*N,3*N+1))
    wall_jac[K,K] = 1
    wall_jac[K,2*N+K] = -1
    wall_jac[N+K,K] = -1
    wall_jac[N+K,2*N+K] = -1
    wall_jac[N+K,-1] = 1
    wall_jac[2*N+K,N+K] = 1
    wall_jac[2*N+K,2*N+K] = -1
    wall_jac[3*N+K,N+K] = -1
    wall_jac[3*N+K,2*N+K] = -1
    wall_jac[3*N+K,-1] = -1
    pair_jac = np.zeros((M,3*N+1))
    pair_jac[np.arange(M),2*N+I] = -1
    pair_jac[np.arange(M),2*N+J] = -1
    cache_v = None
    cache_c = None
    cache_j = None

    def calculate(v):
        nonlocal cache_v, cache_c, cache_j
        check_time()
        if cache_v is not None and np.array_equal(v,cache_v):
            return
        x,y,r,w = v[:N],v[N:2*N],v[2*N:3*N],v[-1]
        dx,dy = x[I]-x[J],y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy)
        safe = np.maximum(d,1e-15)
        jj = pair_jac.copy()
        rows = np.arange(M)
        jj[rows,I] = dx/safe
        jj[rows,J] = -dx/safe
        jj[rows,N+I] = dy/safe
        jj[rows,N+J] = -dy/safe
        cache_c = np.r_[d-r[I]-r[J],x-r,w-x-r,y-r,2-w-y-r]
        cache_j = np.vstack((jj,wall_jac))
        cache_v = v.copy()

    def con(v):
        calculate(v)
        return cache_c

    def jac(v):
        calculate(v)
        return cache_j

    def objective(v):
        check_time()
        return -float(v[2*N:3*N].sum())

    def callback(v):
        check_time()
        retain(v)

    bounds = [(None,None)]*(2*N)+[(1e-7,None)]*N+[(.15,1.85)]
    attempt = 0
    try:
        while time.monotonic() < deadline:
            if attempt:
                if best_z is None:
                    break
                z = best_z.copy()
                amplitude = float(params['restart_noise'])*(1.65**((attempt-1)%4))
                count = min(N, 4+3*(attempt%4))
                chosen = rng.choice(N,count,replace=False)
                z[chosen] += rng.normal(0,amplitude,count)
                z[N+chosen] += rng.normal(0,amplitude,count)
                z[2*N:3*N] *= .97
            result = minimize(objective,z,jac=lambda v: gradient,
                constraints={'type':'ineq','fun':con,'jac':jac},
                bounds=bounds,method='SLSQP',callback=callback,
                options={'maxiter':int(params['first_iterations'] if attempt==0 else params['restart_iterations']),
                         'ftol':float(params['ftol']), 'disp':False})
            retain(result.x)
            attempt += 1
    except StopSearch:
        pass
    if best is None:
        # Universally valid emergency construction.
        safe = np.array([[(i%7)*.12,(i//7)*.12,.03] for i in range(N)])
        best = {'circles':safe.tolist()}
    return best
