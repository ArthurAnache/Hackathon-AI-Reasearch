import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'attempts': (4, 1, 12),
    'slice': (0.31, 0.1, 0.6),
    'radius_factor': (0.94, 0.7, 1.0),
    'maxiter': (100, 20, 250)
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
K = np.arange(N)

class StopSearch(Exception):
    pass

def exact_score(c):
    a = np.asarray(c['circles'], dtype=float)
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2] + a[J, 2]):
        return -np.inf
    x = list(map(lambda v: Fraction(float(v)), a[:, 0]))
    y = list(map(lambda v: Fraction(float(v)), a[:, 1]))
    r = list(map(lambda v: Fraction(float(v)), a[:, 2]))
    perimeter = (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
                 + max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)))
    if perimeter > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))

def repair(z):
    p = np.column_stack((z[:N], z[N:2*N]))
    r = np.maximum(z[2*N:3*N], 1e-12).copy()
    if not np.isfinite(z).all():
        return None
    d = np.sqrt((p[I, 0]-p[J, 0])**2+(p[I, 1]-p[J, 1])**2)
    factor = min(1.0, float(np.min(d/(r[I]+r[J]))))
    if factor <= 0:
        return None
    r *= factor*(1-8e-15)
    lo = np.min(p-r[:, None], axis=0)
    hi = np.max(p+r[:, None], axis=0)
    scale = 2.0/float(np.sum(hi-lo))*(1-8e-15)
    p = (p-lo)*scale
    r *= scale
    return {'circles': np.column_stack((p,r)).tolist()}

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a = np.empty((N,3))
        a[:,0] = (np.arange(N)%5+0.5)/5
        a[:,1] = (np.arange(N)//5+0.5)/5
        a[:,2] = 0.09
        init = {'circles': a.tolist()}
    best = init
    best_score = exact_score(init)
    a = np.asarray(init['circles'], dtype=float)
    lo = np.min(a[:,:2]-a[:,2,None], axis=0)
    hi = np.max(a[:,:2]+a[:,2,None], axis=0)
    scale = 2/np.sum(hi-lo)
    p0 = (a[:,:2]-lo)*scale
    r0 = a[:,2]*scale
    sides = (hi-lo)*scale
    wide = int(np.argmax(sides))
    grad = np.zeros(64)
    grad[42:63] = -1
    linear = np.zeros((4*N,64))
    linear[K,K] = 1
    linear[K,42+K] = -1
    linear[N+K,K] = -1
    linear[N+K,42+K] = -1
    linear[N+K,63] = 1
    linear[2*N+K,N+K] = 1
    linear[2*N+K,42+K] = -1
    linear[3*N+K,N+K] = -1
    linear[3*N+K,42+K] = -1
    linear[3*N+K,63] = -1
    offset = np.zeros(4*N)
    offset[3*N:] = 2
    row = np.arange(M)
    bounds = [(None,None)]*42+[(1e-8,0.5)]*21+[(0.15,1.85)]

    def consider(z):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        c = repair(z)
        if c is not None:
            score = exact_score(c)
            if score > best_score:
                best, best_score = c, score

    for attempt in range(int(params['attempts'])):
        if time.monotonic() >= deadline-0.025:
            break
        local_end = min(deadline-0.012, time.monotonic()+float(params['slice']))
        p = p0.copy()
        axes = [wide] if attempt == 0 else ([1-wide] if attempt == 1 else [0,1])
        for axis in axes:
            if time.monotonic() >= local_end:
                break
            vals = np.sort(p0[:,axis])
            gaps = np.diff(vals)
            mids = (vals[1:]+vals[:-1])/2
            eligible = (mids > 0.22*sides[axis]) & (mids < 0.78*sides[axis])
            merit = np.where(eligible, gaps, -1.0)
            if attempt < 3:
                cut = mids[int(np.argmax(merit))]
            else:
                cut = float(rng.uniform(0.25,0.75))*sides[axis]
            p[:,axis] = np.mod(p[:,axis]-cut, sides[axis])
        z = np.r_[p[:,0],p[:,1],r0*float(params['radius_factor']),sides[0]]
        last = [z.copy()]
        count = [0]
        cache_z = [None]
        cache_value = [None]
        cache_jac = [None]

        def check():
            if time.monotonic() >= local_end:
                raise StopSearch()

        def objective(v):
            check()
            return -float(np.sum(v[42:63]))

        def objective_jac(v):
            check()
            return grad

        def prepare(v):
            check()
            if cache_z[0] is not None and np.array_equal(v,cache_z[0]):
                return
            dx = v[I]-v[J]
            dy = v[N+I]-v[N+J]
            d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-15)
            values = np.r_[d-v[42+I]-v[42+J],linear@v+offset]
            jac = np.zeros((M+4*N,64))
            jac[row,I] = dx/d
            jac[row,J] = -dx/d
            jac[row,N+I] = dy/d
            jac[row,N+J] = -dy/d
            jac[row,42+I] = -1
            jac[row,42+J] = -1
            jac[M:] = linear
            cache_z[0] = v.copy()
            cache_value[0] = values
            cache_jac[0] = jac

        def constraints(v):
            prepare(v)
            return cache_value[0]

        def jacobian(v):
            prepare(v)
            return cache_jac[0]

        def callback(v):
            check()
            last[0] = v.copy()
            count[0] += 1
            if count[0] % 8 == 0:
                consider(v)

        try:
            result = minimize(objective,z,jac=objective_jac,method='SLSQP',
                              bounds=bounds,
                              constraints={'type':'ineq','fun':constraints,'jac':jacobian},
                              callback=callback,
                              options={'maxiter':int(params['maxiter']),'ftol':2e-12,'disp':False})
            last[0] = result.x
        except StopSearch:
            pass
        consider(last[0])
    return best
