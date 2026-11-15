import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'maxiter': (110, 30, 250),
    'restarts': (3, 1, 6),
    'jitter': (0.018, 0.0, 0.08),
    'ftol': (2e-10, 1e-12, 1e-6, 'log'),
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
    r = a[:, 2]
    d = np.sqrt((a[I, 0] - a[J, 0])**2 + (a[I, 1] - a[J, 1])**2)
    if np.any(d < r[I] + r[J]):
        return -np.inf
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    q = [Fraction(float(v)) for v in r]
    w = max(u+v for u,v in zip(x,q)) - min(u-v for u,v in zip(x,q))
    h = max(u+v for u,v in zip(y,q)) - min(u-v for u,v in zip(y,q))
    if w+h > 2:
        return -np.inf
    return float(np.sum(r))

def repair(a):
    a = np.asarray(a, dtype=float).copy()
    if a.shape != (21,3) or not np.isfinite(a).all():
        return None
    a[:, :2] -= a[:, :2].mean(axis=0)
    r = np.maximum(a[:, 2], 1e-10)
    d = np.sqrt((a[I,0]-a[J,0])**2 + (a[I,1]-a[J,1])**2)
    if np.min(d) <= 0:
        return None
    factor = min(1.0, float(np.min(d/(r[I]+r[J]))))
    a[:, 2] = r * factor * (1.0-3e-12)
    lo = np.min(a[:, :2]-a[:, 2,None], axis=0)
    hi = np.max(a[:, :2]+a[:, 2,None], axis=0)
    perimeter = float(np.sum(hi-lo))
    if perimeter <= 0:
        return None
    a *= (2.0/perimeter)*(1.0-3e-12)
    return a

def pack(a):
    lo = np.min(a[:,:2]-a[:,2,None], axis=0)
    hi = np.max(a[:,:2]+a[:,2,None], axis=0)
    p = a[:,:2]-lo
    w = float(hi[0]-lo[0])
    return np.r_[p[:,0],p[:,1],a[:,2],w]

def unpack(z):
    return np.column_stack((z[:N],z[N:2*N],z[2*N:3*N]))

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    best = None
    best_score = -np.inf
    best_array = None

    def offer(a, original=None):
        nonlocal best, best_score, best_array
        if a is None:
            return
        score = exact_score(a)
        if score > best_score:
            best_score = score
            best_array = a.copy()
            best = original if original is not None else {'circles': a.tolist()}

    initial = None
    if init is not None:
        try:
            initial = np.asarray(init['circles'], dtype=float)
            offer(initial, init)
        except (ValueError, TypeError, KeyError, IndexError):
            initial = None

    # A deterministic staggered packing provides a strong feasible cold fallback.
    points = []
    counts = [4,5,4,5,3]
    for row, count in enumerate(counts):
        if time.monotonic() >= deadline:
            break
        offset = 1.0 if row % 2 == 0 else 0.0
        points.extend((2.0*j+offset, np.sqrt(3.0)*row, 1.0) for j in range(count))
    if len(points) != N:
        # This branch also guarantees a valid result for unusually tiny budgets.
        grid = np.column_stack((np.linspace(0.025,1.975,N), np.zeros(N), np.full(N,0.02)))
    else:
        grid = repair(np.array(points, dtype=float))
    offer(grid)
    if initial is not None and initial.shape == (21,3) and time.monotonic() < deadline:
        offer(repair(initial))

    linear_jac = np.zeros((4*N,3*N+1))
    linear_jac[K,K] = 1.0
    linear_jac[K,2*N+K] = -1.0
    linear_jac[N+K,K] = -1.0
    linear_jac[N+K,2*N+K] = -1.0
    linear_jac[N+K,3*N] = 1.0
    linear_jac[2*N+K,N+K] = 1.0
    linear_jac[2*N+K,2*N+K] = -1.0
    linear_jac[3*N+K,N+K] = -1.0
    linear_jac[3*N+K,2*N+K] = -1.0
    linear_jac[3*N+K,3*N] = -1.0
    objective_jac = np.r_[np.zeros(2*N),-np.ones(N),0.0]
    pair_rows = np.arange(M)
    bounds = [(None,None)]*(2*N) + [(1e-7,0.5)]*N + [(0.05,1.95)]

    for attempt in range(int(params['restarts'])):
        now = time.monotonic()
        if now >= deadline-0.025:
            break
        if attempt == 0:
            a = best_array.copy()
        elif attempt == 1 and initial is not None and initial.shape == (21,3) and best_score < 2.30:
            a = repair(initial)
            if a is None:
                a = best_array.copy()
            a[:,2] = np.maximum(a[:,2],0.06)
        else:
            a = best_array.copy()
            a[:,:2] += rng.normal(0,float(params['jitter']),size=(N,2))
        z = pack(a)
        # Reserve time for another topology if the first local solve is slow.
        local_end = deadline if attempt else min(deadline, now+0.48*budget_s)
        latest = [z.copy()]
        iteration = [0]

        def check():
            if time.monotonic() >= local_end:
                raise StopSearch()

        def objective(v):
            check()
            return -float(np.sum(v[2*N:3*N]))

        def constraints(v):
            check()
            x,y,r,w = v[:N],v[N:2*N],v[2*N:3*N],v[-1]
            dx,dy = x[I]-x[J],y[I]-y[J]
            d = np.sqrt(dx*dx+dy*dy+1e-30)
            return np.r_[d-r[I]-r[J],x-r,w-x-r,y-r,2-w-y-r]

        def jacobian(v):
            check()
            x,y = v[:N],v[N:2*N]
            dx,dy = x[I]-x[J],y[I]-y[J]
            d = np.sqrt(dx*dx+dy*dy+1e-30)
            jac = np.zeros((M+4*N,3*N+1))
            jac[pair_rows,I] = dx/d
            jac[pair_rows,J] = -dx/d
            jac[pair_rows,N+I] = dy/d
            jac[pair_rows,N+J] = -dy/d
            jac[pair_rows,2*N+I] = -1.0
            jac[pair_rows,2*N+J] = -1.0
            jac[M:] = linear_jac
            return jac

        def callback(v):
            check()
            latest[0] = v.copy()
            iteration[0] += 1
            if iteration[0] % 5 == 0:
                offer(repair(unpack(v)))

        try:
            result = minimize(objective,z,jac=lambda v: objective_jac,method='SLSQP',
                              bounds=bounds,
                              constraints={'type':'ineq','fun':constraints,'jac':jacobian},
                              callback=callback,
                              options={'maxiter':int(params['maxiter']), 'ftol':float(params['ftol']), 'disp':False})
            latest[0] = result.x
        except StopSearch:
            pass
        if time.monotonic() < start+0.77*budget_s:
            offer(repair(unpack(latest[0])))
    return best
