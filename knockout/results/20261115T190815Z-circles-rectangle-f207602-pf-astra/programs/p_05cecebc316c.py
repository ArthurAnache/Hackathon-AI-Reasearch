import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'maxiter': (150, 40, 400),
    'jitter': (0.004, 0.0, 0.03),
    'restart_jitter': (0.018, 0.001, 0.08),
    'time_fraction': (0.70, 0.5, 0.75),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
K = np.arange(N)
ROWS = np.arange(M)

class StopSearch(Exception):
    pass

def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -float('inf')
    r = a[:, 2]
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < r[I] + r[J]):
        return -float('inf')
    xs = list(map(Fraction, map(float, a[:, 0])))
    ys = list(map(Fraction, map(float, a[:, 1])))
    rs = list(map(Fraction, map(float, r)))
    span = (max(map(lambda t: t[0]+t[1], zip(xs, rs)))
            - min(map(lambda t: t[0]-t[1], zip(xs, rs)))
            + max(map(lambda t: t[0]+t[1], zip(ys, rs)))
            - min(map(lambda t: t[0]-t[1], zip(ys, rs))))
    if span > 2:
        return -float('inf')
    return float(np.sum(r))

def repair(z):
    p = np.column_stack((z[:N], z[N:2*N]))
    r = np.maximum(z[2*N:3*N], 1e-10).copy()
    if not np.isfinite(p).all() or not np.isfinite(r).all():
        return None
    d = np.sqrt((p[I, 0]-p[J, 0])**2 + (p[I, 1]-p[J, 1])**2)
    factor = min(1.0, float(np.min(d / (r[I]+r[J]))))
    if factor <= 0:
        return None
    r *= factor * (1-2e-13)
    lo = np.min(p-r[:, None], axis=0)
    hi = np.max(p+r[:, None], axis=0)
    span = float(np.sum(hi-lo))
    if not span > 0:
        return None
    scale = (2.0-2e-12)/span
    p = (p-lo)*scale
    r *= scale*(1-2e-13)
    return np.column_stack((p, r))

def encode(a):
    p, r = a[:, :2].copy(), a[:, 2].copy()
    lo = np.min(p-r[:, None], axis=0)
    side = np.max(p+r[:, None], axis=0)-lo
    scale = 2.0/float(np.sum(side))
    p = (p-lo)*scale
    return np.concatenate((p[:, 0], p[:, 1], r*scale, [side[0]*scale]))

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + float(params['time_fraction'])*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -float('inf')
    best_array = None
    if init is not None:
        try:
            a = np.asarray(init['circles'], dtype=float)
            score = exact_score(a)
            if np.isfinite(score):
                best_score, best_array = score, a.copy()
        except (ValueError, TypeError, KeyError, OverflowError):
            pass

    def keep(z):
        nonlocal best, best_score, best_array
        if time.monotonic() >= deadline:
            raise StopSearch()
        a = repair(z)
        if a is not None and float(np.sum(a[:, 2])) > best_score:
            score = exact_score(a)
            if score > best_score:
                best_score = score
                best_array = a.copy()
                best = {'circles': a.tolist()}

    # A guaranteed valid fallback, independent of the supplied start.
    ids = np.arange(N)
    fallback = np.column_stack(((ids % 7 + 0.5)*0.2,
                                (ids // 7 + 0.5)*0.2,
                                np.full(N, 0.1-1e-12)))
    score = exact_score(fallback)
    if score > best_score:
        best_score, best_array = score, fallback.copy()
        best = {'circles': fallback.tolist()}

    # A 21-point triangular lattice, with opposite corners removed.
    xx = np.array([2,4,6,8, 1,3,5,7, 0,2,4,6,8, 1,3,5,7, 0,2,4,6], dtype=float)
    yy = np.repeat(np.arange(5), [4,4,5,4,4])*np.sqrt(3.0)
    lattice = np.column_stack((xx, yy, np.ones(N)))
    initial = encode(lattice)
    initial[:2*N] += rng.normal(0, float(params['jitter']), 2*N)
    if best_score > 2.23:
        initial = encode(best_array)

    objective_gradient = np.zeros(3*N+1)
    objective_gradient[2*N:3*N] = -1
    fixed = np.zeros((M+4*N, 3*N+1))
    fixed[ROWS, 2*N+I] = -1
    fixed[ROWS, 2*N+J] = -1
    fixed[M+K, K] = 1
    fixed[M+K, 2*N+K] = -1
    fixed[M+N+K, K] = -1
    fixed[M+N+K, 2*N+K] = -1
    fixed[M+N+K, 3*N] = 1
    fixed[M+2*N+K, N+K] = 1
    fixed[M+2*N+K, 2*N+K] = -1
    fixed[M+3*N+K, N+K] = -1
    fixed[M+3*N+K, 2*N+K] = -1
    fixed[M+3*N+K, 3*N] = -1
    cached_z = None
    cached_c = None
    cached_j = None

    def geometry(z):
        nonlocal cached_z, cached_c, cached_j
        if time.monotonic() >= deadline:
            raise StopSearch()
        if cached_z is not None and np.array_equal(z, cached_z):
            return cached_c, cached_j
        x, y, r, w = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
        dx, dy = x[I]-x[J], y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy)
        safe = np.maximum(d, 1e-15)
        c = np.concatenate((d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r))
        jac = fixed.copy()
        jac[ROWS, I] = dx/safe
        jac[ROWS, J] = -dx/safe
        jac[ROWS, N+I] = dy/safe
        jac[ROWS, N+J] = -dy/safe
        cached_z, cached_c, cached_j = z.copy(), c, jac
        return c, jac

    def objective(z):
        if time.monotonic() >= deadline:
            raise StopSearch()
        return -float(np.sum(z[2*N:3*N])), objective_gradient

    bounds = [(None, None)]*(2*N) + [(1e-7, 0.5)]*N + [(0.25, 1.75)]
    constraints = {'type': 'ineq', 'fun': lambda z: geometry(z)[0],
                   'jac': lambda z: geometry(z)[1]}
    attempt = 0
    try:
        while time.monotonic() < deadline:
            if attempt == 0:
                z = initial.copy()
            else:
                z = encode(best_array)
                amplitude = float(params['restart_jitter'])*(1+0.35*(attempt-1))
                z[:2*N] += rng.normal(0, amplitude, 2*N)
                z[2*N:3*N] *= rng.uniform(0.92, 1.08, N)
            keep(z)
            cached_z = None
            result = minimize(objective, z, jac=True, method='SLSQP',
                              bounds=bounds, constraints=constraints, callback=keep,
                              options={'maxiter': int(params['maxiter']),
                                       'ftol': 2e-11, 'disp': False})
            keep(result.x)
            attempt += 1
    except StopSearch:
        pass
    return best
