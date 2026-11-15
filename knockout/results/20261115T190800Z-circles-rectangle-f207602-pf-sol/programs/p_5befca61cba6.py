import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'concave_power': (0.25, 0.05, 0.9),
    'convex_power': (2.2, 1.1, 4.0),
    'excursion_iterations': (38, 10, 100),
    'recovery_iterations': (65, 15, 150),
}
N = 21
I, J = np.triu_indices(N, 1)
K = np.arange(N)
M = len(I)

class StopSearch(Exception):
    pass

def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2] + a[J, 2]):
        return -np.inf
    x = list(map(lambda v: Fraction(float(v)), a[:, 0]))
    y = list(map(lambda v: Fraction(float(v)), a[:, 1]))
    r = list(map(lambda v: Fraction(float(v)), a[:, 2]))
    perimeter = (max(u+v for u, v in zip(x, r)) - min(u-v for u, v in zip(x, r))
                 + max(u+v for u, v in zip(y, r)) - min(u-v for u, v in zip(y, r)))
    if perimeter > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))

def repair(z):
    p = np.column_stack((z[:N], z[N:2*N])).copy()
    r = np.maximum(z[2*N:3*N], 1e-12).copy()
    if not np.isfinite(p).all() or not np.isfinite(r).all():
        return None
    d = np.sqrt(np.sum((p[I]-p[J])**2, axis=1))
    factor = min(1.0, float(np.min(d/(r[I]+r[J]))))
    if factor <= 0:
        return None
    r *= factor * (1.0-8e-15)
    lo = np.min(p-r[:, None], axis=0)
    hi = np.max(p+r[:, None], axis=0)
    scale = (2.0-3e-14)/float(np.sum(hi-lo))
    p = (p-lo)*scale
    r *= scale
    return np.column_stack((p, r))

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + 0.69*budget_s
    rng = np.random.default_rng(seed)
    if init is not None:
        base = np.asarray(init['circles'], dtype=float).copy()
        best = init
        best_score = exact_score(base)
    else:
        base = np.zeros((N, 3))
        base[:, 0] = (np.arange(N) % 5 + 0.5)/5
        base[:, 1] = (np.arange(N) // 5 + 0.5)/5
        base[:, 2] = 0.035
        best = {'circles': base.tolist()}
        best_score = exact_score(base)
    lo = np.min(base[:, :2]-base[:, 2, None], axis=0)
    hi = np.max(base[:, :2]+base[:, 2, None], axis=0)
    scale = 2.0/float(np.sum(hi-lo))
    p = (base[:, :2]-lo)*scale
    r = base[:, 2]*scale
    z = np.r_[p[:, 0], p[:, 1], r, (hi[0]-lo[0])*scale]
    original = z.copy()
    last = z.copy()
    reference = float(np.mean(r))
    lower = np.r_[np.zeros(2*N), np.full(N, 1e-7), 0.08]
    upper = np.r_[np.full(2*N, 2.0), np.full(N, 0.9), 1.92]
    bounds = list(zip(lower, upper))
    cached_z = None
    cached_c = None
    cached_j = None

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def constraints(v):
        nonlocal cached_z, cached_c, cached_j
        check()
        if cached_z is not None and np.array_equal(v, cached_z):
            return cached_c
        x, y, rr, w = v[:N], v[N:2*N], v[2*N:3*N], v[-1]
        dx, dy = x[I]-x[J], y[I]-y[J]
        dd = np.sqrt(dx*dx+dy*dy+1e-30)
        cached_c = np.r_[dd-rr[I]-rr[J], x-rr, w-x-rr, y-rr, 2-w-y-rr]
        a = np.zeros((M+4*N, 3*N+1))
        rows = np.arange(M)
        a[rows, I] = dx/dd
        a[rows, J] = -dx/dd
        a[rows, N+I] = dy/dd
        a[rows, N+J] = -dy/dd
        a[rows, 2*N+I] = -1
        a[rows, 2*N+J] = -1
        rows = M+K
        a[rows, K] = 1
        a[rows, 2*N+K] = -1
        rows = M+N+K
        a[rows, K] = -1
        a[rows, 2*N+K] = -1
        a[rows, -1] = 1
        rows = M+2*N+K
        a[rows, N+K] = 1
        a[rows, 2*N+K] = -1
        rows = M+3*N+K
        a[rows, N+K] = -1
        a[rows, 2*N+K] = -1
        a[rows, -1] = -1
        cached_z = v.copy()
        cached_j = a
        return cached_c

    def jacobian(v):
        constraints(v)
        return cached_j

    def consider(v):
        nonlocal best, best_score
        a = repair(v)
        if a is not None:
            score = exact_score(a)
            if score > best_score:
                best_score = score
                best = {'circles': a.tolist()}

    def callback(v):
        nonlocal last
        check()
        last = v.copy()

    path = [(float(params['concave_power']), int(params['excursion_iterations'])),
            (1.0, int(params['recovery_iterations'])),
            (float(params['convex_power']), int(params['excursion_iterations'])),
            (1.0, int(params['recovery_iterations']))]
    try:
        for stage, (power, iterations) in enumerate(path):
            check()
            if stage == 2:
                z = original.copy()
            def objective(v):
                check()
                rr = np.maximum(v[2*N:3*N], 1e-12)
                ratios = rr/reference
                value = -reference*np.sum(ratios**power)/power
                gradient = np.zeros(3*N+1)
                gradient[2*N:3*N] = -(ratios**(power-1))
                return float(value), gradient
            result = minimize(objective, z, method='SLSQP', jac=True,
                              constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                              bounds=bounds, callback=callback,
                              options={'maxiter': iterations, 'ftol': 2e-12, 'disp': False})
            z = result.x.copy()
            last = z.copy()
            consider(z)
    except StopSearch:
        pass
    consider(last)
    return best
