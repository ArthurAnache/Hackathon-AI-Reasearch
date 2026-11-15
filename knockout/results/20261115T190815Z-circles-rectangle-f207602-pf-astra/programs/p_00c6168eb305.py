import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'excursion': (0.19, 0.05, 0.45),
    'fixed_iterations': (32, 10, 90),
    'free_iterations': (65, 20, 150),
    'trials': (6, 2, 12),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
K = np.arange(N)

class StopSearch(Exception):
    pass

def exact_score(c):
    a = np.asarray(c['circles'], dtype=np.float64)
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    p, r = a[:, :2], a[:, 2]
    d = np.sqrt((p[I, 0] - p[J, 0]) ** 2 + (p[I, 1] - p[J, 1]) ** 2)
    if np.any(d < r[I] + r[J]):
        return -np.inf
    x = list(map(lambda v: Fraction(float(v)), p[:, 0]))
    y = list(map(lambda v: Fraction(float(v)), p[:, 1]))
    q = list(map(lambda v: Fraction(float(v)), r))
    perimeter = (max(a+b for a,b in zip(x,q)) - min(a-b for a,b in zip(x,q))
                 + max(a+b for a,b in zip(y,q)) - min(a-b for a,b in zip(y,q)))
    if perimeter > 2:
        return -np.inf
    return float(np.sum(r))

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a = np.zeros((21, 3))
        a[:, 0] = (np.arange(21) % 7 + 0.5) * 0.2
        a[:, 1] = (np.arange(21) // 7 + 0.5) * 0.2
        a[:, 2] = 0.099999999999
        init = {'circles': a.tolist()}
    best = init
    best_score = exact_score(init)
    a = np.array(init['circles'], dtype=float)
    lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
    sides = np.max(a[:, :2] + a[:, 2, None], axis=0) - lo
    scale = 2.0 / np.sum(sides)
    p = (a[:, :2] - lo) * scale
    r = a[:, 2] * scale
    w0 = float(sides[0] * scale)
    base = np.r_[p[:, 0], p[:, 1], r, w0]
    objective_gradient = np.r_[np.zeros(42), -np.ones(21), 0.0]
    static = np.zeros((M + 4*N, 64))
    rows = np.arange(M)
    static[rows, 42+I] = -1
    static[rows, 42+J] = -1
    static[M+K, K] = 1
    static[M+K, 42+K] = -1
    static[M+N+K, K] = -1
    static[M+N+K, 42+K] = -1
    static[M+N+K, 63] = 1
    static[M+2*N+K, 21+K] = 1
    static[M+2*N+K, 42+K] = -1
    static[M+3*N+K, 21+K] = -1
    static[M+3*N+K, 42+K] = -1
    static[M+3*N+K, 63] = -1
    cache_z = None
    cache_c = None
    cache_j = None
    last = base.copy()

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def objective(z):
        check()
        return -float(np.sum(z[42:63]))

    def gradient(z):
        check()
        return objective_gradient

    def constraints(z):
        nonlocal cache_z, cache_c, cache_j
        check()
        if cache_z is not None and np.array_equal(cache_z, z):
            return cache_c
        x, y, q, w = z[:21], z[21:42], z[42:63], z[63]
        dx, dy = x[I]-x[J], y[I]-y[J]
        dist = np.sqrt(dx*dx + dy*dy)
        safe = np.maximum(dist, 1e-30)
        jac = static.copy()
        jac[rows, I] = dx/safe
        jac[rows, J] = -dx/safe
        jac[rows, 21+I] = dy/safe
        jac[rows, 21+J] = -dy/safe
        cache_z = z.copy()
        cache_c = np.r_[dist-q[I]-q[J], x-q, w-x-q, y-q, 2-w-y-q]
        cache_j = jac
        return cache_c

    def jacobian(z):
        constraints(z)
        return cache_j

    def consider(z):
        nonlocal best, best_score
        if time.monotonic() >= start + 0.77*budget_s:
            return
        if not np.isfinite(z).all():
            return
        q = z[42:63].copy()
        if np.any(q <= 0):
            return
        points = np.column_stack((z[:21], z[21:42]))
        dist = np.sqrt((points[I,0]-points[J,0])**2 + (points[I,1]-points[J,1])**2)
        shrink = min(1.0, float(np.min(dist/(q[I]+q[J]))))
        q *= shrink * (1.0-3e-14)
        lower = np.min(points-q[:,None], axis=0)
        upper = np.max(points+q[:,None], axis=0)
        factor = min(1.0, 2.0/float(np.sum(upper-lower))) * (1.0-3e-14)
        points = (points-lower)*factor
        q *= factor
        if float(np.sum(q)) <= best_score:
            return
        candidate = {'circles': np.column_stack((points,q)).tolist()}
        score = exact_score(candidate)
        if score > best_score:
            best, best_score = candidate, score

    count = 0
    def callback(z):
        nonlocal last, count
        last = z.copy()
        count += 1
        check()
        if count % 7 == 0:
            consider(z)

    common_bounds = [(0.0, 2.0)]*42 + [(1e-7, 0.5)]*21
    cons = {'type': 'ineq', 'fun': constraints, 'jac': jacobian}
    direction = 1 if rng.integers(2) else -1
    try:
        for trial in range(int(params['trials'])):
            check()
            magnitude = float(params['excursion']) * (1 + trial//2 * 0.65)
            sign = direction if trial % 2 == 0 else -direction
            # Excursions are in log aspect ratio, avoiding asymmetric treatment
            # of wide and tall rectangles.
            ratio = w0/(2-w0) * np.exp(sign * 2*magnitude)
            target = float(np.clip(2*ratio/(1+ratio), 0.35, 1.65))
            z = base.copy()
            sx, sy = target/w0, (2-target)/(2-w0)
            z[:21] *= sx
            z[21:42] *= sy
            z[42:63] *= min(sx,sy)
            z[63] = target
            last = z.copy()
            fixed = minimize(objective, z, jac=gradient, constraints=cons,
                             bounds=common_bounds+[(target,target)], method='SLSQP',
                             callback=callback,
                             options={'maxiter': int(params['fixed_iterations']), 'ftol': 2e-10})
            consider(fixed.x)
            check()
            last = fixed.x.copy()
            free = minimize(objective, fixed.x, jac=gradient, constraints=cons,
                            bounds=common_bounds+[(0.2,1.8)], method='SLSQP',
                            callback=callback,
                            options={'maxiter': int(params['free_iterations']), 'ftol': 2e-13})
            consider(free.x)
    except StopSearch:
        consider(last)
    return best
