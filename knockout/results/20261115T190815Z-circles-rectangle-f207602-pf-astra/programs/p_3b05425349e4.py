import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    "jitter": (0.008, 0.0, 0.04),
    "maxiter": (160, 40, 400),
    "ftol": (2e-11, 1e-13, 1e-7, "log"),
    "time_fraction": (0.70, 0.5, 0.78)
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
K = np.arange(N)
E = np.eye(N)

class StopSearch(Exception):
    pass


def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0] - a[J, 0]) ** 2 + (a[I, 1] - a[J, 1]) ** 2)
    if np.any(d < a[I, 2] + a[J, 2]):
        return -np.inf
    x = list(map(Fraction, map(float, a[:, 0])))
    y = list(map(Fraction, map(float, a[:, 1])))
    r = list(map(Fraction, map(float, a[:, 2])))
    perimeter = (max(map(lambda t: t[0] + t[1], zip(x, r)))
                 - min(map(lambda t: t[0] - t[1], zip(x, r)))
                 + max(map(lambda t: t[0] + t[1], zip(y, r)))
                 - min(map(lambda t: t[0] - t[1], zip(y, r))))
    if perimeter > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))


def repaired(z):
    p = np.column_stack((z[:N], z[N:2*N]))
    r = np.maximum(z[2*N:3*N], 1e-10).copy()
    if not np.isfinite(p).all() or not np.isfinite(r).all():
        return None
    d = np.sqrt((p[I, 0] - p[J, 0]) ** 2 + (p[I, 1] - p[J, 1]) ** 2)
    ratio = float(np.min(d / (r[I] + r[J])))
    if ratio <= 0:
        return None
    r *= min(1.0, ratio) * (1.0 - 2e-12)
    lo = np.min(p - r[:, None], axis=0)
    hi = np.max(p + r[:, None], axis=0)
    scale = (2.0 - 8e-12) / float(np.sum(hi - lo))
    return np.column_stack(((p - lo) * scale, r * scale))


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + float(params['time_fraction']) * budget_s
    rng = np.random.default_rng(seed)
    # A deterministic, strictly feasible emergency packing.
    fallback = np.column_stack((np.linspace(0.025, 1.975, N), np.zeros(N), np.full(N, 0.02)))
    fallback *= 0.95
    best = {'circles': fallback.tolist()}
    best_score = exact_score(fallback)
    original = None
    if init is not None:
        try:
            original = np.asarray(init['circles'], dtype=float)
            score = exact_score(original)
            if score > best_score:
                best, best_score = init, score
        except (ValueError, TypeError, KeyError, OverflowError):
            original = None

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def consider(z):
        nonlocal best, best_score
        check()
        a = repaired(z)
        if a is not None and float(np.sum(a[:, 2])) > best_score:
            s = exact_score(a)
            if s > best_score:
                best_score = s
                best = {'circles': a.tolist()}

    # Constraints use non-squared distances for consistent conditioning.
    base = np.zeros((M + 4*N, 3*N + 1))
    base[np.arange(M), 2*N + I] = -1
    base[np.arange(M), 2*N + J] = -1
    base[M:M+N, :N] = E
    base[M:M+N, 2*N:3*N] = -E
    base[M+N:M+2*N, :N] = -E
    base[M+N:M+2*N, 2*N:3*N] = -E
    base[M+N:M+2*N, -1] = 1
    base[M+2*N:M+3*N, N:2*N] = E
    base[M+2*N:M+3*N, 2*N:3*N] = -E
    base[M+3*N:, N:2*N] = -E
    base[M+3*N:, 2*N:3*N] = -E
    base[M+3*N:, -1] = -1
    cache_z = None
    cache_c = None
    cache_j = None

    def constraints(z):
        nonlocal cache_z, cache_c, cache_j
        check()
        if cache_z is None or not np.array_equal(z, cache_z):
            x, y, r, w = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
            dx, dy = x[I] - x[J], y[I] - y[J]
            d = np.sqrt(dx*dx + dy*dy)
            safe = np.maximum(d, 1e-15)
            c = np.concatenate((d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r))
            jac = base.copy()
            jac[np.arange(M), I] = dx/safe
            jac[np.arange(M), J] = -dx/safe
            jac[np.arange(M), N+I] = dy/safe
            jac[np.arange(M), N+J] = -dy/safe
            cache_z, cache_c, cache_j = z.copy(), c, jac
        return cache_c

    def jacobian(z):
        constraints(z)
        return cache_j

    objective_gradient = np.r_[np.zeros(2*N), -np.ones(N), 0.0]

    def objective(z):
        check()
        return -float(np.sum(z[2*N:3*N])), objective_gradient

    row = np.repeat(np.arange(5), [4, 5, 4, 5, 3])
    col = np.concatenate((np.arange(4)+0.5, np.arange(5), np.arange(4)+0.5,
                          np.arange(5), np.arange(3)+0.5))
    p = np.column_stack((2*col+1, np.sqrt(3)*row+1))
    p += rng.normal(0, float(params['jitter']), p.shape)
    rr = np.ones(N)
    lo = (p-rr[:, None]).min(axis=0)
    sides = (p+rr[:, None]).max(axis=0)-lo
    scale = 2/float(sides.sum())
    p = (p-lo)*scale
    z_grid = np.r_[p[:, 0], p[:, 1], rr*scale*0.995, sides[0]*scale]
    starts = [z_grid]
    if original is not None and original.shape == (21, 3) and np.isfinite(original).all():
        p = original[:, :2].copy()
        r = np.maximum(np.abs(original[:, 2]), 1e-6)
        lo = (p-r[:, None]).min(axis=0)
        sides = (p+r[:, None]).max(axis=0)-lo
        scale = 2/max(float(sides.sum()), 1e-10)
        p = (p-lo)*scale
        zi = np.r_[p[:, 0], p[:, 1], r*scale, sides[0]*scale]
        if best_score > 2.25:
            starts.insert(0, zi)
        else:
            starts.append(zi)
    bounds = [(None, None)]*(2*N) + [(1e-8, 0.5)]*N + [(0.05, 1.95)]
    try:
        for z in starts:
            check()
            consider(z)
            cache_z = None
            result = minimize(objective, z, jac=True, method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                              callback=consider,
                              options={'maxiter': int(params['maxiter']),
                                       'ftol': float(params['ftol']), 'disp': False})
            consider(result.x)
    except StopSearch:
        pass
    return best
