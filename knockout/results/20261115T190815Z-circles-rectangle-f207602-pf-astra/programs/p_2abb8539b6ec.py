import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'excursion': (1.65, 1.1, 2.5),
    'fixed_fraction': (0.38, 0.15, 0.65),
    'trials': (4, 2, 8),
    'maxiter': (110, 30, 250)
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
K = np.arange(N)
OBJ = np.r_[np.zeros(42), -np.ones(21), 0.0]
B = np.zeros((84, 64))
B[K, K] = 1
B[K, 42 + K] = -1
B[21 + K, K] = -1
B[21 + K, 42 + K] = -1
B[21 + K, 63] = 1
B[42 + K, 21 + K] = 1
B[42 + K, 42 + K] = -1
B[63 + K, 21 + K] = -1
B[63 + K, 42 + K] = -1
B[63 + K, 63] = -1
OFFSET = np.r_[np.zeros(63), np.full(21, 2.0)]

class StopSearch(Exception):
    pass


def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0] - a[J, 0]) ** 2 + (a[I, 1] - a[J, 1]) ** 2)
    if np.any(d < a[I, 2] + a[J, 2]):
        return -np.inf
    x = list(map(lambda v: Fraction(float(v)), a[:, 0]))
    y = list(map(lambda v: Fraction(float(v)), a[:, 1]))
    r = list(map(lambda v: Fraction(float(v)), a[:, 2]))
    perimeter = (max(map(lambda t: t[0] + t[1], zip(x, r)))
                 - min(map(lambda t: t[0] - t[1], zip(x, r)))
                 + max(map(lambda t: t[0] + t[1], zip(y, r)))
                 - min(map(lambda t: t[0] - t[1], zip(y, r))))
    if perimeter > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))


def repaired(z):
    a = np.column_stack((z[:21], z[21:42], z[42:63])).copy()
    if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return None
    d = np.sqrt((a[I, 0] - a[J, 0]) ** 2 + (a[I, 1] - a[J, 1]) ** 2)
    ratio = min(1.0, float(np.min(d / (a[I, 2] + a[J, 2]))))
    a[:, 2] *= ratio * (1.0 - 1e-14)
    low = np.min(a[:, :2] - a[:, 2, None], axis=0)
    a[:, :2] -= low
    sides = np.max(a[:, :2] + a[:, 2, None], axis=0)
    a *= (2.0 / np.sum(sides)) * (1.0 - 1e-14)
    return a


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a0 = np.column_stack((np.tile(np.arange(3), 7) * 0.2 + 0.1,
                              np.repeat(np.arange(7), 3) * 0.2 + 0.1,
                              np.full(21, 0.1 - 1e-14)))
        best = {'circles': a0.tolist()}
    else:
        a0 = np.asarray(init['circles'], dtype=float).copy()
        best = init
    best_score = exact_score(a0)
    lo = np.min(a0[:, :2] - a0[:, 2, None], axis=0)
    side = np.max(a0[:, :2] + a0[:, 2, None], axis=0) - lo
    a0[:, :2] -= lo
    a0 *= 2.0 / side.sum()
    side *= 2.0 / side.sum()
    if side[0] > side[1]:
        a0[:, :2] = a0[:, [1, 0]]
        side = side[::-1].copy()
    base_w = float(side[0])
    ntrials = int(params['trials'])
    excursion = float(params['excursion'])
    # Opposite aspect excursions establish different boundary contact patterns.
    factors = [excursion, 1.0 / excursion, excursion ** 0.5, excursion ** -0.5]
    if seed % 2:
        factors[0], factors[1] = factors[1], factors[0]
    bounds = [(0.0, 2.0)] * 42 + [(1e-8, 0.5)] * 21 + [(0.05, 1.95)]

    def consider(z):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        a = repaired(z)
        if a is not None:
            score = exact_score(a)
            if score > best_score:
                best_score = score
                best = {'circles': a.tolist()}

    def solve(z, until, fixed):
        last = z.copy()
        cache_z = None
        cache_c = None
        cache_j = None

        def calculate(v):
            nonlocal cache_z, cache_c, cache_j
            if time.monotonic() >= until:
                raise StopSearch()
            if cache_z is not None and np.array_equal(v, cache_z):
                return
            dx = v[I] - v[J]
            dy = v[21 + I] - v[21 + J]
            d = np.maximum(np.sqrt(dx * dx + dy * dy), 1e-15)
            c = np.r_[d - v[42 + I] - v[42 + J], B @ v + OFFSET]
            jac = np.zeros((M + 84, 64))
            rows = np.arange(M)
            jac[rows, I] = dx / d
            jac[rows, J] = -dx / d
            jac[rows, 21 + I] = dy / d
            jac[rows, 21 + J] = -dy / d
            jac[rows, 42 + I] = -1
            jac[rows, 42 + J] = -1
            jac[M:] = B
            cache_z, cache_c, cache_j = v.copy(), c, jac

        def constraints(v):
            calculate(v)
            return cache_c

        def jacobian(v):
            calculate(v)
            return cache_j

        def callback(v):
            nonlocal last
            last = v.copy()
            if time.monotonic() >= until:
                raise StopSearch()

        bb = list(bounds)
        if fixed:
            bb[-1] = (float(z[-1]), float(z[-1]))
        try:
            result = minimize(lambda v: float(OBJ @ v), z, jac=lambda v: OBJ,
                              bounds=bb, method='SLSQP',
                              constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                              callback=callback,
                              options={'maxiter': int(params['maxiter']), 'ftol': 2e-12, 'disp': False})
            last = result.x
        except StopSearch:
            pass
        return last

    for trial in range(ntrials):
        now = time.monotonic()
        if now >= deadline - 0.025:
            break
        remaining = deadline - now - 0.015
        allocation = remaining / (ntrials - trial)
        factor = factors[trial % 4]
        if trial >= 4:
            factor *= float(rng.uniform(0.85, 1.15))
        aspect = base_w / (2.0 - base_w) * factor
        w = 2.0 * aspect / (1.0 + aspect)
        sx, sy = w / base_w, (2.0 - w) / (2.0 - base_w)
        p = a0[:, :2] * np.array([sx, sy])
        r = a0[:, 2] * min(sx, sy)
        z = np.r_[p[:, 0], p[:, 1], r, w]
        fixed_end = min(deadline - 0.015, now + allocation * float(params['fixed_fraction']))
        z = solve(z, fixed_end, True)
        consider(z)
        if time.monotonic() >= deadline - 0.015:
            break
        free_end = min(deadline - 0.008, now + allocation)
        z = solve(z, free_end, False)
        consider(z)
    return best
