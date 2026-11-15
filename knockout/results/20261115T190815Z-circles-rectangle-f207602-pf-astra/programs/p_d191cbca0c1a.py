import time
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize

PARAMS = {
    'trust': (0.065, 0.015, 0.18),
    'linear_fraction': (0.64, 0.35, 0.85),
    'linear_iterations': (160, 20, 300),
    'polish_iterations': (100, 20, 200),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
K = np.arange(N)
DIM = 64
OBJ = np.r_[np.zeros(42), -np.ones(21), 0.0]

class Deadline(Exception):
    pass


def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    r = a[:, 2]
    d = np.sqrt((a[I, 0] - a[J, 0]) ** 2 + (a[I, 1] - a[J, 1]) ** 2)
    if np.any(d < r[I] + r[J]):
        return -np.inf
    x = list(map(lambda v: Fraction(float(v)), a[:, 0]))
    y = list(map(lambda v: Fraction(float(v)), a[:, 1]))
    q = list(map(lambda v: Fraction(float(v)), r))
    perimeter = (max(u + v for u, v in zip(x, q)) - min(u - v for u, v in zip(x, q))
                 + max(u + v for u, v in zip(y, q)) - min(u - v for u, v in zip(y, q)))
    if perimeter > 2:
        return -np.inf
    return float(np.sum(r))


def repair(z):
    p = np.column_stack((z[:21], z[21:42])).copy()
    r = np.maximum(z[42:63], 1e-12).copy()
    if not np.isfinite(p).all() or not np.isfinite(r).all():
        return None
    p -= np.mean(p, axis=0)
    d = np.sqrt((p[I, 0] - p[J, 0]) ** 2 + (p[I, 1] - p[J, 1]) ** 2)
    ratio = min(1.0, float(np.min(d / (r[I] + r[J]))))
    if ratio <= 0:
        return None
    r *= ratio * (1.0 - 2e-13)
    sides = np.max(p + r[:, None], axis=0) - np.min(p - r[:, None], axis=0)
    scale = min(1.0, 2.0 / float(np.sum(sides))) * (1.0 - 2e-13)
    return np.column_stack((p * scale, r * scale))


def encode(a):
    p = a[:, :2].copy()
    r = np.maximum(np.abs(a[:, 2]), 1e-9)
    lo = np.min(p - r[:, None], axis=0)
    sides = np.max(p + r[:, None], axis=0) - lo
    scale = 2.0 / float(np.sum(sides))
    p = (p - lo) * scale
    return np.r_[p[:, 0], p[:, 1], r * scale, sides[0] * scale]


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70 * budget_s
    linear_end = start + 0.70 * budget_s * float(params['linear_fraction'])
    best = None
    best_score = -np.inf

    def offer(z):
        nonlocal best, best_score
        if time.monotonic() >= end:
            raise Deadline()
        a = repair(z)
        if a is not None:
            score = exact_score(a)
            if score > best_score:
                best_score = score
                best = {'circles': a.tolist()}

    # A deterministic valid fallback, also useful when init is malformed.
    grid = np.column_stack(((np.arange(21) % 5 + 0.5) * 0.2,
                            (np.arange(21) // 5 + 0.5) * 0.2,
                            np.full(21, 0.1 * (1.0 - 1e-12))))
    best = {'circles': grid.tolist()}
    best_score = exact_score(grid)
    a0 = grid
    if init is not None:
        try:
            aa = np.asarray(init['circles'], dtype=float)
            s = exact_score(aa)
            if s > best_score:
                best, best_score = init, s
            if aa.shape == (21, 3) and np.isfinite(aa).all():
                a0 = aa
        except (ValueError, TypeError, KeyError):
            pass
    z = encode(a0)

    # Wall inequalities in the box [0,w] x [0,2-w].
    A = np.zeros((M + 84, DIM))
    b = np.zeros(M + 84)
    A[np.arange(M), 42 + I] = 1.0
    A[np.arange(M), 42 + J] = 1.0
    A[M + K, K] = -1.0
    A[M + K, 42 + K] = 1.0
    A[M + 21 + K, K] = 1.0
    A[M + 21 + K, 42 + K] = 1.0
    A[M + 21 + K, 63] = -1.0
    A[M + 42 + K, 21 + K] = -1.0
    A[M + 42 + K, 42 + K] = 1.0
    A[M + 63 + K, 21 + K] = 1.0
    A[M + 63 + K, 42 + K] = 1.0
    A[M + 63 + K, 63] = 1.0
    b[M + 63:] = 2.0
    pair_rows = np.arange(M)
    trust = float(params['trust'])
    previous = -np.inf

    try:
        offer(z)
        for iteration in range(int(params['linear_iterations'])):
            now = time.monotonic()
            if now >= linear_end or end - now < 0.08:
                break
            dx = z[I] - z[J]
            dy = z[21 + I] - z[21 + J]
            d = np.maximum(np.hypot(dx, dy), 1e-15)
            ux, uy = dx / d, dy / d
            A[pair_rows, I] = -ux
            A[pair_rows, J] = ux
            A[pair_rows, 21 + I] = -uy
            A[pair_rows, 21 + J] = uy
            lower = np.r_[np.maximum(z[:42] - trust, 0.0), np.full(21, 1e-8), max(0.1, z[63] - trust)]
            upper = np.r_[np.minimum(z[:42] + trust, 2.0), np.full(21, 0.5), min(1.9, z[63] + trust)]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result = linprog(OBJ, A_ub=A, b_ub=b,
                    bounds=np.column_stack((lower, upper)), method='highs',
                    options={'threads': 1, 'primal_feasibility_tolerance': 1e-9,
                             'dual_feasibility_tolerance': 1e-9,
                             'time_limit': max(0.005, min(linear_end - now, end - now - 0.04))})
            if not result.success or result.x is None:
                break
            z = result.x
            offer(z)
            current = float(np.sum(z[42:63]))
            gain = current - previous
            previous = current
            if iteration >= 12 and gain < 2e-9:
                break

        # Full smooth constraints, with a cached analytic Jacobian.
        wall_jac = -A[M:].copy()
        cache_z = None
        cache_c = None
        cache_j = None

        def constraints(v):
            nonlocal cache_z, cache_c, cache_j
            if time.monotonic() >= end:
                raise Deadline()
            if cache_z is not None and np.array_equal(v, cache_z):
                return cache_c
            dx = v[I] - v[J]
            dy = v[21 + I] - v[21 + J]
            d = np.maximum(np.hypot(dx, dy), 1e-15)
            c = np.r_[d - v[42 + I] - v[42 + J], b[M:] - A[M:] @ v]
            jac = np.zeros((M + 84, DIM))
            jac[pair_rows, I] = dx / d
            jac[pair_rows, J] = -dx / d
            jac[pair_rows, 21 + I] = dy / d
            jac[pair_rows, 21 + J] = -dy / d
            jac[pair_rows, 42 + I] = -1.0
            jac[pair_rows, 42 + J] = -1.0
            jac[M:] = wall_jac
            cache_z, cache_c, cache_j = v.copy(), c, jac
            return c

        def jacobian(v):
            constraints(v)
            return cache_j

        def objective(v):
            if time.monotonic() >= end:
                raise Deadline()
            return -float(np.sum(v[42:63]))

        def callback(v):
            offer(v)

        if time.monotonic() < end - 0.045:
            result = minimize(objective, z, jac=lambda v: OBJ, method='SLSQP',
                bounds=[(0.0, 2.0)] * 42 + [(1e-8, 0.5)] * 21 + [(0.1, 1.9)],
                constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                callback=callback,
                options={'maxiter': int(params['polish_iterations']), 'ftol': 2e-13, 'disp': False})
            offer(result.x)
    except Deadline:
        pass
    return best
