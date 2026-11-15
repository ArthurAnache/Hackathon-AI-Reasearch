import time
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog

PARAMS = {
    'trust': (0.12, 0.02, 0.3),
    'restart_noise': (0.012, 0.001, 0.06),
    'iterations': (180, 20, 400),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
K = np.arange(N)


def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    dx = a[I, 0] - a[J, 0]
    dy = a[I, 1] - a[J, 1]
    if np.any(np.sqrt(dx * dx + dy * dy) < a[I, 2] + a[J, 2]):
        return -np.inf
    x = list(map(lambda v: Fraction(float(v)), a[:, 0]))
    y = list(map(lambda v: Fraction(float(v)), a[:, 1]))
    r = list(map(lambda v: Fraction(float(v)), a[:, 2]))
    width = max(map(lambda t: t[0] + t[1], zip(x, r))) - min(map(lambda t: t[0] - t[1], zip(x, r)))
    height = max(map(lambda t: t[0] + t[1], zip(y, r))) - min(map(lambda t: t[0] - t[1], zip(y, r)))
    if width + height > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))


def repair(a):
    a = np.array(a, dtype=float, copy=True)
    if not np.isfinite(a).all():
        return None
    a[:, 2] = np.maximum(a[:, 2], 1e-10)
    a[:, :2] -= np.mean(a[:, :2], axis=0)
    d = np.sqrt((a[I, 0] - a[J, 0]) ** 2 + (a[I, 1] - a[J, 1]) ** 2)
    factor = min(1.0, float(np.min(d / (a[I, 2] + a[J, 2]))))
    if factor <= 0:
        return None
    a[:, 2] *= factor * (1.0 - 3e-13)
    lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
    hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
    a[:, :2] -= lo
    a *= (2.0 - 3e-13) / float(np.sum(hi - lo))
    return a


def construct(params, seed, budget_s, init=None):
    begin = time.monotonic()
    deadline = begin + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    # A certified fallback is available before any numerical optimization.
    grid = np.column_stack(((2 * (np.arange(N) % 7) + 1) * 0.1,
                            (2 * (np.arange(N) // 7) + 1) * 0.1,
                            np.full(N, 0.1 - 1e-13)))
    best_a = grid
    best_score = exact_score(grid)
    best_object = None
    initial = None
    if init is not None:
        try:
            raw = np.asarray(init['circles'], dtype=float)
            val = exact_score(raw)
            if val > best_score:
                best_a, best_score, best_object = raw.copy(), val, init
            if raw.shape == (N, 3) and np.isfinite(raw).all():
                initial = repair(raw)
        except (ValueError, KeyError, TypeError, OverflowError):
            pass
    if initial is None:
        # Staggered rows, with enough radius variation available to the LP.
        counts = [4, 5, 4, 5, 3]
        pts = []
        for row, count in enumerate(counts):
            if time.monotonic() >= deadline:
                break
            for col in range(count):
                if time.monotonic() >= deadline:
                    break
                pts.append((col + (5 - count) / 2, row * np.sqrt(3) / 2, 0.45))
        initial = repair(np.asarray(pts)) if len(pts) == N else grid.copy()

    # Variables are x[21], y[21], r[21], and rectangle width.
    # Rectangle height is 2-width.
    A = np.zeros((M + 4 * N, 3 * N + 1))
    b = np.zeros(M + 4 * N)
    rows = np.arange(M)
    A[rows, 2 * N + I] = 1
    A[rows, 2 * N + J] = 1
    q = M + K
    A[q, K] = -1
    A[q, 2 * N + K] = 1
    q = M + N + K
    A[q, K] = 1
    A[q, 2 * N + K] = 1
    A[q, -1] = -1
    q = M + 2 * N + K
    A[q, N + K] = -1
    A[q, 2 * N + K] = 1
    q = M + 3 * N + K
    A[q, N + K] = 1
    A[q, 2 * N + K] = 1
    A[q, -1] = 1
    b[q] = 2
    objective = np.zeros(3 * N + 1)
    objective[2 * N:3 * N] = -1
    p = initial[:, :2].copy()
    trust = float(params['trust'])
    previous = -np.inf
    stagnant = 0
    restarts = 0
    for iteration in range(int(params['iterations'])):
        now = time.monotonic()
        if now >= deadline - 0.025:
            break
        dx = p[I, 0] - p[J, 0]
        dy = p[I, 1] - p[J, 1]
        d = np.maximum(np.sqrt(dx * dx + dy * dy), 1e-15)
        ux, uy = dx / d, dy / d
        A[rows, I] = -ux
        A[rows, J] = ux
        A[rows, N + I] = -uy
        A[rows, N + J] = uy
        bounds = np.empty((3 * N + 1, 2))
        bounds[:N, 0] = np.maximum(0, p[:, 0] - trust)
        bounds[:N, 1] = np.minimum(2, p[:, 0] + trust)
        bounds[N:2 * N, 0] = np.maximum(0, p[:, 1] - trust)
        bounds[N:2 * N, 1] = np.minimum(2, p[:, 1] + trust)
        bounds[2 * N:3 * N, 0] = 1e-7
        bounds[2 * N:3 * N, 1] = 0.5
        bounds[-1] = (0.1, 1.9)
        remaining = deadline - time.monotonic()
        if remaining <= 0.025:
            break
        result = linprog(objective, A_ub=A, b_ub=b, bounds=bounds,
                         method='highs', options={
                             'threads': 1,
                             'time_limit': max(0.005, remaining - 0.02),
                             'primal_feasibility_tolerance': 1e-9,
                             'dual_feasibility_tolerance': 1e-9,
                         })
        if result.x is None:
            break
        z = result.x
        candidate = repair(np.column_stack((z[:N], z[N:2 * N], z[2 * N:3 * N])))
        if candidate is None:
            break
        score = exact_score(candidate)
        if score > best_score:
            best_a = candidate.copy()
            best_score = score
            best_object = None
        p = candidate[:, :2].copy()
        gain = score - previous
        stagnant = stagnant + 1 if abs(gain) < 2e-10 else 0
        previous = score
        if gain < -1e-7:
            trust = max(0.015, trust * 0.65)
        elif gain > 1e-4:
            trust = min(float(params['trust']), trust * 1.15)
        if stagnant >= 3:
            if time.monotonic() >= deadline - 0.12:
                break
            restarts += 1
            a = best_a.copy()
            noise = float(params['restart_noise']) * min(2.0, 1 + 0.3 * restarts)
            a[:, :2] += rng.normal(0, noise, (N, 2))
            a = repair(a)
            if a is None:
                break
            p = a[:, :2].copy()
            previous = -np.inf
            stagnant = 0
            trust = float(params['trust'])
    if best_object is not None:
        return best_object
    return {'circles': best_a.tolist()}
