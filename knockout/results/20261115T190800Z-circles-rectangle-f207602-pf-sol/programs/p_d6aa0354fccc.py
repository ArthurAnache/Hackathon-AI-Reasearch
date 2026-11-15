import time
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog

PARAMS = {
    'angle': (43.0, 10.0, 45.0),
    'trust': (0.085, 0.01, 0.25),
    'iterations': (32, 8, 80),
    'restarts': (4, 1, 8),
}
N = 21
I, J = np.triu_indices(N, 1)


def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -float('inf')
    d = np.sqrt((a[I, 0] - a[J, 0]) ** 2 + (a[I, 1] - a[J, 1]) ** 2)
    if np.any(d < a[I, 2] + a[J, 2]):
        return -float('inf')
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    perimeter = (max(u + v for u, v in zip(x, r)) - min(u - v for u, v in zip(x, r))
                 + max(u + v for u, v in zip(y, r)) - min(u - v for u, v in zip(y, r)))
    if perimeter > 2:
        return -float('inf')
    return float(np.sum(a[:, 2]))


def normalize(a):
    a = a.copy()
    lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
    hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
    s = 2.0 / np.sum(hi - lo)
    a[:, :2] = (a[:, :2] - lo) * s
    a[:, 2] *= s
    return a, float((hi[0] - lo[0]) * s)


def repaired(a):
    a = a.copy()
    d = np.sqrt((a[I, 0] - a[J, 0]) ** 2 + (a[I, 1] - a[J, 1]) ** 2)
    q = float(np.min(d / (a[I, 2] + a[J, 2])))
    a[:, 2] *= min(1.0, q) * (1.0 - 3e-13)
    a, _ = normalize(a)
    a *= 1.0 - 3e-13
    return a


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    if init is not None:
        original = np.asarray(init['circles'], dtype=float).copy()
        best = init
        best_score = exact_score(original)
    else:
        original = np.array([[(i % 7 + .5) / 7, (i // 7 + .5) / 3, .065] for i in range(21)])
        best = {'circles': original.tolist()}
        best_score = exact_score(original)
    base, _ = normalize(original)
    # x[0:21], y[21:42], r[42:63], rectangle width[63].
    m = len(I)
    A = np.zeros((m + 84, 64))
    b = np.zeros(m + 84)
    rows = np.arange(m)
    A[rows, 42 + I] = 1.0
    A[rows, 42 + J] = 1.0
    k = np.arange(N)
    A[m + k, k] = -1
    A[m + k, 42 + k] = 1
    A[m + N + k, k] = 1
    A[m + N + k, 42 + k] = 1
    A[m + N + k, 63] = -1
    A[m + 2 * N + k, N + k] = -1
    A[m + 2 * N + k, 42 + k] = 1
    A[m + 3 * N + k, N + k] = 1
    A[m + 3 * N + k, 42 + k] = 1
    A[m + 3 * N + k, 63] = 1
    b[m + 3 * N:] = 2
    objective = np.r_[np.zeros(42), -np.ones(21), 0.0]
    angles = [float(params['angle']), -27.0, 16.0, -38.0, 32.0, -12.0, 22.0, 0.0]
    # Randomly reverse the rotation sense, without changing reproducibility.
    sign = 1 if rng.integers(2) else -1
    for restart in range(int(params['restarts'])):
        if time.monotonic() >= deadline:
            break
        angle = np.deg2rad(sign * angles[restart % len(angles)])
        cs, sn = np.cos(angle), np.sin(angle)
        a = base.copy()
        a[:, :2] = (a[:, :2] - np.mean(a[:, :2], axis=0)) @ np.array([[cs, sn], [-sn, cs]])
        a, w = normalize(a)
        previous = float(a[:, 2].sum())
        stalled = 0
        for iteration in range(int(params['iterations'])):
            now = time.monotonic()
            if now >= deadline - .025:
                break
            dx = a[I, 0] - a[J, 0]
            dy = a[I, 1] - a[J, 1]
            d = np.maximum(np.sqrt(dx * dx + dy * dy), 1e-14)
            nx, ny = dx / d, dy / d
            A[rows, I] = -nx
            A[rows, J] = nx
            A[rows, N + I] = -ny
            A[rows, N + J] = ny
            trust = float(params['trust'])
            z = np.r_[a[:, 0], a[:, 1], a[:, 2], w]
            bounds = [(max(0.0, float(v - trust)), min(2.0, float(v + trust))) for v in z[:42]]
            bounds += [(1e-8, .5)] * 21
            bounds += [(max(.08, w - trust), min(1.92, w + trust))]
            res = linprog(objective, A_ub=A, b_ub=b, bounds=bounds, method='highs',
                          options={'time_limit': max(.001, deadline - now - .015),
                                   'dual_feasibility_tolerance': 1e-9,
                                   'primal_feasibility_tolerance': 1e-9})
            if not res.success:
                break
            z = res.x
            a = np.column_stack((z[:21], z[21:42], z[42:63]))
            w = float(z[63])
            value = float(np.sum(a[:, 2]))
            if value > best_score - 1e-8 and time.monotonic() < deadline:
                candidate = repaired(a)
                score = exact_score(candidate)
                if score > best_score:
                    best_score = score
                    best = {'circles': candidate.tolist()}
            if value - previous < 2e-10:
                stalled += 1
            else:
                stalled = 0
            previous = value
            if stalled >= 3:
                break
    return best
