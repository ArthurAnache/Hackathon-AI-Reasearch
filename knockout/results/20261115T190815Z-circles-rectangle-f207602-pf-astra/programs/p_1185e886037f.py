import time
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog

PARAMS = {
    "iterations": (48, 15, 100),
    "jitter": (0.035, 0.001, 0.15),
    "minimum_radius": (0.015, 0.0001, 0.05),
    "restart_noise": (0.022, 0.001, 0.08),
}

N = 21
I, J = np.triu_indices(N, 1)
M = len(I)


def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or not (a[:, 2] > 0).all():
        return -float('inf')
    d = np.sqrt((a[I, 0] - a[J, 0])**2 + (a[I, 1] - a[J, 1])**2)
    if np.any(d < a[I, 2] + a[J, 2]):
        return -float('inf')
    x = list(map(Fraction, map(float, a[:, 0])))
    y = list(map(Fraction, map(float, a[:, 1])))
    r = list(map(Fraction, map(float, a[:, 2])))
    hp = (max(u + v for u, v in zip(x, r)) - min(u - v for u, v in zip(x, r))
          + max(u + v for u, v in zip(y, r)) - min(u - v for u, v in zip(y, r)))
    if hp > 2:
        return -float('inf')
    return float(np.sum(a[:, 2]))


def repair(a):
    a = np.array(a, dtype=float, copy=True)
    if a.shape != (N, 3) or not np.isfinite(a).all():
        return None
    a[:, 2] = np.maximum(a[:, 2], 1e-10)
    d = np.sqrt((a[I, 0] - a[J, 0])**2 + (a[I, 1] - a[J, 1])**2)
    factor = min(1.0, float(np.min(d / (a[I, 2] + a[J, 2]))))
    if factor <= 0:
        return None
    a[:, 2] *= factor * (1.0 - 2e-13)
    lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
    hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
    hp = float(np.sum(hi - lo))
    if hp <= 0:
        return None
    a[:, :2] -= lo
    a *= (2.0 / hp) * (1.0 - 2e-13)
    return a


def encode(a):
    return {"x": a[:, 0].tolist(), "y": a[:, 1].tolist(), "r": a[:, 2].tolist()}


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best = None
    best_a = None
    best_score = -float('inf')
    initial = None
    if init is not None:
        try:
            initial = np.asarray(init['circles'], dtype=float)
            val = exact_score(initial)
            if val > best_score:
                best, best_a, best_score = init, initial.copy(), val
        except (KeyError, TypeError, ValueError, OverflowError):
            initial = None

    # Explicit fallback has ample strict geometric slack.
    fallback = np.column_stack((np.arange(N) * 0.085, np.zeros(N), np.full(N, 0.039)))
    fallback = repair(fallback)
    val = exact_score(fallback)
    if val > best_score:
        best, best_a, best_score = encode(fallback), fallback.copy(), val

    rows = np.arange(M)
    ids = np.arange(N)
    A = np.zeros((M + 4*N, 3*N + 1))
    b = np.zeros(M + 4*N)
    A[rows, 2*N + I] = 1.0
    A[rows, 2*N + J] = 1.0
    q = M + ids
    A[q, ids] = -1.0
    A[q, 2*N + ids] = 1.0
    q = M + N + ids
    A[q, ids] = 1.0
    A[q, 2*N + ids] = 1.0
    A[q, 3*N] = -1.0
    q = M + 2*N + ids
    A[q, N + ids] = -1.0
    A[q, 2*N + ids] = 1.0
    q = M + 3*N + ids
    A[q, N + ids] = 1.0
    A[q, 2*N + ids] = 1.0
    A[q, 3*N] = 1.0
    b[q] = 2.0
    objective = np.r_[np.zeros(2*N), -np.ones(N), 0.0]
    bounds = [(0.0, 2.0)] * (2*N) + [(float(params['minimum_radius']), 0.5)] * N + [(0.25, 1.75)]

    patterns = [[3, 4, 3, 4, 3, 4], [4, 5, 4, 5, 3], [4, 4, 5, 4, 4]]
    restart = 0
    while time.monotonic() < deadline:
        if restart < len(patterns):
            counts = patterns[restart]
            pts = []
            for k, count in enumerate(counts):
                if time.monotonic() >= deadline:
                    break
                offset = 1.0 if count < max(counts) else 0.0
                if restart == 2:
                    offset = 1.0 if k % 2 == 0 else 0.0
                pts.extend((2.0*j + offset, np.sqrt(3.0)*k) for j in range(count))
            if len(pts) != N:
                break
            p = np.asarray(pts, dtype=float)
            p += rng.normal(0.0, float(params['jitter']), p.shape)
            a = repair(np.column_stack((p, np.ones(N))))
        elif restart == 3 and initial is not None and initial.shape == (N, 3):
            a = repair(initial)
        else:
            a = best_a.copy()
            a[:, :2] += rng.normal(0.0, float(params['restart_noise']), (N, 2))
            a = repair(a)
        restart += 1
        if a is None:
            continue
        val = exact_score(a)
        if val > best_score:
            best, best_a, best_score = encode(a), a.copy(), val
        p = a[:, :2].copy()
        previous = -float('inf')
        stagnant = 0
        for iteration in range(int(params['iterations'])):
            now = time.monotonic()
            if now >= deadline - 0.018:
                break
            delta = p[I] - p[J]
            distance = np.sqrt(np.sum(delta*delta, axis=1))
            if np.any(distance < 1e-12):
                break
            u = delta / distance[:, None]
            A[rows, I] = -u[:, 0]
            A[rows, J] = u[:, 0]
            A[rows, N + I] = -u[:, 1]
            A[rows, N + J] = u[:, 1]
            result = linprog(objective, A_ub=A, b_ub=b, bounds=bounds,
                             method='highs', options={
                                 'threads': 1,
                                 'time_limit': max(0.001, deadline - now - 0.012),
                                 'dual_feasibility_tolerance': 1e-8,
                                 'primal_feasibility_tolerance': 1e-8})
            if result.x is None:
                break
            z = result.x
            a = repair(np.column_stack((z[:N], z[N:2*N], z[2*N:3*N])))
            if a is None:
                break
            val = exact_score(a)
            if val > best_score:
                best, best_a, best_score = encode(a), a.copy(), val
            p = a[:, :2].copy()
            if val <= previous + 2e-9:
                stagnant += 1
            else:
                stagnant = 0
            previous = val
            if stagnant >= 4:
                break
        if time.monotonic() >= deadline - 0.018:
            break
    return best
