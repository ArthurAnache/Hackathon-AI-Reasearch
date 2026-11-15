import time
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog

PARAMS = {
    'excursion_weight': (0.65, 0.1, 1.5),
    'excursion_steps': (3, 1, 8),
    'recovery_steps': (12, 4, 30),
    'trust': (0.16, 0.02, 0.5),
    'cluster_scale': (0.25, 0.08, 0.7),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)


def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    r = a[:, 2]
    d = np.sqrt((a[I, 0] - a[J, 0]) ** 2 + (a[I, 1] - a[J, 1]) ** 2)
    if np.any(d < r[I] + r[J]):
        return -np.inf
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    q = [Fraction(float(v)) for v in r]
    perimeter = (max(v + t for v, t in zip(x, q)) - min(v - t for v, t in zip(x, q))
                 + max(v + t for v, t in zip(y, q)) - min(v - t for v, t in zip(y, q)))
    if perimeter > 2:
        return -np.inf
    return float(np.sum(r))


def repair(z, deadline):
    a = np.column_stack((z[:N], z[N:2*N], z[2*N:3*N])).copy()
    if not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
        return None
    d = np.sqrt((a[I, 0] - a[J, 0]) ** 2 + (a[I, 1] - a[J, 1]) ** 2)
    ratio = float(np.min(d / (a[I, 2] + a[J, 2])))
    a[:, 2] *= min(1.0, ratio) * (1.0 - 3e-14)
    lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
    hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
    scale = min(1.0, 2.0 / float(np.sum(hi - lo))) * (1.0 - 3e-15)
    a *= scale
    for k in range(5):
        if time.monotonic() >= deadline:
            return None
        if np.isfinite(exact_score(a)):
            return a
        a[:, 2] *= 1.0 - 2e-13
        a[:, :2] *= 1.0 - 2e-13
    return None


def normalize(a):
    lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
    hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
    side = hi - lo
    s = 2.0 / float(np.sum(side))
    p = (a[:, :2] - lo) * s
    return np.r_[p[:, 0], p[:, 1], a[:, 2] * s, side[0] * s]


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.69 * budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a = np.array([[(k % 5 + 0.5) / 5, (k // 5 + 0.5) / 5, 0.07] for k in range(N)])
        best = {'circles': a.tolist()}
    else:
        best = init
        a = np.asarray(init['circles'], dtype=float).copy()
    best_score = exact_score(a)
    base = normalize(a)
    z = base.copy()

    # The distance supporting planes are inner approximations: satisfying
    # one guarantees separation, even after a large simultaneous displacement.
    A = np.zeros((M + 4*N, 3*N + 1))
    b = np.zeros(M + 4*N)
    ids = np.arange(N)
    pairs = np.arange(M)
    A[pairs, 2*N + I] = 1.0
    A[pairs, 2*N + J] = 1.0
    for j in range(4):
        rows = M + j*N + ids
        A[rows, 2*N + ids] = 1.0
    A[M + ids, ids] = -1.0
    A[M + N + ids, ids] = 1.0
    A[M + N + ids, -1] = -1.0
    A[M + 2*N + ids, N + ids] = -1.0
    A[M + 3*N + ids, N + ids] = 1.0
    A[M + 3*N + ids, -1] = 1.0
    b[M + 3*N:] = 2.0
    objective = np.zeros(3*N + 1)
    objective[2*N:3*N] = -1.0
    cycle = 0

    while time.monotonic() < deadline - 0.025:
        if cycle == 0:
            phases = [(np.ones(N), int(params['recovery_steps']))]
        else:
            # Restart each excursion at the best verified geometry, rather
            # than spending the remaining budget recovering a poor basin.
            z = base.copy()
            center = np.array([z[int(rng.integers(N))], z[N + int(rng.integers(N))]])
            p = np.column_stack((z[:N], z[N:2*N]))
            dist2 = np.sum((p - center) ** 2, axis=1)
            field = np.exp(-dist2 / (2 * float(params['cluster_scale']) ** 2))
            field -= np.mean(field)
            field /= max(float(np.std(field)), 0.1)
            if cycle % 2 == 0:
                field = -field
            weights = np.exp(np.clip(float(params['excursion_weight']) * field, -1.8, 1.8))
            phases = [(weights, int(params['excursion_steps'])), (np.ones(N), int(params['recovery_steps']))]
        for weights, steps in phases:
            if time.monotonic() >= deadline - 0.025:
                break
            objective[2*N:3*N] = -weights
            for step in range(steps):
                now = time.monotonic()
                if now >= deadline - 0.025:
                    break
                dx = z[I] - z[J]
                dy = z[N+I] - z[N+J]
                length = np.maximum(np.sqrt(dx*dx + dy*dy), 1e-15)
                nx, ny = dx/length, dy/length
                A[pairs, I] = -nx
                A[pairs, J] = nx
                A[pairs, N+I] = -ny
                A[pairs, N+J] = ny
                trust = float(params['trust'])
                bounds = [(max(0.0, float(v)-trust), min(2.0, float(v)+trust)) for v in z[:2*N]]
                bounds += [(1e-7, 0.5)] * N + [(0.12, 1.88)]
                res = linprog(objective, A_ub=A, b_ub=b, bounds=bounds,
                              method='highs', options={'time_limit': max(0.001, deadline-now-0.012),
                              'primal_feasibility_tolerance': 1e-9,
                              'dual_feasibility_tolerance': 1e-9})
                if time.monotonic() >= deadline:
                    break
                if not res.success:
                    break
                old = z
                z = res.x
                candidate = repair(z, deadline)
                if candidate is not None:
                    value = exact_score(candidate)
                    if value > best_score:
                        best_score = value
                        best = {'circles': candidate.tolist()}
                        base = normalize(candidate)
                if np.max(np.abs(z-old)) < 2e-9:
                    break
        cycle += 1
    return best
