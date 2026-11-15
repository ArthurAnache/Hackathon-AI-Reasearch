import time
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog

PARAMS = {
    'cluster_max': (8, 4, 12),
    'trust': (0.13, 0.03, 0.4),
    'steps': (22, 6, 40),
}
N = 21
I, J = np.triu_indices(N, 1)


def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0] - a[J, 0]) ** 2 + (a[I, 1] - a[J, 1]) ** 2)
    if np.any(d < a[I, 2] + a[J, 2]):
        return -np.inf
    x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
    perimeter = (max(u + v for u, v in zip(x, r)) - min(u - v for u, v in zip(x, r))
                 + max(u + v for u, v in zip(y, r)) - min(u - v for u, v in zip(y, r)))
    if perimeter > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))


def repair(a):
    a = a.copy()
    if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return None
    d = np.sqrt((a[I, 0] - a[J, 0]) ** 2 + (a[I, 1] - a[J, 1]) ** 2)
    ratio = float(np.min(d / (a[I, 2] + a[J, 2])))
    a[:, 2] *= min(1.0, ratio) * (1.0 - 8e-15)
    lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
    hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
    a[:, :2] -= lo
    a *= min(1.0, 2.0 / float(np.sum(hi - lo))) * (1.0 - 8e-15)
    return a


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.69 * budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a = np.array([[(k % 5 + 0.5) / 5, (k // 5 + 0.5) / 5, 0.045] for k in range(N)])
        best = {'circles': a.tolist()}
    else:
        a = np.asarray(init['circles'], dtype=float).copy()
        best = init
    best_score = exact_score(a)
    base = a.copy()
    lo = np.min(base[:, :2] - base[:, 2, None], axis=0)
    base[:, :2] -= lo
    span = np.max(base[:, :2] + base[:, 2, None], axis=0)
    base *= 2.0 / np.sum(span)

    # Variables: all x, all y, all radii, rectangle width.
    m = len(I)
    A = np.zeros((m + 4 * N, 3 * N + 1))
    B = np.zeros(m + 4 * N)
    rows = np.arange(m)
    A[rows, 2 * N + I] = 1
    A[rows, 2 * N + J] = 1
    for k in range(N):
        if time.monotonic() >= end:
            return best
        s = m + 4 * k
        A[s, k] = -1
        A[s, 2 * N + k] = 1
        A[s + 1, k] = 1
        A[s + 1, 2 * N + k] = 1
        A[s + 1, -1] = -1
        A[s + 2, N + k] = -1
        A[s + 2, 2 * N + k] = 1
        A[s + 3, N + k] = 1
        A[s + 3, 2 * N + k] = 1
        A[s + 3, -1] = 1
        B[s + 3] = 2
    objective = np.r_[np.zeros(2 * N), -np.ones(N), 0.0]
    distances = np.linalg.norm(base[I, :2] - base[J, :2], axis=1)
    gaps = distances - base[I, 2] - base[J, 2]
    edges = np.argsort(gaps)[:65]
    attempt = 0
    while time.monotonic() < end:
        p = base[:, :2].copy()
        if attempt:
            edge = int(edges[int(rng.integers(len(edges)))])
            i, j = int(I[edge]), int(J[edge])
            anchor = (p[i] + p[j]) * 0.5
            axis = p[j] - p[i]
            axis /= max(np.linalg.norm(axis), 1e-15)
            normal = np.array([-axis[1], axis[0]])
            signed = (p - anchor) @ normal
            side = 1 if rng.random() < 0.5 else -1
            eligible = np.flatnonzero(signed * side > 1e-7)
            if len(eligible) >= 3:
                order = np.argsort(np.linalg.norm(p[eligible] - anchor, axis=1))
                count = int(rng.integers(3, int(params['cluster_max']) + 1))
                chosen = eligible[order[:count]]
                p[chosen] -= 2 * signed[chosen, None] * normal
            else:
                center = p[int(rng.integers(N))]
                chosen = np.argsort(np.linalg.norm(p - center, axis=1))[:5]
                p[chosen] = 2 * p[chosen].mean(axis=0) - p[chosen]
            p -= p.min(axis=0)
            extent = p.max(axis=0)
            p *= 1.65 / max(float(extent.sum()), 1e-8)
            p += 0.07
        last_sum = -1.0
        for step in range(int(params['steps'])):
            now = time.monotonic()
            if now >= end - 0.025:
                break
            delta = p[I] - p[J]
            length = np.maximum(np.linalg.norm(delta, axis=1), 1e-14)
            normals = delta / length[:, None]
            A[rows, I] = -normals[:, 0]
            A[rows, J] = normals[:, 0]
            A[rows, N + I] = -normals[:, 1]
            A[rows, N + J] = normals[:, 1]
            trust = float(params['trust']) * (1.0 if step < 10 else 0.5)
            lower = np.maximum(p - trust, 0)
            upper = np.minimum(p + trust, 2)
            bounds = list(zip(lower[:, 0], upper[:, 0])) + list(zip(lower[:, 1], upper[:, 1]))
            bounds += [(1e-7, 0.7)] * N + [(0.12, 1.88)]
            result = linprog(objective, A_ub=A, b_ub=B, bounds=bounds, method='highs',
                             options={'time_limit': max(0.001, end - now - 0.015),
                                      'primal_feasibility_tolerance': 1e-9,
                                      'dual_feasibility_tolerance': 1e-9})
            if not result.success:
                break
            z = result.x
            p = np.column_stack((z[:N], z[N:2 * N]))
            candidate = repair(np.column_stack((p, z[2 * N:3 * N])))
            if candidate is not None:
                score = exact_score(candidate)
                if score > best_score:
                    best_score = score
                    best = {'circles': candidate.tolist()}
            total = float(np.sum(z[2 * N:3 * N]))
            if step > 3 and abs(total - last_sum) < 2e-11:
                break
            last_sum = total
        attempt += 1
    return best
