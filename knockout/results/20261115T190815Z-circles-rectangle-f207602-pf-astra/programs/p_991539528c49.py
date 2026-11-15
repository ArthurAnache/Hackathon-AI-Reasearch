import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linprog

PARAMS = {
    'rotation': (0.22, 0.0, 0.6),
    'aspect': (1.0, 0.5, 2.0, 'log'),
    'rows': (5, 3, 7),
    'jitter': (0.03, 0.0, 0.15),
    'mu0': (10.0, 1.0, 1000.0, 'log'),
    'mu_growth': (8.0, 2.0, 30.0),
    'stages': (6, 2, 12),
    'maxiter': (150, 30, 1000)
}
N = 21
I, J = np.triu_indices(N, 1)

class StopSearch(Exception):
    pass


def perimeter(p, r):
    xs = list(map(lambda x: Fraction(float(x)), p[:, 0]))
    ys = list(map(lambda x: Fraction(float(x)), p[:, 1]))
    rs = list(map(lambda x: Fraction(float(x)), r))
    return (max(x + q for x, q in zip(xs, rs))
            - min(x - q for x, q in zip(xs, rs))
            + max(y + q for y, q in zip(ys, rs))
            - min(y - q for y, q in zip(ys, rs)))


def valid(p, r):
    if p.shape != (N, 2) or r.shape != (N,):
        return False
    if not np.isfinite(p).all() or not np.isfinite(r).all() or not (r > 0).all():
        return False
    if perimeter(p, r) > 2:
        return False
    d = np.sqrt((p[I, 0] - p[J, 0]) ** 2 + (p[I, 1] - p[J, 1]) ** 2)
    return not np.any(d < r[I] + r[J])


def output(p, r):
    return {'circles': np.column_stack((p, r)).tolist()}


def penalty(z, mu):
    x, y, r, w = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
    dx, dy = x[I] - x[J], y[I] - y[J]
    d = np.sqrt(dx * dx + dy * dy + 1e-18)
    v = np.maximum(r[I] + r[J] - d, 0)
    b0 = np.maximum(r - x, 0)
    b1 = np.maximum(r + x - w, 0)
    b2 = np.maximum(r - y, 0)
    b3 = np.maximum(r + y - (2 - w), 0)
    neg = np.maximum(-r, 0)
    f = -r.sum() + mu * (v @ v + b0 @ b0 + b1 @ b1 + b2 @ b2 + b3 @ b3 + neg @ neg)
    gx, gy, gr = np.zeros(N), np.zeros(N), -np.ones(N)
    c = 2 * mu * v
    np.add.at(gr, I, c)
    np.add.at(gr, J, c)
    np.add.at(gx, I, -c * dx / d)
    np.add.at(gx, J, c * dx / d)
    np.add.at(gy, I, -c * dy / d)
    np.add.at(gy, J, c * dy / d)
    gr += 2 * mu * (b0 + b1 + b2 + b3 - neg)
    gx += 2 * mu * (-b0 + b1)
    gy += 2 * mu * (-b2 + b3)
    gw = 2 * mu * (-b1.sum() + b3.sum())
    return f, np.r_[gx, gy, gr, gw]


def lp_radii(p, deadline):
    remaining = deadline - time.monotonic()
    if remaining < 0.025:
        return None
    m = len(I)
    a = np.zeros((m + 4*N + 1, N + 4))
    b = np.zeros(m + 4*N + 1)
    a[np.arange(m), I] = 1
    a[np.arange(m), J] = 1
    b[:m] = np.sqrt((p[I, 0] - p[J, 0]) ** 2 + (p[I, 1] - p[J, 1]) ** 2)
    for k, (col, sign, coord) in enumerate(((N, 1, 0), (N+1, -1, 0), (N+2, 1, 1), (N+3, -1, 1))):
        if time.monotonic() >= deadline:
            return None
        rows = m + k*N + np.arange(N)
        a[rows, np.arange(N)] = 1
        a[rows, col] = sign
        b[rows] = sign * p[:, coord]
    a[-1, N:] = [-1, 1, -1, 1]
    b[-1] = 2
    res = linprog(-np.r_[np.ones(N), np.zeros(4)], A_ub=a, b_ub=b,
                  bounds=[(0, None)] * N + [(None, None)] * 4,
                  method='highs', options={'time_limit': max(0.001, deadline - time.monotonic())})
    return res.x[:N] if res.success else None


def repair(p, r, deadline):
    r = np.maximum(np.asarray(r, dtype=float), 1e-12) * (1 - 1e-12)
    d = np.sqrt((p[I, 0] - p[J, 0]) ** 2 + (p[I, 1] - p[J, 1]) ** 2)
    for _ in range(5):
        if time.monotonic() >= deadline:
            return None
        if not np.any(d < r[I] + r[J]):
            break
        for i, j, dd in zip(I, J, d):
            if time.monotonic() >= deadline:
                return None
            if r[i] + r[j] > dd * (1 - 1e-12):
                s = dd * (1 - 1e-12) / (r[i] + r[j])
                r[i] *= s
                r[j] *= s
    for _ in range(60):
        if time.monotonic() >= deadline:
            return None
        excess = perimeter(p, r) - 2
        if excess <= 0:
            break
        r *= 1 - max(2 * float(excess) / float(r.sum()), 1e-12)
    return np.maximum(r, 1e-12)


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    descent_end = deadline - min(0.22, 0.15 * budget_s)
    rng = np.random.default_rng(seed)
    best, best_score = None, -np.inf
    if init is not None:
        a = np.asarray(init['circles'], dtype=float)
        p0, r0 = a[:, :2].copy(), a[:, 2].copy()
        if valid(p0, r0):
            best, best_score = init, float(np.sum(r0))
        # The sole search change: rigidly rotate before normalizing the box.
        theta = float(params['rotation'])
        c, s = np.cos(theta), np.sin(theta)
        rotation = np.array([[c, -s], [s, c]])
        p = (p0 - p0.mean(axis=0)) @ rotation.T
        r = np.abs(r0)
        lo = (p - r[:, None]).min(axis=0)
        side = (p + r[:, None]).max(axis=0) - lo
        scale = 2 / side.sum()
        p, r, w = (p - lo) * scale, r * scale, side[0] * scale
        candidates = [p0]
    else:
        w = 2 * float(params['aspect']) / (1 + float(params['aspect']))
        rows = int(params['rows'])
        per_row = int(np.ceil(N / rows))
        k = np.arange(N)
        p = np.column_stack(((k % per_row + 0.5) / per_row,
                             (k // per_row + 0.5) / rows))
        p[:, 0] += (k // per_row % 2) * 0.25 / per_row
        p = np.clip(p + rng.normal(0, float(params['jitter']), p.shape), 0.02, 0.98)
        p *= [w, 2 - w]
        r = np.full(N, 0.05)
        candidates = []
    if best is None:
        k = np.arange(N)
        fallback_p = np.column_stack(((k % 5 + 0.5) / 5, (k // 5 + 0.5) / 5))
        fallback_r = np.full(N, 0.099999999999)
        best = output(fallback_p, fallback_r)
        best_score = float(np.sum(fallback_r))
    z = np.r_[p[:, 0], p[:, 1], r, w]
    mu = float(params['mu0'])
    last = [z.copy()]
    for _ in range(int(params['stages'])):
        if time.monotonic() >= descent_end:
            break
        def objective(q):
            if time.monotonic() >= descent_end:
                raise StopSearch()
            last[0] = q.copy()
            return penalty(q, mu)
        try:
            z = minimize(objective, z, jac=True, method='L-BFGS-B',
                         options={'maxiter': int(params['maxiter'])}).x
        except StopSearch:
            z = last[0]
            break
        mu *= float(params['mu_growth'])
    # Prioritize the structurally changed packing over the unchanged centres.
    candidates.insert(0, np.column_stack((z[:N], z[N:2*N])))
    for pc in candidates:
        if time.monotonic() >= deadline - 0.015:
            break
        rc = lp_radii(pc, deadline - 0.01)
        if rc is None:
            continue
        rc = repair(pc, rc, deadline)
        if rc is not None and valid(pc, rc):
            score = float(np.sum(rc))
            if score > best_score:
                best, best_score = output(pc, rc), score
    return best
