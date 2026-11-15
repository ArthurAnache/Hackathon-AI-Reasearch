import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linprog

N = 21
PARAMS = {
    'aspect': (1.0, 0.5, 2.0, 'log'),
    'rows': (5, 3, 7),
    'jitter': (0.03, 0.0, 0.15),
    'init_jitter': (0.0, 0.0, 0.05),
    'radius_equalization': (1.0, 0.0, 1.0),
    'mu0': (10.0, 1.0, 1000.0, 'log'),
    'mu_growth': (8.0, 2.0, 30.0),
    'stages': (6, 2, 12),
    'maxiter': (150, 30, 1000)
}
I, J = np.triu_indices(N, 1)

class StopSearch(Exception):
    pass


def half_perimeter(p, r):
    xs = [Fraction(float(v)) for v in p[:, 0]]
    ys = [Fraction(float(v)) for v in p[:, 1]]
    rs = [Fraction(float(v)) for v in r]
    return (max(x + q for x, q in zip(xs, rs))
            - min(x - q for x, q in zip(xs, rs))
            + max(y + q for y, q in zip(ys, rs))
            - min(y - q for y, q in zip(ys, rs)))


def valid(p, r):
    if p.shape != (N, 2) or r.shape != (N,):
        return False
    if not np.isfinite(p).all() or not np.isfinite(r).all() or not (r > 0).all():
        return False
    d = np.sqrt((p[I, 0] - p[J, 0]) ** 2 + (p[I, 1] - p[J, 1]) ** 2)
    if (d < r[I] + r[J]).any():
        return False
    return half_perimeter(p, r) <= 2


def output(p, r):
    return {'circles': np.column_stack((p, r)).tolist()}


def repair(p, r, deadline):
    r = np.maximum(np.asarray(r, dtype=float), 1e-12) * (1 - 1e-12)
    d = np.sqrt((p[I, 0] - p[J, 0]) ** 2 + (p[I, 1] - p[J, 1]) ** 2)
    for unused in range(5):
        if time.monotonic() >= deadline:
            return None
        if not (d < r[I] + r[J]).any():
            break
        for i, j, dd in zip(I, J, d):
            if time.monotonic() >= deadline:
                return None
            if r[i] + r[j] > dd * (1 - 1e-12):
                scale = dd * (1 - 1e-12) / (r[i] + r[j])
                r[i] *= scale
                r[j] *= scale
    for unused in range(60):
        if time.monotonic() >= deadline:
            return None
        excess = half_perimeter(p, r) - 2
        if excess <= 0:
            return r
        r *= 1 - max(2 * float(excess) / float(r.sum()), 1e-12)
    return r


def lp_radii(p, deadline):
    remaining = deadline - time.monotonic()
    if remaining < 0.02:
        return None
    m = len(I)
    a = np.zeros((m + 4 * N + 1, N + 4))
    b = np.zeros(m + 4 * N + 1)
    a[np.arange(m), I] = 1
    a[np.arange(m), J] = 1
    b[:m] = np.sqrt((p[I, 0] - p[J, 0]) ** 2 + (p[I, 1] - p[J, 1]) ** 2)
    for k, (col, sign, coord) in enumerate([(N, 1, 0), (N + 1, -1, 0), (N + 2, 1, 1), (N + 3, -1, 1)]):
        if time.monotonic() >= deadline:
            return None
        rows = m + k * N + np.arange(N)
        a[rows, np.arange(N)] = 1
        a[rows, col] = sign
        b[rows] = sign * p[:, coord]
    a[-1, N:] = [-1, 1, -1, 1]
    b[-1] = 2
    res = linprog(-np.r_[np.ones(N), np.zeros(4)], A_ub=a, b_ub=b,
                  bounds=[(0, None)] * N + [(None, None)] * 4,
                  method='highs', options={'time_limit': max(0.001, deadline - time.monotonic())})
    return res.x[:N] if res.success else None


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


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    search_end = start + 0.57 * budget_s
    rng = np.random.default_rng(seed)
    best, best_score = None, -float('inf')
    if init is not None:
        arr = np.asarray(init['circles'], dtype=float)
        p0, r0 = arr[:, :2].copy(), arr[:, 2].copy()
        if valid(p0, r0):
            best, best_score = init, float(np.sum(r0))
        lo = (p0 - np.abs(r0)[:, None]).min(axis=0)
        side = (p0 + np.abs(r0)[:, None]).max(axis=0) - lo
        scale = 2 / side.sum()
        p, r, w = (p0 - lo) * scale, np.abs(r0) * scale, side[0] * scale
        p += rng.normal(0, float(params['init_jitter']), p.shape)
        # The sole search change: flatten the initial radius distribution.
        alpha = float(params['radius_equalization'])
        r = (1 - alpha) * r + alpha * np.mean(r)
    else:
        aspect = float(params['aspect'])
        w = 2 * aspect / (1 + aspect)
        rows = int(params['rows'])
        cols = int(np.ceil(N / rows))
        pts = []
        for k in range(rows):
            if time.monotonic() >= deadline:
                break
            for j in range(min(cols, N - len(pts))):
                if time.monotonic() >= deadline:
                    break
                pts.append(((j + .5) / cols + (.25 / cols if k % 2 else 0), (k + .5) / rows))
        if len(pts) != N:
            p = np.column_stack((np.arange(N) / 11, np.zeros(N)))
            return output(p, np.full(N, 1e-8))
        p = np.clip(np.asarray(pts) + rng.normal(0, float(params['jitter']), (N, 2)), .02, .98)
        p *= [w, 2 - w]
        r = np.full(N, .05)
    z = np.r_[p[:, 0], p[:, 1], r, w]
    last = [z.copy()]
    mu = float(params['mu0'])

    def objective(v):
        if time.monotonic() >= search_end:
            raise StopSearch
        last[0] = v.copy()
        return penalty(v, mu)

    for unused in range(int(params['stages'])):
        if time.monotonic() >= search_end:
            break
        try:
            z = minimize(objective, z, jac=True, method='L-BFGS-B',
                         options={'maxiter': int(params['maxiter'])}).x
        except StopSearch:
            z = last[0]
            break
        mu *= float(params['mu_growth'])
    pc = np.column_stack((z[:N], z[N:2*N]))
    rc = lp_radii(pc, deadline)
    if rc is not None:
        rc = repair(pc, rc, deadline)
        if rc is not None and valid(pc, rc):
            score = float(np.sum(rc))
            if score > best_score:
                best, best_score = output(pc, rc), score
    if best is not None:
        return best
    # A deterministic valid cold-start fallback, independent of failed optimization.
    return output(np.column_stack((np.arange(N) / 11, np.zeros(N))), np.full(N, 1e-8))
