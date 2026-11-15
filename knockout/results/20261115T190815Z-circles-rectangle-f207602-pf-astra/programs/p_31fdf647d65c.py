import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linprog

PARAMS = {
    'aspect': (1.0, 0.5, 2.0, 'log'),
    'rows': (5, 3, 7),
    'jitter': (0.03, 0.0, 0.15),
    'mu0': (10.0, 1.0, 1000.0, 'log'),
    'mu_growth': (8.0, 2.0, 30.0),
    'stages': (6, 2, 12),
    'maxiter': (150, 30, 1000),
    'soft_factor': (0.02, 0.001, 1.0, 'log'),
}
N = 21
I, J = np.triu_indices(N, 1)

class StopSearch(Exception):
    pass

def perimeter(p, r):
    xs = list(map(lambda v: Fraction(float(v)), p[:, 0]))
    ys = list(map(lambda v: Fraction(float(v)), p[:, 1]))
    rs = list(map(lambda v: Fraction(float(v)), r))
    return (max(x + q for x, q in zip(xs, rs))
            - min(x - q for x, q in zip(xs, rs))
            + max(y + q for y, q in zip(ys, rs))
            - min(y - q for y, q in zip(ys, rs)))

def score(p, r):
    if p.shape != (21, 2) or r.shape != (21,):
        return -np.inf
    if not np.isfinite(p).all() or not np.isfinite(r).all() or not (r > 0).all():
        return -np.inf
    d = np.sqrt((p[I, 0] - p[J, 0]) ** 2 + (p[I, 1] - p[J, 1]) ** 2)
    if np.any(d < r[I] + r[J]) or perimeter(p, r) > 2:
        return -np.inf
    return float(np.sum(r))

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
    if time.monotonic() >= deadline:
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
    remaining = deadline - time.monotonic()
    if remaining <= 0.015:
        return None
    res = linprog(-np.r_[np.ones(N), np.zeros(4)], A_ub=a, b_ub=b,
                  bounds=[(0, None)]*N + [(None, None)]*4,
                  method='highs', options={'time_limit': remaining})
    return res.x[:N] if res.success else None

def repair(p, r, deadline):
    r = np.maximum(np.asarray(r, dtype=float), 1e-12) * (1 - 1e-12)
    d = np.sqrt((p[I, 0] - p[J, 0]) ** 2 + (p[I, 1] - p[J, 1]) ** 2)
    for _ in range(5):
        if time.monotonic() >= deadline:
            return r
        if not np.any(d < r[I] + r[J]):
            break
        for i, j, dd in zip(I, J, d):
            if time.monotonic() >= deadline:
                return r
            if r[i] + r[j] > dd * (1 - 1e-12):
                s = dd * (1 - 1e-12) / (r[i] + r[j])
                r[i] *= s
                r[j] *= s
    for _ in range(60):
        if time.monotonic() >= deadline:
            return r
        excess = perimeter(p, r) - 2
        if excess <= 0:
            break
        r *= 1 - max(2 * float(excess) / float(r.sum()), 1e-12)
    return np.maximum(r, 1e-12)

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    opt_end = start + 0.53 * budget_s
    rng = np.random.default_rng(seed)
    best, best_score = init, -np.inf
    if init is not None:
        arr = np.asarray(init['circles'], dtype=float)
        p0, r0 = arr[:, :2].copy(), arr[:, 2].copy()
        best_score = score(p0, r0)
        lo = (p0 - np.abs(r0)[:, None]).min(axis=0)
        side = (p0 + np.abs(r0)[:, None]).max(axis=0) - lo
        scale = 2 / side.sum()
        p, r, w = (p0 - lo)*scale, np.abs(r0)*scale, side[0]*scale
        original_centres = p.copy()
    else:
        # Valid fallback, independent of optimization success.
        idx = np.arange(N)
        fallback = np.column_stack((0.1 + 0.2*(idx % 5), 0.1 + 0.2*(idx // 5), np.full(N, 0.09)))
        best = {'circles': fallback.tolist()}
        best_score = score(fallback[:, :2], fallback[:, 2])
        w = 2 * float(params['aspect']) / (1 + float(params['aspect']))
        rows = int(params['rows'])
        cols = int(np.ceil(N / rows))
        p = np.column_stack(((idx % cols + 0.5)/cols + 0.25/cols*((idx // cols) % 2), (idx // cols + 0.5)/rows))
        p = np.clip(p + rng.normal(0, float(params['jitter']), p.shape), 0.02, 0.98)
        p *= [w, 2-w]
        r = np.full(N, 0.05)
        original_centres = None
    z = np.r_[p[:, 0], p[:, 1], r, w]
    mu0 = float(params['mu0'])
    # The sole search change: prepend one soft, topology-relaxing stage.
    schedule = np.r_[mu0 * float(params['soft_factor']),
                     mu0 * float(params['mu_growth']) ** np.arange(int(params['stages']))]
    for mu in schedule:
        if time.monotonic() >= opt_end:
            break
        last = [z.copy()]
        def fun(q):
            if time.monotonic() >= opt_end:
                raise StopSearch()
            return penalty(q, mu)
        def callback(q):
            last[0] = q.copy()
            if time.monotonic() >= opt_end:
                raise StopSearch()
        try:
            z = minimize(fun, z, jac=True, method='L-BFGS-B', callback=callback,
                         options={'maxiter': int(params['maxiter'])}).x
        except StopSearch:
            z = last[0]
            break
    candidates = [np.column_stack((z[:N], z[N:2*N]))]
    if original_centres is not None:
        candidates.append(original_centres)
    for pc in candidates:
        if time.monotonic() >= deadline:
            break
        rc = lp_radii(pc, deadline)
        if rc is None:
            continue
        rc = repair(pc, rc, deadline)
        value = score(pc, rc)
        if value > best_score:
            best_score = value
            best = {'circles': np.column_stack((pc, rc)).tolist()}
    return best
