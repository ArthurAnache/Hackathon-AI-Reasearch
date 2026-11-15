import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linprog

PARAMS = {
    'mu0': (10.0, 1.0, 1000.0, 'log'),
    'mu_growth': (8.0, 2.0, 30.0),
    'stages': (6, 2, 12),
    'maxiter': (150, 30, 1000),
    'cold_threshold': (1.5, 0.1, 2.2)
}
N = 21
I, J = np.triu_indices(N, 1)

class StopSearch(Exception):
    pass


def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    x, y, r = ([Fraction(float(v)) for v in a[:, k]] for k in range(3))
    perimeter = (max(u + v for u, v in zip(x, r)) - min(u - v for u, v in zip(x, r))
                 + max(u + v for u, v in zip(y, r)) - min(u - v for u, v in zip(y, r)))
    if perimeter > 2:
        return -np.inf
    d = np.sqrt((a[I, 0] - a[J, 0]) ** 2 + (a[I, 1] - a[J, 1]) ** 2)
    if np.any(d < a[I, 2] + a[J, 2]):
        return -np.inf
    return float(np.sum(a[:, 2]))


def repair(p, r):
    p = np.array(p, dtype=float, copy=True)
    r = np.maximum(np.array(r, dtype=float, copy=True), 1e-12)
    if not np.isfinite(p).all() or not np.isfinite(r).all():
        return None
    d = np.sqrt((p[I, 0] - p[J, 0]) ** 2 + (p[I, 1] - p[J, 1]) ** 2)
    factor = min(1.0, float(np.min(d / (r[I] + r[J]))))
    if factor <= 0:
        return None
    r *= factor * (1 - 2e-12)
    lo = np.min(p - r[:, None], axis=0)
    hi = np.max(p + r[:, None], axis=0)
    scale = min(1.0, 2.0 / float(np.sum(hi - lo))) * (1 - 2e-12)
    p = (p - lo) * scale
    r *= scale
    return np.column_stack((p, r))


def lattice(deadline):
    pts = []
    for row in range(6):
        if time.monotonic() >= deadline:
            raise StopSearch
        count = 4 if row % 2 == 0 else 3
        for col in range(count):
            if time.monotonic() >= deadline:
                raise StopSearch
            pts.append((1 + 2 * col + row % 2, 1 + np.sqrt(3.0) * row))
    scale = 2.0 / (10.0 + 5.0 * np.sqrt(3.0))
    p = np.asarray(pts) * scale
    r = np.full(N, scale * (1 - 2e-12))
    return p, r, 8.0 * scale


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
    c = 2 * mu * v
    gr = -np.ones(N) + np.bincount(I, c, minlength=N) + np.bincount(J, c, minlength=N)
    cx, cy = c * dx / d, c * dy / d
    gx = -np.bincount(I, cx, minlength=N) + np.bincount(J, cx, minlength=N)
    gy = -np.bincount(I, cy, minlength=N) + np.bincount(J, cy, minlength=N)
    gr += 2 * mu * (b0 + b1 + b2 + b3 - neg)
    gx += 2 * mu * (-b0 + b1)
    gy += 2 * mu * (-b2 + b3)
    gw = 2 * mu * (-b1.sum() + b3.sum())
    return f, np.concatenate((gx, gy, gr, [gw]))


def lp_radii(p, deadline):
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
    result = linprog(-np.r_[np.ones(N), np.zeros(4)], A_ub=a, b_ub=b,
                     bounds=[(1e-12, None)]*N + [(None, None)]*4,
                     method='highs', options={'time_limit': max(0.001, remaining * 0.65)})
    return result.x[:N] if result.success else None


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    search_end = start + 0.55 * budget_s
    best, best_score = None, -np.inf
    a0 = None
    if init is not None:
        try:
            a0 = np.asarray(init['circles'], dtype=float)
            score = exact_score(a0)
            if score > best_score:
                best, best_score = init, score
        except (ValueError, TypeError, KeyError, OverflowError):
            a0 = None

    def consider(a):
        nonlocal best, best_score
        if a is None:
            return
        score = exact_score(a)
        if score > best_score:
            best_score = score
            best = {'circles': a.tolist()}

    # Tiny cold circles carry no useful contact graph: start with a compact
    # triangular lattice instead, while retaining init as the exact baseline.
    if a0 is None or best_score < float(params['cold_threshold']):
        try:
            p, r, w = lattice(deadline)
        except StopSearch:
            if best is not None:
                return best
            p = np.column_stack((np.arange(N) / 12.0, np.zeros(N)))
            r = np.full(N, 0.02)
            w = 1.8
    else:
        p, r = a0[:, :2].copy(), a0[:, 2].copy()
        lo = np.min(p - r[:, None], axis=0)
        side = np.max(p + r[:, None], axis=0) - lo
        scale = 2.0 / float(side.sum())
        p, r, w = (p - lo) * scale, r * scale, side[0] * scale
    consider(repair(p, r))
    z = np.concatenate((p[:, 0], p[:, 1], r, [w]))
    accepted = z.copy()
    mu = float(params['mu0'])

    def objective(v):
        if time.monotonic() >= search_end:
            raise StopSearch
        return penalty(v, mu)

    def callback(v):
        nonlocal accepted
        accepted = v.copy()
        if time.monotonic() >= search_end:
            raise StopSearch

    for stage in range(int(params['stages'])):
        if time.monotonic() >= search_end:
            break
        accepted = z.copy()
        try:
            result = minimize(objective, z, jac=True, method='L-BFGS-B', callback=callback,
                              options={'maxiter': int(params['maxiter'])})
            z = result.x
        except StopSearch:
            z = accepted
        p = np.column_stack((z[:N], z[N:2*N]))
        consider(repair(p, z[2*N:3*N]))
        mu *= float(params['mu_growth'])
    if time.monotonic() < deadline:
        p = np.column_stack((z[:N], z[N:2*N]))
        rlp = lp_radii(p, deadline)
        if rlp is not None:
            consider(repair(p, rlp))
    return best
