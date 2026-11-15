import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'mu0': (10.0, 1.0, 1000.0, 'log'),
    'mu_growth': (8.0, 2.0, 30.0),
    'stages': (6, 2, 12),
    'maxiter': (150, 30, 1000),
    'cold_threshold': (1.5, 0.1, 2.2)
}
N = 21
I, J = np.triu_indices(N, 1)
K = np.arange(N)
M = len(I)

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
    return np.column_stack(((p - lo) * scale, r * scale))


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
    return np.asarray(pts) * scale, np.full(N, scale * (1 - 2e-12)), 8.0 * scale


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


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    search_end = start + 0.38 * budget_s
    best, best_score = None, -np.inf
    a0 = None
    if init is not None:
        try:
            a0 = np.asarray(init['circles'], dtype=float)
            score = exact_score(a0)
            if score > best_score:
                best, best_score = init, score
            if a0.shape != (N, 3) or not np.isfinite(a0).all():
                a0 = None
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
        consider(repair(np.column_stack((z[:N], z[N:2*N])), z[2*N:3*N]))
        mu *= float(params['mu_growth'])

    # Joint constrained polishing replaces the fixed-centre LP. The linear
    # boundary rows are constant; only the pairwise distance rows change.
    if time.monotonic() >= deadline:
        return best
    base = np.zeros((M + 4*N, 3*N + 1))
    rows = np.arange(M)
    base[rows, 2*N + I] = -1
    base[rows, 2*N + J] = -1
    base[M+K, K] = 1
    base[M+K, 2*N+K] = -1
    base[M+N+K, K] = -1
    base[M+N+K, 2*N+K] = -1
    base[M+N+K, -1] = 1
    base[M+2*N+K, N+K] = 1
    base[M+2*N+K, 2*N+K] = -1
    base[M+3*N+K, N+K] = -1
    base[M+3*N+K, 2*N+K] = -1
    base[M+3*N+K, -1] = -1
    grad = np.r_[np.zeros(2*N), -np.ones(N), 0.0]

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch

    def constraints(v):
        check()
        x, y, rr, ww = v[:N], v[N:2*N], v[2*N:3*N], v[-1]
        d = np.sqrt((x[I]-x[J])**2 + (y[I]-y[J])**2)
        return np.r_[d-rr[I]-rr[J], x-rr, ww-x-rr, y-rr, 2-ww-y-rr]

    def jacobian(v):
        check()
        dx = v[I] - v[J]
        dy = v[N+I] - v[N+J]
        d = np.maximum(np.sqrt(dx*dx + dy*dy), 1e-15)
        a = base.copy()
        a[rows, I] = dx/d
        a[rows, J] = -dx/d
        a[rows, N+I] = dy/d
        a[rows, N+J] = -dy/d
        return a

    def linear_objective(v):
        check()
        return -float(np.sum(v[2*N:3*N])), grad

    def polish_callback(v):
        check()
        consider(repair(np.column_stack((v[:N], v[N:2*N])), v[2*N:3*N]))

    try:
        result = minimize(linear_objective, z, method='SLSQP', jac=True,
                          bounds=[(None, None)]*(2*N) + [(1e-10, None)]*N + [(0.05, 1.95)],
                          constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                          callback=polish_callback,
                          options={'maxiter': 160, 'ftol': 2e-12})
        v = result.x
        consider(repair(np.column_stack((v[:N], v[N:2*N])), v[2*N:3*N]))
    except StopSearch:
        pass
    return best
