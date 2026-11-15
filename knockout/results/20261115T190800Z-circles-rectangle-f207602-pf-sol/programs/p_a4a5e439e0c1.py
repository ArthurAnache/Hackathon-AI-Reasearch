import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'excursion': (0.23, 0.05, 0.55),
    'penalty_iterations': (75, 20, 160),
    'refine_iterations': (90, 20, 200),
    'initial_penalty': (25.0, 1.0, 200.0, 'log'),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class StopSearch(Exception):
    pass


def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or not (a[:, 2] > 0).all():
        return -float('inf')
    d = np.sqrt((a[I, 0] - a[J, 0]) ** 2 + (a[I, 1] - a[J, 1]) ** 2)
    if np.any(d < a[I, 2] + a[J, 2]):
        return -float('inf')
    x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
    perimeter = (max(u + v for u, v in zip(x, r)) - min(u - v for u, v in zip(x, r))
                 + max(u + v for u, v in zip(y, r)) - min(u - v for u, v in zip(y, r)))
    if perimeter > 2:
        return -float('inf')
    return float(np.sum(a[:, 2]))


def unpack(z):
    return z[:N], z[N:2*N], z[2*N:3*N], z[-1]


def penalty(z, mu):
    x, y, r, w = unpack(z)
    dx, dy = x[I] - x[J], y[I] - y[J]
    d = np.sqrt(dx*dx + dy*dy + 1e-26)
    v = np.maximum(r[I] + r[J] - d, 0.0)
    b0 = np.maximum(r - x, 0.0)
    b1 = np.maximum(r + x - w, 0.0)
    b2 = np.maximum(r - y, 0.0)
    b3 = np.maximum(r + y + w - 2, 0.0)
    c = 2*mu*v
    cx, cy = c*dx/d, c*dy/d
    gx = np.bincount(I, -cx, N) + np.bincount(J, cx, N) + 2*mu*(b1-b0)
    gy = np.bincount(I, -cy, N) + np.bincount(J, cy, N) + 2*mu*(b3-b2)
    gr = -np.ones(N) + np.bincount(I, c, N) + np.bincount(J, c, N) + 2*mu*(b0+b1+b2+b3)
    gw = 2*mu*(b3.sum()-b1.sum())
    f = -r.sum() + mu*(v@v+b0@b0+b1@b1+b2@b2+b3@b3)
    return f, np.r_[gx, gy, gr, gw]


def constraints(z):
    x, y, r, w = unpack(z)
    dx, dy = x[I]-x[J], y[I]-y[J]
    d = np.sqrt(dx*dx+dy*dy+1e-30)
    return np.r_[d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r]


def constraint_jac(z):
    x, y, r, w = unpack(z)
    dx, dy = x[I]-x[J], y[I]-y[J]
    d = np.sqrt(dx*dx+dy*dy+1e-30)
    a = np.zeros((M+4*N, 3*N+1))
    q = np.arange(M)
    a[q, I], a[q, J] = dx/d, -dx/d
    a[q, N+I], a[q, N+J] = dy/d, -dy/d
    a[q, 2*N+I], a[q, 2*N+J] = -1, -1
    k = np.arange(N)
    for b in range(4):
        rows = M+b*N+k
        a[rows, 2*N+k] = -1
        a[rows, (N if b >= 2 else 0)+k] = 1 if b % 2 == 0 else -1
    a[M+N:M+2*N, -1] = 1
    a[M+3*N:M+4*N, -1] = -1
    return a


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        p = np.array([[(i % 5 + .5)/5, (i // 5 + .5)/5] for i in range(N)])
        a0 = np.column_stack((p, np.full(N, .045)))
    else:
        a0 = np.asarray(init['circles'], dtype=float).copy()
    best = init
    best_score = exact_score(a0)
    if best is None and np.isfinite(best_score):
        best = {'circles': a0.tolist()}

    def consider(z):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        x, y, r, w = unpack(z)
        p = np.column_stack((x, y))
        r = np.maximum(r.copy(), 1e-14)
        d = np.sqrt((x[I]-x[J])**2+(y[I]-y[J])**2)
        ratio = min(1.0, float(np.min(d/(r[I]+r[J]))))
        r *= max(0.0, ratio)*(1-2e-14)
        lo = np.min(p-r[:, None], axis=0)
        hi = np.max(p+r[:, None], axis=0)
        size = float(np.sum(hi-lo))
        if size <= 0 or not np.isfinite(size):
            return
        scale = (2.0/size)*(1-2e-14)
        a = np.column_stack(((p-lo)*scale, r*scale))
        s = exact_score(a)
        if s > best_score:
            best_score, best = s, {'circles': a.tolist()}

    lo = np.min(a0[:, :2]-a0[:, 2, None], axis=0)
    hi = np.max(a0[:, :2]+a0[:, 2, None], axis=0)
    scale = 2.0/float(np.sum(hi-lo))
    p = (a0[:, :2]-lo)*scale
    r = a0[:, 2]*scale
    w0 = (hi[0]-lo[0])*scale
    base = np.r_[p[:, 0], p[:, 1], r, w0]
    bounds = [(0.0, 2.0)]*(2*N)+[(1e-10, .6)]*N+[(.15, 1.85)]
    objective_gradient = np.r_[np.zeros(2*N), -np.ones(N), 0.0]
    direction = 1 if rng.random() < .5 else -1

    for attempt in range(2):
        now = time.monotonic()
        if now >= deadline:
            break
        local_end = now + (deadline-now)/(2-attempt)
        penalty_end = now + .57*(local_end-now)
        target = float(np.clip(w0 + direction*((-1)**attempt)*float(params['excursion']), .25, 1.75))
        z = base.copy()
        z[:N] *= target/w0
        z[N:2*N] *= (2-target)/(2-w0)
        z[-1] = target
        latest = [z.copy()]
        try:
            for stage, multiple in enumerate((1.0, 8.0, 64.0)):
                if time.monotonic() >= penalty_end:
                    break
                stage_bounds = bounds.copy()
                if stage == 0:
                    stage_bounds[-1] = (target, target)
                elif stage == 1:
                    stage_bounds[-1] = (max(.15, target-.07), min(1.85, target+.07))
                mu = float(params['initial_penalty'])*multiple
                def fun(v):
                    if time.monotonic() >= penalty_end:
                        raise StopSearch()
                    latest[0] = v.copy()
                    return penalty(v, mu)
                z = minimize(fun, z, jac=True, bounds=stage_bounds, method='L-BFGS-B',
                             options={'maxiter': int(params['penalty_iterations']), 'ftol': 1e-11, 'maxls': 15}).x
                latest[0] = z.copy()
        except StopSearch:
            z = latest[0]
        consider(z)
        if time.monotonic() >= local_end:
            continue
        latest[0] = z.copy()
        def obj(v):
            if time.monotonic() >= local_end:
                raise StopSearch()
            latest[0] = v.copy()
            return -float(v[2*N:3*N].sum())
        def con(v):
            if time.monotonic() >= local_end:
                raise StopSearch()
            return constraints(v)
        try:
            result = minimize(obj, z, jac=lambda v: objective_gradient, method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': con, 'jac': constraint_jac},
                              options={'maxiter': int(params['refine_iterations']), 'ftol': 2e-12})
            z = result.x
        except StopSearch:
            z = latest[0]
        consider(z)
    if best is None:
        return {'circles': a0.tolist()}
    return best
