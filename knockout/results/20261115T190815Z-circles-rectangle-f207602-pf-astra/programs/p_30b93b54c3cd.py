import time
from fractions import Fraction
import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import minimize

PARAMS = {
    'contact_tol': (0.000002, 0.00000001, 0.0001, 'log'),
    'newton_steps': (7, 2, 15),
    'smoothing_mu': (35.0, 5.0, 300.0, 'log'),
    'smoothing_steps': (55, 10, 120),
    'perturbation': (0.012, 0.0, 0.06)
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
K = np.arange(N)

class Deadline(Exception):
    pass

def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return -np.inf
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    w = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
    h = max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
    if w+h > Fraction(2):
        return -np.inf
    return float(np.sum(a[:, 2]))

def unpack(z):
    return np.column_stack((z[:N], z[N:2*N], z[2*N:3*N]))

def encode(a):
    p = a[:, :2].copy()
    r = a[:, 2].copy()
    lo = np.min(p-r[:, None], axis=0)
    sides = np.max(p+r[:, None], axis=0)-lo
    scale = 2.0/float(sides.sum())
    p = (p-lo)*scale
    return np.r_[p[:, 0], p[:, 1], r*scale, sides[0]*scale]

def constraints(z, jac=False):
    x, y, r, w = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
    dx, dy = x[I]-x[J], y[I]-y[J]
    d = np.sqrt(dx*dx+dy*dy)
    g = np.r_[d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r]
    if not jac:
        return g
    a = np.zeros((M+4*N, 3*N+1))
    rows = np.arange(M)
    d = np.maximum(d, 1e-100)
    a[rows, I] = dx/d
    a[rows, J] = -dx/d
    a[rows, N+I] = dy/d
    a[rows, N+J] = -dy/d
    a[rows, 2*N+I] = -1
    a[rows, 2*N+J] = -1
    a[M+K, K] = 1
    a[M+K, 2*N+K] = -1
    a[M+N+K, K] = -1
    a[M+N+K, 2*N+K] = -1
    a[M+N+K, -1] = 1
    a[M+2*N+K, N+K] = 1
    a[M+2*N+K, 2*N+K] = -1
    a[M+3*N+K, N+K] = -1
    a[M+3*N+K, 2*N+K] = -1
    a[M+3*N+K, -1] = -1
    return g, a

def repair(z):
    a = unpack(z)
    if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return None
    p, r = a[:, :2].copy(), a[:, 2].copy()
    d = np.sqrt((p[I, 0]-p[J, 0])**2+(p[I, 1]-p[J, 1])**2)
    shrink = min(1.0, float(np.min(d/(r[I]+r[J]))))
    if shrink <= 0:
        return None
    r *= shrink*(1-8e-15)
    lo = np.min(p-r[:, None], axis=0)
    side = np.max(p+r[:, None], axis=0)-lo
    scale = 2*(1-8e-15)/float(side.sum())
    return np.column_stack(((p-lo)*scale, r*scale))

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    stop = start+0.70*budget_s
    rng = np.random.default_rng(seed)
    ids = np.arange(N)
    fallback = np.column_stack(((ids%7)*0.2, (ids//7)*0.2, np.full(N, 0.1-1e-13)))
    best = {'circles': fallback.tolist()}
    best_score = exact_score(fallback)
    if init is not None:
        original = np.asarray(init['circles'], dtype=float)
        value = exact_score(original)
        if value >= best_score:
            best, best_score = init, value
    z0 = encode(np.asarray(best['circles'], dtype=float))
    grad = np.r_[np.zeros(2*N), -np.ones(N), 0.0]

    def check(limit=stop):
        if time.monotonic() >= limit:
            raise Deadline()

    def consider(z):
        nonlocal best, best_score
        if time.monotonic() >= stop:
            return
        a = repair(z)
        if a is not None:
            value = exact_score(a)
            if value > best_score:
                best_score = value
                best = {'circles': a.tolist()}

    def contact_correct(z, limit):
        z = z.copy()
        active = np.flatnonzero(constraints(z) < float(params['contact_tol']))
        if len(active) < 3*N:
            return z
        for _ in range(int(params['newton_steps'])):
            check(limit)
            g, a = constraints(z, True)
            step, _, _, _ = lstsq(a[active], -g[active], cond=1e-12, lapack_driver='gelsy', check_finite=False)
            if np.linalg.norm(step) > 0.03:
                break
            z += step
            consider(z)
            if np.max(np.abs(step)) < 2e-15:
                break
        return z

    bounds = [(None, None)]*(2*N)+[(1e-8, 0.5)]*N+[(0.2, 1.8)]

    def polish(z, limit, iterations=100):
        last = [z.copy()]
        def fun(v):
            check(limit)
            return -float(np.sum(v[2*N:3*N]))
        def con(v):
            check(limit)
            return constraints(v)
        def jac(v):
            check(limit)
            return constraints(v, True)[1]
        def callback(v):
            check(limit)
            last[0] = v.copy()
            consider(v)
        try:
            result = minimize(fun, z, jac=lambda v: grad, method='SLSQP', bounds=bounds,
                constraints={'type': 'ineq', 'fun': con, 'jac': jac}, callback=callback,
                options={'ftol': 3e-14, 'maxiter': iterations, 'disp': False})
            last[0] = result.x
        except Deadline:
            pass
        consider(last[0])
        return last[0]

    try:
        z = contact_correct(z0, min(stop, start+0.12*budget_s))
    except Deadline:
        z = z0
    if time.monotonic() < start+0.27*budget_s:
        z = polish(z, min(stop, start+0.27*budget_s), 65)
    if time.monotonic() < start+0.32*budget_s:
        try:
            contact_correct(encode(np.asarray(best['circles'], dtype=float)), min(stop, start+0.32*budget_s))
        except Deadline:
            pass

    # Keep penalty smoothing, but use it only to explore before a hard-constrained finish.
    z = encode(np.asarray(best['circles'], dtype=float))
    a = unpack(z)
    anchor = int(rng.integers(N))
    nearest = np.argsort(np.sum((a[:, :2]-a[anchor, :2])**2, axis=1))[:4]
    shift = rng.normal(size=(4, 2))*float(params['perturbation'])
    z[nearest] += shift[:, 0]
    z[N+nearest] += shift[:, 1]
    smooth_end = min(stop, start+0.43*budget_s)
    for factor in (1.0, 8.0):
        if time.monotonic() >= smooth_end:
            break
        mu = float(params['smoothing_mu'])*factor
        last = [z.copy()]
        def penalty(v):
            check(smooth_end)
            last[0] = v.copy()
            g, j = constraints(v, True)
            violation = np.minimum(g, 0)
            f = -float(np.sum(v[2*N:3*N]))+mu*float(violation@violation)
            return f, grad+2*mu*(violation@j)
        try:
            res = minimize(penalty, z, jac=True, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': int(params['smoothing_steps']), 'ftol': 1e-11, 'maxls': 15})
            z = res.x
        except Deadline:
            z = last[0]
        consider(z)
    if time.monotonic() < stop:
        z = polish(z, stop-0.025*budget_s, 150)
    if time.monotonic() < stop-0.006*budget_s:
        try:
            contact_correct(z, stop-0.003*budget_s)
        except Deadline:
            pass
    return best
