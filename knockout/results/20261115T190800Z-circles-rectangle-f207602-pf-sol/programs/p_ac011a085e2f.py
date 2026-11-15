import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linprog

PARAMS = {
    'aspect': (1.0, 0.5, 2.0, 'log'),
    'rows': (5, 3, 7),
    'jitter': (0.03, 0.0, 0.15),
    'init_jitter': (0.0, 0.0, 0.05),
    'mu0': (10.0, 1.0, 1000.0, 'log'),
    'mu_growth': (8.0, 2.0, 30.0),
    'stages': (6, 2, 12),
    'maxiter': (150, 30, 1000),
    'polish_iters': (70, 10, 150),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
PAIR_JAC = np.zeros((M, 64))
PAIR_JAC[np.arange(M), 42 + I] = -1
PAIR_JAC[np.arange(M), 42 + J] = -1
WALL_JAC = np.zeros((84, 64))
K = np.arange(N)
WALL_JAC[K, K] = 1
WALL_JAC[K, 42 + K] = -1
WALL_JAC[21 + K, K] = -1
WALL_JAC[21 + K, 42 + K] = -1
WALL_JAC[21 + K, 63] = 1
WALL_JAC[42 + K, 21 + K] = 1
WALL_JAC[42 + K, 42 + K] = -1
WALL_JAC[63 + K, 21 + K] = -1
WALL_JAC[63 + K, 42 + K] = -1
WALL_JAC[63 + K, 63] = -1
OBJ_JAC = np.r_[np.zeros(42), -np.ones(21), 0.0]

class Timeout(Exception):
    pass


def perimeter(p, r):
    xs = list(map(Fraction, map(float, p[:, 0])))
    ys = list(map(Fraction, map(float, p[:, 1])))
    rs = list(map(Fraction, map(float, r)))
    return (max(map(lambda a, b: a+b, xs, rs)) - min(map(lambda a, b: a-b, xs, rs))
            + max(map(lambda a, b: a+b, ys, rs)) - min(map(lambda a, b: a-b, ys, rs)))


def score(p, r):
    if p.shape != (21, 2) or r.shape != (21,):
        return -np.inf
    if not np.isfinite(p).all() or not np.isfinite(r).all() or not (r > 0).all():
        return -np.inf
    d = np.sqrt((p[I, 0]-p[J, 0])**2 + (p[I, 1]-p[J, 1])**2)
    if np.any(d < r[I]+r[J]) or perimeter(p, r) > 2:
        return -np.inf
    return float(np.sum(r))


def repair(p, r, deadline):
    r = np.maximum(np.asarray(r, dtype=float), 1e-12).copy() * (1-1e-12)
    d = np.sqrt((p[I, 0]-p[J, 0])**2 + (p[I, 1]-p[J, 1])**2)
    for _ in range(5):
        if time.monotonic() >= deadline:
            return r
        if not np.any(d < r[I]+r[J]):
            break
        for i, j, dd in zip(I, J, d):
            if time.monotonic() >= deadline:
                return r
            if r[i]+r[j] > dd*(1-1e-12):
                s = dd*(1-1e-12)/(r[i]+r[j])
                r[i] *= s
                r[j] *= s
    for _ in range(60):
        if time.monotonic() >= deadline:
            break
        excess = perimeter(p, r)-2
        if excess <= 0:
            break
        r *= 1-max(2*float(excess)/float(r.sum()), 1e-12)
    return r


def penalty(z, mu):
    x, y, r, w = z[:21], z[21:42], z[42:63], z[63]
    dx, dy = x[I]-x[J], y[I]-y[J]
    d = np.sqrt(dx*dx+dy*dy+1e-18)
    v = np.maximum(r[I]+r[J]-d, 0)
    b = np.maximum(np.stack((r-x, r+x-w, r-y, r+y-(2-w))), 0)
    neg = np.maximum(-r, 0)
    f = -r.sum()+mu*((v*v).sum()+(b*b).sum()+(neg*neg).sum())
    gx, gy, gr = np.zeros(21), np.zeros(21), -np.ones(21)
    c = 2*mu*v
    np.add.at(gr, I, c)
    np.add.at(gr, J, c)
    np.add.at(gx, I, -c*dx/d)
    np.add.at(gx, J, c*dx/d)
    np.add.at(gy, I, -c*dy/d)
    np.add.at(gy, J, c*dy/d)
    gr += 2*mu*b.sum(axis=0)-2*mu*neg
    gx += 2*mu*(-b[0]+b[1])
    gy += 2*mu*(-b[2]+b[3])
    return f, np.r_[gx, gy, gr, 2*mu*(-b[1].sum()+b[3].sum())]


def constraints(z, jac=False):
    x, y, r, w = z[:21], z[21:42], z[42:63], z[63]
    dx, dy = x[I]-x[J], y[I]-y[J]
    d = np.sqrt(dx*dx+dy*dy)
    if not jac:
        return np.r_[d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r]
    a = PAIR_JAC.copy()
    d = np.maximum(d, 1e-100)
    a[np.arange(M), I] = dx/d
    a[np.arange(M), J] = -dx/d
    a[np.arange(M), 21+I] = dy/d
    a[np.arange(M), 21+J] = -dy/d
    return np.vstack((a, WALL_JAC))


def lp_radii(p, deadline):
    remaining = deadline-time.monotonic()
    if remaining < 0.06:
        return None
    a = np.zeros((M+85, 25))
    b = np.zeros(M+85)
    a[np.arange(M), I] = 1
    a[np.arange(M), J] = 1
    b[:M] = np.sqrt(((p[I]-p[J])**2).sum(axis=1))
    for k, (col, sign, coord) in enumerate(((21, 1, 0), (22, -1, 0), (23, 1, 1), (24, -1, 1))):
        if time.monotonic() >= deadline:
            return None
        rows = M+k*21+K
        a[rows, K] = 1
        a[rows, col] = sign
        b[rows] = sign*p[:, coord]
    a[-1, 21:] = [-1, 1, -1, 1]
    b[-1] = 2
    result = linprog(-np.r_[np.ones(21), np.zeros(4)], A_ub=a, b_ub=b,
                     bounds=[(0, None)]*21+[(None, None)]*4,
                     method='highs', options={'time_limit': max(0.001, remaining*0.6)})
    return result.x[:21] if result.success else None


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start+0.70*budget_s
    continuation_end = start+0.38*budget_s
    optimize_end = start+0.65*budget_s
    rng = np.random.default_rng(seed)
    best, best_score = init, -np.inf
    if init is not None:
        ar = np.asarray(init['circles'], dtype=float)
        p, r = ar[:, :2].copy(), ar[:, 2].copy()
        best_score = score(p, r)
        lo = (p-np.abs(r)[:, None]).min(axis=0)
        sides = (p+np.abs(r)[:, None]).max(axis=0)-lo
        scale = 2/sides.sum()
        p, r, w = (p-lo)*scale, np.abs(r)*scale, sides[0]*scale
        p += rng.normal(0, params['init_jitter'], p.shape)
    else:
        rows = int(params['rows'])
        cols = int(np.ceil(21/rows))
        k = np.arange(21)
        w = 2*params['aspect']/(1+params['aspect'])
        p = np.column_stack(((k % cols+0.5)/cols+0.25/cols*((k//cols) % 2), (k//cols+0.5)/rows))
        p = np.clip(p+rng.normal(0, params['jitter'], p.shape), 0.02, 0.98)*[w, 2-w]
        r = np.full(21, 0.05)
    z0 = np.r_[p[:, 0], p[:, 1], r, w]
    z = z0.copy()

    def consider(zz):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        pp = np.column_stack((zz[:21], zz[21:42]))
        rr = repair(pp, zz[42:63], deadline)
        value = score(pp, rr)
        if value > best_score:
            best_score = value
            best = {'circles': np.column_stack((pp, rr)).tolist()}

    mu = float(params['mu0'])
    for _ in range(int(params['stages'])):
        if time.monotonic() >= continuation_end:
            break
        def fun(v):
            if time.monotonic() >= continuation_end:
                raise Timeout()
            return penalty(v, mu)
        try:
            z = minimize(fun, z, jac=True, method='L-BFGS-B', options={'maxiter': int(params['maxiter'])}).x
        except Timeout:
            break
        mu *= params['mu_growth']
    consider(z)

    # Direct geometric polishing removes the finite-penalty bias. Start with
    # the incumbent as well so that continuation cannot discard its basin.
    for initial in (z0, z):
        if time.monotonic() >= optimize_end:
            break
        last = [initial.copy()]
        def objective(v):
            if time.monotonic() >= optimize_end:
                raise Timeout()
            return -float(v[42:63].sum()), OBJ_JAC
        def callback(v):
            if time.monotonic() >= optimize_end:
                raise Timeout()
            last[0] = v.copy()
        try:
            res = minimize(objective, initial, jac=True, method='SLSQP',
                           bounds=[(None, None)]*42+[(1e-10, None)]*21+[(1e-8, 2-1e-8)],
                           constraints={'type': 'ineq', 'fun': constraints,
                                        'jac': lambda v: constraints(v, True)},
                           callback=callback,
                           options={'maxiter': int(params['polish_iters']), 'ftol': 2e-14})
            last[0] = res.x
        except Timeout:
            pass
        consider(last[0])

    if time.monotonic() < optimize_end:
        pp = np.column_stack((z[:21], z[21:42]))
        rr = lp_radii(pp, optimize_end)
        if rr is not None:
            zz = z.copy()
            zz[42:63] = rr
            consider(zz)
    if best is None:
        # Deterministic valid cold-start fallback, independent of optimization.
        pp = np.column_stack((np.arange(21)*0.08, np.zeros(21)))
        rr = np.full(21, 0.01)
        best = {'circles': np.column_stack((pp, rr)).tolist()}
    return best
