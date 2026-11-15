import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linprog

N = 21
I, J = np.triu_indices(N, 1)
K = np.arange(N)
M = len(I)
PARAMS = {
    'mu0': (10.0, 1.0, 1000.0, 'log'),
    'mu_growth': (8.0, 2.0, 30.0),
    'penalty_stages': (3, 1, 5),
    'maxiter': (130, 30, 400),
    'polish_iterations': (150, 20, 400),
    'polish_ftol': (1e-11, 1e-13, 1e-7, 'log')
}

class TimeLimit(Exception):
    pass


def valid_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    r = a[:, 2]
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < r[I] + r[J]):
        return -np.inf
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    q = [Fraction(float(v)) for v in r]
    width = max(u+v for u, v in zip(x, q))-min(u-v for u, v in zip(x, q))
    height = max(u+v for u, v in zip(y, q))-min(u-v for u, v in zip(y, q))
    if width + height > 2:
        return -np.inf
    return float(np.sum(r))


def safe_candidate(p, r):
    if not np.isfinite(p).all() or not np.isfinite(r).all():
        return None
    r = np.maximum(r, 1e-13).copy()
    d = np.sqrt((p[I, 0]-p[J, 0])**2 + (p[I, 1]-p[J, 1])**2)
    factor = min(1.0, float(np.min(d/(r[I]+r[J]))))
    if factor <= 0:
        return None
    r *= factor * (1-2e-13)
    lo = np.min(p-r[:, None], axis=0)
    hi = np.max(p+r[:, None], axis=0)
    scale = (2.0/(hi-lo).sum()) * (1-2e-13)
    return np.column_stack(((p-lo)*scale, r*scale))


def penalty(z, mu):
    x, y, r, w = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
    dx, dy = x[I]-x[J], y[I]-y[J]
    d = np.sqrt(dx*dx+dy*dy+1e-18)
    v = np.maximum(r[I]+r[J]-d, 0)
    b0 = np.maximum(r-x, 0)
    b1 = np.maximum(r+x-w, 0)
    b2 = np.maximum(r-y, 0)
    b3 = np.maximum(r+y-2+w, 0)
    neg = np.maximum(-r, 0)
    f = -r.sum()+mu*(v@v+b0@b0+b1@b1+b2@b2+b3@b3+neg@neg)
    c = 2*mu*v
    gx = np.bincount(I, -c*dx/d, minlength=N)+np.bincount(J, c*dx/d, minlength=N)
    gy = np.bincount(I, -c*dy/d, minlength=N)+np.bincount(J, c*dy/d, minlength=N)
    gr = -np.ones(N)+np.bincount(I, c, minlength=N)+np.bincount(J, c, minlength=N)
    gx += 2*mu*(-b0+b1)
    gy += 2*mu*(-b2+b3)
    gr += 2*mu*(b0+b1+b2+b3-neg)
    return f, np.r_[gx, gy, gr, 2*mu*(-b1.sum()+b3.sum())]


def constraints(z):
    x, y, r, w = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
    d = np.sqrt((x[I]-x[J])**2+(y[I]-y[J])**2+1e-30)
    return np.r_[d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r]


def constraint_jac(z):
    x, y = z[:N], z[N:2*N]
    dx, dy = x[I]-x[J], y[I]-y[J]
    d = np.sqrt(dx*dx+dy*dy+1e-30)
    a = np.zeros((M+4*N, 3*N+1))
    t = np.arange(M)
    a[t, I], a[t, J] = dx/d, -dx/d
    a[t, N+I], a[t, N+J] = dy/d, -dy/d
    a[t, 2*N+I], a[t, 2*N+J] = -1, -1
    a[M+K, K] = 1
    a[M+N+K, K] = -1
    a[M+N+K, -1] = 1
    a[M+2*N+K, N+K] = 1
    a[M+3*N+K, N+K] = -1
    a[M+3*N+K, -1] = -1
    a[M+K, 2*N+K] = -1
    a[M+N+K, 2*N+K] = -1
    a[M+2*N+K, 2*N+K] = -1
    a[M+3*N+K, 2*N+K] = -1
    return a


def lp_radii(p, remaining):
    a = np.zeros((M+4*N+1, N+4))
    b = np.zeros(M+4*N+1)
    t = np.arange(M)
    a[t, I], a[t, J] = 1, 1
    b[:M] = np.sqrt((p[I, 0]-p[J, 0])**2+(p[I, 1]-p[J, 1])**2)
    a[M+K, K] = 1
    a[M+K, N] = 1
    b[M+K] = p[:, 0]
    a[M+N+K, K] = 1
    a[M+N+K, N+1] = -1
    b[M+N+K] = -p[:, 0]
    a[M+2*N+K, K] = 1
    a[M+2*N+K, N+2] = 1
    b[M+2*N+K] = p[:, 1]
    a[M+3*N+K, K] = 1
    a[M+3*N+K, N+3] = -1
    b[M+3*N+K] = -p[:, 1]
    a[-1, N:] = [-1, 1, -1, 1]
    b[-1] = 2
    res = linprog(-np.r_[np.ones(N), np.zeros(4)], A_ub=a, b_ub=b,
                  bounds=[(1e-12, None)]*N+[(None, None)]*4,
                  method='highs', options={'time_limit': max(0.01, remaining), 'threads': 1})
    return res.x[:N] if res.success else None


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    opt_end = start+0.62*budget_s
    finish = start+0.71*budget_s
    rng = np.random.default_rng(seed)
    best = None
    best_score = -np.inf

    def consider(a):
        nonlocal best, best_score
        if a is None:
            return
        score = valid_score(a)
        if score > best_score:
            best_score = score
            best = {'circles': a.tolist()}

    # An unconditional feasible fallback also protects against malformed starts.
    fallback = np.array([[(i%7+0.5)*0.2, (i//7+0.5)*0.2, 0.1*(1-1e-12)] for i in range(N)])
    consider(fallback)
    a0 = None
    if init is not None:
        try:
            a0 = np.asarray(init['circles'], dtype=float)
            score = valid_score(a0)
            if score >= best_score:
                best, best_score = init, score
            if a0.shape != (N, 3) or not np.isfinite(a0).all():
                a0 = None
        except (ValueError, KeyError, TypeError):
            a0 = None
    if a0 is None:
        p = np.array([[(i%5+0.5+0.25*((i//5)%2))/5, (i//5+0.5)/5] for i in range(N)])
        p += rng.normal(0, 0.03, p.shape)
        r = np.full(N, 0.05)
        w = 1.0
    else:
        r = np.maximum(np.abs(a0[:, 2]), 1e-12)
        lo = np.min(a0[:, :2]-r[:, None], axis=0)
        side = np.max(a0[:, :2]+r[:, None], axis=0)-lo
        scale = 2/side.sum()
        p, r, w = (a0[:, :2]-lo)*scale, r*scale, side[0]*scale
    z = np.r_[p[:, 0], p[:, 1], r, w]
    last = z.copy()

    def check():
        if time.monotonic() >= opt_end:
            raise TimeLimit()

    def callback(v):
        nonlocal last
        check()
        last = v.copy()

    def keep(v):
        consider(safe_candidate(np.column_stack((v[:N], v[N:2*N])), v[2*N:3*N]))

    mu = float(params['mu0'])
    for stage in range(int(params['penalty_stages'])):
        if time.monotonic() >= opt_end:
            break
        def fun(v):
            check()
            return penalty(v, mu)
        try:
            z = minimize(fun, z, jac=True, method='L-BFGS-B', callback=callback,
                         options={'maxiter': int(params['maxiter']), 'ftol': 1e-11}).x
            last = z.copy()
        except TimeLimit:
            z = last.copy()
            keep(z)
            break
        keep(z)
        mu *= float(params['mu_growth'])

    # Replace the stiff final penalty stages with the actual contact inequalities.
    if time.monotonic() < opt_end:
        objective_gradient = np.r_[np.zeros(2*N), -np.ones(N), 0.0]
        def objective(v):
            check()
            return -float(v[2*N:3*N].sum()), objective_gradient
        def con(v):
            check()
            return constraints(v)
        def jac(v):
            check()
            return constraint_jac(v)
        try:
            z = minimize(objective, z, jac=True, method='SLSQP', callback=callback,
                         bounds=[(None, None)]*(2*N)+[(1e-10, 0.5)]*N+[(0.01, 1.99)],
                         constraints={'type': 'ineq', 'fun': con, 'jac': jac},
                         options={'maxiter': int(params['polish_iterations']),
                                  'ftol': float(params['polish_ftol'])}).x
        except TimeLimit:
            z = last.copy()
        keep(z)

    if time.monotonic() < finish-0.06:
        p = np.column_stack((z[:N], z[N:2*N]))
        rlp = lp_radii(p, finish-time.monotonic()-0.02)
        if rlp is not None:
            consider(safe_candidate(p, rlp))
    return best
