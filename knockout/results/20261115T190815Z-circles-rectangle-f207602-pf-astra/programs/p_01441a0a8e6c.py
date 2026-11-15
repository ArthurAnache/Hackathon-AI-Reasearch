import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linprog

PARAMS = {
    'aspect': (1.1, 0.7, 1.5),
    'jitter': (0.008, 0.0, 0.04),
    'mu0': (10.0, 1.0, 1000.0, 'log'),
    'mu_growth': (8.0, 2.0, 30.0),
    'stages': (6, 3, 10),
    'maxiter': (150, 30, 500),
    'symmetric_stages': (4, 1, 6)
}
N = 21
I, J = np.triu_indices(N, 1)
REP = np.r_[np.arange(12), np.arange(3, 12)]

class StopSearch(Exception):
    pass


def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or not (a[:, 2] > 0).all():
        return -np.inf
    r = a[:, 2]
    d = np.sqrt((a[I, 0] - a[J, 0]) ** 2 + (a[I, 1] - a[J, 1]) ** 2)
    if np.any(d < r[I] + r[J]):
        return -np.inf
    x = list(map(lambda v: Fraction(float(v)), a[:, 0]))
    y = list(map(lambda v: Fraction(float(v)), a[:, 1]))
    q = list(map(lambda v: Fraction(float(v)), r))
    w = max(u + v for u, v in zip(x, q)) - min(u - v for u, v in zip(x, q))
    h = max(u + v for u, v in zip(y, q)) - min(u - v for u, v in zip(y, q))
    if w + h > 2:
        return -np.inf
    return float(np.sum(r))


def penalty(z, mu):
    x, y, r, w = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
    dx, dy = x[I] - x[J], y[I] - y[J]
    d = np.sqrt(dx*dx + dy*dy + 1e-18)
    v = np.maximum(r[I] + r[J] - d, 0)
    b0 = np.maximum(r-x, 0)
    b1 = np.maximum(r+x-w, 0)
    b2 = np.maximum(r-y, 0)
    b3 = np.maximum(r+y-(2-w), 0)
    neg = np.maximum(-r, 0)
    f = -r.sum() + mu*(v@v + b0@b0 + b1@b1 + b2@b2 + b3@b3 + neg@neg)
    c = 2*mu*v
    gx = np.bincount(I, weights=-c*dx/d, minlength=N) + np.bincount(J, weights=c*dx/d, minlength=N)
    gy = np.bincount(I, weights=-c*dy/d, minlength=N) + np.bincount(J, weights=c*dy/d, minlength=N)
    gr = -np.ones(N) + np.bincount(I, weights=c, minlength=N) + np.bincount(J, weights=c, minlength=N)
    gx += 2*mu*(-b0+b1)
    gy += 2*mu*(-b2+b3)
    gr += 2*mu*(b0+b1+b2+b3-neg)
    gw = 2*mu*(-b1.sum()+b3.sum())
    return f, np.r_[gx, gy, gr, gw]


def expand(q):
    w = q[-1]
    mid = (2-w)/2
    x = q[:12][REP]
    y = np.r_[np.full(3, mid), mid+q[12:21], mid-q[12:21]]
    r = q[21:33][REP]
    return np.r_[x, y, r, w]


def reduce_gradient(g):
    gx, gy, gr = g[:N], g[N:2*N], g[2*N:3*N]
    return np.r_[np.bincount(REP, weights=gx, minlength=12),
                 gy[3:12]-gy[12:21],
                 np.bincount(REP, weights=gr, minlength=12),
                 g[-1]-0.5*gy.sum()]


def repair(p, r):
    p = np.asarray(p, dtype=float).copy()
    r = np.maximum(np.asarray(r, dtype=float), 1e-13).copy()
    if not np.isfinite(p).all() or not np.isfinite(r).all():
        return None
    d = np.sqrt((p[I, 0]-p[J, 0])**2 + (p[I, 1]-p[J, 1])**2)
    ratio = min(1.0, float(np.min(d/(r[I]+r[J]))))
    if ratio <= 0:
        return None
    r *= ratio*(1-2e-12)
    lo = np.min(p-r[:, None], axis=0)
    hi = np.max(p+r[:, None], axis=0)
    span = float(np.sum(hi-lo))
    if span <= 0:
        return None
    scale = 2*(1-2e-12)/span
    p = (p-lo)*scale
    r *= scale
    return np.column_stack((p, r))


def lp_radii(p):
    m = len(I)
    a = np.zeros((m+4*N+1, N+4))
    b = np.zeros(m+4*N+1)
    a[np.arange(m), I] = 1
    a[np.arange(m), J] = 1
    b[:m] = np.sqrt((p[I, 0]-p[J, 0])**2 + (p[I, 1]-p[J, 1])**2)
    ids = np.arange(N)
    a[m+ids, ids] = 1
    a[m+ids, N] = 1
    b[m+ids] = p[:, 0]
    a[m+N+ids, ids] = 1
    a[m+N+ids, N+1] = -1
    b[m+N+ids] = -p[:, 0]
    a[m+2*N+ids, ids] = 1
    a[m+2*N+ids, N+2] = 1
    b[m+2*N+ids] = p[:, 1]
    a[m+3*N+ids, ids] = 1
    a[m+3*N+ids, N+3] = -1
    b[m+3*N+ids] = -p[:, 1]
    a[-1, N:] = [-1, 1, -1, 1]
    b[-1] = 2
    res = linprog(-np.r_[np.ones(N), np.zeros(4)], A_ub=a, b_ub=b,
                  bounds=[(1e-12, None)]*N+[(None, None)]*4,
                  method='highs', options={'time_limit': 0.12})
    return res.x[:N] if res.success else None


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    finish = start + 0.70*budget_s
    opt_finish = start + 0.59*budget_s
    rng = np.random.default_rng(seed)
    ids = np.arange(N)
    fallback = np.column_stack(((ids % 7)*0.2, (ids // 7)*0.2, np.full(N, 0.1-1e-12)))
    best = {'circles': fallback.tolist()}
    best_score = exact_score(fallback)
    original = None
    if init is not None:
        try:
            original = np.asarray(init['circles'], dtype=float)
            s = exact_score(original)
            if s > best_score:
                best, best_score = init, s
        except (ValueError, TypeError, KeyError):
            original = None

    def consider(a):
        nonlocal best, best_score
        if a is not None:
            s = exact_score(a)
            if s > best_score:
                best_score = s
                best = {'circles': a.tolist()}

    # Cold starts get an independent quotient layout: outer rows of five,
    # inner rows of four, and three circles on the reflection axis.
    use_symmetry = not (original is not None and exact_score(original) > 2.2)
    if use_symmetry:
        w = 2*float(params['aspect'])/(1+float(params['aspect']))
        h = 2-w
        x = np.r_[(np.arange(3)+0.5)/3,
                  np.linspace(0.1, 0.9, 4), np.linspace(0.1, 0.9, 5)]*w
        x += rng.normal(0, float(params['jitter']), 12)
        dy = np.r_[np.full(4, 0.205*h), np.full(5, 0.405*h)]
        dy += rng.normal(0, float(params['jitter']), 9)
        q = np.r_[x, dy, np.full(12, 0.085), w]
        z = expand(q)
    else:
        p, r = original[:, :2].copy(), original[:, 2].copy()
        lo = np.min(p-r[:, None], axis=0)
        sides = np.max(p+r[:, None], axis=0)-lo
        scale = 2/sides.sum()
        p, r = (p-lo)*scale, r*scale
        z = np.r_[p[:, 0], p[:, 1], r, sides[0]*scale]

    mu = float(params['mu0'])
    for stage in range(int(params['stages'])):
        if time.monotonic() >= opt_finish:
            break
        symmetric = use_symmetry and stage < int(params['symmetric_stages'])
        last = [q.copy() if symmetric else z.copy()]

        def objective(v):
            if time.monotonic() >= opt_finish:
                raise StopSearch()
            last[0] = v.copy()
            f, g = penalty(expand(v) if symmetric else v, mu)
            return (f, reduce_gradient(g)) if symmetric else (f, g)

        try:
            result = minimize(objective, last[0], jac=True, method='L-BFGS-B',
                              options={'maxiter': int(params['maxiter']), 'ftol': 1e-12,
                                       'gtol': 1e-7, 'maxls': 25})
            current = result.x
        except StopSearch:
            current = last[0]
        if symmetric:
            q = current
            z = expand(q)
        else:
            z = current
        if time.monotonic() < finish:
            consider(repair(np.column_stack((z[:N], z[N:2*N])), z[2*N:3*N]))
        mu *= float(params['mu_growth'])

    # The final fixed-centre LP is unchanged in purpose from the parent.
    # Start from the best exactly verified geometry, rather than an infeasible
    # last line-search trial, and reserve time for exact certification.
    if time.monotonic()+0.12 < finish:
        a = np.asarray(best['circles'], dtype=float)
        r = lp_radii(a[:, :2])
        if r is not None and time.monotonic() < start+0.88*budget_s:
            consider(repair(a[:, :2], r))
    return best
