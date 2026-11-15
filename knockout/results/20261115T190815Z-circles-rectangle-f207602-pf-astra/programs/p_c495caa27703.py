import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'maxiter': (120, 40, 300),
    'jitter': (0.004, 0.0, 0.03),
    'smooth_mu': (22.0, 5.0, 150.0, 'log'),
    'smooth_iters': (65, 20, 150),
    'restart_jitter': (0.008, 0.0, 0.05),
    'time_fraction': (0.70, 0.5, 0.75),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
K = np.arange(N)
Q = np.arange(M)

class StopSearch(Exception):
    pass


def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -float('inf')
    r = a[:, 2]
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < r[I]+r[J]):
        return -float('inf')
    x = list(map(Fraction, map(float, a[:, 0])))
    y = list(map(Fraction, map(float, a[:, 1])))
    r = list(map(Fraction, map(float, r)))
    span = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
    span += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
    if span > 2:
        return -float('inf')
    return float(np.sum(a[:, 2]))


def encode(a):
    p, r = a[:, :2].copy(), np.maximum(np.abs(a[:, 2]), 1e-10)
    lo = np.min(p-r[:, None], axis=0)
    side = np.max(p+r[:, None], axis=0)-lo
    s = 2.0/float(np.sum(side))
    p = (p-lo)*s
    return np.r_[p[:, 0], p[:, 1], r*s, side[0]*s]


def repair(z):
    p = np.column_stack((z[:N], z[N:2*N]))
    r = np.maximum(z[2*N:3*N], 1e-10).copy()
    if not np.isfinite(p).all() or not np.isfinite(r).all():
        return None
    d = np.sqrt((p[I, 0]-p[J, 0])**2+(p[I, 1]-p[J, 1])**2)
    f = min(1.0, float(np.min(d/(r[I]+r[J]))))
    if f <= 0:
        return None
    r *= f*(1-3e-13)
    lo = np.min(p-r[:, None], axis=0)
    hi = np.max(p+r[:, None], axis=0)
    s = (2-2e-12)/float(np.sum(hi-lo))
    return np.column_stack(((p-lo)*s, r*s*(1-3e-13)))


def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started+float(params['time_fraction'])*budget_s
    phase_end = deadline
    rng = np.random.default_rng(seed)
    ids = np.arange(N)
    fallback = np.column_stack(((ids % 7+0.5)*0.2, (ids//7+0.5)*0.2,
                                np.full(N, 0.1-1e-12)))
    best_array = fallback.copy()
    best_score = exact_score(fallback)
    best = {'circles': fallback.tolist()}
    if init is not None:
        try:
            a = np.asarray(init['circles'], dtype=float)
            score = exact_score(a)
            if score >= best_score:
                best, best_score, best_array = init, score, a.copy()
        except (ValueError, TypeError, KeyError, OverflowError):
            pass

    def check():
        if time.monotonic() >= min(deadline, phase_end):
            raise StopSearch()

    def keep(z):
        check()
        nonlocal best, best_score, best_array
        a = repair(z)
        if a is not None and float(np.sum(a[:, 2])) > best_score:
            score = exact_score(a)
            if score > best_score:
                best_score, best_array = score, a.copy()
                best = {'circles': a.tolist()}

    xx = np.array([2,4,6,8, 1,3,5,7, 0,2,4,6,8, 1,3,5,7, 0,2,4,6], dtype=float)
    yy = np.repeat(np.arange(5), [4,4,5,4,4])*np.sqrt(3.0)
    initial = encode(np.column_stack((xx, yy, np.ones(N))))
    initial[:2*N] += rng.normal(0, float(params['jitter']), 2*N)
    if best_score > 2.23:
        initial = encode(best_array)

    grad = np.r_[np.zeros(2*N), -np.ones(N), 0.0]
    fixed = np.zeros((M+4*N, 3*N+1))
    fixed[Q, 2*N+I] = -1
    fixed[Q, 2*N+J] = -1
    fixed[M+K, K] = 1
    fixed[M+K, 2*N+K] = -1
    fixed[M+N+K, K] = -1
    fixed[M+N+K, 2*N+K] = -1
    fixed[M+N+K, -1] = 1
    fixed[M+2*N+K, N+K] = 1
    fixed[M+2*N+K, 2*N+K] = -1
    fixed[M+3*N+K, N+K] = -1
    fixed[M+3*N+K, 2*N+K] = -1
    fixed[M+3*N+K, -1] = -1
    cached = [None, None, None]

    def geometry(z):
        check()
        if cached[0] is not None and np.array_equal(z, cached[0]):
            return cached[1], cached[2]
        x, y, r, w = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
        dx, dy = x[I]-x[J], y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy)
        den = np.maximum(d, 1e-15)
        c = np.r_[d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r]
        jac = fixed.copy()
        jac[Q, I], jac[Q, J] = dx/den, -dx/den
        jac[Q, N+I], jac[Q, N+J] = dy/den, -dy/den
        cached[:] = [z.copy(), c, jac]
        return c, jac

    def objective(z):
        check()
        return -float(z[2*N:3*N].sum()), grad

    def penalty(z, mu):
        check()
        x, y, r, w = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
        dx, dy = x[I]-x[J], y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-24)
        v = np.maximum(r[I]+r[J]-d, 0)
        b0 = np.maximum(r-x, 0)
        b1 = np.maximum(r+x-w, 0)
        b2 = np.maximum(r-y, 0)
        b3 = np.maximum(r+y-2+w, 0)
        c = 2*mu*v
        gx = np.bincount(I, -c*dx/d, minlength=N)+np.bincount(J, c*dx/d, minlength=N)
        gy = np.bincount(I, -c*dy/d, minlength=N)+np.bincount(J, c*dy/d, minlength=N)
        gr = -np.ones(N)+np.bincount(I, c, minlength=N)+np.bincount(J, c, minlength=N)
        gx += 2*mu*(-b0+b1)
        gy += 2*mu*(-b2+b3)
        gr += 2*mu*(b0+b1+b2+b3)
        f = -r.sum()+mu*(v@v+b0@b0+b1@b1+b2@b2+b3@b3)
        return f, np.r_[gx, gy, gr, 2*mu*(-b1.sum()+b3.sum())]

    bounds = [(None, None)]*(2*N)+[(1e-7, 0.5)]*N+[(0.25, 1.75)]
    constraints = {'type': 'ineq', 'fun': lambda z: geometry(z)[0],
                   'jac': lambda z: geometry(z)[1]}

    def polish(z):
        cached[0] = None
        keep(z)
        result = minimize(objective, z, jac=True, method='SLSQP', bounds=bounds,
                          constraints=constraints, callback=keep,
                          options={'maxiter': int(params['maxiter']), 'ftol': 1e-11})
        keep(result.x)

    phase_end = started+0.39*budget_s
    try:
        polish(initial)
    except StopSearch:
        pass
    attempt = 0
    while time.monotonic() < deadline-0.04:
        z = encode(best_array)
        z[:2*N] += rng.normal(0, float(params['restart_jitter']), 2*N)
        last = [z.copy()]
        phase_end = min(deadline-0.12, time.monotonic()+0.11*budget_s)
        def remember(v):
            check()
            last[0] = v.copy()
        try:
            for mu in (float(params['smooth_mu']), float(params['smooth_mu'])*5):
                check()
                result = minimize(penalty, last[0], args=(mu,), jac=True,
                                  method='L-BFGS-B', bounds=bounds, callback=remember,
                                  options={'maxiter': int(params['smooth_iters']),
                                           'ftol': 1e-10, 'maxls': 20})
                last[0] = result.x.copy()
        except StopSearch:
            pass
        phase_end = deadline
        try:
            polish(last[0])
        except StopSearch:
            break
        attempt += 1
    return best
