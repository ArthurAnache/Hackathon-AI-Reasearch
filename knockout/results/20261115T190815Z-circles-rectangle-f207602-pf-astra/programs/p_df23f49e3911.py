import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'penalty_fraction': (0.27, 0.1, 0.45),
    'mu0': (20.0, 2.0, 100.0, 'log'),
    'growth': (8.0, 3.0, 20.0),
    'penalty_iterations': (80, 30, 200),
    'polish_iterations': (150, 50, 300),
    'restart_jitter': (0.012, 0.001, 0.05),
    'time_fraction': (0.70, 0.5, 0.75),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
K = np.arange(N)
Q = np.arange(M)

class Stop(Exception):
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
    rr = list(map(Fraction, map(float, r)))
    span = (max(u+v for u,v in zip(x,rr))-min(u-v for u,v in zip(x,rr))
            +max(u+v for u,v in zip(y,rr))-min(u-v for u,v in zip(y,rr)))
    return float(np.sum(r)) if span <= 2 else -float('inf')


def encode(a):
    p = a[:, :2].copy()
    r = np.maximum(np.abs(a[:, 2]), 1e-8)
    lo = np.min(p-r[:, None], axis=0)
    side = np.max(p+r[:, None], axis=0)-lo
    scale = 2.0/float(side.sum())
    p = (p-lo)*scale
    return np.r_[p[:, 0], p[:, 1], r*scale, side[0]*scale]


def repair(z):
    p = np.column_stack((z[:N], z[N:2*N]))
    r = np.maximum(z[2*N:3*N], 1e-10).copy()
    if not np.isfinite(p).all() or not np.isfinite(r).all():
        return None
    d = np.sqrt((p[I, 0]-p[J, 0])**2+(p[I, 1]-p[J, 1])**2)
    factor = min(1.0, float(np.min(d/(r[I]+r[J]))))
    if factor <= 0:
        return None
    r *= factor*(1-4e-13)
    lo = np.min(p-r[:, None], axis=0)
    side = np.max(p+r[:, None], axis=0)-lo
    scale = (2.0-3e-12)/float(side.sum())
    return np.column_stack(((p-lo)*scale, r*scale))


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start+float(params['time_fraction'])*budget_s
    rng = np.random.default_rng(seed)
    ids = np.arange(N)
    fallback = np.column_stack(((ids % 7+.5)*.2, (ids//7+.5)*.2, np.full(N, .1-1e-12)))
    best_array = fallback
    best = {'circles': fallback.tolist()}
    best_score = exact_score(fallback)
    original = None
    if init is not None:
        try:
            a = np.asarray(init['circles'], dtype=float)
            if a.shape == (N, 3) and np.isfinite(a).all():
                original = a.copy()
                s = exact_score(a)
                if s > best_score:
                    best, best_array, best_score = init, a.copy(), s
        except (ValueError, TypeError, KeyError, OverflowError):
            pass

    def keep(z):
        nonlocal best, best_array, best_score
        if time.monotonic() >= end:
            raise Stop()
        a = repair(z)
        if a is not None and float(a[:, 2].sum()) > best_score:
            s = exact_score(a)
            if s > best_score:
                best_score, best_array = s, a.copy()
                best = {'circles': a.tolist()}

    xx = np.array([2,4,6,8, 1,3,5,7, 0,2,4,6,8, 1,3,5,7, 0,2,4,6], dtype=float)
    yy = np.repeat(np.arange(5), [4,4,5,4,4])*np.sqrt(3.0)
    lattice = encode(np.column_stack((xx, yy, np.ones(N))))
    z = encode(original) if original is not None else lattice.copy()
    latest = z.copy()
    penalty_end = min(end, start+float(params['penalty_fraction'])*budget_s)

    def penalty(v, mu):
        if time.monotonic() >= penalty_end:
            raise Stop()
        x, y, r, w = v[:N], v[N:2*N], v[2*N:3*N], v[-1]
        dx, dy = x[I]-x[J], y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-24)
        over = np.maximum(r[I]+r[J]-d, 0)
        b0 = np.maximum(r-x, 0)
        b1 = np.maximum(r+x-w, 0)
        b2 = np.maximum(r-y, 0)
        b3 = np.maximum(r+y-2+w, 0)
        c = 2*mu*over
        gx = np.bincount(I, -c*dx/d, minlength=N)+np.bincount(J, c*dx/d, minlength=N)
        gy = np.bincount(I, -c*dy/d, minlength=N)+np.bincount(J, c*dy/d, minlength=N)
        gr = -np.ones(N)+np.bincount(I, c, minlength=N)+np.bincount(J, c, minlength=N)
        gx += 2*mu*(-b0+b1)
        gy += 2*mu*(-b2+b3)
        gr += 2*mu*(b0+b1+b2+b3)
        val = -r.sum()+mu*(over@over+b0@b0+b1@b1+b2@b2+b3@b3)
        return val, np.r_[gx, gy, gr, 2*mu*(-b1.sum()+b3.sum())]

    def penalty_callback(v):
        nonlocal latest
        if time.monotonic() >= penalty_end:
            raise Stop()
        latest = v.copy()

    bounds = [(None, None)]*(2*N)+[(1e-7, .5)]*N+[(.2, 1.8)]
    try:
        keep(z)
        if original is None or exact_score(original) < 2.25:
            mu = float(params['mu0'])
            try:
                for stage in range(5):
                    if time.monotonic() >= penalty_end:
                        break
                    result = minimize(penalty, z, args=(mu,), jac=True, method='L-BFGS-B',
                                      bounds=bounds, callback=penalty_callback,
                                      options={'maxiter': int(params['penalty_iterations']), 'ftol': 1e-12, 'maxls': 20})
                    z = result.x
                    latest = z.copy()
                    keep(z)
                    mu *= float(params['growth'])
            except Stop:
                pass
            z = latest.copy()
            keep(z)

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
        cache_z = None
        cache_c = None
        cache_j = None
        grad = np.r_[np.zeros(2*N), -np.ones(N), 0.0]

        def objective(v):
            if time.monotonic() >= end:
                raise Stop()
            return -float(v[2*N:3*N].sum()), grad

        def geometry(v):
            nonlocal cache_z, cache_c, cache_j
            if time.monotonic() >= end:
                raise Stop()
            if cache_z is not None and np.array_equal(v, cache_z):
                return cache_c, cache_j
            x, y, r, w = v[:N], v[N:2*N], v[2*N:3*N], v[-1]
            dx, dy = x[I]-x[J], y[I]-y[J]
            d = np.sqrt(dx*dx+dy*dy)
            safe = np.maximum(d, 1e-15)
            c = np.r_[d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r]
            jac = fixed.copy()
            jac[Q, I] = dx/safe
            jac[Q, J] = -dx/safe
            jac[Q, N+I] = dy/safe
            jac[Q, N+J] = -dy/safe
            cache_z, cache_c, cache_j = v.copy(), c, jac
            return c, jac

        constraints = {'type': 'ineq', 'fun': lambda v: geometry(v)[0], 'jac': lambda v: geometry(v)[1]}
        attempt = 0
        while time.monotonic() < end:
            if attempt > 0:
                if attempt == 1 and best_score < 2.34:
                    z = lattice.copy()
                    z[:2*N] += rng.normal(0, .002, 2*N)
                else:
                    z = encode(best_array)
                    z[:2*N] += rng.normal(0, float(params['restart_jitter']), 2*N)
            cache_z = None
            result = minimize(objective, z, jac=True, method='SLSQP', bounds=bounds,
                              constraints=constraints, callback=keep,
                              options={'maxiter': int(params['polish_iterations']), 'ftol': 2e-12, 'disp': False})
            keep(result.x)
            attempt += 1
    except Stop:
        pass
    return best
