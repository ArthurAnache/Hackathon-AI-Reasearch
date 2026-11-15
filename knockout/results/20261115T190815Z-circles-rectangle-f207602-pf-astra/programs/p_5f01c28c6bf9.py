import time
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize

PARAMS = {
    'trust': (0.20, 0.04, 0.40),
    'lp_steps': (3, 1, 8),
    'trial_seconds': (0.34, 0.15, 0.60),
    'cavity_grid': (29, 15, 45),
    'ftol': (3e-12, 1e-14, 1e-8, 'log')
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
Q = np.arange(M)
K = np.arange(N)

class Deadline(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    r = a[:, 2]
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < r[I]+r[J]):
        return -np.inf
    x = list(map(lambda v: Fraction(float(v)), a[:, 0]))
    y = list(map(lambda v: Fraction(float(v)), a[:, 1]))
    r = list(map(lambda v: Fraction(float(v)), r))
    w = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
    h = max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
    return float(np.sum(a[:, 2])) if w+h <= 2 else -np.inf

def encode(a):
    r = a[:, 2]
    lo = np.min(a[:, :2]-r[:, None], axis=0)
    hi = np.max(a[:, :2]+r[:, None], axis=0)
    s = 2.0/float(np.sum(hi-lo))
    p = (a[:, :2]-lo)*s
    return np.r_[p[:, 0], p[:, 1], r*s, (hi[0]-lo[0])*s]

def repair(z):
    p = np.column_stack((z[:N], z[N:2*N])).copy()
    r = z[2*N:3*N].copy()
    if not np.isfinite(z).all() or np.any(r <= 0):
        return None
    d = np.sqrt(np.sum((p[I]-p[J])**2, axis=1))
    f = min(1.0, float(np.min(d/(r[I]+r[J]))))
    if f <= 0:
        return None
    r *= f*(1-8e-15)
    lo = np.min(p-r[:, None], axis=0)
    hi = np.max(p+r[:, None], axis=0)
    p -= (lo+hi)*0.5
    s = (2.0-5e-14)/float(np.sum(hi-lo))
    return np.column_stack((p*s, r*s))

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start+0.70*budget_s
    rng = np.random.default_rng(seed)
    ids = np.arange(N)
    fallback = np.column_stack(((ids%7)*0.2, (ids//7)*0.2, np.full(N, 0.1-1e-13)))
    best = {'circles': fallback.tolist()}
    best_score = exact_score(fallback)
    incumbent = encode(fallback)
    if init is not None:
        try:
            a = np.asarray(init['circles'], dtype=float)
            value = exact_score(a)
            if value > best_score:
                best, best_score, incumbent = init, value, encode(a)
        except (ValueError, TypeError, KeyError, OverflowError):
            pass

    def retain(z):
        nonlocal best, best_score, incumbent
        if time.monotonic() >= end:
            return
        a = repair(z)
        if a is None or float(np.sum(a[:, 2])) <= best_score:
            return
        score = exact_score(a)
        if score > best_score:
            best, best_score, incumbent = {'circles': a.tolist()}, score, encode(a)

    objective = np.r_[np.zeros(42), -np.ones(21), 0.0]
    base = np.zeros((M+84, 64))
    base[Q, 42+I] = -1
    base[Q, 42+J] = -1
    base[M+K, K] = 1
    base[M+K, 42+K] = -1
    base[M+21+K, K] = -1
    base[M+21+K, 42+K] = -1
    base[M+21+K, 63] = 1
    base[M+42+K, 21+K] = 1
    base[M+42+K, 42+K] = -1
    base[M+63+K, 21+K] = -1
    base[M+63+K, 42+K] = -1
    base[M+63+K, 63] = -1
    constant = np.zeros(M+84)
    constant[M+63:] = 2.0
    bounds = [(None, None)]*42+[(1e-6, 0.49)]*21+[(0.25, 1.75)]

    def geometry(z):
        dx, dy = z[I]-z[J], z[21+I]-z[21+J]
        d = np.sqrt(dx*dx+dy*dy)
        safe = np.maximum(d, 1e-15)
        jac = base.copy()
        jac[Q, I] = dx/safe
        jac[Q, J] = -dx/safe
        jac[Q, 21+I] = dy/safe
        jac[Q, 21+J] = -dy/safe
        c = base@z+constant
        c[:M] += d
        return c, jac

    def polish(z, stop):
        cache_z = None
        cache_c = None
        cache_j = None
        def check():
            if time.monotonic() >= stop:
                raise Deadline()
        def calc(v):
            nonlocal cache_z, cache_c, cache_j
            check()
            if cache_z is None or not np.array_equal(v, cache_z):
                cache_c, cache_j = geometry(v)
                cache_z = v.copy()
        def fun(v):
            check()
            return float(objective@v), objective
        def con(v):
            calc(v)
            return cache_c
        def jac(v):
            calc(v)
            return cache_j
        def callback(v):
            check()
            retain(v)
        try:
            result = minimize(fun, z, jac=True, method='SLSQP', bounds=bounds,
                constraints={'type': 'ineq', 'fun': con, 'jac': jac},
                callback=callback, options={'maxiter': 130, 'ftol': float(params['ftol']), 'disp': False})
            retain(result.x)
            return result.x
        except Deadline:
            return z

    def linear_ascent(z, stop):
        step = 0
        while step < int(params['lp_steps']) and time.monotonic() < stop-0.025:
            c, jac = geometry(z)
            rhs = c-jac@z
            delta = float(params['trust'])
            b = np.empty((64, 2))
            b[:42, 0], b[:42, 1] = z[:42]-delta, z[:42]+delta
            b[42:63] = [1e-6, 0.49]
            b[63] = [0.25, 1.75]
            remaining = stop-time.monotonic()
            if remaining < 0.025:
                break
            result = linprog(objective, A_ub=-jac, b_ub=rhs, bounds=b,
                method='highs', options={'time_limit': max(0.001, remaining-0.012),
                'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9})
            if result.x is None or not result.success:
                break
            z = result.x
            retain(z)
            step += 1
        return z

    def relocate(z, trial):
        p = np.column_stack((z[:21], z[21:42])).copy()
        r = z[42:63].copy()
        w, h = z[-1], 2-z[-1]
        if trial%3 == 0:
            interior = np.minimum.reduce((p[:, 0]-r, w-p[:, 0]-r, p[:, 1]-r, h-p[:, 1]-r))
            first = int(np.argmax(interior+0.10*r))
        else:
            first = int(rng.integers(N))
        distances = np.sum((p-p[first])**2, axis=1)
        distances[first] = np.inf
        near = np.argsort(distances)[:5]
        second = int(near[trial%len(near)])
        removed = [first, second]
        active = np.ones(N, dtype=bool)
        active[removed] = False
        g = int(params['cavity_grid'])
        xx, yy = np.meshgrid(np.linspace(0.035, w-0.035, g), np.linspace(0.035, h-0.035, g))
        grid = np.column_stack((xx.ravel(), yy.ravel()))
        walls = np.minimum.reduce((grid[:, 0], w-grid[:, 0], grid[:, 1], h-grid[:, 1]))
        for index in removed:
            if time.monotonic() >= end-0.03:
                return z
            d = np.sqrt(np.sum((grid[:, None, :]-p[active][None, :, :])**2, axis=2))-r[active][None, :]
            clearance = np.minimum(walls, np.min(d, axis=1))
            # Exclude the circle's original neighbourhood to force a discrete move.
            clearance[np.sum((grid-p[index])**2, axis=1) < (1.15*r[index])**2] = -1
            ranked = np.argsort(clearance)[::-1]
            pick = int(ranked[0 if trial%2 == 0 else min(5, len(ranked)-1)])
            p[index] = grid[pick]
            r[index] = max(0.003, 0.96*clearance[pick])
            active[index] = True
        return np.r_[p[:, 0], p[:, 1], r, w]

    # Cheap initial polish, then spend the remaining time on changed topologies.
    if time.monotonic() < end-0.04:
        polish(incumbent.copy(), min(end-0.02, start+0.10*budget_s))
    trial = 0
    while time.monotonic() < end-0.065:
        local_start = time.monotonic()
        stop = min(end-0.012, local_start+float(params['trial_seconds']))
        z = relocate(incumbent.copy(), trial)
        if time.monotonic() >= stop-0.025:
            break
        z = linear_ascent(z, min(stop-0.025, local_start+0.38*(stop-local_start)))
        if time.monotonic() < stop-0.012:
            polish(z, stop)
        trial += 1
    return best
