import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'grid': (27, 15, 45),
    'penalty': (70.0, 10.0, 500.0, 'log'),
    'relocation_distance': (0.30, 0.15, 0.65),
    'attempt_seconds': (0.38, 0.20, 0.60),
    'radius_mix': (0.75, 0.3, 1.0)
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
K = np.arange(N)
T = np.arange(M)

class Deadline(Exception):
    pass


def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return -np.inf
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    w = max(u+v for u, v in zip(x, r))-min(u-v for u, v in zip(x, r))
    h = max(u+v for u, v in zip(y, r))-min(u-v for u, v in zip(y, r))
    return float(np.sum(a[:, 2])) if w+h <= 2 else -np.inf


def encode(a):
    lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
    hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
    scale = 2.0/np.sum(hi-lo)
    p = (a[:, :2]-(hi+lo)/2)*scale
    return np.r_[p[:, 0], p[:, 1], a[:, 2]*scale, (hi[0]-lo[0])*scale]


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    ids = np.arange(N)
    fallback = np.column_stack(((ids % 7-3)*0.2, (ids//7-1)*0.2, np.full(N, 0.1-1e-14)))
    best = {'circles': fallback.tolist()}
    score = exact_score(fallback)
    base = encode(fallback)
    if init is not None:
        a = np.asarray(init['circles'], dtype=float)
        s = exact_score(a)
        if s >= score:
            best, score, base = init, s, encode(a)

    wall = np.zeros((84, 64))
    wall[K, K] = -1
    wall[K, 42+K] = -1
    wall[K, 63] = 0.5
    wall[21+K, K] = 1
    wall[21+K, 42+K] = -1
    wall[21+K, 63] = 0.5
    wall[42+K, 21+K] = -1
    wall[42+K, 42+K] = -1
    wall[42+K, 63] = -0.5
    wall[63+K, 21+K] = 1
    wall[63+K, 42+K] = -1
    wall[63+K, 63] = -0.5
    offset = np.r_[np.zeros(42), np.ones(42)]
    template = np.zeros((M+84, 64))
    template[T, 42+I] = -1
    template[T, 42+J] = -1
    template[M:] = wall
    objective = np.r_[np.zeros(42), -np.ones(21), 0.0]
    bounds = [(None, None)]*42 + [(1e-5, 0.45)]*21 + [(0.35, 1.65)]

    def geometry(z):
        dx = z[I]-z[J]
        dy = z[21+I]-z[21+J]
        d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-30)
        v = np.r_[d-z[42+I]-z[42+J], wall@z+offset]
        jac = template.copy()
        jac[T, I] = dx/d
        jac[T, J] = -dx/d
        jac[T, 21+I] = dy/d
        jac[T, 21+J] = -dy/d
        return v, jac

    def consider(z):
        nonlocal best, score, base
        if time.monotonic() >= end-0.004 or not np.isfinite(z).all() or np.any(z[42:63] <= 0):
            return
        a = np.column_stack((z[:21], z[21:42], z[42:63]))
        d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
        factor = min(1.0, float(np.min(d/(a[I, 2]+a[J, 2]))))
        if factor <= 0:
            return
        a[:, 2] *= factor*(1-4e-15)
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        a[:, :2] -= (hi+lo)/2
        a *= 2.0/np.sum(hi-lo)*(1-5e-15)
        if np.sum(a[:, 2]) <= score:
            return
        s = exact_score(a)
        if s > score:
            best, score, base = {'circles': a.tolist()}, s, encode(a)

    def polish(z, stop):
        stop = min(stop, end-0.012)
        last = z.copy()
        cache_z = None
        cache = None
        def calc(v):
            nonlocal cache_z, cache
            if time.monotonic() >= stop:
                raise Deadline()
            if cache_z is None or not np.array_equal(v, cache_z):
                cache_z, cache = v.copy(), geometry(v)
            return cache
        def fun(v):
            if time.monotonic() >= stop:
                raise Deadline()
            return float(objective@v)
        def callback(v):
            nonlocal last
            if time.monotonic() >= stop:
                raise Deadline()
            last = v.copy()
            consider(v)
        try:
            res = minimize(fun, z, jac=lambda v: objective, method='SLSQP', bounds=bounds,
                constraints={'type': 'ineq', 'fun': lambda v: calc(v)[0], 'jac': lambda v: calc(v)[1]},
                callback=callback, options={'maxiter': 85, 'ftol': 1e-13, 'disp': False})
            last = res.x
        except Deadline:
            pass
        consider(last)
        return last

    def smooth(z, stop, mu):
        last = z.copy()
        def fun(v):
            if time.monotonic() >= stop:
                raise Deadline()
            val, jac = geometry(v)
            bad = np.minimum(val, 0)
            return float(objective@v + mu*(bad@bad)), objective + (2*mu*bad)@jac
        def callback(v):
            nonlocal last
            if time.monotonic() >= stop:
                raise Deadline()
            last = v.copy()
        try:
            res = minimize(fun, z, jac=True, method='L-BFGS-B', bounds=bounds, callback=callback,
                options={'maxiter': 90, 'ftol': 1e-10, 'gtol': 2e-6, 'maxls': 15, 'maxcor': 8})
            last = res.x
        except Deadline:
            pass
        return last

    if time.monotonic() < end-0.05:
        polish(base, min(end-0.02, start+0.10*budget_s))

    # Rank nonlocal vacancy insertions, explicitly excluding the removed circle's old neighbourhood.
    z0 = base.copy()
    points = np.column_stack((z0[:21], z0[21:42]))
    w, h = z0[-1], 2-z0[-1]
    g = int(params['grid'])
    xx, yy = np.meshgrid(np.linspace(-w/2+0.03, w/2-0.03, g), np.linspace(-h/2+0.03, h/2-0.03, g))
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    clearance = np.sqrt(np.sum((grid[:, None, :]-points[None, :, :])**2, axis=2))-z0[42:63]
    edge = np.minimum(w/2-np.abs(grid[:, 0]), h/2-np.abs(grid[:, 1]))
    relocations = []
    for k in rng.permutation(N):
        if time.monotonic() >= end-0.05:
            break
        free = np.minimum(np.min(clearance[:, ids != k], axis=1), edge)
        far = np.linalg.norm(grid-points[k], axis=1) > float(params['relocation_distance'])
        free[~far] = -1
        j = int(np.argmax(free))
        if free[j] > 0:
            relocations.append((float(free[j]-z0[42+k]), int(k), grid[j].copy(), float(free[j])))
    relocations.sort(key=lambda q: q[0], reverse=True)
    attempt = 0
    while time.monotonic() < end-0.13 and attempt < len(relocations):
        now = time.monotonic()
        stop = min(end-0.016, now+float(params['attempt_seconds']))
        _, k, target, radius = relocations[attempt]
        z = z0.copy()
        z[k], z[21+k] = target
        mix = float(params['radius_mix'])
        z[42+k] = mix*z[42+k]+(1-mix)*radius
        # The vacancy creates room for a different contact graph; retain all other radii initially.
        span = stop-now
        z = smooth(z, now+0.16*span, float(params['penalty']))
        if time.monotonic() < stop-0.05:
            z = smooth(z, now+0.29*span, 8*float(params['penalty']))
        if time.monotonic() < stop-0.03:
            z = polish(z, stop)
        consider(z)
        attempt += 1
    return best
