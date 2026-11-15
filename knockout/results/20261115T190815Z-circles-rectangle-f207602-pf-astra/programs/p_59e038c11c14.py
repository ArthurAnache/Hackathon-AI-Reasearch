import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'weight': (0.32, 0.05, 0.8),
    'patch_size': (4, 3, 8),
    'paths': (3, 1, 6),
    'maxiter': (85, 20, 200),
    'ftol': (2e-13, 1e-14, 1e-8, 'log'),
}
N = 21
I, J = np.triu_indices(N, 1)
P = len(I)
K = np.arange(N)
ROWS = np.arange(P)

class Finished(Exception):
    pass

def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    r = a[:, 2]
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < r[I]+r[J]):
        return -np.inf
    x = list(map(lambda t: Fraction(float(t)), a[:, 0]))
    y = list(map(lambda t: Fraction(float(t)), a[:, 1]))
    q = list(map(lambda t: Fraction(float(t)), r))
    w = max(u+v for u, v in zip(x, q))-min(u-v for u, v in zip(x, q))
    h = max(u+v for u, v in zip(y, q))-min(u-v for u, v in zip(y, q))
    return float(np.sum(r)) if w+h <= 2 else -np.inf

def repaired(a):
    a = np.array(a, dtype=float, copy=True)
    if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return None
    a[:, :2] -= np.mean(a[:, :2], axis=0)
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    f = min(1.0, float(np.min(d/(a[I, 2]+a[J, 2]))))
    if f <= 0:
        return None
    a[:, 2] *= f*(1-2e-14)
    extent = np.max(a[:, :2]+a[:, 2, None], axis=0)-np.min(a[:, :2]-a[:, 2, None], axis=0)
    a *= (2/float(np.sum(extent)))*(1-2e-14)
    return a

def pack(a):
    lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
    hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
    xy = a[:, :2]-lo
    return np.r_[xy[:, 0], xy[:, 1], a[:, 2], hi[0]-lo[0]]

def unpack(z):
    return np.column_stack((z[:N], z[N:2*N], z[2*N:3*N]))

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start+0.70*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf
    best_a = None

    def offer(a, original=None):
        nonlocal best, best_score, best_a
        if a is None:
            return
        score = exact_score(a)
        if score > best_score:
            best_score = score
            best_a = a.copy()
            best = original if original is not None else {'circles': a.tolist()}

    if init is not None:
        offer(np.asarray(init['circles'], dtype=float), init)
    if best_a is None:
        k = np.arange(N)
        a = np.column_stack((0.1+0.2*(k % 7), 0.1+0.2*(k // 7), np.full(N, 0.1-1e-13)))
        offer(a)
    if time.monotonic() >= deadline:
        return best

    boundary = np.zeros((4*N, 3*N+1))
    boundary[K, K] = 1
    boundary[K, 2*N+K] = -1
    boundary[N+K, K] = -1
    boundary[N+K, 2*N+K] = -1
    boundary[N+K, -1] = 1
    boundary[2*N+K, N+K] = 1
    boundary[2*N+K, 2*N+K] = -1
    boundary[3*N+K, N+K] = -1
    boundary[3*N+K, 2*N+K] = -1
    boundary[3*N+K, -1] = -1
    offset = np.r_[np.zeros(3*N), np.full(N, 2.0)]
    bounds = [(None, None)]*(2*N)+[(1e-6, 0.6)]*N+[(0.1, 1.9)]

    def solve(z, weights, end, certify):
        last = [z.copy()]
        iterations = [0]
        grad = np.r_[np.zeros(2*N), -weights, 0.0]
        cache = [None, None, None]

        def check():
            if time.monotonic() >= end:
                raise Finished()

        def calculate(v):
            check()
            if cache[0] is not None and np.array_equal(v, cache[0]):
                return cache[1], cache[2]
            x, y, r = v[:N], v[N:2*N], v[2*N:3*N]
            dx, dy = x[I]-x[J], y[I]-y[J]
            d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-20)
            c = np.r_[d-r[I]-r[J], boundary@v+offset]
            jac = np.zeros((P+4*N, 3*N+1))
            jac[ROWS, I] = dx/d
            jac[ROWS, J] = -dx/d
            jac[ROWS, N+I] = dy/d
            jac[ROWS, N+J] = -dy/d
            jac[ROWS, 2*N+I] = -1
            jac[ROWS, 2*N+J] = -1
            jac[P:] = boundary
            cache[:] = [v.copy(), c, jac]
            return c, jac

        def objective(v):
            check()
            return float(grad@v)

        def callback(v):
            check()
            last[0] = v.copy()
            iterations[0] += 1
            if certify and iterations[0] % 7 == 0:
                offer(repaired(unpack(v)))

        try:
            result = minimize(objective, z, jac=lambda v: grad, method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': lambda v: calculate(v)[0],
                             'jac': lambda v: calculate(v)[1]},
                callback=callback,
                options={'maxiter': int(params['maxiter']), 'ftol': float(params['ftol']), 'disp': False})
            last[0] = result.x.copy()
        except Finished:
            pass
        if time.monotonic() < deadline:
            offer(repaired(unpack(last[0])))
        return last[0]

    ones = np.ones(N)
    z = pack(best_a)
    solve(z, ones, min(deadline, start+0.13*budget_s), True)
    reference = best_a.copy()
    # Each path loads a spatial patch, then removes that load before scoring.
    anchors = rng.permutation(N)
    paths = int(params['paths'])
    for attempt in range(paths):
        now = time.monotonic()
        if now >= deadline-0.025:
            break
        remaining = paths-attempt
        path_end = now+(deadline-now)/remaining
        anchor = int(anchors[attempt % N])
        base = reference if attempt % 2 == 0 else best_a
        distance = np.sum((base[:, :2]-base[anchor, :2])**2, axis=1)
        patch = np.argsort(distance)[:int(params['patch_size'])]
        weights = ones.copy()
        sign = 1 if attempt % 2 == 0 else -1
        weights[patch] += sign*float(params['weight'])
        weights /= np.mean(weights)
        z = pack(base)
        loaded = solve(z, weights, now+0.37*(path_end-now), False)
        if time.monotonic() >= deadline-0.01:
            break
        solve(loaded, ones, min(path_end, deadline), True)
    return best
