import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'iterations': (135, 30, 250),
    'starts': (4, 1, 8),
    'jitter': (0.012, 0.0, 0.05)
}
N = 21
I, J = np.triu_indices(N, 1)
K = np.arange(N)
A = np.zeros((84, 64))
A[K, K] = 1
A[K, 42 + K] = -1
A[21 + K, K] = -1
A[21 + K, 42 + K] = -1
A[21 + K, 63] = 1
A[42 + K, 21 + K] = 1
A[42 + K, 42 + K] = -1
A[63 + K, 21 + K] = -1
A[63 + K, 42 + K] = -1
A[63 + K, 63] = -1
B = np.zeros(84)
B[63:] = 2
G = np.zeros(64)
G[42:63] = -1


def score(c):
    if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((c[I, 0] - c[J, 0]) ** 2 + (c[I, 1] - c[J, 1]) ** 2)
    if np.any(d < c[I, 2] + c[J, 2]):
        return -np.inf
    x = [Fraction(float(v)) for v in c[:, 0]]
    y = [Fraction(float(v)) for v in c[:, 1]]
    r = [Fraction(float(v)) for v in c[:, 2]]
    perimeter = max(a+b for a,b in zip(x,r)) - min(a-b for a,b in zip(x,r)) + max(a+b for a,b in zip(y,r)) - min(a-b for a,b in zip(y,r))
    return float(np.sum(c[:, 2])) if perimeter <= 2 else -np.inf


def repair(z):
    p = np.column_stack((z[:21], z[21:42]))
    r = np.maximum(z[42:63], 1e-10).copy()
    if not np.isfinite(p).all() or not np.isfinite(r).all():
        return None
    d = np.sqrt(np.sum((p[I] - p[J]) ** 2, axis=1))
    factor = min(1.0, float(np.min(d / (r[I] + r[J]))))
    if factor <= 0:
        return None
    r *= factor * (1 - 2e-12)
    lo = np.min(p - r[:, None], axis=0)
    hi = np.max(p + r[:, None], axis=0)
    scale = 2.0 / np.sum(hi - lo) * (1 - 2e-12)
    return np.column_stack(((p - lo) * scale, r * scale))


def encode(c):
    lo = np.min(c[:, :2] - c[:, 2, None], axis=0)
    hi = np.max(c[:, :2] + c[:, 2, None], axis=0)
    return {'centers': c[:, :2].tolist(), 'radii': c[:, 2].tolist(), 'box': (hi-lo).tolist()}


class StopSearch(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    end = time.monotonic() + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best = None
    best_score = -np.inf
    initial = None
    if init is not None:
        try:
            initial = np.asarray(init['circles'], dtype=float)
            value = score(initial)
            if value > best_score:
                best, best_score = initial.copy(), value
        except (KeyError, ValueError, TypeError):
            initial = None
    patterns = [[5,4,5,4,3], [4,5,4,5,3], [6,5,5,5], [4,4,5,4,4]]
    bounds = [(0.0, 2.0)] * 42 + [(1e-9, 0.5)] * 21 + [(0.1, 1.9)]

    def con(z):
        if time.monotonic() >= end:
            raise StopSearch
        dx = z[I] - z[J]
        dy = z[21+I] - z[21+J]
        d = np.sqrt(dx*dx + dy*dy + 1e-30)
        return np.concatenate((A @ z + B, d - z[42+I] - z[42+J]))

    def jac(z):
        if time.monotonic() >= end:
            raise StopSearch
        dx = z[I] - z[J]
        dy = z[21+I] - z[21+J]
        d = np.sqrt(dx*dx + dy*dy + 1e-30)
        q = np.zeros((len(I), 64))
        rows = np.arange(len(I))
        q[rows, I] = dx/d
        q[rows, J] = -dx/d
        q[rows, 21+I] = dy/d
        q[rows, 21+J] = -dy/d
        q[rows, 42+I] = -1
        q[rows, 42+J] = -1
        return np.vstack((A, q))

    for attempt in range(int(params['starts'])):
        if time.monotonic() >= end:
            break
        counts = patterns[attempt % len(patterns)]
        points = []
        for row, count in enumerate(counts):
            if time.monotonic() >= end:
                break
            offset = row % 2
            points.extend((2*j + offset, np.sqrt(3.0)*row) for j in range(count))
        if len(points) != N:
            break
        p = np.asarray(points, dtype=float)
        r = np.ones(N)
        lo = np.min(p-r[:,None], axis=0)
        side = np.max(p+r[:,None], axis=0)-lo
        scale = 2/side.sum()
        p = (p-lo)*scale
        r *= scale
        w = side[0]*scale
        z = np.r_[p[:,0], p[:,1], r, w]
        c = repair(z)
        if c is not None:
            value = score(c)
            if value > best_score:
                best, best_score = c, value
        if attempt == 1 and initial is not None and score(initial) > 2.2:
            p = initial[:,:2].copy()
            r = initial[:,2].copy()
            lo = np.min(p-r[:,None], axis=0)
            side = np.max(p+r[:,None], axis=0)-lo
            scale = 2/side.sum()
            p = (p-lo)*scale
            z = np.r_[p[:,0], p[:,1], r*scale, side[0]*scale]
        elif attempt:
            z[:42] += rng.normal(0, float(params['jitter']), 42)
            z[:42] = np.clip(z[:42], 0.001, 1.999)
        latest = [z.copy()]
        def callback(v):
            latest[0] = v.copy()
            if time.monotonic() >= end:
                raise StopSearch
        try:
            result = minimize(lambda v: -np.sum(v[42:63]), z, jac=lambda v: G,
                              method='SLSQP', bounds=bounds,
                              constraints={'type':'ineq', 'fun':con, 'jac':jac},
                              callback=callback,
                              options={'maxiter':int(params['iterations']), 'ftol':2e-11, 'disp':False})
            latest[0] = result.x
        except StopSearch:
            pass
        c = repair(latest[0])
        if c is not None:
            value = score(c)
            if value > best_score:
                best, best_score = c, value
    if best is None:
        p = np.column_stack((np.arange(N)%5, np.arange(N)//5))*0.2 + 0.1
        best = np.column_stack((p, np.full(N, 0.099999999)))
    return encode(best)
