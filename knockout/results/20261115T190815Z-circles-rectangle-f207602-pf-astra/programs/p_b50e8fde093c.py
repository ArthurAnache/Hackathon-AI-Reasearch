import time
import operator
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize

PARAMS = {
    'radius_floor': (0.025, 0.001, 0.06),
    'lp_iterations': (48, 12, 100),
    'jitter': (0.018, 0.0, 0.08),
    'polish_fraction': (0.23, 0.1, 0.4)
}

N = 21
I, J = np.triu_indices(N, 1)
K = len(I)

class StopSearch(Exception):
    pass


def exact_score(c):
    c = np.asarray(c, dtype=float)
    if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
        return -float('inf')
    dx = c[I, 0] - c[J, 0]
    dy = c[I, 1] - c[J, 1]
    if np.any(np.sqrt(dx**2 + dy**2) < c[I, 2] + c[J, 2]):
        return -float('inf')
    x = list(map(Fraction, map(float, c[:, 0])))
    y = list(map(Fraction, map(float, c[:, 1])))
    r = list(map(Fraction, map(float, c[:, 2])))
    w = max(map(operator.add, x, r)) - min(map(operator.sub, x, r))
    h = max(map(operator.add, y, r)) - min(map(operator.sub, y, r))
    if w + h > 2:
        return -float('inf')
    return float(np.sum(c[:, 2]))


def construct(params, seed, budget_s, init=None):
    begin = time.monotonic()
    end = begin + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best = None
    best_score = -float('inf')
    original = None
    if init is not None:
        try:
            original = np.asarray(init['circles'] if 'circles' in init else np.asarray(init['coordinate_rows']).T, dtype=float)
            best_score = exact_score(original)
            if np.isfinite(best_score):
                best = original.copy()
        except (KeyError, ValueError, TypeError):
            original = None

    def retain(c):
        nonlocal best, best_score
        if time.monotonic() >= end:
            return
        c = np.asarray(c, dtype=float).copy()
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return
        d = np.sqrt((c[I, 0]-c[J, 0])**2 + (c[I, 1]-c[J, 1])**2)
        ratio = float(np.min(d / (c[I, 2]+c[J, 2])))
        c[:, 2] *= min(1.0, ratio) * (1.0-2e-12)
        if np.any(c[:, 2] <= 0):
            return
        lo = np.min(c[:, :2] - c[:, 2, None], axis=0)
        hi = np.max(c[:, :2] + c[:, 2, None], axis=0)
        c[:, :2] -= (lo + hi) / 2
        c *= (2.0-2e-13) / float(np.sum(hi-lo))
        if float(np.sum(c[:, 2])) > best_score:
            s = exact_score(c)
            if s > best_score:
                best, best_score = c, s

    # A guaranteed elementary fallback.
    ix = np.arange(21) % 5
    iy = np.arange(21) // 5
    fallback = np.column_stack((0.2*ix, 0.2*iy, np.full(21, 0.099999999999)))
    retain(fallback)

    rows = np.arange(K)
    A = np.zeros((K+84, 64))
    b = np.zeros(K+84)
    A[rows, 42+I] = 1
    A[rows, 42+J] = 1
    q = np.arange(21)
    A[K+q, q] = -1
    A[K+q, 42+q] = 1
    A[K+21+q, q] = 1
    A[K+21+q, 42+q] = 1
    A[K+21+q, 63] = -1
    A[K+42+q, 21+q] = -1
    A[K+42+q, 42+q] = 1
    A[K+63+q, 21+q] = 1
    A[K+63+q, 42+q] = 1
    A[K+63+q, 63] = 1
    b[K+63:] = 2
    objective = np.zeros(64)
    objective[42:63] = -1
    bounds = [(0, 2)]*42 + [(float(params['radius_floor']), 0.5)]*21 + [(0.2, 1.8)]

    def encode(c):
        c = c.copy()
        lo = np.min(c[:, :2]-c[:, 2,None], axis=0)
        hi = np.max(c[:, :2]+c[:, 2,None], axis=0)
        size = hi-lo
        scale = 2 / np.sum(size)
        c[:, :2] -= lo
        c *= scale
        return np.r_[c[:, 0], c[:, 1], c[:, 2], size[0]*scale]

    def decode(z):
        return np.column_stack((z[:21], z[21:42], z[42:63]))

    starts = []
    if original is not None and original.shape == (21, 3) and np.isfinite(original).all():
        if best_score > 2.3:
            starts.append(encode(original))
    patterns = [[4, 3, 4, 3, 4, 3], [4, 5, 3, 5, 4], [3, 4, 3, 4, 3, 4]]
    for pattern in patterns:
        if time.monotonic() >= end:
            break
        points = []
        for k, count in enumerate(pattern):
            if time.monotonic() >= end:
                break
            xx = 2*np.arange(count) - (count-1)
            yy = np.full(count, k*np.sqrt(3.0))
            points.append(np.column_stack((xx, yy)))
        if len(points) != len(pattern):
            break
        xy = np.concatenate(points)
        xy += rng.normal(0, float(params['jitter']), xy.shape)
        starts.append(encode(np.column_stack((xy, np.full(21, 0.8)))))
    if best_score < 2.3 and original is not None and original.shape == (21, 3) and np.isfinite(original).all():
        starts.append(encode(original))

    lp_end = begin + 0.70*budget_s*(1-float(params['polish_fraction']))
    for start_index, z in enumerate(starts):
        now = time.monotonic()
        if now >= lp_end:
            break
        stage_end = now + (lp_end-now)/(len(starts)-start_index)
        prev = -1.0
        stalled = 0
        local_bounds = bounds
        for iteration in range(int(params['lp_iterations'])):
            now = time.monotonic()
            if now >= stage_end:
                break
            dx = z[I]-z[J]
            dy = z[21+I]-z[21+J]
            dist = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-15)
            ux, uy = dx/dist, dy/dist
            A[rows, I] = -ux
            A[rows, J] = ux
            A[rows, 21+I] = -uy
            A[rows, 21+J] = uy
            result = linprog(objective, A_ub=A, b_ub=b, bounds=local_bounds,
                method='highs', options={'time_limit': max(0.001, min(0.09, stage_end-now)),
                'dual_feasibility_tolerance': 1e-8, 'primal_feasibility_tolerance': 1e-8})
            if not result.success:
                if iteration == 0:
                    local_bounds = [(0, 2)]*42 + [(1e-5, 0.5)]*21 + [(0.2, 1.8)]
                    continue
                break
            z = result.x
            retain(decode(z))
            val = -result.fun
            if val-prev < 2e-9:
                stalled += 1
            else:
                stalled = 0
            prev = val
            if stalled >= 3:
                break

    if best is not None and time.monotonic() < end-0.025:
        z0 = encode(best)
        wall_A = A[K:].copy()
        wall_b = b[K:].copy()
        def fun(z):
            if time.monotonic() >= end-0.008:
                raise StopSearch()
            return -float(np.sum(z[42:63]))
        def jac(z):
            if time.monotonic() >= end-0.008:
                raise StopSearch()
            return objective
        def con(z):
            if time.monotonic() >= end-0.008:
                raise StopSearch()
            dx = z[I]-z[J]
            dy = z[21+I]-z[21+J]
            d = np.sqrt(dx*dx+dy*dy)
            return np.r_[d-z[42+I]-z[42+J], wall_b-wall_A@z]
        def cjac(z):
            if time.monotonic() >= end-0.008:
                raise StopSearch()
            dx = z[I]-z[J]
            dy = z[21+I]-z[21+J]
            d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-15)
            out = np.zeros((K+84, 64))
            out[rows,I] = dx/d
            out[rows,J] = -dx/d
            out[rows,21+I] = dy/d
            out[rows,21+J] = -dy/d
            out[rows,42+I] = -1
            out[rows,42+J] = -1
            out[K:] = -wall_A
            return out
        def callback(z):
            if time.monotonic() >= end-0.008:
                raise StopSearch()
            retain(decode(z))
        try:
            result = minimize(fun, z0, jac=jac, method='SLSQP',
                bounds=[(0,2)]*42+[(1e-7,0.5)]*21+[(0.2,1.8)],
                constraints={'type':'ineq','fun':con,'jac':cjac}, callback=callback,
                options={'maxiter':65,'ftol':2e-12,'disp':False})
            retain(decode(result.x))
        except StopSearch:
            pass
    if best is None:
        return {'circles': fallback.tolist()}
    return {'coordinate_rows': best.T.tolist()}
