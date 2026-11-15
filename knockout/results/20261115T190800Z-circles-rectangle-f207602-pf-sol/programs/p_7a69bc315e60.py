import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {'probe_iters': (28, 12, 60), 'polish_iters': (75, 30, 140), 'probe_seconds': (0.18, 0.08, 0.35), 'jitter': (0.006, 0.001, 0.025)}


def canonical(obj):
    if 'circles' in obj:
        return obj
    out = []
    for x, y, r, m in obj['orbits']:
        out.append([x, y, r])
        if m == 2:
            out.append([x, -y, r])
    return {'circles': out}


def score(obj):
    try:
        a = np.asarray(canonical(obj)['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
        span = max(u+v for u, v in zip(x, r))-min(u-v for u, v in zip(x, r))
        span += max(u+v for u, v in zip(y, r))-min(u-v for u, v in zip(y, r))
        if span > 2:
            return -float('inf')
        i, j = np.triu_indices(21, 1)
        d = np.sqrt((a[i, 0]-a[j, 0])**2+(a[i, 1]-a[j, 1])**2)
        if np.any(d < a[i, 2]+a[j, 2]):
            return -float('inf')
        return float(np.sum(a[:, 2]))
    except Exception:
        return -float('inf')


class TimeLimit(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_value = score(init) if init is not None else -float('inf')
    local_deadline = deadline
    groups = np.r_[np.arange(3), np.repeat(np.arange(3, 12), 2)]
    signs = np.r_[np.zeros(3), np.tile([1., -1.], 9)]
    X = np.zeros((21, 34)); Y = X.copy(); R = X.copy()
    X[np.arange(21), groups] = 1
    R[np.arange(21), groups+21] = 1
    mask = groups >= 3
    Y[np.flatnonzero(mask), groups[mask]+9] = signs[mask]
    ai, aj = np.triu_indices(21, 1)
    keys = np.column_stack((np.minimum(groups[ai], groups[aj]), np.maximum(groups[ai], groups[aj]), signs[ai]*signs[aj]))
    _, keep = np.unique(keys, axis=0, return_index=True)
    pi, pj = ai[keep], aj[keep]
    DX = X[pi]-X[pj]; DY = Y[pi]-Y[pj]; RS = R[pi]+R[pj]
    W = np.zeros((36, 34)); off = np.zeros(36)
    q = np.arange(12)
    W[q, q] = 1; W[q, q+21] = -1; W[q, 33] = .5
    W[q+12, q] = -1; W[q+12, q+21] = -1; W[q+12, 33] = .5
    W[q+24, q+21] = -1; W[q+24, 33] = -.5
    W[27+np.arange(9), 12+np.arange(9)] = -1
    off[24:] = 1
    objective = np.zeros(34); objective[21:33] = -np.r_[np.ones(3), np.full(9, 2.)]
    bounds = [(-1., 1.)]*12+[(0., 1.)]*9+[(1e-8, .5)]*12+[(.15, 1.85)]
    saved = []

    def tick():
        if time.monotonic() >= local_deadline:
            raise TimeLimit()

    def repair(v):
        if v is None or not np.isfinite(v).all():
            return None
        a = np.column_stack((X@v, Y@v, R@v))
        if np.any(a[:, 2] <= 0):
            return None
        d = np.sqrt((a[ai, 0]-a[aj, 0])**2+(a[ai, 1]-a[aj, 1])**2)
        f = min(1., float(np.min(d/(a[ai, 2]+a[aj, 2]))))*(1-3e-14)
        if f <= 0:
            return None
        a[:, 2] *= f
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        s = 2./float(np.sum(hi-lo))*(1-3e-14)
        a *= s
        z = v.copy(); z[:33] *= s; z[21:33] *= f
        z[:12] -= .5*(lo[0]+hi[0])*s
        z[33] = (hi[0]-lo[0])*s
        a[:, 0] -= .5*(lo[0]+hi[0])*s
        return a, z

    def retain(v):
        nonlocal best, best_value
        result = repair(v)
        if result is None:
            return -float('inf'), None
        a, z = result
        val = float(np.sum(a[:, 2]))
        if val > best_value:
            obj = {'circles': a.tolist()}
            exact = score(obj)
            if exact > best_value:
                best, best_value = obj, exact
        return val, z

    cache_v = None; cache_c = None; cache_j = None
    def calc(v):
        nonlocal cache_v, cache_c, cache_j
        tick()
        if cache_v is not None and np.array_equal(v, cache_v):
            return
        dx, dy = DX@v, DY@v
        d = np.sqrt(dx*dx+dy*dy)
        dd = np.maximum(d, 1e-25)
        cache_c = np.r_[d-RS@v, W@v+off]
        cache_j = np.vstack(((dx/dd)[:, None]*DX+(dy/dd)[:, None]*DY-RS, W))
        cache_v = v.copy()
    def con(v):
        calc(v)
        return cache_c
    def cjac(v):
        calc(v)
        return cache_j
    def fun(v):
        tick()
        return float(objective@v)
    def jac(v):
        tick()
        return objective

    def solve(v, iterations, seconds):
        nonlocal local_deadline, cache_v
        local_deadline = min(deadline, time.monotonic()+seconds)
        cache_v = None
        last = v.copy()
        local_best_value, local_best = retain(v)
        count = 0
        def cb(z):
            nonlocal last, count, local_best_value, local_best
            last = z.copy()
            count += 1
            if count % 5 == 0:
                val, feasible = retain(z)
                if val > local_best_value:
                    local_best_value, local_best = val, feasible
            tick()
        try:
            result = minimize(fun, v, jac=jac, bounds=bounds, method='SLSQP', constraints={'type': 'ineq', 'fun': con, 'jac': cjac}, callback=cb, options={'maxiter': int(iterations), 'ftol': 3e-12, 'disp': False})
            last = result.x
        except (TimeLimit, ValueError, FloatingPointError):
            pass
        val, feasible = retain(last)
        if val > local_best_value:
            local_best_value, local_best = val, feasible
        local_deadline = deadline
        return local_best_value, local_best if local_best is not None else v

    def lattice(kind):
        if kind == 0:
            xx = np.r_[[-2., 0., 2.], [-4., -2., 0., 2., 4.], [-3., -1., 1., 3.]]
            yy = np.r_[np.full(5, np.sqrt(3.)), np.full(4, 2*np.sqrt(3.))]
        elif kind == 1:
            xx = np.r_[[-2., 0., 2.], [-3., -1., 1., 3.], [-4., -2., 0., 2., 4.]]
            yy = np.r_[np.full(4, np.sqrt(3.)), np.full(5, 2*np.sqrt(3.))]
        else:
            xx = np.r_[[-2., 0., 2.], [-1., 1., 3.], [-2., 0., 2.], [-1., 1., 3.]]
            yy = np.repeat(np.arange(1, 4)*np.sqrt(3.), 3)
        width = np.max(xx)-np.min(xx)+2
        height = 2*(np.max(yy)+1)
        s = 2/(width+height)*(1-1e-13)
        z = np.zeros(34)
        z[:12] = (xx-.5*(np.max(xx)+np.min(xx)))*s
        z[12:21] = yy*s; z[21:33] = s; z[33] = width*s
        return z

    starts = [lattice(0), lattice(1), lattice(2)]
    for z in starts:
        if time.monotonic() >= deadline:
            break
        retain(z)
    projections = []
    if init is not None and time.monotonic() < deadline:
        try:
            a = np.asarray(canonical(init)['circles'], float).copy()
            if a.shape == (21, 3) and np.isfinite(a).all():
                low = np.min(a[:, :2]-a[:, 2, None], axis=0)
                high = np.max(a[:, :2]+a[:, 2, None], axis=0)
                a[:, :2] -= (low+high)/2
                for swap in (False, True):
                    if time.monotonic() >= deadline:
                        break
                    b = a[:, [1, 0, 2]].copy() if swap else a.copy()
                    order = np.argsort(b[:, 1])
                    lower, axis, upper = order[:9], order[9:12], order[12:]
                    reflected = b[lower].copy(); reflected[:, 1] *= -1
                    costs = np.sum((b[upper, None, :]-reflected[None, :, :])**2, axis=2)
                    u, v = linear_sum_assignment(costs)
                    g = np.vstack((b[axis], (b[upper[u]]+reflected[v])/2))
                    err = np.sum(costs[u, v])+np.sum(b[axis, 1]**2)
                    z = np.zeros(34)
                    z[:12] = g[:, 0]; z[12:21] = np.abs(g[3:, 1]); z[21:33] = np.maximum(g[:, 2], 1e-7)
                    z[33] = high[1 if swap else 0]-low[1 if swap else 0]
                    projections.append((err, z))
                projections.sort(key=lambda x: x[0])
        except (ValueError, KeyError, TypeError):
            pass
    if projections:
        if score(init) > 2.2:
            starts.insert(0, projections[0][1])
        else:
            starts.insert(1, projections[0][1])
        if len(projections) > 1 and projections[1][0] < 0.025:
            starts.append(projections[1][1])

    for z in starts:
        if time.monotonic() >= deadline-0.04:
            break
        remaining = deadline-time.monotonic()
        val, point = solve(z, params['probe_iters'], min(params['probe_seconds'], remaining*.65))
        saved.append((val, point))
    saved.sort(key=lambda p: p[0], reverse=True)
    round_number = 0
    while saved and time.monotonic() < deadline-0.025:
        now = time.monotonic()
        idx = 0 if round_number % 3 != 1 or len(saved) == 1 else 1
        previous, z = saved[idx]
        z = z.copy()
        if round_number >= 3:
            z[:21] += rng.normal(0, params['jitter'], 21)
            z[12:21] = np.maximum(z[12:21], z[24:33])
        val, point = solve(z, params['polish_iters'], min(.32, deadline-now))
        if val > previous:
            saved[idx] = (val, point)
        saved.sort(key=lambda p: p[0], reverse=True)
        round_number += 1
    return best
