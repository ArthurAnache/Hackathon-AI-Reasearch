import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {'release': (0.014, 0.002, 0.05), 'contacts': (4, 3, 6), 'probe_iters': (16, 8, 35), 'restore_iters': (44, 20, 90)}

class StopSearch(Exception):
    pass

def canonical(obj):
    if 'circles' in obj:
        return obj
    rows = []
    for x, y, r, m in obj['orbits']:
        rows.append([x, y, r])
        if m == 2:
            rows.append([x, -y, r])
    return {'circles': rows}

def score(obj):
    try:
        a = np.asarray(canonical(obj)['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        w = max(u+v for u, v in zip(x, r))-min(u-v for u, v in zip(x, r))
        h = max(u+v for u, v in zip(y, r))-min(u-v for u, v in zip(y, r))
        if w+h > 2:
            return -float('inf')
        i, j = np.triu_indices(21, 1)
        d = np.sqrt((a[i, 0]-a[j, 0])**2+(a[i, 1]-a[j, 1])**2)
        if np.any(d < a[i, 2]+a[j, 2]):
            return -float('inf')
        return float(np.sum(a[:, 2]))
    except Exception:
        return -float('inf')

def construct(params, seed, budget_s, init=None):
    deadline = time.monotonic()+0.69*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_value = score(init) if init is not None else -float('inf')
    def tick():
        if time.monotonic() >= deadline:
            raise StopSearch()
    fallback = {'circles': [[(i % 7)*0.2, (i//7)*0.2, 0.09999999999999] for i in range(21)]}
    if best is None:
        best, best_value = fallback, score(fallback)
    groups = np.r_[np.arange(3), np.repeat(np.arange(3, 12), 2)]
    signs = np.r_[np.zeros(3), np.tile([1., -1.], 9)]
    X = np.zeros((21, 34)); Y = X.copy(); R = X.copy()
    X[np.arange(21), groups] = 1
    R[np.arange(21), 21+groups] = 1
    mask = groups >= 3
    Y[np.flatnonzero(mask), groups[mask]+9] = signs[mask]
    ai, aj = np.triu_indices(21, 1)
    keys = np.column_stack((np.minimum(groups[ai], groups[aj]), np.maximum(groups[ai], groups[aj]), signs[ai]*signs[aj]))
    _, keep = np.unique(keys, axis=0, return_index=True)
    pi, pj = ai[keep], aj[keep]
    DX, DY, RS = X[pi]-X[pj], Y[pi]-Y[pj], R[pi]+R[pj]
    W = np.zeros((36, 34)); off = np.zeros(36)
    q = np.arange(12)
    W[q, q] = 1; W[q, 21+q] = -1; W[q, 33] = .5
    W[12+q, q] = -1; W[12+q, 21+q] = -1; W[12+q, 33] = .5
    W[24+q, 21+q] = -1; W[24+q, 33] = -.5
    W[27+np.arange(9), 12+np.arange(9)] = -1
    off[24:] = 1
    objective = np.zeros(34); objective[21:33] = -np.r_[np.ones(3), np.full(9, 2.)]
    bounds = [(-1., 1.)]*12+[(0., 1.)]*9+[(1e-7, .5)]*12+[(.12, 1.88)]
    release = np.zeros(len(pi)+36)
    cache_v = None; cache_c = None; cache_j = None
    last = None
    anchor_best = None

    def geometry(v):
        nonlocal cache_v, cache_c, cache_j
        tick()
        if cache_v is None or not np.array_equal(v, cache_v):
            dx, dy = DX@v, DY@v
            d = np.sqrt(dx*dx+dy*dy)
            ds = np.maximum(d, 1e-25)
            cache_c = np.r_[d-RS@v, W@v+off]
            cache_j = np.vstack(((dx/ds)[:, None]*DX+(dy/ds)[:, None]*DY-RS, W))
            cache_v = v.copy()
        return cache_c, cache_j

    def retain(v):
        nonlocal best, best_value, anchor_best
        if v is None or not np.isfinite(v).all():
            return
        a = np.column_stack((X@v, Y@v, R@v))
        if np.any(a[:, 2] <= 0):
            return
        d = np.sqrt((a[ai, 0]-a[aj, 0])**2+(a[ai, 1]-a[aj, 1])**2)
        f = min(1., float(np.min(d/(a[ai, 2]+a[aj, 2]))))*(1.-3e-15)
        if f <= 0:
            return
        a[:, 2] *= f
        span = np.max(a[:, :2]+a[:, 2, None], axis=0)-np.min(a[:, :2]-a[:, 2, None], axis=0)
        s = 2./float(np.sum(span))*(1.-3e-15)
        a *= s
        if float(np.sum(a[:, 2])) <= best_value:
            return
        obj = {'circles': a.tolist()}
        val = score(obj)
        if val > best_value:
            best, best_value = obj, val
            anchor_best = v.copy()

    def fun(v):
        tick()
        return float(objective@v)
    def jac(v):
        tick()
        return objective
    def con(v):
        return geometry(v)[0]-release
    def cjac(v):
        return geometry(v)[1]
    callbacks = 0
    def callback(v):
        nonlocal last, callbacks
        tick()
        last = v.copy(); callbacks += 1
        if callbacks % 10 == 0:
            retain(v)
    def solve(v, n):
        nonlocal last
        tick()
        last = v.copy()
        result = minimize(fun, v, jac=jac, bounds=bounds, method='SLSQP', constraints={'type': 'ineq', 'fun': con, 'jac': cjac}, callback=callback, options={'maxiter': int(n), 'ftol': 3e-14, 'disp': False})
        retain(result.x)
        return result.x

    try:
        a = np.asarray(canonical(best)['circles'], dtype=float).copy()
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        a[:, :2] -= (lo+hi)/2
        projections = []
        for swap in (False, True):
            tick()
            b = a[:, [1, 0, 2]].copy() if swap else a.copy()
            order = np.argsort(b[:, 1]); lower, axis, upper = order[:9], order[9:12], order[12:]
            reflected = b[lower].copy(); reflected[:, 1] *= -1
            costs = np.sum((b[upper, None, :]-reflected[None, :, :])**2, axis=2)
            u, v = linear_sum_assignment(costs)
            g = np.vstack((b[axis], (b[upper[u]]+reflected[v])/2))
            z = np.zeros(34)
            z[:12] = g[:, 0]; z[12:21] = np.abs(g[3:, 1]); z[21:33] = g[:, 2]
            z[33] = hi[1 if swap else 0]-lo[1 if swap else 0]
            projections.append((float(np.sum(costs[u, v])+np.sum(b[axis, 1]**2)), z))
        anchor = min(projections, key=lambda item: item[0])[1]
        retain(anchor)
        anchor = solve(anchor, 55)
        # Newton refinement of an isolated active contact system.
        for _ in range(5):
            tick()
            c, J = geometry(anchor)
            active = np.flatnonzero(c < 2e-7)
            if len(active) < 33:
                break
            delta = np.linalg.lstsq(J[active], -c[active], rcond=1e-11)[0]
            if np.max(np.abs(delta)) > .001:
                break
            candidate = anchor+delta
            if np.min(geometry(candidate)[0]) < -1e-9:
                break
            anchor = candidate
            retain(anchor)
        base = anchor.copy()
        attempt = 0
        while time.monotonic() < deadline:
            tick()
            release[:] = 0
            c, J = geometry(base)
            active = np.flatnonzero(c < 2e-5)
            if len(active) < 3:
                base = solve(base, 35)
                attempt += 1
                continue
            multipliers = np.linalg.lstsq(J[active].T, objective, rcond=1e-10)[0]
            weakness = 1./(.04+np.abs(multipliers))
            weakness /= np.sum(weakness)
            count = min(int(params['contacts']), len(active))
            if attempt % 3 == 0:
                chosen = rng.choice(active, size=count, replace=False, p=weakness)
            else:
                first = int(rng.choice(active, p=weakness))
                similarity = np.abs(J[active]@J[first])
                probabilities = weakness*(.1+similarity)
                probabilities[active == first] = 0
                probabilities /= np.sum(probabilities)
                chosen = np.r_[first, rng.choice(active, count-1, replace=False, p=probabilities)]
            release[chosen] = float(params['release'])*rng.uniform(.65, 1.35, len(chosen))
            trial = base.copy()
            if attempt % 4 == 3:
                target = release[active]
                displacement = np.linalg.lstsq(J[active], target, rcond=1e-9)[0]
                magnitude = np.max(np.abs(displacement))
                if magnitude > .06:
                    displacement *= .06/magnitude
                trial += displacement
            trial = solve(trial, params['probe_iters'])
            release[:] = 0
            trial = solve(trial, params['restore_iters'])
            if np.min(geometry(trial)[0]) > -2e-9 and objective@trial < objective@base-1e-11:
                base = trial.copy()
            elif anchor_best is not None and attempt % 4 == 2:
                base = anchor_best.copy()
            attempt += 1
    except (StopSearch, ValueError, FloatingPointError, np.linalg.LinAlgError):
        pass
    retain(last)
    return best
