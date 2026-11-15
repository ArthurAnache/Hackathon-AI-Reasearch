import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {'screen_iterations': (38, 15, 70), 'polish_iterations': (110, 50, 180), 'restart_scale': (0.003, 0.0001, 0.02)}

class StopSearch(Exception):
    pass

def canonical(obj):
    if obj is None:
        return None
    if 'circles' in obj:
        return obj
    rows = []
    for x, y, r, m in obj['orbits']:
        rows.append([x, y, r])
        if m == 2:
            rows.append([x, -y, r])
    return {'circles': rows}

def exact_score(obj):
    try:
        a = np.asarray(canonical(obj)['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        xs, ys, rs = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
        w = max(x+r for x, r in zip(xs, rs))-min(x-r for x, r in zip(xs, rs))
        h = max(y+r for y, r in zip(ys, rs))-min(y-r for y, r in zip(ys, rs))
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
    started = time.monotonic()
    deadline = started + 0.70*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = exact_score(init)
    if not np.isfinite(best_score):
        a0 = np.array([[2*(k % 7), 2*(k // 7), 1.] for k in range(21)], dtype=float)/10.00000000001
        best = {'circles': a0.tolist()}
        best_score = exact_score(best)
    if time.monotonic() >= deadline:
        return best
    try:
        a = np.asarray(canonical(init)['circles'], dtype=float).copy()
        if a.shape != (21, 3) or not np.isfinite(a).all():
            raise ValueError()
    except Exception:
        a = np.asarray(canonical(best)['circles'], dtype=float).copy()
    a[:, :2] -= (np.max(a[:, :2]+a[:, 2, None], axis=0)+np.min(a[:, :2]-a[:, 2, None], axis=0))/2
    groups = np.array([0, 1, 2]+[g for g in range(3, 12) for _ in range(2)])
    signs = np.array([0., 0., 0.]+[s for _ in range(9) for s in (1., -1.)])
    X = np.zeros((21, 34)); Y = np.zeros_like(X); R = np.zeros_like(X)
    for k in range(21):
        if time.monotonic() >= deadline:
            return best
        g = groups[k]
        X[k, g] = 1
        R[k, 21+g] = 1
        if g >= 3:
            Y[k, 9+g] = signs[k]
    ai, aj = np.triu_indices(21, 1)
    seen = set(); pairs = []
    for i, j in zip(ai, aj):
        if time.monotonic() >= deadline:
            return best
        g, h = sorted((int(groups[i]), int(groups[j])))
        key = (g, h, int(signs[i]*signs[j]))
        if key not in seen:
            seen.add(key); pairs.append((i, j))
    pi, pj = np.asarray(pairs).T
    DX = X[pi]-X[pj]; DY = Y[pi]-Y[pj]; RS = R[pi]+R[pj]
    walls = np.zeros((36, 34)); offsets = np.zeros(36)
    for g in range(12):
        if time.monotonic() >= deadline:
            return best
        walls[g, g] = 1; walls[g, 21+g] = -1; walls[g, 33] = .5
        walls[12+g, g] = -1; walls[12+g, 21+g] = -1; walls[12+g, 33] = .5
        walls[24+g, 21+g] = -1; walls[24+g, 33] = -.5
        offsets[24+g] = 1
        if g >= 3:
            walls[24+g, 9+g] = -1
    weights = np.array([1.]*3+[2.]*9)
    grad = np.zeros(34); grad[21:33] = -weights
    bounds = [(-1., 1.)]*12+[(0., 1.)]*9+[(1e-8, .65)]*12+[(.1, 1.9)]
    anchor = None; anchor_score = -float('inf')
    local_deadline = deadline
    cache_v = None; cache_c = None; cache_j = None

    def tick():
        if time.monotonic() >= local_deadline:
            raise StopSearch()

    def retain(v):
        nonlocal best, best_score, anchor, anchor_score
        if not np.isfinite(v).all() or np.min(v[21:33]) <= 0:
            return
        xx = v[:12].copy(); yy = np.r_[np.zeros(3), np.abs(v[12:21])]
        rr = v[21:33].copy()
        cx = xx[groups]; cy = yy[groups]*signs; cr = rr[groups]
        dist = np.sqrt((cx[ai]-cx[aj])**2+(cy[ai]-cy[aj])**2)
        ratio = float(np.min(dist/(cr[ai]+cr[aj])))
        rr *= min(1., ratio)*(1.-2e-14)
        if np.min(rr) <= 0:
            return
        width = float(np.max(xx+rr)-np.min(xx-rr))
        height = float(2*np.max(yy+rr))
        scale = 2/(width+height)*(1.-2e-14)
        estimate = float(weights@rr)*scale
        if estimate > anchor_score:
            anchor_score = estimate
            anchor = v.copy()
            anchor[:12] = xx*scale
            anchor[12:21] = yy[3:]*scale
            anchor[21:33] = rr*scale
            anchor[33] = width*scale
        if estimate <= best_score:
            return
        obj = {'orbits': [[float(xx[g]*scale), float(yy[g]*scale), float(rr[g]*scale), 1 if g < 3 else 2] for g in range(12)]}
        val = exact_score(obj)
        if val > best_score:
            best = obj; best_score = val

    def update(v):
        nonlocal cache_v, cache_c, cache_j
        tick()
        if cache_v is not None and np.array_equal(v, cache_v):
            return
        dx = DX@v; dy = DY@v
        d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-30)
        cache_c = np.concatenate((d-RS@v, walls@v+offsets))
        cache_j = np.vstack(((dx/d)[:, None]*DX+(dy/d)[:, None]*DY-RS, walls))
        cache_v = v.copy()

    def fun(v):
        tick()
        return float(grad@v)

    def jac(v):
        tick()
        return grad

    def cons(v):
        update(v)
        return cache_c

    def cjac(v):
        update(v)
        return cache_j

    def callback(v):
        tick()
        retain(v)

    def solve(v, iterations, seconds, tolerance=1e-12):
        nonlocal local_deadline, cache_v
        if time.monotonic() >= deadline:
            return
        local_deadline = min(deadline, time.monotonic()+max(.001, seconds))
        cache_v = None
        v = np.array(v, dtype=float, copy=True)
        v[12:21] = np.clip(v[12:21], 0, 1)
        v[21:33] = np.clip(v[21:33], 1e-8, .65)
        v[33] = np.clip(v[33], .1, 1.9)
        retain(v)
        try:
            result = minimize(fun, v, jac=jac, method='SLSQP', bounds=bounds,
                constraints={'type': 'ineq', 'fun': cons, 'jac': cjac}, callback=callback,
                options={'maxiter': int(iterations), 'ftol': tolerance, 'disp': False})
            retain(result.x)
        except StopSearch:
            pass
        local_deadline = deadline

    def vector(g):
        z = np.zeros(34)
        z[:12] = g[:, 0]; z[12:21] = np.abs(g[3:, 1]); z[21:33] = np.maximum(g[:, 2], 1e-6)
        z[33] = np.clip(np.max(g[:, 0]+g[:, 2])-np.min(g[:, 0]-g[:, 2]), .1, 1.9)
        return z

    projections = []
    try:
        for swap in (False, True):
            tick()
            b = a[:, [1, 0, 2]].copy() if swap else a.copy()
            order = np.argsort(b[:, 1]); lower = order[:9]; singles = order[9:12]; upper = order[12:]
            reflected = b[lower].copy(); reflected[:, 1] *= -1
            # Pair primarily by horizontal position, retaining radius evidence.
            cost = np.sum((b[upper, None, :]-reflected[None, :, :])**2, axis=2)
            u, v = linear_sum_assignment(cost)
            g = [[b[k, 0], 0., b[k, 2]] for k in singles]
            for p, q in zip(u, v):
                tick()
                g.append(((b[upper[p]]+reflected[q])/2).tolist())
            error = float(np.sum(cost[u, v])+np.sum(b[singles, 1]**2))
            projections.append((error, vector(np.asarray(g))))
        projections.sort(key=lambda item: item[0])
        # Recovery first, then a bounded challenger, with an untouched final reserve.
        solve(projections[0][1], params['polish_iterations'], .32*budget_s)
        if best_score < 2.35 and projections[1][0] < 2.5*max(projections[0][0], 1e-12):
            solve(projections[1][1], params['screen_iterations'], .11*budget_s)
        if best_score < 2.36583237 and time.monotonic() < started+.49*budget_s:
            g = [[x, 0., 1.] for x in (-np.sqrt(3.), 0., np.sqrt(3.))]
            for x, ys in ((-2*np.sqrt(3.), (1., 3.)), (-np.sqrt(3.), (2., 4.)), (0., (2.,)), (np.sqrt(3.), (2., 4.)), (2*np.sqrt(3.), (1., 3.))):
                tick()
                for y in ys:
                    tick()
                    g.append([x, y, 1.])
            g = np.asarray(g)* (2/(4*np.sqrt(3.)+12))
            solve(vector(g), params['screen_iterations'], .15*budget_s)
        if anchor is not None:
            solve(anchor, params['polish_iterations'], max(.001, deadline-time.monotonic()), 8e-14)
        attempt = 0
        while time.monotonic() < deadline-.015 and anchor is not None:
            tick()
            z = anchor.copy()
            amplitude = params['restart_scale']*(1+attempt % 3)
            z[:21] += rng.normal(0., amplitude, 21)
            z[12:21] = np.maximum(z[12:21], 0.)
            solve(z, params['screen_iterations'], min(.12*budget_s, deadline-time.monotonic()), 2e-13)
            attempt += 1
    except (StopSearch, ValueError, FloatingPointError):
        pass
    return best
