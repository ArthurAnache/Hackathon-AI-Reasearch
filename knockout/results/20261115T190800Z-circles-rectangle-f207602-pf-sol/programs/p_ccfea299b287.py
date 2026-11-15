import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {'project_time': (0.29, 0.15, 0.38), 'seed_time': (0.18, 0.10, 0.25), 'iterations': (125, 60, 220), 'jitter': (0.004, 0.001, 0.02)}

class Stop(Exception):
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

def score(obj):
    try:
        a = np.asarray(canonical(obj)['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        i, j = np.triu_indices(21, 1)
        d = np.sqrt((a[i, 0]-a[j, 0])**2 + (a[i, 1]-a[j, 1])**2)
        if np.any(d < a[i, 2]+a[j, 2]):
            return -float('inf')
        x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
        w = max(u+v for u, v in zip(x, r))-min(u-v for u, v in zip(x, r))
        h = max(u+v for u, v in zip(y, r))-min(u-v for u, v in zip(y, r))
        if w+h > 2:
            return -float('inf')
        return float(np.sum(a[:, 2]))
    except Exception:
        return -float('inf')

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    end = started + 0.70*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = score(init)
    def tick():
        if time.monotonic() >= end:
            raise Stop()
    if best_score == -float('inf'):
        a = np.array([[.2*(k % 7-3), .2*(k//7-1), .0999999999999] for k in range(21)])
        best = {'circles': a.tolist()}
        best_score = score(best)
    try:
        a = np.asarray(canonical(init)['circles'], float).copy()
        if a.shape != (21, 3) or not np.isfinite(a).all():
            raise ValueError()
    except Exception:
        a = np.asarray(canonical(best)['circles'], float).copy()
    lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
    hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
    a[:, :2] -= (hi+lo)/2
    groups = np.r_[np.arange(3), np.repeat(np.arange(3, 12), 2)]
    signs = np.r_[np.zeros(3), np.tile([1., -1.], 9)]
    X = np.zeros((21, 34)); Y = X.copy(); R = X.copy()
    for k in range(21):
        tick()
        g = groups[k]
        X[k, g] = 1.
        R[k, 21+g] = 1.
        if g >= 3:
            Y[k, 9+g] = signs[k]
    ii, jj = np.triu_indices(21, 1)
    seen = set(); pairs = []
    for i, j in zip(ii, jj):
        tick()
        key = (int(groups[i]), int(groups[j]), float(signs[i]*signs[j]))
        if key not in seen:
            seen.add(key)
            pairs.append((i, j))
    pi, pj = np.asarray(pairs).T
    DX = X[pi]-X[pj]; DY = Y[pi]-Y[pj]; RS = R[pi]+R[pj]
    W = np.zeros((36, 34)); off = np.zeros(36)
    for g in range(12):
        tick()
        W[g, g] = 1.; W[g, 21+g] = -1.; W[g, 33] = .5
        W[12+g, g] = -1.; W[12+g, 21+g] = -1.; W[12+g, 33] = .5
        W[24+g, 21+g] = -1.; W[24+g, 33] = -.5
        off[24+g] = 1.
        if g >= 3:
            W[24+g, 9+g] = -1.
    weights = np.r_[np.ones(3), np.full(9, 2.)]
    grad = np.zeros(34); grad[21:33] = -weights
    bounds = [(-1., 1.)]*12 + [(0., 1.)]*9 + [(1e-7, .6)]*12 + [(.1, 1.9)]
    anchor = None
    anchor_value = -float('inf')
    def retain(v):
        nonlocal best, best_score, anchor, anchor_value
        tick()
        if not np.isfinite(v).all() or np.any(v[21:33] <= 0):
            return
        xx = v[:12].copy()
        yy = np.r_[np.zeros(3), np.abs(v[12:21])]
        rr = v[21:33].copy()
        cx = xx[groups]; cy = yy[groups]*signs; cr = rr[groups]
        d = np.sqrt((cx[ii]-cx[jj])**2+(cy[ii]-cy[jj])**2)
        rr *= min(1., float(np.min(d/(cr[ii]+cr[jj]))))*(1.-2e-14)
        if np.any(rr <= 0):
            return
        left = np.min(xx-rr); right = np.max(xx+rr)
        width = right-left
        height = 2.*np.max(yy+rr)
        scale = 2./(width+height)*(1.-2e-14)
        xx -= (left+right)/2
        z = np.r_[xx*scale, yy[3:]*scale, rr*scale, width*scale]
        value = float(weights@rr)*scale
        if value > anchor_value:
            anchor_value = value
            anchor = z.copy()
        if value <= best_score:
            return
        obj = {'orbits': [[float(z[g]), float(0. if g < 3 else z[9+g]), float(z[21+g]), 1 if g < 3 else 2] for g in range(12)]}
        exact = score(obj)
        if exact > best_score:
            best, best_score = obj, exact
    cache_z = None; cache_c = None; cache_j = None
    local_end = end
    def check_local():
        if time.monotonic() >= min(end, local_end):
            raise Stop()
    def update(z):
        nonlocal cache_z, cache_c, cache_j
        check_local()
        if cache_z is not None and np.array_equal(z, cache_z):
            return
        dx = DX@z; dy = DY@z
        d = np.sqrt(dx*dx+dy*dy)
        safe = np.maximum(d, 1e-25)
        cache_c = np.r_[d-RS@z, W@z+off]
        cache_j = np.vstack(((dx/safe)[:, None]*DX+(dy/safe)[:, None]*DY-RS, W))
        cache_z = z.copy()
    def fun(z):
        check_local()
        return float(grad@z)
    def jac(z):
        check_local()
        return grad
    def con(z):
        update(z)
        return cache_c
    def cj(z):
        update(z)
        return cache_j
    def solve(z, seconds, iterations=None):
        nonlocal local_end
        tick()
        local_end = min(end, time.monotonic()+seconds)
        last = [z.copy()]; count = [0]
        def callback(v):
            last[0] = v.copy()
            count[0] += 1
            check_local()
            if count[0] % 4 == 0:
                retain(v)
        retain(z)
        try:
            result = minimize(fun, z, jac=jac, method='SLSQP', bounds=bounds,
                constraints={'type':'ineq', 'fun':con, 'jac':cj}, callback=callback,
                options={'maxiter':int(iterations or params['iterations']), 'ftol':8e-14, 'disp':False})
            last[0] = result.x.copy()
        except Stop:
            pass
        if time.monotonic() < end:
            retain(last[0])
        return last[0]
    def vector(g):
        z = np.zeros(34)
        z[:12] = g[:, 0]
        z[12:21] = np.abs(g[3:, 1])
        z[21:33] = np.maximum(g[:, 2], 1e-6)
        z[33] = np.clip(np.max(g[:, 0]+g[:, 2])-np.min(g[:, 0]-g[:, 2]), .1, 1.9)
        return z
    try:
        projections = []
        for swap in (False, True):
            tick()
            b = a[:, [1, 0, 2]].copy() if swap else a.copy()
            order = np.argsort(b[:, 1])
            lower, singles, upper = order[:9], order[9:12], order[12:]
            b[:, 1] -= np.mean(b[singles, 1])
            reflected = b[lower].copy(); reflected[:, 1] *= -1
            cost = np.sum((b[upper, None, :]-reflected[None, :, :])**2, axis=2)
            u, v = linear_sum_assignment(cost)
            g = [[b[k, 0], 0., b[k, 2]] for k in singles]
            for p, q in zip(u, v):
                tick()
                g.append(((b[upper[p]]+reflected[q])/2).tolist())
            error = float(np.sum(cost[u, v])+np.sum(b[singles, 1]**2))
            projections.append((error, vector(np.asarray(g))))
        projections.sort(key=lambda t:t[0])
        solve(projections[0][1], float(params['project_time'])*budget_s)
        # Independent five-row seed from the second parent's construction.
        g = np.zeros((12, 3)); g[:, 2] = .095
        g[:3, 0] = [-.30, 0., .30]
        g[3:7, 0] = np.linspace(-.45, .45, 4); g[3:7, 1] = .18
        g[7:, 0] = np.linspace(-.44, .44, 5); g[7:, 1] = .37
        if best_score < 2.3658323758:
            solve(vector(g), float(params['seed_time'])*budget_s)
        if best_score < 2.35 and time.monotonic() < end:
            solve(projections[1][1], .09*budget_s)
        attempt = 0
        while time.monotonic() < end:
            tick()
            if anchor is None:
                break
            z = anchor.copy()
            if attempt > 0:
                z[:21] += rng.normal(0., float(params['jitter'])*(1.+attempt % 3), 21)
                z[12:21] = np.maximum(z[12:21], 0.)
            solve(z, min(.16*budget_s, end-time.monotonic()))
            attempt += 1
    except (Stop, ValueError, FloatingPointError):
        pass
    return best
