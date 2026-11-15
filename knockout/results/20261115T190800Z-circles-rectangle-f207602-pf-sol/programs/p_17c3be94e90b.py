import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {
    'screen_iterations': (28, 15, 50),
    'polish_iterations': (95, 50, 150),
    'release': (0.009, 0.002, 0.025),
    'screen_fraction': (0.32, 0.2, 0.45)
}

class SearchStop(Exception):
    pass

class SliceStop(Exception):
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
        a = np.asarray(canonical(obj)['circles'], dtype=np.float64)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        i, j = np.triu_indices(21, 1)
        if np.any(np.sqrt((a[i, 0]-a[j, 0])**2 + (a[i, 1]-a[j, 1])**2) < a[i, 2]+a[j, 2]):
            return -float('inf')
        x = list(map(Fraction, map(float, a[:, 0])))
        y = list(map(Fraction, map(float, a[:, 1])))
        r = list(map(Fraction, map(float, a[:, 2])))
        w = max(u+v for u, v in zip(x, r))-min(u-v for u, v in zip(x, r))
        h = max(u+v for u, v in zip(y, r))-min(u-v for u, v in zip(y, r))
        if w+h > 2:
            return -float('inf')
        return float(np.sum(a[:, 2]))
    except (TypeError, ValueError, KeyError, OverflowError):
        return -float('inf')

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = exact_score(init)
    def tick():
        if time.monotonic() >= deadline:
            raise SearchStop()
    fallback = {'circles': [[.2*(k % 7-3), .2*(k//7-1), .09999999999998] for k in range(21)]}
    fs = exact_score(fallback)
    if fs > best_score:
        best, best_score = fallback, fs
    try:
        tick()
        gid = np.r_[np.arange(3), np.repeat(np.arange(3, 12), 2)]
        sign = np.r_[np.zeros(3), np.tile([1., -1.], 9)]
        X = np.zeros((21, 34))
        Y = np.zeros_like(X)
        R = np.zeros_like(X)
        X[np.arange(21), gid] = 1.
        R[np.arange(21), 21+gid] = 1.
        q = np.flatnonzero(gid >= 3)
        Y[q, 9+gid[q]] = sign[q]
        pairs = []
        seen = set()
        for i in range(21):
            tick()
            for j in range(i):
                tick()
                key = (min(gid[i], gid[j]), max(gid[i], gid[j]), sign[i]*sign[j])
                if key not in seen:
                    seen.add(key)
                    pairs.append((i, j))
        ii, jj = np.asarray(pairs).T
        DX, DY, RS = X[ii]-X[jj], Y[ii]-Y[jj], R[ii]+R[jj]
        W = np.zeros((36, 34))
        off = np.zeros(36)
        for g in range(12):
            tick()
            W[g, g] = 1.
            W[g, 21+g] = -1.
            W[g, 33] = .5
            W[12+g, g] = -1.
            W[12+g, 21+g] = -1.
            W[12+g, 33] = .5
            W[24+g, 21+g] = -1.
            W[24+g, 33] = -.5
            off[24+g] = 1.
            if g >= 3:
                W[24+g, 9+g] = -1.
        grad = np.zeros(34)
        grad[21:33] = -np.r_[np.ones(3), np.full(9, 2.)]
        bounds = [(-1., 1.)]*12 + [(0., 1.)]*9 + [(1e-7, .5)]*12 + [(.05, 1.95)]
        ai, aj = np.triu_indices(21, 1)
        release = np.zeros(len(ii))
        global_z = None
        global_z_score = -float('inf')

        def repaired(v):
            if not np.isfinite(v).all() or np.min(v[21:33]) <= 0:
                return None
            a = np.column_stack((X@v, Y@v, R@v))
            d = np.sqrt((a[ai, 0]-a[aj, 0])**2 + (a[ai, 1]-a[aj, 1])**2)
            ratio = float(np.min(d/(a[ai, 2]+a[aj, 2])))
            if ratio <= 0:
                return None
            a[:, 2] *= min(1., ratio)*(1.-2e-14)
            wh = np.max(a[:, :2]+a[:, 2, None], axis=0)-np.min(a[:, :2]-a[:, 2, None], axis=0)
            a *= min(1., 2./float(np.sum(wh)))*(1.-2e-14)
            return a

        def retain(v):
            nonlocal best, best_score, global_z, global_z_score
            tick()
            a = repaired(v)
            if a is None:
                return -float('inf')
            estimate = float(np.sum(a[:, 2]))
            if estimate <= min(best_score, global_z_score):
                return estimate
            obj = {'circles': a.tolist()}
            score = exact_score(obj)
            if score > global_z_score:
                global_z_score, global_z = score, v.copy()
            if score > best_score:
                best_score, best = score, obj
            return score

        def pack(g):
            v = np.zeros(34)
            v[:12] = g[:, 0]
            v[12:21] = np.abs(g[3:, 1])
            v[21:33] = g[:, 2]
            v[33] = np.clip(np.max(g[:, 0]+g[:, 2])-np.min(g[:, 0]-g[:, 2]), .05, 1.95)
            return v

        def solve(v, iterations, until):
            last = v.copy()
            cache_v = None
            cache_c = None
            cache_j = None
            def check():
                tick()
                if time.monotonic() >= until:
                    raise SliceStop()
            def update(z):
                nonlocal cache_v, cache_c, cache_j
                check()
                if cache_v is None or not np.array_equal(z, cache_v):
                    dx, dy = DX@z, DY@z
                    d = np.sqrt(dx*dx+dy*dy)
                    safe = np.maximum(d, 1e-25)
                    cache_c = np.r_[d-RS@z-release, W@z+off]
                    cache_j = np.vstack(((dx/safe)[:, None]*DX+(dy/safe)[:, None]*DY-RS, W))
                    cache_v = z.copy()
            def fun(z):
                check()
                return float(grad@z)
            def jac(z):
                check()
                return grad
            def con(z):
                update(z)
                return cache_c
            def cj(z):
                update(z)
                return cache_j
            count = 0
            def cb(z):
                nonlocal last, count
                last = z.copy()
                count += 1
                if count % 6 == 0:
                    retain(last)
                check()
            try:
                out = minimize(fun, v, jac=jac, bounds=bounds, method='SLSQP',
                    constraints={'type': 'ineq', 'fun': con, 'jac': cj}, callback=cb,
                    options={'maxiter': iterations, 'ftol': 1e-12, 'disp': False})
                last = out.x.copy()
            except SliceStop:
                pass
            score = retain(last)
            return last, score

        starts = []
        raw = canonical(init)
        if raw is not None:
            a = np.asarray(raw['circles'], dtype=float)
            if a.shape == (21, 3) and np.isfinite(a).all():
                a = a.copy()
                lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
                hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
                a[:, :2] -= (lo+hi)/2
                projected = []
                for swap in (False, True):
                    tick()
                    b = a.copy()
                    if swap:
                        b[:, [0, 1]] = b[:, [1, 0]]
                    order = np.argsort(b[:, 1])
                    lower, middle, upper = order[:9], order[9:12], order[12:]
                    reflected = b[lower].copy()
                    reflected[:, 1] *= -1
                    cost = np.sum((b[upper, None, :]-reflected[None, :, :])**2, axis=2)
                    u, v = linear_sum_assignment(cost)
                    g = np.zeros((12, 3))
                    g[:3] = b[middle]
                    g[:3, 1] = 0.
                    g[3:] = (b[upper[u]]+reflected[v])/2
                    err = float(np.sum(cost[u, v])+np.sum(b[middle, 1]**2))
                    projected.append((err, pack(g)))
                projected.sort(key=lambda t: t[0])
                starts.append(projected[0][1])
        g = np.zeros((12, 3))
        g[:, 2] = .095
        g[:3, 0] = [-.32, 0., .32]
        g[3:7, 0] = np.linspace(-.42, .42, 4)
        g[3:7, 1] = .17
        g[7:, 0] = np.linspace(-.42, .42, 5)
        g[7:, 1] = .35
        starts.append(pack(g))
        if raw is not None and 'projected' in locals():
            starts.append(projected[1][1])
        g = np.zeros((12, 3))
        g[:, 2] = .095
        g[:3, 0] = [-.26, 0., .26]
        g[3:, 0] = np.tile([-.26, 0., .26], 3)
        g[3:, 1] = np.repeat([.15, .30, .45], 3)
        starts.append(pack(g))

        beam = []
        screen_end = start+float(params['screen_fraction'])*budget_s
        for k, z in enumerate(starts):
            tick()
            now = time.monotonic()
            if now >= screen_end:
                break
            allowance = (screen_end-now)/max(1, len(starts)-k)
            z, score = solve(z, int(params['screen_iterations']), min(screen_end, now+allowance))
            beam.append((score, z))
        beam.sort(key=lambda p: p[0], reverse=True)
        for k, (_, z) in enumerate(beam[:2]):
            tick()
            remaining = deadline-time.monotonic()
            cap = time.monotonic()+remaining*(.64 if k == 0 else .72)
            z, score = solve(z, int(params['polish_iterations']), cap)
        if global_z is not None:
            tick()
            release[:] = 0.
            z, score = solve(global_z.copy(), 70, time.monotonic()+.48*(deadline-time.monotonic()))
        attempt = 0
        while global_z is not None and time.monotonic() < deadline:
            tick()
            anchor = global_z.copy()
            dx, dy = DX@anchor, DY@anchor
            gaps = np.sqrt(dx*dx+dy*dy)-RS@anchor
            near = np.flatnonzero(gaps < .003)
            if len(near) < 3:
                break
            chosen = rng.choice(near, min(4, len(near)), replace=False)
            release[:] = 0.
            release[chosen] = float(params['release'])*(1.+.3*(attempt % 3))
            remaining = deadline-time.monotonic()
            moved, _ = solve(anchor, 16, time.monotonic()+.30*remaining)
            release[:] = 0.
            solve(moved, 65, deadline)
            attempt += 1
    except (SearchStop, ValueError, FloatingPointError):
        pass
    return best
