import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {"screen_iterations": (65, 30, 120), "polish_iterations": (110, 40, 200), "release": (0.012, 0.002, 0.04)}


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
        x, y, r = [[Fraction(float(t)) for t in a[:, k]] for k in range(3)]
        if max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))+max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)) > 2:
            return -float('inf')
        for i in range(21):
            for j in range(i):
                dx, dy = a[i, :2]-a[j, :2]
                if np.sqrt(dx*dx+dy*dy) < a[i, 2]+a[j, 2]:
                    return -float('inf')
        return float(np.sum(a[:, 2]))
    except (ValueError, TypeError, KeyError, OverflowError):
        return -float('inf')


class StopSearch(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.69*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = score(init) if init is not None else -float('inf')

    def tick():
        if time.monotonic() >= deadline:
            raise StopSearch()

    # A certified fallback is independent of the quality of the input.
    fallback = {'circles': [[(i % 7-3)*0.19999999999999, (i//7-1)*0.19999999999999, 0.09999999999999] for i in range(21)]}
    fs = score(fallback)
    if fs > best_score:
        best, best_score = fallback, fs

    group = np.array([0, 1, 2]+[g for g in range(3, 12) for _ in range(2)])
    sign = np.array([0., 0., 0.]+[s for _ in range(9) for s in (1., -1.)])
    X = np.zeros((21, 34)); Y = np.zeros_like(X); R = np.zeros_like(X)
    for k in range(21):
        tick()
        g = group[k]
        X[k, g] = 1
        R[k, 21+g] = 1
        if g >= 3:
            Y[k, 9+g] = sign[k]
    pairs = []
    seen = set()
    for i in range(21):
        tick()
        for j in range(i):
            key = (min(group[i], group[j]), max(group[i], group[j]), int(sign[i]*sign[j]))
            if key not in seen:
                seen.add(key)
                pairs.append((i, j))
    pi, pj = np.array(pairs).T
    DX, DY, RS = X[pi]-X[pj], Y[pi]-Y[pj], R[pi]+R[pj]
    walls = np.zeros((36, 34)); offset = np.zeros(36)
    for g in range(12):
        tick()
        walls[g, g] = 1
        walls[g, 21+g] = -1
        walls[g, 33] = .5
        walls[12+g, g] = -1
        walls[12+g, 21+g] = -1
        walls[12+g, 33] = .5
        walls[24+g, 21+g] = -1
        walls[24+g, 33] = -.5
        offset[24+g] = 1
        if g >= 3:
            walls[24+g, 9+g] = -1
    weights = np.array([1.]*3+[2.]*9)
    objective = np.zeros(34); objective[21:33] = -weights
    bounds = [(-1., 1.)]*12+[(0., 1.)]*9+[(1e-7, .6)]*12+[(.08, 1.92)]
    release = np.zeros(len(pi))
    ai, aj = np.triu_indices(21, 1)
    champion = None
    champion_value = -float('inf')

    def retain(v):
        nonlocal best, best_score, champion, champion_value
        tick()
        if not np.isfinite(v).all() or np.any(v[21:33] <= 0):
            return
        a = np.column_stack((X@v, Y@v, R@v))
        d = a[ai, :2]-a[aj, :2]
        factor = min(1., float(np.min(np.sqrt(np.sum(d*d, axis=1))/(a[ai, 2]+a[aj, 2]))))
        r = v[21:33]*factor*(1-2e-14)
        xx = v[:12]
        yy = np.r_[np.zeros(3), v[12:21]]
        perimeter = np.max(xx+r)-np.min(xx-r)+2*np.max(yy+r)
        scale = min(1., 2./perimeter)*(1-2e-14)
        value = float(weights@r)*scale
        if value > champion_value:
            champion_value = value
            champion = v.copy()
        if value <= best_score:
            return
        obj = {'orbits': [[float(xx[g]*scale), float(yy[g]*scale), float(r[g]*scale), 1 if g < 3 else 2] for g in range(12)]}
        exact = score(obj)
        if exact > best_score:
            best, best_score = obj, exact

    def fun(v):
        tick()
        return float(objective@v)

    def jac(v):
        tick()
        return objective

    def con(v):
        tick()
        dx, dy = DX@v, DY@v
        return np.r_[np.sqrt(dx*dx+dy*dy)-RS@v-release, walls@v+offset]

    def cjac(v):
        tick()
        dx, dy = DX@v, DY@v
        dd = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-24)
        return np.vstack(((dx/dd)[:, None]*DX+(dy/dd)[:, None]*DY-RS, walls))

    def solve(v, n):
        tick()
        counter = 0
        def callback(z):
            nonlocal counter
            tick()
            counter += 1
            if counter % 5 == 0:
                retain(z)
        result = minimize(fun, v, method='SLSQP', jac=jac, bounds=bounds,
                          constraints={'type': 'ineq', 'fun': con, 'jac': cjac},
                          callback=callback,
                          options={'maxiter': n, 'ftol': 2e-12, 'disp': False})
        retain(result.x)
        return result.x

    def lattice(rows, stretch=1.):
        data = [[2.*(j-1), 0., 1.] for j in range(3)]
        for k, count in enumerate(rows):
            tick()
            for j in range(count):
                data.append([2.*(j-(count-1)/2), (k+1)*np.sqrt(3.)*stretch, 1.])
        g = np.asarray(data)
        w = np.ptp(g[:, 0])+2
        h = 2*np.max(g[:, 1]+g[:, 2])
        g *= 2/(w+h)
        v = np.zeros(34)
        v[:12] = g[:, 0]
        v[12:21] = g[3:, 1]
        v[21:33] = g[:, 2]
        v[33] = w*2/(w+h)
        return v

    try:
        starts = []
        # Use a meaningful incumbent before any exploratory starts.
        if init is not None and score(init) > 2.15:
            a = np.asarray(canonical(init)['circles'], float).copy()
            lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
            hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
            a[:, :2] -= (lo+hi)/2
            projections = []
            for swap in (False, True):
                tick()
                b = a.copy()
                if swap:
                    b[:, [0, 1]] = b[:, [1, 0]]
                order = np.argsort(b[:, 1])
                lower, singles, upper = order[:9], order[9:12], order[12:]
                ref = b[lower].copy(); ref[:, 1] *= -1
                cost = np.sum((b[upper, None, :]-ref[None, :, :])**2, axis=2)
                u, vv = linear_sum_assignment(cost)
                g = np.vstack((np.column_stack((b[singles, 0], np.zeros(3), b[singles, 2])), (b[upper[u]]+ref[vv])/2))
                v = np.zeros(34)
                v[:12] = g[:, 0]; v[12:21] = np.maximum(g[3:, 1], 0)
                v[21:33] = g[:, 2]
                v[33] = np.max(g[:, 0]+g[:, 2])-np.min(g[:, 0]-g[:, 2])
                projections.append((float(np.sum(cost[u, vv])+np.sum(b[singles, 1]**2)), v))
            starts.append(min(projections, key=lambda p: p[0])[1])
        starts.extend([lattice((4, 3, 2)), lattice((3, 3, 3)), lattice((2, 3, 4)), lattice((5, 4))])
        survivors = []
        for v in starts:
            tick()
            if time.monotonic()-start > .43*budget_s and survivors:
                break
            v = solve(v, int(params['screen_iterations']))
            survivors.append(v)
        if champion is not None:
            release[:] = 0
            anchor = solve(champion.copy(), int(params['polish_iterations']))
        else:
            anchor = starts[0]
        attempt = 0
        while time.monotonic() < deadline:
            tick()
            release[:] = 0
            gaps = con(anchor)[:len(pi)]
            near = np.flatnonzero(gaps < .004)
            if len(near) < 3:
                break
            selected = rng.choice(near, min(4, len(near)), replace=False)
            release[selected] = float(params['release'])*rng.uniform(.7, 1.3, len(selected))
            v = solve(anchor.copy(), 18)
            release[:] = 0
            v = solve(v, 55)
            if champion is not None:
                anchor = champion.copy()
            attempt += 1
    except (StopSearch, ValueError, FloatingPointError):
        pass
    return best
