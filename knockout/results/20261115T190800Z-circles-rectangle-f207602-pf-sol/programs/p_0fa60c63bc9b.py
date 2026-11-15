import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {'release': (0.023, 0.003, 0.08), 'contacts': (4, 3, 5), 'probe': (18, 8, 40), 'recover': (42, 15, 90)}


def canonical(obj):
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
        for i in range(21):
            for j in range(i):
                dx = a[i, 0]-a[j, 0]
                dy = a[i, 1]-a[j, 1]
                if np.sqrt(dx*dx+dy*dy) < a[i, 2]+a[j, 2]:
                    return -float('inf')
        x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
        w = max(u+v for u, v in zip(x, r))-min(u-v for u, v in zip(x, r))
        h = max(u+v for u, v in zip(y, r))-min(u-v for u, v in zip(y, r))
        if w+h > 2:
            return -float('inf')
        return float(np.sum(a[:, 2]))
    except (TypeError, KeyError, ValueError, OverflowError):
        return -float('inf')


class StopSearch(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    deadline = time.monotonic()+0.69*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = exact_score(init) if init is not None else -float('inf')

    def tick():
        if time.monotonic() >= deadline:
            raise StopSearch()

    if best_score == -float('inf'):
        best = {'circles': [[(i % 7-3)*0.17, (i//7-1)*0.17, 0.08] for i in range(21)]}
        best_score = exact_score(best)
    a = np.asarray(canonical(best)['circles'], float).copy()
    lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
    hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
    a[:, :2] -= (lo+hi)/2
    projections = []
    for swap in (False, True):
        if time.monotonic() >= deadline:
            return best
        b = a.copy()
        if swap:
            b[:, [0, 1]] = b[:, [1, 0]]
        order = np.argsort(b[:, 1])
        low, singles, high = order[:9], order[9:12], order[12:]
        reflected = b[low].copy()
        reflected[:, 1] *= -1
        cost = np.sum((b[high, None, :]-reflected[None, :, :])**2, axis=2)
        p, q = linear_sum_assignment(cost)
        g = np.vstack((np.column_stack((b[singles, 0], np.zeros(3), b[singles, 2])), (b[high[p]]+reflected[q])/2))
        projections.append((float(np.sum(cost[p, q])+np.sum(b[singles, 1]**2)), g))
    projections.sort(key=lambda t: t[0])

    oi = np.r_[np.arange(12), np.arange(3, 12)]
    sg = np.r_[np.ones(12), -np.ones(9)]
    X = np.zeros((21, 34))
    Y = np.zeros_like(X)
    R = np.zeros_like(X)
    for i, g in enumerate(oi):
        if time.monotonic() >= deadline:
            return best
        X[i, g] = 1
        R[i, 21+g] = 1
        if g >= 3:
            Y[i, 12+g-3] = sg[i]
    edges, seen = [], set()
    for i in range(21):
        if time.monotonic() >= deadline:
            return best
        for j in range(i):
            if time.monotonic() >= deadline:
                return best
            key = (min(oi[i], oi[j]), max(oi[i], oi[j]), int(sg[i]*sg[j]) if oi[i] >= 3 and oi[j] >= 3 else 0)
            if key not in seen:
                seen.add(key)
                edges.append((i, j))
    p, q = np.array(edges).T
    DX, DY, RR = X[p]-X[q], Y[p]-Y[q], R[p]+R[q]
    L = np.zeros((36, 34))
    offset = np.zeros(36)
    for g in range(12):
        if time.monotonic() >= deadline:
            return best
        L[g, g], L[g, 21+g], L[g, 33] = 1, -1, 1
        L[12+g, g], L[12+g, 21+g], L[12+g, 33] = -1, -1, 1
        L[24+g, 21+g], L[24+g, 33] = -1, -1
        offset[24+g] = 1
        if g >= 3:
            L[24+g, 12+g-3] = -1
    weights = np.r_[np.ones(3), np.full(9, 2.)]
    gradient = np.zeros(34)
    gradient[21:33] = -weights
    bounds = [(-1., 1.)]*12+[(0., 1.)]*9+[(1e-5, .5)]*12+[(.08, .92)]
    release = np.zeros(len(p))
    cache_z = None
    cache_c = None
    cache_j = None
    ii, jj = np.triu_indices(21, 1)
    steps = 0

    def retain(z):
        nonlocal best, best_score
        tick()
        if not np.isfinite(z).all() or np.any(z[21:33] <= 0):
            return
        c = np.column_stack((X@z, Y@z, R@z))
        d = c[ii, :2]-c[jj, :2]
        ratio = float(np.min(np.sqrt(d[:, 0]**2+d[:, 1]**2)/(c[ii, 2]+c[jj, 2])))
        if ratio <= 0:
            return
        c[:, 2] *= min(1., ratio)*(1.-6e-15)
        low = np.min(c[:, :2]-c[:, 2, None], axis=0)
        high = np.max(c[:, :2]+c[:, 2, None], axis=0)
        scale = 2./float(np.sum(high-low))*(1.-8e-15)
        c[:, :2] -= (low+high)/2
        c *= scale
        if float(np.sum(c[:, 2])) <= best_score:
            return
        obj = {'circles': c.tolist()}
        s = exact_score(obj)
        if s > best_score:
            best, best_score = obj, s

    def constraints(z):
        nonlocal cache_z, cache_c, cache_j
        tick()
        if cache_z is None or not np.array_equal(z, cache_z):
            dx, dy = DX@z, DY@z
            dist = np.sqrt(dx*dx+dy*dy)
            safe = np.maximum(dist, 1e-30)
            cache_c = np.r_[dist-RR@z, L@z+offset]
            cache_j = np.vstack(((dx/safe)[:, None]*DX+(dy/safe)[:, None]*DY-RR, L))
            cache_z = z.copy()
        out = cache_c.copy()
        out[:len(p)] -= release
        return out

    def jac(z):
        constraints(z)
        return cache_j

    def callback(z):
        nonlocal steps
        tick()
        steps += 1
        if steps % 5 == 0:
            retain(z)

    def objective(z):
        tick()
        return float(gradient@z)

    def solve(z, iterations):
        tick()
        result = minimize(objective, z, jac=lambda z: gradient, bounds=bounds, method='SLSQP', constraints={'type': 'ineq', 'fun': constraints, 'jac': jac}, callback=callback, options={'maxiter': iterations, 'ftol': 8e-14, 'disp': False})
        retain(result.x)
        return result.x

    try:
        g = projections[0][1]
        anchor = np.zeros(34)
        anchor[:12] = g[:, 0]
        anchor[12:21] = np.maximum(g[3:, 1], 0.)
        anchor[21:33] = g[:, 2]
        anchor[33] = np.clip((np.max(g[:, 0]+g[:, 2])-np.min(g[:, 0]-g[:, 2]))/2, .08, .92)
        anchor = solve(anchor, 55)
        attempt = 0
        while time.monotonic() < deadline:
            tick()
            release[:] = 0
            gaps = constraints(anchor)[:len(p)]
            near = np.flatnonzero(gaps < .004)
            if len(near) < 3:
                break
            root = int(rng.choice(near))
            shared = np.isin(oi[p[near]], [oi[p[root]], oi[q[root]]]) | np.isin(oi[q[near]], [oi[p[root]], oi[q[root]]])
            probabilities = np.where(shared, 4., 1.)
            probabilities /= probabilities.sum()
            chosen = rng.choice(near, min(int(params['contacts']), len(near)), replace=False, p=probabilities)
            release[chosen] = float(params['release'])*rng.uniform(.7, 1.3, len(chosen))*(.65+0.45*(attempt % 3))
            moved = solve(anchor.copy(), int(params['probe']))
            release[:] = 0
            recovered = solve(moved, int(params['recover']))
            if np.min(constraints(recovered)) > -1e-8 and gradient@recovered < gradient@anchor-1e-10:
                anchor = recovered.copy()
            attempt += 1
    except (StopSearch, ValueError, FloatingPointError):
        pass
    return best
