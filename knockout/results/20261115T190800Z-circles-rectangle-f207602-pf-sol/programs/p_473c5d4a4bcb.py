import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {'probe': (40, 15, 75), 'lattice': (75, 35, 120), 'polish': (65, 20, 110), 'release': (0.019, 0.003, 0.06), 'contacts': (4, 3, 5)}


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
        a = np.asarray(canonical(obj)['circles'], dtype=np.float64)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        for i in range(21):
            for j in range(i):
                dx, dy = a[i, :2]-a[j, :2]
                if np.sqrt(dx*dx+dy*dy) < a[i, 2]+a[j, 2]:
                    return -float('inf')
        x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
        w = max(u+v for u, v in zip(x, r))-min(u-v for u, v in zip(x, r))
        h = max(u+v for u, v in zip(y, r))-min(u-v for u, v in zip(y, r))
        if w+h > 2:
            return -float('inf')
        return float(np.sum(a[:, 2]))
    except (TypeError, ValueError, KeyError, OverflowError):
        return -float('inf')


class Finished(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    deadline = time.monotonic()+0.69*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = score(init) if init is not None else -float('inf')

    def tick():
        if time.monotonic() >= deadline:
            raise Finished()

    if best_score == -float('inf'):
        best = {'circles': [[(i % 7-3)*0.17, (i//7-1)*0.17, 0.08] for i in range(21)]}
        best_score = score(best)
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
        lower, singles, upper = order[:9], order[9:12], order[12:]
        reflected = b[lower].copy()
        reflected[:, 1] *= -1
        cost = np.sum((b[upper, None, :]-reflected[None, :, :])**2, axis=2)
        p, q = linear_sum_assignment(cost)
        g = np.vstack((np.column_stack((b[singles, 0], np.zeros(3), b[singles, 2])), (b[upper[p]]+reflected[q])/2))
        error = float(np.sum(cost[p, q])+np.sum(b[singles, 1]**2))
        projections.append((error, g))
    projections.sort(key=lambda t: t[0])

    oi = np.r_[np.arange(12), np.arange(3, 12)]
    sg = np.r_[np.ones(12), -np.ones(9)]
    X = np.zeros((21, 34)); Y = X.copy(); R = X.copy()
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
    p, q = np.asarray(edges).T
    DX, DY, RR = X[p]-X[q], Y[p]-Y[q], R[p]+R[q]
    L = np.zeros((36, 34)); offset = np.zeros(36)
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
    gradient = np.zeros(34); gradient[21:33] = -weights
    bounds = [(-1., 1.)]*12+[(0., 1.)]*9+[(1e-6, .5)]*12+[(.06, .94)]
    release = np.zeros(len(p))
    ii, jj = np.triu_indices(21, 1)
    cache_z = cache_c = cache_j = None
    local_best = None
    local_score = -float('inf')
    steps = 0

    def retain(z):
        nonlocal best, best_score, local_best, local_score
        tick()
        if not np.isfinite(z).all() or np.any(z[21:33] <= 0):
            return
        c = np.column_stack((X@z, Y@z, R@z))
        d = c[ii, :2]-c[jj, :2]
        ratio = float(np.min(np.sqrt(d[:, 0]**2+d[:, 1]**2)/(c[ii, 2]+c[jj, 2])))
        if ratio <= 0:
            return
        c[:, 2] *= min(1., ratio)*(1.-1.2e-14)
        low = np.min(c[:, :2]-c[:, 2, None], axis=0)
        high = np.max(c[:, :2]+c[:, 2, None], axis=0)
        scale = 2./float(np.sum(high-low))*(1.-1.2e-14)
        c[:, :2] -= (low+high)/2
        c *= scale
        estimate = float(np.sum(c[:, 2]))
        if estimate <= min(best_score, local_score):
            return
        obj = {'circles': c.tolist()}
        value = score(obj)
        if value > local_score:
            local_score, local_best = value, z.copy()
        if value > best_score:
            best, best_score = obj, value

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

    def objective(z):
        tick()
        return float(gradient@z)

    def objective_jac(z):
        tick()
        return gradient

    def callback(z):
        nonlocal steps
        tick()
        steps += 1
        if steps % 4 == 0:
            retain(z)

    def solve(z, iterations):
        tick()
        result = minimize(objective, z, jac=objective_jac, bounds=bounds, method='SLSQP', constraints={'type': 'ineq', 'fun': constraints, 'jac': jac}, callback=callback, options={'maxiter': iterations, 'ftol': 4e-14, 'disp': False})
        retain(result.x)
        return result.x

    def encode(g):
        z = np.zeros(34)
        z[:12] = g[:, 0]
        z[12:21] = np.maximum(g[3:, 1], 0.)
        z[21:33] = np.maximum(g[:, 2], 1e-6)
        z[33] = np.clip((np.max(g[:, 0]+g[:, 2])-np.min(g[:, 0]-g[:, 2]))/2, .06, .94)
        return z

    try:
        lattice = np.array([[-2., 0., 1.], [0., 0., 1.], [2., 0., 1.], [-3., np.sqrt(3.), 1.], [-1., np.sqrt(3.), 1.], [1., np.sqrt(3.), 1.], [3., np.sqrt(3.), 1.], [-4., 2*np.sqrt(3.), 1.], [-2., 2*np.sqrt(3.), 1.], [0., 2*np.sqrt(3.), 1.], [2., 2*np.sqrt(3.), 1.], [4., 2*np.sqrt(3.), 1.]])
        lattice *= 2./(12.+4.*np.sqrt(3.))
        candidates = [(projections[0][1], int(params['probe'])), (lattice, int(params['lattice'])), (projections[1][1], int(params['probe']))]
        racers = []
        for g, iterations in candidates:
            tick()
            local_best, local_score = None, -float('inf')
            z = solve(encode(g), iterations)
            racers.append((local_score, local_best.copy() if local_best is not None else z.copy()))
        racers.sort(key=lambda t: t[0], reverse=True)
        local_best, local_score = None, -float('inf')
        anchor = solve(racers[0][1], int(params['polish']))
        attempt = 0
        while time.monotonic() < deadline:
            tick()
            release[:] = 0
            gaps = constraints(anchor)[:len(p)]
            near = np.flatnonzero(gaps < .003)
            if len(near) < 3:
                break
            root = int(rng.choice(near))
            shared = np.isin(oi[p[near]], [oi[p[root]], oi[q[root]]]) | np.isin(oi[q[near]], [oi[p[root]], oi[q[root]]])
            probabilities = np.where(shared, 4., 1.)
            probabilities /= probabilities.sum()
            chosen = rng.choice(near, min(int(params['contacts']), len(near)), replace=False, p=probabilities)
            release[chosen] = float(params['release'])*rng.uniform(.7, 1.3, len(chosen))*(.7+.4*(attempt % 3))
            moved = solve(anchor.copy(), 20)
            release[:] = 0
            recovered = solve(moved, int(params['polish']))
            if np.min(constraints(recovered)) > -1e-8 and gradient@recovered < gradient@anchor-1e-10:
                anchor = recovered.copy()
            attempt += 1
    except (Finished, ValueError, FloatingPointError):
        pass
    return best
