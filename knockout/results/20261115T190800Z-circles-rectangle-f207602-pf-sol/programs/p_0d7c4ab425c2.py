import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {'release': (0.019, 0.003, 0.07), 'contacts': (4, 3, 5), 'probe': (42, 15, 80), 'polish': (70, 25, 120)}


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
        x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
        w = max(u+v for u, v in zip(x, r))-min(u-v for u, v in zip(x, r))
        h = max(u+v for u, v in zip(y, r))-min(u-v for u, v in zip(y, r))
        if w+h > 2:
            return -float('inf')
        for i in range(21):
            for j in range(i):
                dx = a[i, 0]-a[j, 0]
                dy = a[i, 1]-a[j, 1]
                if np.sqrt(dx*dx+dy*dy) < a[i, 2]+a[j, 2]:
                    return -float('inf')
        return float(np.sum(a[:, 2]))
    except (TypeError, ValueError, KeyError, OverflowError):
        return -float('inf')


class StopSearch(Exception):
    pass


class EndSlice(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started+0.69*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = score(init) if init is not None else -float('inf')

    def tick():
        if time.monotonic() >= deadline:
            raise StopSearch()

    if best_score == -float('inf'):
        best = {'circles': [[(i % 7-3)*.2, (i//7-1)*.2, .09999999999999] for i in range(21)]}
        best_score = score(best)
    try:
        a = np.asarray(canonical(init)['circles'], float).copy() if init is not None else np.asarray(best['circles'], float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            a = np.asarray(canonical(best)['circles'], float).copy()
        low = np.min(a[:, :2]-a[:, 2, None], axis=0)
        high = np.max(a[:, :2]+a[:, 2, None], axis=0)
        a[:, :2] -= (low+high)/2
        projections = []
        for swap in (False, True):
            tick()
            b = a.copy()
            if swap:
                b[:, [0, 1]] = b[:, [1, 0]]
            order = np.argsort(b[:, 1])
            lower, singles, upper = order[:9], order[9:12], order[12:]
            ref = b[lower].copy()
            ref[:, 1] *= -1
            cost = np.sum((b[upper, None, :]-ref[None, :, :])**2, axis=2)
            p, q = linear_sum_assignment(cost)
            groups = np.vstack((np.column_stack((b[singles, 0], np.zeros(3), b[singles, 2])), (b[upper[p]]+ref[q])/2))
            projections.append((float(cost[p, q].sum()+np.sum(b[singles, 1]**2)), groups))
        projections.sort(key=lambda t: t[0])

        oi = np.r_[np.arange(12), np.arange(3, 12)]
        sg = np.r_[np.zeros(3), np.ones(9), -np.ones(9)]
        X = np.zeros((21, 34)); Y = X.copy(); R = X.copy()
        for i, g in enumerate(oi):
            tick()
            X[i, g] = 1
            R[i, 21+g] = 1
            if g >= 3:
                Y[i, 9+g] = sg[i]
        edges, seen = [], set()
        for i in range(21):
            tick()
            for j in range(i):
                tick()
                key = (min(oi[i], oi[j]), max(oi[i], oi[j]), int(sg[i]*sg[j]))
                if key not in seen:
                    seen.add(key)
                    edges.append((i, j))
        p, q = np.array(edges).T
        DX, DY, RR = X[p]-X[q], Y[p]-Y[q], R[p]+R[q]
        L = np.zeros((36, 34)); offset = np.zeros(36)
        for g in range(12):
            tick()
            L[g, g], L[g, 21+g], L[g, 33] = 1, -1, 1
            L[12+g, g], L[12+g, 21+g], L[12+g, 33] = -1, -1, 1
            L[24+g, 21+g], L[24+g, 33] = -1, -1
            offset[24+g] = 1
            if g >= 3:
                L[24+g, 9+g] = -1
        weights = np.r_[np.ones(3), np.full(9, 2.)]
        gradient = np.zeros(34); gradient[21:33] = -weights
        bounds = [(-1., 1.)]*12+[(0., 1.)]*9+[(1e-6, .5)]*12+[(.06, .94)]
        release = np.zeros(len(p))
        ii, jj = np.triu_indices(21, 1)
        cache_z = None
        cache_c = cache_j = None
        slice_end = deadline
        last = None
        steps = 0
        local_value = -float('inf')
        local_point = None

        def retain(z):
            nonlocal best, best_score, local_value, local_point
            tick()
            if not np.isfinite(z).all() or np.any(z[21:33] <= 0):
                return
            c = np.column_stack((X@z, Y@z, R@z))
            d = c[ii, :2]-c[jj, :2]
            ratio = float(np.min(np.sqrt(d[:, 0]**2+d[:, 1]**2)/(c[ii, 2]+c[jj, 2])))
            if ratio <= 0:
                return
            c[:, 2] *= min(1., ratio)*(1.-1.2e-14)
            lo = np.min(c[:, :2]-c[:, 2, None], axis=0)
            hi = np.max(c[:, :2]+c[:, 2, None], axis=0)
            scale = 2./float(np.sum(hi-lo))*(1.-1.2e-14)
            c[:, :2] -= (lo+hi)/2
            c *= scale
            estimate = float(np.sum(c[:, 2]))
            if estimate <= min(best_score, local_value):
                return
            obj = {'circles': c.tolist()}
            value = score(obj)
            if value > local_value:
                local_value, local_point = value, z.copy()
            if value > best_score:
                best, best_score = obj, value

        def check_slice():
            tick()
            if time.monotonic() >= slice_end:
                raise EndSlice()

        def constraints(z):
            nonlocal cache_z, cache_c, cache_j
            check_slice()
            if cache_z is None or not np.array_equal(z, cache_z):
                dx, dy = DX@z, DY@z
                dist = np.sqrt(dx*dx+dy*dy)
                safe = np.maximum(dist, 1e-30)
                cache_c = np.r_[dist-RR@z, L@z+offset]
                cache_j = np.vstack(((dx/safe)[:, None]*DX+(dy/safe)[:, None]*DY-RR, L))
                cache_z = z.copy()
            result = cache_c.copy()
            result[:len(p)] -= release
            return result

        def derivatives(z):
            constraints(z)
            return cache_j

        def objective(z):
            check_slice()
            return float(gradient@z)

        def callback(z):
            nonlocal last, steps
            check_slice()
            last = z.copy()
            steps += 1
            if steps % 5 == 0:
                retain(z)

        def solve(z, iterations, seconds):
            nonlocal slice_end, last
            tick()
            slice_end = min(deadline, time.monotonic()+seconds)
            last = z.copy()
            try:
                result = minimize(objective, z, jac=lambda z: gradient, bounds=bounds, method='SLSQP', constraints={'type': 'ineq', 'fun': constraints, 'jac': derivatives}, callback=callback, options={'maxiter': iterations, 'ftol': 3e-14, 'disp': False})
                last = result.x.copy()
            except EndSlice:
                pass
            slice_end = deadline
            retain(last)
            return last.copy()

        def vector(g):
            z = np.zeros(34)
            z[:12] = g[:, 0]
            z[12:21] = np.maximum(g[3:, 1], 0.)
            z[21:33] = np.maximum(g[:, 2], 1e-6)
            z[33] = np.clip((np.max(g[:, 0]+g[:, 2])-np.min(g[:, 0]-g[:, 2]))/2, .06, .94)
            return z

        # Compact five-row construction, with three circles on the axis.
        g = np.zeros((12, 3))
        g[:3, 0] = [-2.5, 0., 2.5]
        g[3:8, 0] = [-4., -2., 0., 2., 4.]
        g[3:8, 1] = np.sqrt(3.)
        g[8:, 0] = [-3., -1., 1., 3.]
        g[8:, 1] = 2*np.sqrt(3.)
        g[:, 2] = 1.
        g *= 2./(10.+2.+4*np.sqrt(3.))
        starts = [vector(projections[0][1]), vector(g), vector(projections[1][1])]
        racers = []
        for k, z in enumerate(starts):
            tick()
            local_value, local_point = -float('inf'), None
            z = solve(z, int(params['probe'])+(18 if k == 1 else 0), budget_s*(.19 if k < 2 else .10))
            racers.append((local_value, local_point.copy() if local_point is not None else z.copy()))
        racers.sort(key=lambda t: t[0], reverse=True)
        local_value, local_point = -float('inf'), None
        anchor = solve(racers[0][1], int(params['polish']), .16*budget_s)
        if local_point is not None:
            anchor = local_point.copy()
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
            probability = np.where(shared, 4., 1.)
            probability /= probability.sum()
            chosen = rng.choice(near, min(int(params['contacts']), len(near)), replace=False, p=probability)
            release[chosen] = float(params['release'])*rng.uniform(.7, 1.3, len(chosen))*(.7+.4*(attempt % 3))
            moved = solve(anchor.copy(), 24, .07*budget_s)
            release[:] = 0
            recovered = solve(moved, int(params['polish']), .12*budget_s)
            if np.min(constraints(recovered)) > -1e-8 and gradient@recovered < gradient@anchor-1e-10:
                anchor = recovered.copy()
            attempt += 1
    except (StopSearch, EndSlice, ValueError, FloatingPointError):
        pass
    return best
