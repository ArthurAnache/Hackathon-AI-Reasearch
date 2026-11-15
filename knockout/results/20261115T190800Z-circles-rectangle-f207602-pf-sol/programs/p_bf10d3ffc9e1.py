import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {
    "release": (0.018, 0.003, 0.06),
    "release_iterations": (23, 10, 55),
    "restore_iterations": (85, 30, 160),
    "safety": (0.69, 0.55, 0.72)
}


class StopSearch(Exception):
    pass


class SliceEnd(Exception):
    pass


def canonical(obj):
    if obj is None or 'circles' in obj:
        return obj
    out = []
    for x, y, r, m in obj['orbits']:
        out.append([x, y, r])
        if m == 2:
            out.append([x, -y, r])
    return {'circles': out}


def exact_score(obj):
    try:
        a = np.asarray(canonical(obj)['circles'], dtype=np.float64)
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
    except (TypeError, ValueError, KeyError, OverflowError):
        return -float('inf')


def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + float(params['safety'])*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = exact_score(init) if init is not None else -float('inf')
    ii, jj = np.triu_indices(21, 1)

    def tick():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def consider(a):
        nonlocal best, best_score
        tick()
        if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        a = a.copy()
        a[:, :2] -= (np.max(a[:, :2]+a[:, 2, None], axis=0)+np.min(a[:, :2]-a[:, 2, None], axis=0))/2
        d = np.sqrt((a[ii, 0]-a[jj, 0])**2+(a[ii, 1]-a[jj, 1])**2)
        factor = min(1., float(np.min(d/(a[ii, 2]+a[jj, 2]))))
        if factor <= 0:
            return -float('inf')
        a[:, 2] *= factor*(1-8e-15)
        ext = np.max(a[:, :2]+a[:, 2, None], axis=0)-np.min(a[:, :2]-a[:, 2, None], axis=0)
        a *= (2/float(np.sum(ext)))*(1-8e-15)
        value = float(np.sum(a[:, 2]))
        if value > best_score:
            obj = {'circles': a.tolist()}
            verified = exact_score(obj)
            if verified > best_score:
                best, best_score = obj, verified
        return value

    try:
        tick()
        if best is None:
            fallback = np.array([[(k % 7-3)*.2, (k//7-1)*.2, .0999999999999] for k in range(21)])
            consider(fallback)

        # Twelve reflection orbits: three singletons and nine mirrored pairs.
        pi, pj, signs = [], [], []
        for i in range(12):
            tick()
            for j in range(i+1, 12):
                tick()
                pi.append(i); pj.append(j); signs.append(1.)
                if i >= 3:
                    pi.append(i); pj.append(j); signs.append(-1.)
        pi = np.asarray(pi); pj = np.asarray(pj); signs = np.asarray(signs)
        m = len(pi)
        rows = np.arange(m)
        imask, jmask = pi >= 3, pj >= 3
        L = np.zeros((45, 34)); c = np.zeros(45)
        z = np.arange(12)
        L[z, z] = 1; L[z, 21+z] = -1
        L[12+z, z] = -1; L[12+z, 21+z] = -1; L[12+z, 33] = 1
        L[24+z, 21+z] = -1; L[24+z, 33] = -.5; c[24+z] = 1
        q = np.arange(9)
        L[27+q, 12+q] = -1
        L[36+q, 12+q] = 1; L[36+q, 24+q] = -1
        objective = np.zeros(34)
        objective[21:24] = -1; objective[24:33] = -2
        bounds = [(0., 2.)]*12+[(0., 1.)]*9+[(1e-7, .6)]*12+[(.12, 1.88)]
        release = np.zeros(m+45)
        cached_v = None
        cached_g = None
        cached_j = None
        champion = None
        champion_value = -float('inf')

        def calc(v):
            nonlocal cached_v, cached_g, cached_j
            tick()
            if cached_v is not None and np.array_equal(v, cached_v):
                return cached_g, cached_j
            y = np.zeros(12); y[3:] = v[12:21]
            dx = v[pi]-v[pj]
            dy = y[pi]-signs*y[pj]
            dist = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-25)
            g = dist-v[21+pi]-v[21+pj]
            J = np.zeros((m, 34))
            J[rows, pi] = dx/dist; J[rows, pj] = -dx/dist
            J[rows[imask], 9+pi[imask]] = dy[imask]/dist[imask]
            J[rows[jmask], 9+pj[jmask]] = -signs[jmask]*dy[jmask]/dist[jmask]
            J[rows, 21+pi] = -1; J[rows, 21+pj] = -1
            cached_v = v.copy()
            cached_g = np.r_[g, L@v+c]
            cached_j = np.vstack((J, L))
            return cached_g, cached_j

        def expand(v):
            a = np.zeros((21, 3))
            a[:3, 0] = v[:3]; a[:3, 2] = v[21:24]
            a[3::2, 0] = v[3:12]; a[4::2, 0] = v[3:12]
            a[3::2, 1] = v[12:21]; a[4::2, 1] = -v[12:21]
            a[3::2, 2] = v[24:33]; a[4::2, 2] = v[24:33]
            return a

        def retain(v):
            nonlocal champion, champion_value
            tick()
            if not np.isfinite(v).all():
                return
            value = consider(expand(v))
            if value > champion_value:
                champion_value = value
                champion = v.copy()

        def solve(v, iterations, seconds, tolerance=2e-12):
            tick()
            local_end = min(deadline, time.monotonic()+seconds)
            last = v.copy()
            count = 0
            def callback(w):
                nonlocal last, count
                tick()
                last = w.copy()
                count += 1
                if count % 9 == 0:
                    retain(w)
                if time.monotonic() >= local_end:
                    raise SliceEnd()
            def fun(w):
                tick()
                return float(objective@w)
            try:
                result = minimize(fun, v, method='SLSQP', jac=lambda w: objective,
                    bounds=bounds,
                    constraints={'type': 'ineq', 'fun': lambda w: calc(w)[0]-release,
                                 'jac': lambda w: calc(w)[1]},
                    callback=callback,
                    options={'maxiter': int(iterations), 'ftol': tolerance, 'disp': False})
                last = result.x
            except SliceEnd:
                pass
            retain(last)
            return last

        def lattice(counts):
            tick()
            data = [[2.*(j-1), 0., 1.] for j in range(3)]
            for row, count in enumerate(counts):
                tick()
                for j in range(count):
                    tick()
                    data.append([2.*(j-(count-1)/2), (row+1)*np.sqrt(3.), 1.])
            g = np.asarray(data)
            w = np.ptp(g[:, 0])+2
            h = 2*np.max(g[:, 1]+g[:, 2])
            g *= 2/(w+h)
            v = np.zeros(34)
            v[33] = w*2/(w+h)
            v[:12] = g[:, 0]+v[33]/2
            v[12:21] = g[3:, 1]
            v[21:33] = g[:, 2]*.995
            return v

        projections = []
        if init is not None:
            a = np.asarray(canonical(init)['circles'], float).copy()
            if a.shape == (21, 3):
                lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
                hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
                a[:, :2] -= (lo+hi)/2
                for swap in (False, True):
                    tick()
                    b = a.copy()
                    if swap:
                        b[:, [0, 1]] = b[:, [1, 0]]
                    order = np.argsort(b[:, 1])
                    lower, singles, upper = order[:9], order[9:12], order[12:]
                    ref = b[lower].copy(); ref[:, 1] *= -1
                    cost = np.sum((b[upper, None, :]-ref[None, :, :])**2, axis=2)
                    u, vj = linear_sum_assignment(cost)
                    g = np.vstack((np.column_stack((b[singles, 0], np.zeros(3), b[singles, 2])),
                                   (b[upper[u]]+ref[vj])/2))
                    v = np.zeros(34)
                    left = np.min(g[:, 0]-g[:, 2])
                    v[:12] = g[:, 0]-left
                    v[12:21] = np.maximum(g[3:, 1], 0)
                    v[21:33] = g[:, 2]
                    v[33] = np.max(g[:, 0]+g[:, 2])-left
                    residual = float(np.sum(cost[u, vj])+np.sum(b[singles, 1]**2))
                    projections.append((residual, v))

        if projections and best_score > 2.15:
            anchor = min(projections, key=lambda t: t[0])[1]
            anchor = solve(anchor, 100, .20*budget_s, 2e-14)
        else:
            anchor = solve(lattice((4, 5)), 130, .27*budget_s)
            if projections and time.monotonic() < started+.40*budget_s:
                tick()
                solve(min(projections, key=lambda t: t[0])[1], 90, .13*budget_s)
        if champion is not None:
            anchor = champion.copy()

        attempt = 0
        while time.monotonic() < deadline-.035:
            tick()
            release[:] = 0
            gaps, J = calc(anchor)
            near = np.flatnonzero(gaps[:m] < .002)
            if len(near) < 3:
                break
            if attempt % 3 != 2:
                pivot = int(rng.integers(0, 12))
                incident = near[(pi[near] == pivot) | (pj[near] == pivot)]
                count = min(3+attempt % 3, len(near))
                if len(incident) >= 2:
                    chosen = list(rng.choice(incident, min(count-1, len(incident)), replace=False))
                    remaining = np.setdiff1d(near, np.asarray(chosen))
                    if len(chosen) < count:
                        chosen.extend(rng.choice(remaining, count-len(chosen), replace=False).tolist())
                    selected = np.asarray(chosen, dtype=int)
                else:
                    selected = rng.choice(near, count, replace=False)
            else:
                selected = rng.choice(near, min(5, len(near)), replace=False)
            amplitude = float(params['release'])*(1., 1.6, .65, 2.2)[attempt % 4]
            release[selected] = amplitude*rng.uniform(.8, 1.2, len(selected))
            v = anchor.copy()
            # A linearized coordinated displacement starts the release solve
            # away from the incumbent without moving unrelated contacts far.
            if attempt % 2 == 0:
                active = np.flatnonzero(gaps < 2e-6)
                if len(active) > 0:
                    target = release[active]
                    direction = np.linalg.lstsq(J[active], target, rcond=1e-9)[0]
                    norm = float(np.max(np.abs(direction)))
                    if norm > .07:
                        direction *= .07/norm
                    v += .55*direction
                    v = np.clip(v, np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds]))
            v = solve(v, params['release_iterations'], .105*budget_s)
            release[:] = 0
            v = solve(v, params['restore_iterations'], .17*budget_s, 5e-14)
            if champion is not None:
                anchor = champion.copy()
            attempt += 1
    except (StopSearch, ValueError, FloatingPointError, np.linalg.LinAlgError):
        pass
    return best
