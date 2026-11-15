import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {
    'projection_fraction': (0.17, 0.08, 0.25),
    'lattice_fraction': (0.24, 0.12, 0.32),
    'release': (0.021, 0.004, 0.07),
    'contact_threshold': (2e-6, 1e-8, 1e-4, 'log'),
    'jitter': (0.028, 0.005, 0.10)
}

class StopSearch(Exception):
    pass

class SliceEnd(Exception):
    pass

def canonical(obj):
    if obj is None or 'circles' in obj:
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
        xs = [Fraction(float(v)) for v in a[:, 0]]
        ys = [Fraction(float(v)) for v in a[:, 1]]
        rs = [Fraction(float(v)) for v in a[:, 2]]
        w = max(x+r for x, r in zip(xs, rs))-min(x-r for x, r in zip(xs, rs))
        h = max(y+r for y, r in zip(ys, rs))-min(y-r for y, r in zip(ys, rs))
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
    start = time.monotonic()
    deadline = start + 0.69*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_value = exact_score(init)
    ii, jj = np.triu_indices(21, 1)

    def tick():
        if time.monotonic() >= deadline:
            raise StopSearch()

    fallback = np.column_stack(((np.arange(21) % 7-3)*0.2,
                                (np.arange(21)//7-1)*0.2,
                                np.full(21, 0.0999999999999)))
    fb = {'circles': fallback.tolist()}
    fv = exact_score(fb)
    if fv > best_value:
        best, best_value = fb, fv

    def consider(a):
        nonlocal best, best_value
        tick()
        if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        a = a.copy()
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        a[:, :2] -= (lo+hi)/2
        dist = np.sqrt((a[ii, 0]-a[jj, 0])**2+(a[ii, 1]-a[jj, 1])**2)
        factor = min(1., float(np.min(dist/(a[ii, 2]+a[jj, 2]))))
        if factor <= 0:
            return -float('inf')
        a[:, 2] *= factor*(1.-1.5e-14)
        ext = np.max(a[:, :2]+a[:, 2, None], axis=0)-np.min(a[:, :2]-a[:, 2, None], axis=0)
        a *= (2./float(np.sum(ext)))*(1.-1.5e-14)
        value = float(np.sum(a[:, 2]))
        if value > best_value:
            obj = {'circles': a.tolist()}
            verified = exact_score(obj)
            if verified > best_value:
                best, best_value = obj, verified
        return value

    try:
        tick()
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
        L = np.zeros((45, 34)); offset = np.zeros(45)
        k = np.arange(12)
        L[k, k] = 1.; L[k, 21+k] = -1.
        L[12+k, k] = -1.; L[12+k, 21+k] = -1.; L[12+k, 33] = 1.
        L[24+k, 21+k] = -1.; L[24+k, 33] = -.5; offset[24+k] = 1.
        k = np.arange(9)
        L[27+k, 12+k] = -1.
        L[36+k, 12+k] = 1.; L[36+k, 24+k] = -1.
        objective = np.zeros(34)
        objective[21:24] = -1.; objective[24:33] = -2.
        bounds = [(0., 2.)]*12+[(0., 1.)]*9+[(1e-7, .6)]*12+[(.12, 1.88)]
        lower = np.asarray([b[0] for b in bounds])
        upper = np.asarray([b[1] for b in bounds])
        release = np.zeros(m+45)
        cached_v = None
        cached_g = None
        cached_J = None
        champion = None
        champion_value = -float('inf')

        def calc(v):
            nonlocal cached_v, cached_g, cached_J
            tick()
            if cached_v is not None and np.array_equal(v, cached_v):
                return cached_g, cached_J
            y = np.zeros(12); y[3:] = v[12:21]
            dx = v[pi]-v[pj]
            dy = y[pi]-signs*y[pj]
            d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-25)
            g = d-v[21+pi]-v[21+pj]
            J = np.zeros((m, 34))
            J[rows, pi] = dx/d; J[rows, pj] = -dx/d
            J[rows[imask], 9+pi[imask]] = dy[imask]/d[imask]
            J[rows[jmask], 9+pj[jmask]] = -signs[jmask]*dy[jmask]/d[jmask]
            J[rows, 21+pi] = -1.; J[rows, 21+pj] = -1.
            cached_v = v.copy()
            cached_g = np.r_[g, L@v+offset]
            cached_J = np.vstack((J, L))
            return cached_g, cached_J

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
            value = consider(expand(v))
            if value > champion_value:
                champion_value = value
                champion = v.copy()
            return value

        def polish(v):
            tick()
            g, J = calc(v)
            if np.min(g) < -1e-5:
                return v
            active = np.flatnonzero(np.abs(g) < float(params['contact_threshold']))
            if len(active) < 34:
                return v
            original = v.copy()
            for _ in range(5):
                tick()
                g, J = calc(v)
                delta, _, rank, _ = np.linalg.lstsq(J[active], -g[active], rcond=1e-11)
                if rank < 34 or np.max(np.abs(delta)) > .003:
                    return original
                v = v+delta
                if np.any(v[21:33] <= 0) or not np.isfinite(v).all():
                    return original
                retain(v)
                if np.max(np.abs(g[active])) < 3e-16:
                    break
            if np.min(calc(v)[0]) >= -1e-10 and objective@v <= objective@original+1e-11:
                return v
            return original

        def solve(v, seconds, iterations=100, do_polish=True):
            tick()
            local_end = min(deadline-.004, time.monotonic()+seconds)
            last = np.clip(v, lower, upper)
            count = 0
            def fun(w):
                tick()
                if time.monotonic() >= local_end:
                    raise SliceEnd()
                return float(objective@w)
            def jac(w):
                tick()
                return objective
            def callback(w):
                nonlocal last, count
                tick()
                last = w.copy(); count += 1
                if count % 8 == 0:
                    retain(w)
                if time.monotonic() >= local_end:
                    raise SliceEnd()
            try:
                result = minimize(fun, last, jac=jac, method='SLSQP', bounds=bounds,
                    constraints={'type': 'ineq', 'fun': lambda w: calc(w)[0]-release,
                                 'jac': lambda w: calc(w)[1]}, callback=callback,
                    options={'maxiter': iterations, 'ftol': 4e-14, 'disp': False})
                last = result.x
            except SliceEnd:
                pass
            retain(last)
            if do_polish and not np.any(release):
                last = polish(last)
            return last

        def lattice(counts):
            tick()
            data = [[2.*(j-1), 0., 1.] for j in range(3)]
            for level, count in enumerate(counts, 1):
                tick()
                for j in range(count):
                    tick()
                    data.append([2.*j-(count-1), level*np.sqrt(3.), 1.])
            g = np.asarray(data)
            w = np.ptp(g[:, 0])+2.
            h = 2.*np.max(g[:, 1]+g[:, 2])
            scale = 2./(w+h)
            g *= scale
            v = np.zeros(34)
            v[33] = w*scale
            v[:12] = g[:, 0]+v[33]/2
            v[12:21] = g[3:, 1]
            v[21:33] = g[:, 2]*.999
            return v

        projections = []
        try:
            a = np.asarray(canonical(init)['circles'], float).copy()
            if a.shape == (21, 3) and np.isfinite(a).all() and np.all(a[:, 2] > 0):
                a[:, :2] -= (np.max(a[:, :2]+a[:, 2, None], axis=0)+np.min(a[:, :2]-a[:, 2, None], axis=0))/2
                for swap in (False, True):
                    tick()
                    b = a.copy()
                    if swap:
                        b[:, [0, 1]] = b[:, [1, 0]]
                    order = np.argsort(b[:, 1])
                    low, axis, high = order[:9], order[9:12], order[12:]
                    reflected = b[low].copy(); reflected[:, 1] *= -1
                    cost = np.sum((b[high, None, :]-reflected[None, :, :])**2, axis=2)
                    u, j = linear_sum_assignment(cost)
                    g = np.vstack((np.column_stack((b[axis, 0], np.zeros(3), b[axis, 2])),
                                   (b[high[u]]+reflected[j])/2))
                    v = np.zeros(34)
                    left = np.min(g[:, 0]-g[:, 2])
                    v[:12] = g[:, 0]-left
                    v[12:21] = np.maximum(g[3:, 1], 0.)
                    v[21:33] = g[:, 2]
                    v[33] = np.max(g[:, 0]+g[:, 2])-left
                    error = float(np.sum(cost[u, j])+np.sum(b[axis, 1]**2))
                    projections.append((error, v))
        except (TypeError, ValueError, KeyError):
            pass
        projections.sort(key=lambda item: item[0])
        if projections:
            solve(projections[0][1], float(params['projection_fraction'])*budget_s, 105)
        tick()
        solve(lattice((4, 5)), float(params['lattice_fraction'])*budget_s, 145)
        if champion_value < 2.363 and time.monotonic() < start+.49*budget_s:
            alternative = projections[1][1] if len(projections) > 1 and projections[1][0] < .001 else lattice((5, 4))
            solve(alternative, .105*budget_s, 85)
        if champion is not None:
            anchor = solve(champion.copy(), min(.12*budget_s, max(.005, deadline-time.monotonic()-.07*budget_s)), 100)
        else:
            anchor = lattice((4, 5))
        attempt = 0
        while time.monotonic() < deadline-.025:
            tick()
            anchor = champion.copy() if champion is not None else anchor
            release[:] = 0.
            gaps, J = calc(anchor)
            near = np.flatnonzero(gaps[:m] < .002)
            if len(near) < 3:
                break
            remaining = deadline-time.monotonic()
            if remaining < .10*budget_s or attempt % 3 == 2:
                v = anchor.copy()
                v[:21] += rng.normal(0., float(params['jitter']), 21)
                v = np.clip(v, lower, upper)
                solve(v, max(.005, remaining-.012), 100)
            else:
                pivot = int(rng.integers(0, 12))
                incident = near[(pi[near] == pivot) | (pj[near] == pivot)]
                count = min(3+attempt % 2, len(near))
                selected = rng.choice(near, count, replace=False)
                if len(incident) >= count:
                    selected = rng.choice(incident, count, replace=False)
                release[selected] = float(params['release'])*(1. if attempt % 2 == 0 else 1.6)
                v = anchor.copy()
                active = np.flatnonzero(gaps < 2e-6)
                if len(active) >= 20:
                    tick()
                    direction = np.linalg.lstsq(J[active], release[active], rcond=1e-10)[0]
                    direction *= min(.7, .06/max(float(np.max(np.abs(direction))), 1e-20))
                    v = np.clip(v+direction, lower, upper)
                v = solve(v, min(.07*budget_s, remaining*.30), 24, False)
                release[:] = 0.
                solve(v, min(.13*budget_s, max(.006, deadline-time.monotonic()-.012)), 90)
            attempt += 1
    except (StopSearch, ValueError, FloatingPointError, np.linalg.LinAlgError):
        pass
    return best
