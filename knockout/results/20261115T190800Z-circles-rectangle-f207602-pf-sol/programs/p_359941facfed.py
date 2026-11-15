import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'search_fraction': (0.70, 0.50, 0.75),
    'stage_seconds': (0.16, 0.06, 0.35),
    'iterations': (100, 30, 180),
    'perturbation': (0.012, 0.001, 0.08),
}

class StopSearch(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + float(params['search_fraction']) * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    best = init
    best_score = -float('inf')

    def check(end=None):
        if time.monotonic() >= (deadline if end is None else min(end, deadline)):
            raise StopSearch()

    def exact(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        delta = a[ii, :2] - a[jj, :2]
        if np.any(np.sqrt(delta[:, 0]**2 + delta[:, 1]**2) < a[ii, 2] + a[jj, 2]):
            return -float('inf')
        xs = [Fraction(float(v)) for v in a[:, 0]]
        ys = [Fraction(float(v)) for v in a[:, 1]]
        rs = [Fraction(float(v)) for v in a[:, 2]]
        width = max(x+r for x, r in zip(xs, rs)) - min(x-r for x, r in zip(xs, rs))
        height = max(y+r for y, r in zip(ys, rs)) - min(y-r for y, r in zip(ys, rs))
        if width + height > 2:
            return -float('inf')
        return float(np.sum(a[:, 2]))

    if init is not None:
        try:
            best_score = exact(np.asarray(init['circles'], dtype=float))
        except (ValueError, TypeError, KeyError, OverflowError):
            pass

    def retain(a):
        nonlocal best, best_score
        check()
        a = np.array(a, dtype=float, copy=True)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return
        delta = a[ii, :2] - a[jj, :2]
        dist = np.sqrt(delta[:, 0]**2 + delta[:, 1]**2)
        factor = min(1.0, float(np.min(dist / (a[ii, 2] + a[jj, 2]))))
        if factor <= 0:
            return
        a[:, 2] *= factor * (1.0 - 5e-15)
        lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
        hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
        a[:, :2] -= (hi + lo) * 0.5
        a *= (2.0 / float(np.sum(hi-lo))) * (1.0 - 8e-15)
        if float(np.sum(a[:, 2])) <= best_score:
            return
        candidate = {'circles': a.tolist()}
        score = exact(a)
        if score > best_score:
            best_score, best = score, candidate

    def make_model(specs, width):
        check()
        nv = sum(3 if p else 2 for p, x, y, r in specs) + 1
        X = np.zeros((21, nv)); Y = X.copy(); R = X.copy()
        q = np.zeros(nv)
        mirror = []
        reps = []
        bounds = []
        c = k = 0
        for paired, x, y, r in specs:
            check()
            if paired:
                X[c, k] = 1; X[c+1, k] = -1
                Y[c:c+2, k+1] = 1; R[c:c+2, k+2] = 1
                q[k:k+3] = [abs(x), y, r]
                reps.append((k, k+1, k+2))
                bounds.extend([(0, 2), (-1, 3), (1e-7, 0.8)])
                mirror.extend([c+1, c])
                k += 3; c += 2
            else:
                Y[c, k] = 1; R[c, k+1] = 1
                q[k:k+2] = [y, r]
                reps.append((-1, k, k+1))
                bounds.extend([(-1, 3), (1e-7, 0.8)])
                mirror.append(c)
                k += 2; c += 1
        if c != 21:
            raise ValueError('wrong circle count')
        q[-1] = width
        bounds.append((0.15, 1.85))
        edges = []
        for a, b in zip(ii, jj):
            check()
            if (int(a), int(b)) <= tuple(sorted((mirror[a], mirror[b]))):
                edges.append((a, b))
        a, b = np.asarray(edges).T
        DX = X[a]-X[b]; DY = Y[a]-Y[b]; SR = R[a]+R[b]
        walls = []; offsets = []
        for xi, yi, ri in reps:
            check()
            v = np.zeros(nv); v[-1] = 0.5; v[ri] = -1
            if xi >= 0:
                v[xi] = -1
            walls.append(v); offsets.append(0.)
            v = np.zeros(nv); v[yi] = 1; v[ri] = -1
            walls.append(v); offsets.append(0.)
            v = np.zeros(nv); v[-1] = -1; v[yi] = -1; v[ri] = -1
            walls.append(v); offsets.append(2.)
        W = np.asarray(walls); off = np.asarray(offsets)
        obj = -np.sum(R, axis=0)
        return dict(q=q, X=X, Y=Y, R=R, dx=DX, dy=DY, sr=SR,
                    W=W, off=off, obj=obj, bounds=bounds, score=-1.)

    def decode(f, q):
        return np.column_stack((f['X']@q, f['Y']@q, f['R']@q))

    def run(f, seconds, polish=False):
        end = min(deadline, time.monotonic() + seconds)
        cached = [None, None, None]
        def calc(q):
            check(end)
            if cached[0] is None or not np.array_equal(q, cached[0]):
                dx = f['dx']@q; dy = f['dy']@q
                d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-16)
                cached[0] = q.copy()
                cached[1] = np.concatenate((d-f['sr']@q, f['W']@q+f['off']))
                cached[2] = np.vstack(((dx/d)[:, None]*f['dx'] + (dy/d)[:, None]*f['dy'] - f['sr'], f['W']))
            return cached[1], cached[2]
        def callback(q):
            check(end)
            f['q'] = q.copy()
            v, _ = calc(q)
            if np.min(v) > -1e-7:
                f['score'] = max(f['score'], float(-f['obj']@q))
            retain(decode(f, q))
        try:
            if polish:
                q = f['q'].copy()
                v, J = calc(q)
                active = np.where(v < 2e-7)[0]
                if len(active) >= len(q):
                    for _ in range(5):
                        check(end)
                        v, J = calc(q)
                        step = np.linalg.lstsq(J[active], -v[active], rcond=1e-12)[0]
                        if np.max(np.abs(step)) > 0.005:
                            break
                        q += step
                        retain(decode(f, q))
                        if np.max(np.abs(step)) < 2e-15:
                            break
                    if np.min(calc(q)[0]) > -1e-8:
                        f['q'] = q
            result = minimize(lambda q: float(f['obj']@q), f['q'], jac=lambda q: f['obj'],
                method='SLSQP', bounds=f['bounds'],
                constraints={'type': 'ineq', 'fun': lambda q: calc(q)[0], 'jac': lambda q: calc(q)[1]},
                callback=callback, options={'maxiter': int(params['iterations']), 'ftol': 1e-14, 'disp': False})
            f['q'] = result.x.copy()
            retain(decode(f, result.x))
        except StopSearch:
            pass

    families = []
    try:
        if init is not None:
            a0 = np.asarray(init['circles'], dtype=float)
            variants = []
            for swap in (False, True):
                check()
                a = a0.copy()
                if swap:
                    a[:, [0, 1]] = a[:, [1, 0]]
                lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
                hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
                a[:, 0] -= (lo[0]+hi[0])/2
                a[:, 1] -= lo[1]
                reflected = a.copy(); reflected[:, 0] *= -1
                distances = np.sum((a[:, None, :]-reflected[None, :, :])**2, axis=2)
                mapping = np.argmin(distances, axis=1)
                error = float(np.max(np.min(distances, axis=1)))
                variants.append((error, a, mapping, float(hi[0]-lo[0])))
            error, a, mapping, width = min(variants, key=lambda x: x[0])
            if error < 1e-10:
                specs = []; used = set()
                for i in range(21):
                    check()
                    if i in used:
                        continue
                    j = int(mapping[i]); used.add(i); used.add(j)
                    if j == i:
                        specs.append((False, 0., a[i, 1], a[i, 2]))
                    else:
                        specs.append((True, (abs(a[i, 0])+abs(a[j, 0]))/2,
                                      (a[i, 1]+a[j, 1])/2, (a[i, 2]+a[j, 2])/2))
                f = make_model(specs, width)
                families.append(f)
                run(f, 0.26, True)

        layouts = [(4, 5, 4, 5, 3), (3, 4, 5, 4, 5), (5, 4, 3, 4, 5),
                   (4, 5, 3, 5, 4), (3, 4, 3, 4, 3, 4), (5, 6, 5, 5)]
        for rows in layouts:
            check()
            r = 1.0/(max(rows)+1+(len(rows)-1)*np.sqrt(3)/2)
            specs = []
            for row, count in enumerate(rows):
                check()
                y = r + row*np.sqrt(3)*r
                if count % 2:
                    specs.append((False, 0., y, r*.98))
                for k in range(count//2):
                    check()
                    specs.append((True, (2*k+1+count%2)*r, y, r*.98))
            f = make_model(specs, 2*max(rows)*r)
            families.append(f)
            run(f, float(params['stage_seconds']))
        turn = 0
        while time.monotonic() < deadline and families:
            check()
            families.sort(key=lambda f: f['score'], reverse=True)
            f = families[turn % min(3, len(families))]
            old = f['q'].copy()
            if turn:
                f['q'] += rng.normal(0., float(params['perturbation']), len(old))
                for k, (low, high) in enumerate(f['bounds']):
                    check()
                    f['q'][k] = np.clip(f['q'][k], low+1e-8, high-1e-8)
            run(f, 0.19, turn == 0)
            turn += 1
    except (StopSearch, ValueError, np.linalg.LinAlgError):
        pass
    if best is None:
        a = np.array([[i*.2, j*.2, .099999999999] for j in range(7) for i in range(3)])
        candidate = {'circles': a.tolist()}
        if exact(a) > best_score:
            best = candidate
    return best
