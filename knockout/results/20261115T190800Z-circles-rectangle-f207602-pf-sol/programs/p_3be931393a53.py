import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'explore_iterations': (48, 20, 100),
    'polish_iterations': (100, 40, 180),
    'release': (0.018, 0.003, 0.06),
    'contacts': (4, 3, 6)
}

class StopSearch(Exception):
    pass

def canonical(obj):
    if obj is None:
        return None
    if 'circles' in obj:
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
        xs = [Fraction(float(x)) for x in a[:, 0]]
        ys = [Fraction(float(y)) for y in a[:, 1]]
        rs = [Fraction(float(r)) for r in a[:, 2]]
        width = max(x+r for x, r in zip(xs, rs))-min(x-r for x, r in zip(xs, rs))
        height = max(y+r for y, r in zip(ys, rs))-min(y-r for y, r in zip(ys, rs))
        if width+height > 2:
            return -float('inf')
        for i in range(21):
            for j in range(i):
                dx = a[i, 0]-a[j, 0]
                dy = a[i, 1]-a[j, 1]
                if np.sqrt(dx*dx+dy*dy) < a[i, 2]+a[j, 2]:
                    return -float('inf')
        return float(np.sum(a[:, 2]))
    except (ValueError, TypeError, KeyError, OverflowError):
        return -float('inf')

def construct(params, seed, budget_s, init=None):
    deadline = time.monotonic()+0.70*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_value = exact_score(init)
    fallback = {'circles': [[(i % 7)*0.2, (i//7)*0.2, 0.09999999999999] for i in range(21)]}
    fallback_value = exact_score(fallback)
    if fallback_value > best_value:
        best, best_value = fallback, fallback_value

    def tick():
        if time.monotonic() >= deadline:
            raise StopSearch()

    try:
        tick()
        groups = np.array([0, 1, 2]+[g for g in range(3, 12) for _ in range(2)])
        signs = np.array([0., 0., 0.]+[s for _ in range(9) for s in (1., -1.)])
        X = np.zeros((21, 34))
        Y = np.zeros_like(X)
        R = np.zeros_like(X)
        for i in range(21):
            tick()
            g = groups[i]
            X[i, g] = 1.
            R[i, 21+g] = 1.
            if g >= 3:
                Y[i, 9+g] = signs[i]
        pair_list = []
        seen = set()
        for i in range(21):
            tick()
            for j in range(i):
                tick()
                key = (min(groups[i], groups[j]), max(groups[i], groups[j]), signs[i]*signs[j])
                if key not in seen:
                    seen.add(key)
                    pair_list.append((i, j))
        pi, pj = np.array(pair_list).T
        DX, DY, RS = X[pi]-X[pj], Y[pi]-Y[pj], R[pi]+R[pj]
        walls = np.zeros((36, 34))
        offsets = np.zeros(36)
        for g in range(12):
            tick()
            walls[g, g] = 1.
            walls[g, 21+g] = -1.
            walls[g, 33] = .5
            walls[12+g, g] = -1.
            walls[12+g, 21+g] = -1.
            walls[12+g, 33] = .5
            walls[24+g, 21+g] = -1.
            walls[24+g, 33] = -.5
            offsets[24+g] = 1.
            if g >= 3:
                walls[24+g, 9+g] = -1.
        weights = np.array([1.]*3+[2.]*9)
        gradient = np.zeros(34)
        gradient[21:33] = -weights
        bounds = [(-1., 1.)]*12+[(0., 1.)]*9+[(1e-7, .5)]*12+[(.05, 1.95)]
        release = np.zeros(len(pi))
        ii, jj = np.triu_indices(21, 1)
        best_z = None

        def retain(z):
            nonlocal best, best_value, best_z
            tick()
            if not np.isfinite(z).all() or np.any(z[21:33] <= 0):
                return
            a = np.column_stack((X@z, Y@z, R@z))
            dd = a[ii, :2]-a[jj, :2]
            ratios = np.sqrt(np.sum(dd*dd, axis=1))/(a[ii, 2]+a[jj, 2])
            shrink = min(1., float(np.min(ratios)))*(1.-2e-14)
            if shrink <= 0:
                return
            a[:, 2] *= shrink
            lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
            hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
            scale = 2./float(np.sum(hi-lo))*(1.-2e-14)
            value = float(np.sum(a[:, 2]))*scale
            if value <= best_value:
                return
            a[:, :2] -= (lo+hi)/2
            a *= scale
            obj = {'circles': a.tolist()}
            value = exact_score(obj)
            if value > best_value:
                best, best_value = obj, value
                best_z = z.copy()
                best_z[:12] *= scale
                best_z[12:21] *= scale
                best_z[21:33] *= shrink*scale
                best_z[33] = hi[0]*scale-lo[0]*scale

        def objective(z):
            tick()
            return float(gradient@z)

        def jacobian(z):
            tick()
            return gradient

        def constraints(z):
            tick()
            dx, dy = DX@z, DY@z
            return np.concatenate((np.sqrt(dx*dx+dy*dy)-RS@z-release, walls@z+offsets))

        def constraint_jac(z):
            tick()
            dx, dy = DX@z, DY@z
            d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-25)
            return np.vstack(((dx/d)[:, None]*DX+(dy/d)[:, None]*DY-RS, walls))

        def solve(z, iterations):
            count = 0
            def callback(v):
                nonlocal count
                tick()
                count += 1
                if count % 6 == 0:
                    retain(v)
            tick()
            result = minimize(objective, z, jac=jacobian, method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': constraints, 'jac': constraint_jac},
                              callback=callback,
                              options={'maxiter': int(iterations), 'ftol': 8e-13, 'disp': False})
            retain(result.x)
            return result.x

        def make_seed(kind):
            tick()
            singles, pairs = [], []
            if kind == 0:
                for j in range(6):
                    tick()
                    n = 6-j
                    for k in range(n):
                        tick()
                        x, y = j*np.sqrt(3.), 2*k-(n-1)
                        if y == 0:
                            singles.append([x, 0., 1.])
                        elif y > 0:
                            pairs.append([x, y, 1.])
            else:
                singles = [[2.*k, 0., 1.] for k in range(3)]
                for row in range(1, 4):
                    tick()
                    shift = float(row % 2) if kind == 1 else -float(row % 2)
                    for col in range(3):
                        tick()
                        pairs.append([2.*col+shift, row*np.sqrt(3.), 1.])
            g = np.array(singles+pairs)
            xmin = np.min(g[:, 0]-g[:, 2])
            xmax = np.max(g[:, 0]+g[:, 2])
            width = xmax-xmin
            height = 2*np.max(g[:, 1]+g[:, 2])
            g[:, 0] -= (xmin+xmax)/2
            scale = 2./(width+height)
            g *= scale
            z = np.zeros(34)
            z[:12] = g[:, 0]
            z[12:21] = g[3:, 1]
            z[21:33] = g[:, 2]
            z[33] = width*scale
            return z

        # Structured starts are optimized before any contact surgery.
        starts = [make_seed(0), make_seed(1)]
        if init is not None and best_value > 2.2:
            a = np.asarray(canonical(init)['circles'], float).copy()
            lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
            hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
            a[:, :2] -= (lo+hi)/2
            for swap in (False, True):
                tick()
                b = a[:, [1, 0, 2]] if swap else a
                order = np.argsort(np.abs(b[:, 1]))
                axis = order[:3]
                upper = np.flatnonzero(b[:, 1] > 1e-8)
                if len(upper) == 9 and np.max(np.abs(b[axis, 1])) < 1e-7:
                    g = np.vstack((b[axis], b[upper]))
                    z = np.zeros(34)
                    z[:12] = g[:, 0]
                    z[12:21] = g[3:, 1]
                    z[21:33] = g[:, 2]
                    z[33] = np.max(b[:, 0]+b[:, 2])-np.min(b[:, 0]-b[:, 2])
                    starts.insert(0, z)
                    break
        for z in starts:
            tick()
            retain(z)
            solve(z, params['explore_iterations'])
        if best_z is not None:
            solve(best_z.copy(), params['polish_iterations'])
        attempt = 0
        while time.monotonic() < deadline and best_z is not None:
            tick()
            anchor = best_z.copy()
            release[:] = 0.
            dx, dy = DX@anchor, DY@anchor
            gaps = np.sqrt(dx*dx+dy*dy)-RS@anchor
            near = np.flatnonzero(gaps < .004)
            if len(near) < 3:
                break
            k = min(int(params['contacts']), len(near))
            chosen = rng.choice(near, k, replace=False)
            release[chosen] = float(params['release'])*rng.uniform(.6, 1.4, k)
            displaced = solve(anchor, 22)
            release[:] = 0.
            solve(displaced, params['polish_iterations'])
            attempt += 1
    except (StopSearch, ValueError, FloatingPointError):
        pass
    return best
