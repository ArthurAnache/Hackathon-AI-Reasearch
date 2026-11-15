import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'search_fraction': (0.70, 0.50, 0.80),
    'iterations': (105, 30, 180),
    'tolerance': (2e-12, 1e-14, 1e-8, 'log'),
}

class SearchEnd(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    stop = start + float(params['search_fraction']) * budget_s
    ii, jj = np.triu_indices(21, 1)
    best = init
    best_score = -float('inf')

    def exact(data):
        try:
            a = np.asarray(data['circles'], dtype=float)
            if a.shape != (21, 3) or not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
                return -float('inf')
            dx = a[ii, 0] - a[jj, 0]
            dy = a[ii, 1] - a[jj, 1]
            if np.any(np.sqrt(dx**2 + dy**2) < a[ii, 2] + a[jj, 2]):
                return -float('inf')
            xs = list(map(Fraction, map(float, a[:, 0])))
            ys = list(map(Fraction, map(float, a[:, 1])))
            rs = list(map(Fraction, map(float, a[:, 2])))
            w = max(x+r for x, r in zip(xs, rs)) - min(x-r for x, r in zip(xs, rs))
            h = max(y+r for y, r in zip(ys, rs)) - min(y-r for y, r in zip(ys, rs))
            if w+h > 2:
                return -float('inf')
            return float(np.sum(a[:, 2]))
        except (ValueError, TypeError, KeyError, OverflowError):
            return -float('inf')

    if init is not None:
        best_score = exact(init)

    def consider(a):
        nonlocal best, best_score
        if time.monotonic() >= stop:
            return
        a = np.array(a, dtype=float, copy=True)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return
        dx = a[ii, 0] - a[jj, 0]
        dy = a[ii, 1] - a[jj, 1]
        distances = np.sqrt(dx**2 + dy**2)
        f = min(1.0, float(np.min(distances / (a[ii, 2] + a[jj, 2]))))
        if f <= 0:
            return
        a[:, 2] *= f * (1.0 - 8e-14)
        lower = np.min(a[:, :2] - a[:, 2, None], axis=0)
        upper = np.max(a[:, :2] + a[:, 2, None], axis=0)
        perimeter = float(np.sum(upper-lower))
        if perimeter <= 0:
            return
        a[:, :2] -= (upper+lower)*0.5
        a *= (2.0/perimeter)*(1.0-8e-14)
        if float(np.sum(a[:, 2])) <= best_score:
            return
        data = {'circles': a.tolist()}
        score = exact(data)
        if score > best_score:
            best_score, best = score, data

    fallback = np.array([[0.2*x, 0.2*y, 0.1] for y in range(3) for x in range(7)], dtype=float)
    consider(fallback)
    rng = np.random.default_rng(seed)
    solutions = []

    def make_problem(rows):
        # Each group is either an axis circle or a reflected pair.
        coords = []
        multiplicity = []
        for row, count in enumerate(rows):
            if time.monotonic() >= stop:
                raise SearchEnd
            for k in range((count+1)//2):
                if time.monotonic() >= stop:
                    raise SearchEnd
                x = float(2*k + (0 if count % 2 else 1))
                coords.append((x, 1.0 + row*np.sqrt(3.0)))
                multiplicity.append(1 if x == 0 else 2)
        coords = np.asarray(coords)
        mult = np.asarray(multiplicity)
        m = len(mult)
        paired = np.flatnonzero(mult == 2)
        p = len(paired)
        d = p + 2*m + 1
        X = np.zeros((m, d))
        Y = np.zeros((m, d))
        R = np.zeros((m, d))
        X[paired, np.arange(p)] = 1
        Y[np.arange(m), p+np.arange(m)] = 1
        R[np.arange(m), p+m+np.arange(m)] = 1
        pairs = []
        for i in range(m):
            if time.monotonic() >= stop:
                raise SearchEnd
            for j in range(i+1, m):
                if time.monotonic() >= stop:
                    raise SearchEnd
                pairs.append((i,j,1))
                if mult[i] == 2 and mult[j] == 2:
                    pairs.append((i,j,-1))
        for i in paired:
            if time.monotonic() >= stop:
                raise SearchEnd
            pairs.append((i,i,-1))
        u, v, signs = np.asarray(pairs, dtype=int).T
        DX = X[u] - signs[:, None]*X[v]
        DY = Y[u] - Y[v]
        RR = R[u] + R[v]
        walls = np.concatenate((-X-R, Y-R, -Y-R), axis=0)
        walls[:m, -1] = 0.5
        walls[2*m:, -1] = -1
        constants = np.concatenate((np.zeros(2*m), np.full(m, 2.0)))
        gradient = -(mult @ R)
        width = 2.0*max(rows)
        height = 2.0 + (len(rows)-1)*np.sqrt(3.0)
        scale = 2.0/(width+height)
        z = np.zeros(d)
        z[:p] = coords[paired, 0]*scale
        z[p:p+m] = coords[:, 1]*scale
        z[p+m:p+2*m] = scale
        z[-1] = width*scale
        bounds = [(1e-7, 1.0)]*p + [(0.0, 2.0)]*m + [(1e-6, 0.6)]*m + [(0.25, 1.75)]
        cache_z = None
        cache_c = None
        cache_j = None

        def calculate(z):
            nonlocal cache_z, cache_c, cache_j
            if time.monotonic() >= stop:
                raise SearchEnd
            if cache_z is not None and np.array_equal(cache_z, z):
                return
            dx = DX @ z
            dy = DY @ z
            length = np.sqrt(dx*dx+dy*dy)
            safe = np.maximum(length, 1e-20)
            cache_c = np.concatenate((length-RR@z, walls@z+constants))
            deriv = (dx/safe)[:, None]*DX + (dy/safe)[:, None]*DY - RR
            cache_j = np.concatenate((deriv, walls), axis=0)
            cache_z = z.copy()

        def con(z):
            calculate(z)
            return cache_c

        def jac(z):
            calculate(z)
            return cache_j

        def expand(z):
            x, y, r = X@z, Y@z, R@z
            a = np.column_stack((x,y,r))
            reflected = np.column_stack((-x[paired],y[paired],r[paired]))
            return np.concatenate((a, reflected), axis=0)

        return z, bounds, gradient, con, jac, expand

    families = [(5,4,3,4,5), (4,5,3,5,4), (5,4,5,4,3), (4,5,4,5,3), (4,4,5,4,4), (3,5,5,5,3)]
    try:
        for family in families:
            if time.monotonic() >= stop - 0.035:
                break
            z, bounds, gradient, con, jac, expand = make_problem(family)
            last = [z.copy()]
            counter = [0]
            def callback(q):
                if time.monotonic() >= stop:
                    raise SearchEnd
                last[0] = q.copy()
                counter[0] += 1
                if counter[0] % 9 == 0:
                    consider(expand(q))
            try:
                result = minimize(lambda q: float(gradient@q), z, jac=lambda q: gradient,
                                  bounds=bounds, constraints={'type':'ineq','fun':con,'jac':jac},
                                  method='SLSQP', callback=callback,
                                  options={'maxiter':int(params['iterations']), 'ftol':float(params['tolerance']), 'disp':False})
                q = result.x
                consider(expand(q))
                solutions.append((float(-gradient@q), family, q.copy()))
            except SearchEnd:
                break
        solutions.sort(key=lambda item: item[0], reverse=True)
        for value, family, q in solutions[:2]:
            if time.monotonic() >= stop - 0.04:
                break
            z, bounds, gradient, con, jac, expand = make_problem(family)
            def callback2(q):
                if time.monotonic() >= stop:
                    raise SearchEnd
            result = minimize(lambda q: float(gradient@q), q, jac=lambda q: gradient,
                              bounds=bounds, constraints={'type':'ineq','fun':con,'jac':jac},
                              method='SLSQP', callback=callback2,
                              options={'maxiter':65,'ftol':2e-14,'disp':False})
            consider(expand(result.x))
    except SearchEnd:
        pass
    return best
