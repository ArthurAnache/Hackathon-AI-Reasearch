import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'search_fraction': (0.70, 0.55, 0.74),
    'maxiter': (100, 40, 180),
    'jitter': (0.025, 0.0, 0.12)
}

class StopSearch(Exception):
    pass


def exact_score(data):
    try:
        a = np.asarray(data['circles'], dtype=np.float64)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        x = list(map(lambda v: Fraction(float(v)), a[:, 0]))
        y = list(map(lambda v: Fraction(float(v)), a[:, 1]))
        r = list(map(lambda v: Fraction(float(v)), a[:, 2]))
        w = max(map(sum, zip(x, r))) - min(map(lambda p: p[0]-p[1], zip(x, r)))
        h = max(map(sum, zip(y, r))) - min(map(lambda p: p[0]-p[1], zip(y, r)))
        if w+h > 2:
            return -float('inf')
        i, j = np.triu_indices(21, 1)
        d = np.sqrt((a[i, 0]-a[j, 0])**2 + (a[i, 1]-a[j, 1])**2)
        if np.any(d < a[i, 2]+a[j, 2]):
            return -float('inf')
        return float(np.sum(a[:, 2]))
    except Exception:
        return -float('inf')


def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + float(params['search_fraction'])*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = exact_score(init) if init is not None else -float('inf')
    pi, pj = np.triu_indices(21, 1)

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def retain(a):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        a = np.asarray(a, dtype=float).copy()
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return
        d = np.sqrt((a[pi, 0]-a[pj, 0])**2 + (a[pi, 1]-a[pj, 1])**2)
        ratio = min(1.0, float(np.min(d/(a[pi, 2]+a[pj, 2]))))
        if ratio <= 0:
            return
        a[:, 2] *= ratio*(1.0-3e-13)
        a[:, 0] -= (np.max(a[:, 0])+np.min(a[:, 0]))/2
        a[:, 1] -= (np.max(a[:, 1])+np.min(a[:, 1]))/2
        w = np.max(a[:, 0]+a[:, 2])-np.min(a[:, 0]-a[:, 2])
        h = np.max(a[:, 1]+a[:, 2])-np.min(a[:, 1]-a[:, 2])
        a *= (2.0/(w+h))*(1.0-3e-14)
        if float(np.sum(a[:, 2])) <= best_score:
            return
        obj = {'circles': a.tolist()}
        score = exact_score(obj)
        if score > best_score:
            best, best_score = obj, score

    # A guaranteed finite fallback, independent of the supplied cold start.
    fallback = np.empty((21, 3))
    fallback[:, 0] = np.arange(21) % 5
    fallback[:, 1] = np.arange(21) // 5
    fallback[:, 2] = 0.5
    retain(fallback)

    def make_problem(rows, jitter=0.0):
        check()
        gx, gy, mult = [], [], []
        for row, count in enumerate(rows):
            check()
            if count % 2:
                gx.append(0.0)
                gy.append(row*np.sqrt(3.0)/2)
                mult.append(1)
            for k in range(count//2):
                check()
                gx.append(k + (1.0 if count % 2 else 0.5))
                gy.append(row*np.sqrt(3.0)/2)
                mult.append(2)
        gx, gy, mult = np.array(gx), np.array(gy), np.array(mult)
        g = len(gx)
        pairs = np.flatnonzero(mult == 2)
        p = len(pairs)
        nv = p+2*g+2
        X = np.zeros((21, nv))
        Y = np.zeros((21, nv))
        R = np.zeros((21, nv))
        G = np.zeros((g, nv))
        G[pairs, np.arange(p)] = 1
        off = 0
        for k in range(g):
            check()
            m = int(mult[k])
            X[off] = G[k]
            Y[off, p+k] = 1
            R[off, p+g+k] = 1
            if m == 2:
                X[off+1] = -G[k]
                Y[off+1, p+k] = 1
                R[off+1, p+g+k] = 1
            off += m
        DX, DY, RS = X[pi]-X[pj], Y[pi]-Y[pj], R[pi]+R[pj]
        # Canonicalize signs to remove constraints duplicated by reflection.
        ix = np.argmax(DX != 0, axis=1)
        iy = np.argmax(DY != 0, axis=1)
        sx = np.sign(DX[np.arange(len(DX)), ix])
        sy = np.sign(DY[np.arange(len(DY)), iy])
        sx[sx == 0] = 1
        sy[sy == 0] = 1
        DX *= sx[:, None]
        DY *= sy[:, None]
        _, unique = np.unique(np.concatenate((DX, DY, RS), axis=1), axis=0, return_index=True)
        DX, DY, RS = DX[unique], DY[unique], RS[unique]
        B = np.zeros((3*g, nv))
        B[:g] = -G
        B[np.arange(g), p+g+np.arange(g)] = -1
        B[:g, -2] = 1
        B[g+np.arange(g), p+np.arange(g)] = 1
        B[g+np.arange(g), p+g+np.arange(g)] = -1
        B[2*g+np.arange(g), p+np.arange(g)] = -1
        B[2*g+np.arange(g), p+g+np.arange(g)] = -1
        B[2*g:, -1] = 1
        E = np.zeros(nv)
        E[p+g:p+2*g] = mult
        objective = np.zeros(nv)
        objective[-2:] = [2, 1]
        z = np.zeros(nv)
        z[:p] = gx[pairs]/10.5
        z[p:p+g] = (gy+0.5)/10.5
        z[p+g:p+2*g] = 0.5/10.5
        if jitter:
            z[:p+g] += rng.normal(0, jitter/10.5, p+g)
            z[:p+g] = np.maximum(z[:p+g], 1e-5)
        z[-2] = (max(gx)+0.5)/10.5
        z[-1] = (max(gy)+1.0)/10.5
        bounds = [(0.0, None)]*(p+g)+[(1e-7, None)]*g+[(1e-5, None)]*2
        cache_z = None
        cache_c = None
        cache_j = None

        def calc(v):
            nonlocal cache_z, cache_c, cache_j
            check()
            if cache_z is not None and np.array_equal(v, cache_z):
                return
            dx, dy = DX@v, DY@v
            dist = np.sqrt(dx*dx+dy*dy)
            inv = 1.0/np.maximum(dist, 1e-18)
            cache_c = np.concatenate((dist-RS@v, B@v))
            cache_j = np.vstack((DX*(dx*inv)[:, None]+DY*(dy*inv)[:, None]-RS, B))
            cache_z = v.copy()

        def fun(v):
            calc(v)
            return cache_c

        def jac(v):
            calc(v)
            return cache_j

        def unpack(v):
            return np.column_stack((X@v, Y@v, R@v))

        last = [z.copy()]
        def callback(v):
            check()
            last[0] = v.copy()

        constraints = [
            {'type': 'eq', 'fun': lambda v: float(E@v-1), 'jac': lambda v: E},
            {'type': 'ineq', 'fun': fun, 'jac': jac}
        ]
        return z, objective, bounds, constraints, unpack, callback, last

    families = [
        [3, 4, 3, 4, 3, 4],
        [4, 5, 3, 5, 4],
        [5, 4, 3, 4, 5],
        [3, 4, 4, 3, 4, 3],
        [4, 3, 4, 3, 3, 4],
        [5, 3, 5, 3, 5],
        [3, 4, 3, 4, 3, 4]
    ]
    saved = []
    try:
        for index, rows in enumerate(families):
            check()
            if deadline-time.monotonic() < 0.045:
                break
            z, obj, bounds, cons, unpack, callback, last = make_problem(rows, float(params['jitter']) if index == 6 else 0.0)
            retain(unpack(z))
            try:
                result = minimize(lambda v: float(obj@v), z, jac=lambda v: obj,
                    method='SLSQP', bounds=bounds, constraints=cons, callback=callback,
                    options={'ftol': 2e-11, 'maxiter': int(params['maxiter']), 'disp': False})
                retain(unpack(result.x))
                saved.append((float(obj@result.x), result.x.copy(), obj, bounds, cons, unpack, callback))
            except StopSearch:
                break
        # Spend any remaining time polishing the best completed family.
        if saved and time.monotonic() < deadline-0.06:
            saved.sort(key=lambda t: t[0])
            _, z, obj, bounds, cons, unpack, callback = saved[0]
            result = minimize(lambda v: float(obj@v), z, jac=lambda v: obj,
                method='SLSQP', bounds=bounds, constraints=cons, callback=callback,
                options={'ftol': 1e-13, 'maxiter': 60, 'disp': False})
            retain(unpack(result.x))
    except StopSearch:
        pass
    return best
