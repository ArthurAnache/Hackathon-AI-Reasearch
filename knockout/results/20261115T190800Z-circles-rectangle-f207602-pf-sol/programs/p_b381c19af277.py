import time
import math
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"first_share": (0.34, 0.2, 0.5), "perturb": (0.002, 0.0001, 0.02)}


def exact_score(data):
    try:
        a = np.asarray(data['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return -float('inf')
        i, j = np.triu_indices(21, 1)
        d = np.sqrt((a[i, 0]-a[j, 0])**2 + (a[i, 1]-a[j, 1])**2)
        if np.any(d < a[i, 2]+a[j, 2]):
            return -float('inf')
        xx, yy, rr = [], [], []
        for x, y, r in a:
            time.monotonic()
            xx.append(Fraction(float(x)))
            yy.append(Fraction(float(y)))
            rr.append(Fraction(float(r)))
        width = max(x+r for x, r in zip(xx, rr))-min(x-r for x, r in zip(xx, rr))
        height = max(y+r for y, r in zip(yy, rr))-min(y-r for y, r in zip(yy, rr))
        if width+height > 2:
            return -float('inf')
        return float(np.sum(a[:, 2]))
    except Exception:
        return -float('inf')


class StopSearch(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    best_data = init
    best_score = exact_score(init) if init is not None else -float('inf')
    ii, jj = np.triu_indices(21, 1)
    states = []

    def retain(a):
        nonlocal best_data, best_score
        if not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return -float('inf')
        a = a.copy()
        dist = np.sqrt(np.sum((a[ii, :2]-a[jj, :2])**2, axis=1))
        shrink = min(1.0, float(np.min(dist/(a[ii, 2]+a[jj, 2]))))
        if shrink <= 0:
            return -float('inf')
        a[:, 2] *= shrink*(1.0-2e-13)
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        extent = float(np.sum(hi-lo))
        if extent <= 0:
            return -float('inf')
        a[:, :2] -= (lo+hi)*0.5
        a *= (2.0/extent)*(1.0-2e-13)
        value = float(np.sum(a[:, 2]))
        if value > best_score:
            data = {'circles': a.tolist()}
            score = exact_score(data)
            if score > best_score:
                best_score, best_data = score, data
        return value

    # A guaranteed fallback, independent of the quality of the supplied start.
    fallback = np.zeros((21, 3))
    fallback[:, 0] = np.tile(np.arange(7), 3)
    fallback[:, 1] = np.repeat(np.arange(3), 7)
    fallback[:, 2] = 0.5
    retain(fallback)

    def make_model(rows):
        orbits = []
        pairs = []
        singles = []
        for row, count in enumerate(rows):
            if time.monotonic() >= deadline:
                raise StopSearch
            for k in range(count//2):
                time.monotonic()
                x = k+0.5 if count % 2 == 0 else k+1.0
                pairs.append((x, row*math.sqrt(0.75)))
            if count % 2:
                singles.append((0.0, row*math.sqrt(0.75)))
        orbits = pairs+singles
        p, m = len(pairs), len(orbits)
        nv = p+2*m+1
        X, Y, R = [], [], []
        mirror = []
        for k in range(m):
            if time.monotonic() >= deadline:
                raise StopSearch
            first = len(X)
            signs = (1.0, -1.0) if k < p else (0.0,)
            for sign in signs:
                time.monotonic()
                x = np.zeros(nv); y = np.zeros(nv); r = np.zeros(nv)
                if k < p:
                    x[k] = sign
                y[p+k] = 1.0
                r[p+m+k] = 1.0
                X.append(x); Y.append(y); R.append(r)
            mirror.extend([first+1, first] if k < p else [first])
        X, Y, R = np.array(X), np.array(Y), np.array(R)
        edge_i, edge_j = [], []
        for i in range(21):
            if time.monotonic() >= deadline:
                raise StopSearch
            for j in range(i+1, 21):
                time.monotonic()
                mirrored = tuple(sorted((mirror[i], mirror[j])))
                if (i, j) <= mirrored:
                    edge_i.append(i); edge_j.append(j)
        ei, ej = np.array(edge_i), np.array(edge_j)
        BX, BY, BR = X[ei]-X[ej], Y[ei]-Y[ej], R[ei]+R[ej]
        boundary = []
        offset = []
        for k in range(m):
            if time.monotonic() >= deadline:
                raise StopSearch
            line = np.zeros(nv)
            line[-1] = 0.5
            line[p+m+k] = -1
            if k < p:
                line[k] = -1
            boundary.append(line); offset.append(0.0)
            for sign in (-1, 1):
                time.monotonic()
                line = np.zeros(nv)
                line[-1] = -0.5
                line[p+k] = sign
                line[p+m+k] = -1
                boundary.append(line); offset.append(1.0)
        B, off = np.array(boundary), np.array(offset)
        objective = -np.sum(R, axis=0)
        z = np.zeros(nv)
        for k, (x, y) in enumerate(orbits):
            if time.monotonic() >= deadline:
                raise StopSearch
            if k < p:
                z[k] = x
            z[p+k] = y-(len(rows)-1)*math.sqrt(0.75)*0.5
            z[p+m+k] = 0.5
        width = max(rows)
        height = 1+(len(rows)-1)*math.sqrt(0.75)
        z *= 2/(width+height)
        z[-1] = 2*width/(width+height)
        z[p:p+m] += rng.normal(0, params['perturb'], m)
        bounds = [(0.0, 1.0)]*p + [(-1.0, 1.0)]*m + [(1e-5, 0.5)]*m + [(0.1, 1.9)]
        return (X, Y, R, BX, BY, BR, B, off, objective, bounds), z

    def run(model, z, until, maxiter=180):
        X, Y, R, BX, BY, BR, B, off, objective, bounds = model
        cache_z = None
        cache_c = None
        cache_j = None
        local_best = [-float('inf'), z.copy()]
        last = [z.copy()]

        def calculate(v):
            nonlocal cache_z, cache_c, cache_j
            if time.monotonic() >= until:
                raise StopSearch
            if cache_z is not None and np.array_equal(v, cache_z):
                return
            dx, dy = BX@v, BY@v
            d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-15)
            cache_c = np.concatenate((d-BR@v, B@v+off))
            cache_j = np.vstack(((dx/d)[:, None]*BX+(dy/d)[:, None]*BY-BR, B))
            cache_z = v.copy()

        def con(v):
            calculate(v)
            return cache_c

        def jac(v):
            calculate(v)
            return cache_j

        def callback(v):
            if time.monotonic() >= until:
                raise StopSearch
            last[0] = v.copy()
            a = np.column_stack((X@v, Y@v, R@v))
            value = retain(a)
            if value > local_best[0]:
                local_best[:] = [value, v.copy()]

        try:
            callback(z)
            result = minimize(lambda v: float(objective@v), z, jac=lambda v: objective,
                              method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': con, 'jac': jac},
                              callback=callback,
                              options={'maxiter': maxiter, 'ftol': 2e-12, 'disp': False})
            callback(result.x)
        except StopSearch:
            pass
        return local_best[0], local_best[1], last[0]

    families = ([4, 5, 3, 5, 4], [5, 4, 5, 4, 3], [3, 4, 3, 4, 3, 4])
    try:
        for index, rows in enumerate(families):
            now = time.monotonic()
            if now >= deadline-0.08:
                break
            model, z = make_model(rows)
            share = params['first_share'] if index == 0 else 0.23
            until = min(deadline-0.025, now+share*0.70*budget_s)
            val, zb, last = run(model, z, until)
            states.append((val, model, zb, last))
        if states and time.monotonic() < deadline-0.02:
            states.sort(key=lambda v: v[0], reverse=True)
            _, model, z, last = states[0]
            run(model, z, deadline-0.005, 250)
    except StopSearch:
        pass
    return best_data
