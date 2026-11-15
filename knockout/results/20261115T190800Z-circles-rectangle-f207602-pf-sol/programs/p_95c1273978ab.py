import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    "iterations": (95, 30, 200),
    "search_fraction": (0.69, 0.4, 0.75),
    "perturbation": (0.025, 0.001, 0.1),
}


class StopSearch(Exception):
    pass


def exact_score(data):
    try:
        c = np.asarray(data['circles'], dtype=float)
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return -float('inf')
        x = [Fraction(float(v)) for v in c[:, 0]]
        y = [Fraction(float(v)) for v in c[:, 1]]
        r = [Fraction(float(v)) for v in c[:, 2]]
        w = max(a+b for a, b in zip(x, r))-min(a-b for a, b in zip(x, r))
        h = max(a+b for a, b in zip(y, r))-min(a-b for a, b in zip(y, r))
        if w+h > 2:
            return -float('inf')
        for i, j in itertools.combinations(range(21), 2):
            d = np.sqrt((c[i, 0]-c[j, 0])**2+(c[i, 1]-c[j, 1])**2)
            if d < c[i, 2]+c[j, 2]:
                return -float('inf')
        return float(np.sum(c[:, 2]))
    except (ValueError, TypeError, KeyError, OverflowError):
        return -float('inf')


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + float(params['search_fraction'])*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = exact_score(init) if init is not None else -float('inf')
    ii, jj = np.triu_indices(21, 1)
    saved = None

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch

    def consider(c):
        nonlocal best, best_score
        check()
        c = np.array(c, dtype=float, copy=True)
        if not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return
        d = np.sqrt(np.sum((c[ii, :2]-c[jj, :2])**2, axis=1))
        ratio = float(np.min(d/(c[ii, 2]+c[jj, 2])))
        c[:, 2] *= min(1.0, ratio)*(1.0-3e-14)
        if np.min(c[:, 2]) <= 0:
            return
        lo = np.min(c[:, :2]-c[:, 2, None], axis=0)
        hi = np.max(c[:, :2]+c[:, 2, None], axis=0)
        scale = (2.0-1e-13)/float(np.sum(hi-lo))
        c[:, :2] -= (hi+lo)/2
        c *= scale
        approximate = float(np.sum(c[:, 2]))
        if approximate <= best_score:
            return
        candidate = {'circles': c.tolist()}
        score = exact_score(candidate)
        if score > best_score:
            best, best_score = candidate, score

    # A robust fallback is independent of the supplied cold configuration.
    fallback = np.array([[i*.2, j*.2, .1] for j in range(7) for i in range(3)])
    try:
        consider(fallback)
    except StopSearch:
        return best

    def model(rows):
        check()
        specs = []
        initial = []
        nr = len(rows)
        maxrow = max(rows)
        unit = 2.0/(maxrow + 1 + (nr-1)*np.sqrt(3)/2)
        radius = unit/2
        for row, count in enumerate(rows):
            check()
            yy = radius + row*unit*np.sqrt(3)/2
            if count % 2:
                specs.append((False, 0.0, yy))
            for k in range(count//2):
                check()
                xx = (k+(.5 if count % 2 == 0 else 1))*unit
                specs.append((True, xx, yy))
        nvar = sum(3 if p else 2 for p, _, _ in specs)+1
        widx = nvar-1
        z = np.zeros(nvar)
        maps = []
        representatives = []
        mirror = []
        bounds = []
        offset = 0
        for paired, x, y in specs:
            check()
            if paired:
                xi, yi, ri = offset, offset+1, offset+2
                z[[xi, yi, ri]] = [x, y, radius*.97]
                bounds.extend([(0, 2), (-1, 3), (1e-8, 1)])
                offset += 3
                k = len(maps)
                maps.extend([(xi, 1, yi, ri), (xi, -1, yi, ri)])
                mirror.extend([k+1, k])
                representatives.append((xi, yi, ri))
            else:
                yi, ri = offset, offset+1
                z[[yi, ri]] = [y, radius*.97]
                bounds.extend([(-1, 3), (1e-8, 1)])
                offset += 2
                k = len(maps)
                maps.append((-1, 0, yi, ri))
                mirror.append(k)
                representatives.append((-1, yi, ri))
        bounds.append((.15, 1.85))
        z[widx] = maxrow*unit
        X = np.zeros((21, nvar))
        Y = np.zeros_like(X)
        R = np.zeros_like(X)
        for k, (xi, sign, yi, ri) in enumerate(maps):
            check()
            if xi >= 0:
                X[k, xi] = sign
            Y[k, yi] = 1
            R[k, ri] = 1
        edges = []
        for a, b in zip(ii, jj):
            check()
            reflected = tuple(sorted((mirror[a], mirror[b])))
            if (int(a), int(b)) <= reflected:
                edges.append((a, b))
        a, b = np.array(edges).T
        DX, DY, DR = X[a]-X[b], Y[a]-Y[b], R[a]+R[b]
        L = []
        constants = []
        for xi, yi, ri in representatives:
            check()
            v = np.zeros(nvar)
            v[widx], v[ri] = .5, -1
            if xi >= 0:
                v[xi] = -1
            L.append(v)
            constants.append(0.)
            v = np.zeros(nvar)
            v[yi], v[ri] = 1, -1
            L.append(v)
            constants.append(0.)
            v = np.zeros(nvar)
            v[widx], v[yi], v[ri] = -1, -1, -1
            L.append(v)
            constants.append(2.)
        L, constants = np.array(L), np.array(constants)
        objective = -np.sum(R, axis=0)
        cache_z = None
        cache_v = None
        cache_j = None

        def calculate(v):
            nonlocal cache_z, cache_v, cache_j
            check()
            if cache_z is None or not np.array_equal(v, cache_z):
                dx, dy = DX@v, DY@v
                dist = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-15)
                cache_v = np.concatenate((L@v+constants, dist-DR@v))
                cache_j = np.vstack((L, (dx/dist)[:, None]*DX+(dy/dist)[:, None]*DY-DR))
                cache_z = v.copy()
            return cache_v, cache_j

        def decode(v):
            return np.column_stack((X@v, Y@v, R@v))

        def run(v, iterations):
            nonlocal saved
            last = [v.copy()]
            def callback(q):
                check()
                last[0] = q.copy()
                consider(decode(q))
            result = minimize(lambda q: float(objective@q), v,
                jac=lambda q: objective, method='SLSQP', bounds=bounds,
                constraints={'type': 'ineq', 'fun': lambda q: calculate(q)[0],
                             'jac': lambda q: calculate(q)[1]},
                callback=callback,
                options={'maxiter': int(iterations), 'ftol': 2e-11, 'disp': False})
            consider(decode(result.x))
            return result.x
        return z, run

    patterns = [(3, 4, 3, 4, 3, 4), (3, 4, 5, 4, 5),
                (5, 4, 3, 4, 5), (4, 5, 4, 5, 3),
                (5, 6, 5, 5), (4, 4, 5, 4, 4)]
    trajectories = []
    try:
        for rows in patterns:
            check()
            z, run = model(rows)
            before = best_score
            q = run(z, params['iterations'])
            trajectories.append((best_score if best_score > before else -1, q, run))
        trajectories.sort(key=lambda item: item[0], reverse=True)
        count = 0
        while time.monotonic() < deadline:
            check()
            _, q, run = trajectories[count % min(3, len(trajectories))]
            perturb = rng.normal(0, float(params['perturbation']), len(q))
            perturb[-1] *= .5
            trial = np.maximum(q+perturb, 1e-7)
            result = run(trial, params['iterations'])
            trajectories[count % min(3, len(trajectories))] = (best_score, result, run)
            count += 1
    except StopSearch:
        pass
    return best
