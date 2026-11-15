import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'warm_iterations': (65, 20, 140),
    'family_iterations': (85, 30, 180),
    'slack': (3e-15, 1e-15, 1e-12, 'log'),
    'noise': (0.009, 0.001, 0.05, 'log')
}

class SearchStopped(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.695 * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    best = init
    best_score = -float('inf')
    champion = None

    def check():
        if time.monotonic() >= deadline:
            raise SearchStopped()

    def exact(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        d = a[ii, :2] - a[jj, :2]
        if np.any(np.sqrt(d[:, 0]**2 + d[:, 1]**2) < a[ii, 2] + a[jj, 2]):
            return -float('inf')
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        w = max(u+v for u, v in zip(x, r)) - min(u-v for u, v in zip(x, r))
        h = max(u+v for u, v in zip(y, r)) - min(u-v for u, v in zip(y, r))
        if w+h > 2:
            return -float('inf')
        return float(np.sum(a[:, 2]))

    if init is not None:
        best_score = exact(np.asarray(init['circles'], dtype=float))

    X = np.zeros((21, 34))
    Y = np.zeros((21, 34))
    R = np.zeros((21, 34))
    mirror = np.arange(21)
    reps = [0, 1, 2]
    for k in range(3):
        check()
        Y[k, 9+k] = 1
        R[k, 21+k] = 1
    for k in range(9):
        check()
        a, b = 3+2*k, 4+2*k
        X[a, k], X[b, k] = 1, -1
        Y[a, 12+k] = Y[b, 12+k] = 1
        R[a, 24+k] = R[b, 24+k] = 1
        mirror[a], mirror[b] = b, a
        reps.append(a)
    pairs = []
    for a, b in zip(ii, jj):
        check()
        if (int(a), int(b)) <= tuple(sorted((int(mirror[a]), int(mirror[b])))):
            pairs.append((a, b))
    pi, pj = np.array(pairs).T
    DX, DY, SR = X[pi]-X[pj], Y[pi]-Y[pj], R[pi]+R[pj]
    walls, constants = [], []
    for a in reps:
        check()
        v = -X[a]-R[a]
        v[33] += 0.5
        walls.append(v)
        constants.append(0.)
        walls.append(Y[a]-R[a])
        constants.append(0.)
        v = -Y[a]-R[a]
        v[33] -= 1
        walls.append(v)
        constants.append(2.)
    W, wc = np.asarray(walls), np.asarray(constants)
    objective = -R.sum(axis=0)
    bounds = [(0., 1.)]*9 + [(0., 2.)]*12 + [(1e-7, 0.7)]*12 + [(0.15, 1.85)]

    def con(z):
        check()
        dx, dy = DX@z, DY@z
        return np.concatenate((W@z+wc, np.sqrt(dx*dx+dy*dy)-SR@z))

    def jac(z):
        check()
        dx, dy = DX@z, DY@z
        d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-18)
        return np.vstack((W, (dx/d)[:, None]*DX+(dy/d)[:, None]*DY-SR))

    def retain(z):
        nonlocal best, best_score, champion
        check()
        a = np.column_stack((X@z, Y@z, R@z))
        if not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return
        delta = a[ii, :2]-a[jj, :2]
        distances = np.sqrt(delta[:, 0]**2+delta[:, 1]**2)
        factor = min(1., float(np.min(distances/(a[ii, 2]+a[jj, 2]))))
        if factor <= 0:
            return
        a[:, 2] *= factor*(1-params['slack'])
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        span = float(np.sum(hi-lo))
        if span <= 0:
            return
        a[:, :2] -= (hi+lo)/2
        a *= (2/span)*(1-params['slack'])
        if float(np.sum(a[:, 2])) <= best_score:
            return
        value = exact(a)
        if value > best_score:
            best_score = value
            best = {'circles': a.tolist()}
            champion = z.copy()

    def polish(z):
        c = con(z)
        active = np.flatnonzero(c < 3e-8)
        if len(active) < 30:
            return z
        for unused in range(5):
            check()
            residual = con(z)[active]
            step = np.linalg.lstsq(jac(z)[active], -residual, rcond=1e-11)[0]
            if not np.isfinite(step).all() or np.linalg.norm(step) > 0.015:
                break
            z = z+step
            retain(z)
            if np.max(np.abs(residual)) < 2e-16:
                break
        return z

    def optimize(z, iterations):
        last = z.copy()
        count = 0
        def callback(v):
            nonlocal count, last
            check()
            last = v.copy()
            count += 1
            if count % 7 == 0:
                retain(v)
        check()
        result = minimize(lambda v: float(objective@v), z,
                          jac=lambda v: objective, method='SLSQP', bounds=bounds,
                          constraints={'type': 'ineq', 'fun': con, 'jac': jac},
                          callback=callback,
                          options={'maxiter': int(iterations), 'ftol': 4e-14, 'disp': False})
        last = result.x
        retain(last)
        return polish(last.copy())

    def extract(a):
        choices = []
        for swap in (False, True):
            check()
            b = a.copy()
            if swap:
                b[:, :2] = b[:, [1, 0]]
            lo = np.min(b[:, :2]-b[:, 2, None], axis=0)
            hi = np.max(b[:, :2]+b[:, 2, None], axis=0)
            b[:, 0] -= (hi[0]+lo[0])/2
            b[:, 1] -= lo[1]
            singles = list(np.argsort(np.abs(b[:, 0]))[:3])
            remaining = set(range(21))-set(singles)
            groups = []
            error = float(np.sum(b[singles, 0]**2))
            while remaining:
                check()
                i = min(remaining)
                remaining.remove(i)
                target = b[i].copy()
                target[0] *= -1
                j = min(remaining, key=lambda q: float(np.sum((b[q]-target)**2)))
                remaining.remove(j)
                error += float(np.sum((b[j]-target)**2))
                groups.append((i, j))
            z = np.zeros(34)
            for k, i in enumerate(sorted(singles, key=lambda q: b[q, 1])):
                check()
                z[9+k], z[21+k] = b[i, 1], b[i, 2]
            for k, (i, j) in enumerate(groups):
                check()
                z[k] = abs(b[i, 0]-b[j, 0])/2
                z[12+k] = (b[i, 1]+b[j, 1])/2
                z[24+k] = (b[i, 2]+b[j, 2])/2
            z[33] = hi[0]-lo[0]
            choices.append((error, z))
        return min(choices, key=lambda item: item[0])[1]

    def lattice(rows):
        z = np.zeros(34)
        singles, pairs = [], []
        y = 1.
        previous = None
        for count in rows:
            check()
            if previous is not None:
                y += np.sqrt(3.) if count % 2 != previous % 2 else 2.
            if count % 2:
                singles.append(y)
            for x in range(2 if count % 2 else 1, count, 2):
                check()
                pairs.append((x, y))
            previous = count
        width = 2*(max(p[0] for p in pairs)+1)
        scale = 2/(width+y+1)
        for k, yy in enumerate(singles):
            check()
            z[9+k], z[21+k] = yy*scale, scale
        for k, (xx, yy) in enumerate(pairs):
            check()
            z[k], z[12+k], z[24+k] = xx*scale, yy*scale, scale
        z[33] = width*scale
        return z

    archive = []
    try:
        if init is not None:
            z = extract(np.asarray(init['circles'], dtype=float))
            retain(z)
            z = optimize(z, params['warm_iterations'])
            archive.append(z.copy())
        families = [(5, 4, 5, 4, 3), (5, 4, 3, 4, 5), (3, 6, 3, 6, 3), (7, 7, 7), (3, 4, 3, 4, 3, 4)]
        for rows in families:
            check()
            if time.monotonic()-start > 0.55*budget_s:
                break
            z = lattice(rows)
            retain(z)
            archive.append(optimize(z, params['family_iterations']))
        attempt = 0
        while time.monotonic() < deadline:
            check()
            if not archive:
                break
            base = champion if champion is not None and attempt % 2 == 0 else archive[attempt % len(archive)]
            z = base.copy()
            z[:33] += rng.normal(0, params['noise'], 33)
            for k, (lo, hi) in enumerate(bounds):
                check()
                z[k] = min(hi-1e-9, max(lo+1e-9, z[k]))
            optimize(z, params['family_iterations'])
            attempt += 1
    except SearchStopped:
        pass
    if best is None:
        best = {'circles': [[(k % 7)*0.14, (k//7)*0.14, 0.069] for k in range(21)]}
    return best
