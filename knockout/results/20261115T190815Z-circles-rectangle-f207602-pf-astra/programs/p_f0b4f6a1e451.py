import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'time_fraction': (0.69, 0.5, 0.7),
    'max_iterations': (160, 30, 300),
    'perturbation': (0.035, 0.0, 0.15),
}


def exact_score(data):
    try:
        a = np.asarray(data['circles'], dtype=np.float64)
        if a.shape != (21, 3) or not np.isfinite(a).all() or not (a[:, 2] > 0).all():
            return -float('inf')
        xs = [Fraction(float(v)) for v in a[:, 0]]
        ys = [Fraction(float(v)) for v in a[:, 1]]
        rs = [Fraction(float(v)) for v in a[:, 2]]
        w = max(x+r for x, r in zip(xs, rs)) - min(x-r for x, r in zip(xs, rs))
        h = max(y+r for y, r in zip(ys, rs)) - min(y-r for y, r in zip(ys, rs))
        if w+h > Fraction(2):
            return -float('inf')
        for i, j in itertools.combinations(range(21), 2):
            dx = a[i, 0]-a[j, 0]
            dy = a[i, 1]-a[j, 1]
            if np.sqrt(dx**2+dy**2) < a[i, 2]+a[j, 2]:
                return -float('inf')
        return float(np.sum(a[:, 2]))
    except Exception:
        return -float('inf')


class StopSearch(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + float(params['time_fraction'])*budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    best = init
    best_score = exact_score(init) if init is not None else -float('inf')

    def check_time():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def save(a):
        nonlocal best, best_score
        if not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return
        a = a.copy()
        dx = a[ii, 0]-a[jj, 0]
        dy = a[ii, 1]-a[jj, 1]
        distances = np.sqrt(dx**2+dy**2)
        factor = min(1.0, float(np.min(distances/(a[ii, 2]+a[jj, 2]))))
        if factor <= 0:
            return
        a[:, 2] *= factor*(1.0-3e-14)
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        a[:, :2] -= (hi+lo)*0.5
        a *= (2.0/float(np.sum(hi-lo)))*(1.0-3e-14)
        if float(np.sum(a[:, 2])) <= best_score:
            return
        candidate = {'circles': a.tolist()}
        score = exact_score(candidate)
        if score > best_score:
            best, best_score = candidate, score

    # A guaranteed nondegenerate fallback, independent of the supplied start.
    fallback = np.zeros((21, 3))
    fallback[:, 0] = np.arange(21) % 7
    fallback[:, 1] = np.arange(21) // 7
    fallback[:, 2] = 0.5
    save(fallback)

    def make_family(rows):
        groups = []
        pair_count = sum(k//2 for k in rows)
        group_count = sum((k+1)//2 for k in rows)
        nv = pair_count + 2*group_count + 1
        X = np.zeros((21, nv))
        Y = np.zeros((21, nv))
        R = np.zeros((21, nv))
        v = np.zeros(nv)
        circle = 0
        group = 0
        pair = 0
        for row, count in enumerate(rows):
            check_time()
            y = (row-(len(rows)-1)*0.5)*np.sqrt(3.0)
            if count % 2:
                Y[circle, pair_count+group] = 1
                R[circle, pair_count+group_count+group] = 1
                v[pair_count+group] = y
                v[pair_count+group_count+group] = 1
                circle += 1
                group += 1
            for p in range(count//2):
                check_time()
                offset = 2*(p+1) if count % 2 else 2*p+1
                X[circle, pair] = -1
                X[circle+1, pair] = 1
                Y[circle:circle+2, pair_count+group] = 1
                R[circle:circle+2, pair_count+group_count+group] = 1
                v[pair] = offset
                v[pair_count+group] = y
                v[pair_count+group_count+group] = 1
                circle += 2
                group += 1
                pair += 1
        width = 2*max(rows)
        height = 2+(len(rows)-1)*np.sqrt(3.0)
        v *= 2/(width+height)
        v[-1] = 2*width/(width+height)
        return X, Y, R, v, pair_count, group_count

    def solve(rows, noise=0.0):
        check_time()
        X, Y, R, v, npair, ngroup = make_family(rows)
        nv = len(v)
        if noise:
            v[npair:npair+ngroup] += rng.normal(0, noise, ngroup)
            v[npair+ngroup:-1] *= np.exp(rng.normal(0, noise*2, ngroup))
        DX = X[ii]-X[jj]
        DY = Y[ii]-Y[jj]
        SR = R[ii]+R[jj]
        B = np.vstack((-X-R, X-R, -Y-R, Y-R))
        B[:42, -1] += 0.5
        B[42:, -1] -= 0.5
        b = np.zeros(84)
        b[42:] = 1
        objective_jac = -np.sum(R, axis=0)
        cached_v = None
        cached_c = None
        cached_j = None

        def unpack(z):
            return np.column_stack((X@z, Y@z, R@z))

        def update(z):
            nonlocal cached_v, cached_c, cached_j
            check_time()
            if cached_v is not None and np.array_equal(cached_v, z):
                return
            dx = DX@z
            dy = DY@z
            d = np.sqrt(dx*dx+dy*dy)
            dsafe = np.maximum(d, 1e-15)
            cached_c = np.concatenate((d-SR@z, B@z+b))
            cached_j = np.vstack(((dx/dsafe)[:, None]*DX+(dy/dsafe)[:, None]*DY-SR, B))
            cached_v = z.copy()

        def constraints(z):
            update(z)
            return cached_c

        def jacobian(z):
            update(z)
            return cached_j

        def objective(z):
            check_time()
            return float(objective_jac@z)

        def callback(z):
            check_time()
            save(unpack(z))

        bounds = [(1e-7, 1.0)]*npair + [(-1.0, 1.0)]*ngroup + [(1e-6, 0.6)]*ngroup + [(0.15, 1.85)]
        save(unpack(v))
        result = minimize(objective, v, jac=lambda z: objective_jac,
                          method='SLSQP', bounds=bounds,
                          constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                          callback=callback,
                          options={'maxiter': int(params['max_iterations']), 'ftol': 2e-12, 'disp': False})
        check_time()
        save(unpack(result.x))

    families = [
        [4, 3, 4, 3, 4, 3],
        [3, 4, 5, 4, 5],
        [5, 4, 3, 4, 5],
        [4, 5, 4, 5, 3],
        [4, 4, 5, 4, 4],
        [3, 4, 3, 4, 3, 4],
    ]
    try:
        for rows in families:
            check_time()
            solve(rows)
        while time.monotonic() < deadline:
            check_time()
            rows = families[int(rng.integers(len(families)))]
            solve(rows, float(params['perturbation']))
    except StopSearch:
        pass
    return best
