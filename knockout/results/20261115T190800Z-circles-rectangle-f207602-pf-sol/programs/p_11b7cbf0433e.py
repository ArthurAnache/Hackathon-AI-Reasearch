import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'time_fraction': (0.69, 0.4, 0.71),
    'contact_margin': (0.065, 0.02, 0.15),
    'iterations': (110, 40, 220),
    'cluster_move': (0.025, 0.005, 0.07)
}
N = 21
I, J = np.triu_indices(N, 1)
X = 3*np.arange(N)
R = X+2


def exact_score(candidate):
    try:
        c = np.asarray(candidate['circles'], dtype=float)
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return -float('inf')
        a = [[Fraction(float(v)) for v in row] for row in c]
        width = max(q[0]+q[2] for q in a)-min(q[0]-q[2] for q in a)
        height = max(q[1]+q[2] for q in a)-min(q[1]-q[2] for q in a)
        if width+height > 2:
            return -float('inf')
        for u, v in itertools.combinations(c, 2):
            if np.sqrt((u[0]-v[0])**2+(u[1]-v[1])**2) < u[2]+v[2]:
                return -float('inf')
        return float(np.sum(c[:, 2]))
    except Exception:
        return -float('inf')


def repair(c):
    c = np.asarray(c, dtype=float).reshape(21, 3).copy()
    if not np.isfinite(c).all():
        return None
    c[:, 2] = np.maximum(c[:, 2], 1e-9)
    d = c[I, :2]-c[J, :2]
    distances = np.sqrt(np.sum(d*d, axis=1))
    ratio = float(np.min(distances/(c[I, 2]+c[J, 2])))
    if ratio <= 0:
        return None
    c[:, 2] *= min(1., ratio)*(1.-8e-13)
    lo = np.min(c[:, :2]-c[:, 2, None], axis=0)
    hi = np.max(c[:, :2]+c[:, 2, None], axis=0)
    c[:, :2] -= lo
    c *= (2.-2e-12)/float(np.sum(hi-lo))
    return {'circles': c.tolist()}


class StopSearch(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    deadline = time.monotonic()+float(params['time_fraction'])*budget_s
    rng = np.random.default_rng(seed)
    best = init
    score_best = exact_score(init)
    pending = None
    pending_score = score_best
    archive = []

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def retain(c):
        nonlocal pending, pending_score
        candidate = repair(c)
        if candidate is None:
            return
        score = float(np.sum(np.asarray(candidate['circles'])[:, 2]))
        if score > pending_score:
            pending_score = score
            pending = candidate

    patterns = [
        [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)],
        [(0, 0), (0, 1), (1, 2), (2, 0), (2, 2)],
        [(0, 0), (0, 2), (1, 0), (1, 2), (2, 1)]
    ]

    def lattice(k):
        c = [[1+2*x, 1+2*y, 1.] for y in range(4) for x in range(4)]
        c += [[2+2*x, 2+2*y, np.sqrt(2)-1] for x, y in patterns[k]]
        return np.asarray(c)/8.

    retain(lattice(0))
    gradient = np.zeros(64)
    gradient[R] = -1
    bounds = [(0., 2.), (0., 2.), (1e-8, 0.7)]*21+[(0.15, 1.85)]

    def objective(z):
        check()
        return -float(np.sum(z[R]))

    def derivative(z):
        check()
        return gradient

    def normalize(c):
        c = c.copy()
        lo = np.min(c[:, :2]-c[:, 2, None], axis=0)
        hi = np.max(c[:, :2]+c[:, 2, None], axis=0)
        f = 2./float(np.sum(hi-lo))
        c[:, :2] -= lo
        c *= f
        return np.r_[c.ravel(), (hi[0]-lo[0])*f]

    attempt = 0
    try:
        while time.monotonic() < deadline:
            check()
            if attempt == 0 and score_best > 2.1:
                c = np.asarray(init['circles'], dtype=float).copy()
            elif attempt < 3:
                c = lattice(attempt)
                c[:, :2] += rng.normal(0., 0.002, (21, 2))
            else:
                anchor = pending if pending is not None else best
                c = np.asarray(anchor['circles'], dtype=float).copy()
                # Move a connected cluster, rather than independently shaking every circle.
                delta = c[I, :2]-c[J, :2]
                gap = np.sqrt(np.sum(delta*delta, axis=1))-c[I, 2]-c[J, 2]
                edges = list(zip(I[gap < 0.012], J[gap < 0.012]))
                cluster = {int(rng.integers(21))}
                target = int(rng.integers(3, 7))
                while len(cluster) < target and time.monotonic() < deadline:
                    frontier = [int(b if a in cluster else a) for a, b in edges if (a in cluster) != (b in cluster)]
                    if not frontier:
                        break
                    cluster.add(frontier[int(rng.integers(len(frontier)))])
                ids = np.array(sorted(cluster), dtype=int)
                move = float(params['cluster_move'])*(1.+0.35*(attempt % 3))
                c[ids, :2] += rng.normal(0., move, (len(ids), 2))
                c[ids, 2] *= rng.uniform(0.86, 1.12, len(ids))
            z = normalize(c)
            cc = z[:63].reshape(21, 3)
            d = cc[I, :2]-cc[J, :2]
            gap = np.sqrt(np.sum(d*d, axis=1))-cc[I, 2]-cc[J, 2]
            active = gap < float(params['contact_margin'])
            for stage in range(5):
                check()
                ai, aj = I[active], J[active]
                m = len(ai)
                rows = np.arange(m)
                wall = m+np.arange(21)

                def constraints(v):
                    check()
                    q = v[:63].reshape(21, 3)
                    x, y, r = q.T
                    dx, dy = x[ai]-x[aj], y[ai]-y[aj]
                    return np.r_[np.sqrt(dx*dx+dy*dy)-r[ai]-r[aj], x-r, v[63]-x-r, y-r, 2-v[63]-y-r]

                def jacobian(v):
                    check()
                    q = v[:63].reshape(21, 3)
                    dx = q[ai, 0]-q[aj, 0]
                    dy = q[ai, 1]-q[aj, 1]
                    dist = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-20)
                    a = np.zeros((m+84, 64))
                    a[rows, 3*ai] = dx/dist
                    a[rows, 3*aj] = -dx/dist
                    a[rows, 3*ai+1] = dy/dist
                    a[rows, 3*aj+1] = -dy/dist
                    a[rows, 3*ai+2] = -1
                    a[rows, 3*aj+2] = -1
                    a[wall, X] = 1
                    a[wall, R] = -1
                    a[wall+21, X] = -1
                    a[wall+21, R] = -1
                    a[wall+21, 63] = 1
                    a[wall+42, X+1] = 1
                    a[wall+42, R] = -1
                    a[wall+63, X+1] = -1
                    a[wall+63, R] = -1
                    a[wall+63, 63] = -1
                    return a

                def callback(v):
                    check()
                    retain(v[:63].reshape(21, 3))

                result = minimize(objective, z, jac=derivative, method='SLSQP', bounds=bounds,
                    constraints=[{'type': 'ineq', 'fun': constraints, 'jac': jacobian}],
                    callback=callback, options={'maxiter': int(params['iterations']), 'ftol': 8e-12, 'disp': False})
                z = result.x
                retain(z[:63].reshape(21, 3))
                q = z[:63].reshape(21, 3)
                delta = q[I, :2]-q[J, :2]
                gap = np.sqrt(np.sum(delta*delta, axis=1))-q[I, 2]-q[J, 2]
                missing = (~active) & (gap < 1e-5)
                if not np.any(missing):
                    break
                active |= missing
            attempt += 1
    except (StopSearch, ValueError, FloatingPointError):
        pass
    if pending is not None and exact_score(pending) > score_best:
        best = pending
    return best
