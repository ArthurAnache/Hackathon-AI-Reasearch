import time
import math
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog

PARAMS = {
    'kick': (0.065, 0.005, 0.25),
    'iterations': (19, 5, 40),
    'safety': (4e-14, 1e-15, 1e-10, 'log'),
}


def exact_score(obj):
    try:
        a = np.asarray(obj['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -math.inf
        xs = list(map(Fraction, map(float, a[:, 0])))
        ys = list(map(Fraction, map(float, a[:, 1])))
        rs = list(map(Fraction, map(float, a[:, 2])))
        w = max(x+r for x, r in zip(xs, rs))-min(x-r for x, r in zip(xs, rs))
        h = max(y+r for y, r in zip(ys, rs))-min(y-r for y, r in zip(ys, rs))
        if w+h > 2:
            return -math.inf
        for i in range(21):
            for j in range(i):
                dx = a[i, 0]-a[j, 0]
                dy = a[i, 1]-a[j, 1]
                if np.sqrt(dx*dx+dy*dy) < a[i, 2]+a[j, 2]:
                    return -math.inf
        return float(np.sum(a[:, 2]))
    except Exception:
        return -math.inf


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.69*budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        init = {'circles': [[(i % 7)*0.2, (i//7)*0.2, 0.09999999999999] for i in range(21)]}
    best = init
    best_score = exact_score(best)
    base = np.array(init['circles'], dtype=float, copy=True)
    base[:, :2] -= np.min(base[:, :2]-base[:, 2, None], axis=0)
    ii, jj = np.triu_indices(21, 1)
    rows = np.arange(len(ii))
    A = np.zeros((len(ii)+84, 64))
    b = np.zeros(len(ii)+84)
    A[rows, 42+ii] = 1
    A[rows, 42+jj] = 1
    k = len(ii)
    for i in range(21):
        if time.monotonic() >= deadline:
            return best
        A[k, i] = -1; A[k, 42+i] = 1
        A[k+1, i] = 1; A[k+1, 42+i] = 1; A[k+1, 63] = -1
        A[k+2, 21+i] = -1; A[k+2, 42+i] = 1
        A[k+3, 21+i] = 1; A[k+3, 42+i] = 1; A[k+3, 63] = 1
        b[k+3] = 2
        k += 4
    cost = np.zeros(64)
    cost[42:63] = -1
    bounds = [(0, 2)]*42 + [(1e-7, 0.5)]*21 + [(0.2, 1.8)]

    # Infer the incumbent's reflection involution, in either coordinate direction.
    pairings = []
    for axis in range(2):
        if time.monotonic() >= deadline:
            return best
        center = (np.max(base[:, axis]+base[:, 2])+np.min(base[:, axis]-base[:, 2]))/2
        reflected = base.copy()
        reflected[:, axis] = 2*center-reflected[:, axis]
        ds = np.sum((reflected[:, None, :]-base[None, :, :])**2, axis=2)
        partner = np.argmin(ds, axis=1)
        pairings.append((float(np.sum(np.min(ds, axis=1))), axis, partner))
    _, axis, partner = min(pairings, key=lambda q: q[0])
    groups = []
    used = set()
    for i in range(21):
        if time.monotonic() >= deadline:
            return best
        if i not in used:
            j = int(partner[i])
            if int(partner[j]) != i:
                j = i
            groups.append((i, j))
            used.add(i); used.add(j)

    def repaired(a):
        a = a.copy()
        a[:, :2] -= np.mean(a[:, :2], axis=0)
        delta = a[ii, :2]-a[jj, :2]
        dist = np.sqrt(np.sum(delta*delta, axis=1))
        factor = min(1.0, float(np.min(dist/(a[ii, 2]+a[jj, 2]))))
        a[:, 2] *= factor*(1-float(params['safety']))
        extent = np.max(a[:, :2]+a[:, 2, None], axis=0)-np.min(a[:, :2]-a[:, 2, None], axis=0)
        scale = min(1.0, 2.0/float(np.sum(extent)))
        a *= scale*(1-float(params['safety']))
        return {'circles': a.tolist()}

    attempt = 0
    archive = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        while time.monotonic() < deadline:
            a = base.copy()
            if attempt:
                if archive and attempt % 5 == 0:
                    a = archive[int(rng.integers(len(archive)))].copy()
                amplitude = float(params['kick'])*[0.45, 0.8, 1.3, 2.0, 0.25][(attempt-1) % 5]
                if attempt % 4 != 0:
                    count = min(len(groups), int(rng.integers(3, 7)))
                    chosen = rng.choice(len(groups), count, replace=False)
                    for gi in chosen:
                        if time.monotonic() >= deadline:
                            return best
                        i, j = groups[int(gi)]
                        move = rng.normal(size=2)*amplitude
                        if i == j:
                            move[axis] = 0
                            a[i, :2] += move
                        else:
                            a[i, :2] += move
                            move[axis] *= -1
                            a[j, :2] += move
                else:
                    a[:, :2] += rng.normal(size=(21, 2))*amplitude
            previous = -math.inf
            last_value = -math.inf
            for iteration in range(int(params['iterations'])):
                if time.monotonic() >= deadline:
                    return best
                dx = a[ii, 0]-a[jj, 0]
                dy = a[ii, 1]-a[jj, 1]
                norm = np.maximum(np.hypot(dx, dy), 1e-14)
                ux = dx/norm; uy = dy/norm
                A[rows, ii] = -ux
                A[rows, jj] = ux
                A[rows, 21+ii] = -uy
                A[rows, 21+jj] = uy
                result = linprog(cost, A_ub=A, b_ub=b, bounds=bounds, method='highs-ds',
                    options={'threads': 1, 'primal_feasibility_tolerance': 1e-9,
                             'dual_feasibility_tolerance': 1e-9,
                             'time_limit': max(0.001, deadline-time.monotonic())})
                if time.monotonic() >= deadline:
                    return best
                if not result.success:
                    break
                z = result.x
                a = np.column_stack((z[:21], z[21:42], z[42:63]))
                value = float(np.sum(a[:, 2]))
                last_value = value
                if value > best_score:
                    candidate = repaired(a)
                    score = exact_score(candidate)
                    if score > best_score:
                        best, best_score = candidate, score
                        base = np.asarray(candidate['circles'], dtype=float)
                        base[:, :2] -= np.min(base[:, :2]-base[:, 2, None], axis=0)
                if iteration >= 3 and value-previous < 2e-11:
                    break
                previous = value
            if last_value > best_score-0.002 and attempt > 0:
                archive.append(a.copy())
                if len(archive) > 4:
                    archive.pop(0)
            attempt += 1
    return best
