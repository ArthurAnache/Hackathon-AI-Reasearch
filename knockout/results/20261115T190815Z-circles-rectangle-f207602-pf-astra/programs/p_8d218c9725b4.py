import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"iterations": (85, 30, 150), "time_fraction": (0.70, 0.4, 0.75)}


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + float(params["time_fraction"]) * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    best = init
    best_score = -float('inf')

    class Stop(Exception):
        pass

    def check():
        if time.monotonic() >= deadline:
            raise Stop()

    def exact(data):
        try:
            a = np.asarray(data['circles'], dtype=float)
            if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
                return -float('inf')
            xs = list(map(Fraction, map(float, a[:, 0])))
            ys = list(map(Fraction, map(float, a[:, 1])))
            rs = list(map(Fraction, map(float, a[:, 2])))
            w = max(map(lambda t: t[0]+t[1], zip(xs, rs))) - min(map(lambda t: t[0]-t[1], zip(xs, rs)))
            h = max(map(lambda t: t[0]+t[1], zip(ys, rs))) - min(map(lambda t: t[0]-t[1], zip(ys, rs)))
            if w + h > 2:
                return -float('inf')
            for p, q in itertools.combinations(a, 2):
                time.monotonic()
                d = np.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2)
                if d < p[2] + q[2]:
                    return -float('inf')
            return float(np.sum(a[:, 2]))
        except (ValueError, TypeError, KeyError, OverflowError):
            return -float('inf')

    if init is not None:
        best_score = exact(init)

    def retain(a):
        nonlocal best, best_score
        if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return
        a = a.copy()
        a[:, :2] -= np.mean(a[:, :2], axis=0)
        dx = a[ii, 0] - a[jj, 0]
        dy = a[ii, 1] - a[jj, 1]
        d = np.sqrt(dx*dx + dy*dy)
        factor = min(1.0, float(np.min(d / (a[ii, 2] + a[jj, 2]))))
        if factor <= 0:
            return
        a[:, 2] *= factor * (1.0 - 2e-13)
        lower = np.min(a[:, :2] - a[:, 2, None], axis=0)
        upper = np.max(a[:, :2] + a[:, 2, None], axis=0)
        a[:, :2] -= (lower + upper) / 2
        a *= (2.0 / float(np.sum(upper-lower))) * (1.0 - 2e-14)
        if float(np.sum(a[:, 2])) <= best_score:
            return
        data = {'circles': a.tolist()}
        score = exact(data)
        if score > best_score:
            best_score, best = score, data

    # A guaranteed feasible fallback independent of the incoming candidate.
    k = np.arange(21)
    fallback = np.column_stack(((k % 7) * 0.2, (k // 7) * 0.2, np.full(21, 0.1)))
    retain(fallback)

    objjac = np.zeros(64)
    objjac[42:63] = -1.0
    boundary = np.zeros((84, 64))
    ar = np.arange(21)
    boundary[ar, ar] = 1
    boundary[ar, 42+ar] = -1
    boundary[21+ar, ar] = -1
    boundary[21+ar, 42+ar] = -1
    boundary[21+ar, 63] = 1
    boundary[42+ar, 21+ar] = 1
    boundary[42+ar, 42+ar] = -1
    boundary[63+ar, 21+ar] = -1
    boundary[63+ar, 42+ar] = -1
    boundary[63+ar, 63] = -1
    rows = np.arange(m)

    def objective(z):
        check()
        return -float(np.sum(z[42:63]))

    def constraints(z):
        check()
        x, y, r, w = z[:21], z[21:42], z[42:63], z[63]
        dx, dy = x[ii]-x[jj], y[ii]-y[jj]
        return np.concatenate((x-r, w-x-r, y-r, 2-w-y-r,
                               np.sqrt(dx*dx+dy*dy)-r[ii]-r[jj]))

    def jacobian(z):
        check()
        dx = z[ii] - z[jj]
        dy = z[21+ii] - z[21+jj]
        d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-20)
        j = np.zeros((84+m, 64))
        j[:84] = boundary
        j[84+rows, ii] = dx/d
        j[84+rows, jj] = -dx/d
        j[84+rows, 21+ii] = dy/d
        j[84+rows, 21+jj] = -dy/d
        j[84+rows, 42+ii] = -1
        j[84+rows, 42+jj] = -1
        return j

    def callback(z):
        check()
        retain(np.column_stack((z[:21], z[21:42], z[42:63])))

    def optimize(a):
        check()
        a = np.asarray(a, dtype=float).copy()
        lower = np.min(a[:, :2]-a[:, 2, None], axis=0)
        upper = np.max(a[:, :2]+a[:, 2, None], axis=0)
        scale = 2 / np.sum(upper-lower)
        a[:, :2] -= lower
        a *= scale
        z = np.concatenate((a[:, 0], a[:, 1], a[:, 2], [(upper[0]-lower[0])*scale]))
        result = minimize(objective, z, jac=lambda z: objjac,
                          constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                          bounds=[(0.0, 2.0)]*42 + [(1e-7, 0.7)]*21 + [(0.25, 1.75)],
                          method='SLSQP', callback=callback,
                          options={'maxiter': int(params['iterations']), 'ftol': 2e-11, 'disp': False})
        callback(result.x)

    patterns = [(4, 5, 3, 5, 4), (4, 5, 4, 5, 3), (5, 4, 3, 4, 5)]
    try:
        # A high-quality incumbent deserves a polishing pass rather than a restart.
        if best_score > 2.30:
            optimize(best['circles'])
        for attempt in range(7):
            check()
            points = []
            if attempt == 3:
                for y in range(5):
                    check()
                    for x in range(5):
                        check()
                        if x in (0, 4) and y in (0, 4):
                            continue
                        points.append([2.0*x, 2.0*y, 1.0])
            else:
                pattern = patterns[attempt % 3]
                for row, count in enumerate(pattern):
                    check()
                    offset = (5-count)
                    for col in range(count):
                        check()
                        points.append([offset+2.0*col, np.sqrt(3.0)*row, 1.0])
            a = np.asarray(points, dtype=float)
            if attempt >= 4:
                a[:, :2] += rng.normal(0, 0.13, (21, 2))
                a[:, 2] *= rng.uniform(0.86, 1.0, 21)
            retain(a)
            optimize(a)
    except Stop:
        pass
    return best
