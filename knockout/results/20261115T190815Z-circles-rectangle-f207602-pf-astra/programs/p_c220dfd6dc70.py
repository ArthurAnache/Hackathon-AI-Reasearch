import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'time_fraction': (0.70, 0.50, 0.75),
    'restart_displacement': (0.018, 0.002, 0.08),
    'iterations': (180, 40, 300),
    'ftol': (2e-11, 1e-13, 1e-7, 'log')
}

class Deadline(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + params['time_fraction'] * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    rows = np.arange(len(ii))
    ids = np.arange(21)
    best = init
    best_score = -float('inf')

    def check():
        if time.monotonic() >= deadline:
            raise Deadline()

    def exact(candidate):
        try:
            a = np.asarray(candidate['circles'], dtype=np.float64)
            if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
                return -float('inf')
            xs = [Fraction(float(v)) for v in a[:, 0]]
            ys = [Fraction(float(v)) for v in a[:, 1]]
            rs = [Fraction(float(v)) for v in a[:, 2]]
            width = max(x+r for x, r in zip(xs, rs)) - min(x-r for x, r in zip(xs, rs))
            height = max(y+r for y, r in zip(ys, rs)) - min(y-r for y, r in zip(ys, rs))
            if width + height > Fraction(2):
                return -float('inf')
            for p, q in itertools.combinations(a, 2):
                if time.monotonic() >= started + 0.92 * budget_s:
                    return -float('inf')
                dist = np.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2)
                if dist < p[2] + q[2]:
                    return -float('inf')
            return float(np.sum(a[:, 2]))
        except (ValueError, TypeError, KeyError, OverflowError):
            return -float('inf')

    if init is not None:
        best_score = exact(init)

    def consider(a):
        nonlocal best, best_score
        check()
        a = np.asarray(a, dtype=float).copy()
        if a.shape != (21, 3) or not np.isfinite(a).all():
            return
        a[:, 2] = np.maximum(a[:, 2], 1e-10)
        a[:, :2] -= np.mean(a[:, :2], axis=0)
        dx = a[ii, 0] - a[jj, 0]
        dy = a[ii, 1] - a[jj, 1]
        distances = np.sqrt(dx*dx + dy*dy)
        factor = min(1.0, float(np.min(distances / (a[ii, 2] + a[jj, 2]))))
        if factor <= 0:
            return
        a[:, 2] *= factor * (1.0 - 2e-13)
        extent = np.max(a[:, :2] + a[:, 2, None], axis=0) - np.min(a[:, :2] - a[:, 2, None], axis=0)
        total = float(np.sum(extent))
        if total <= 0:
            return
        a *= (2.0 / total) * (1.0 - 2e-13)
        if float(np.sum(a[:, 2])) <= best_score:
            return
        candidate = {'circles': a.tolist()}
        score = exact(candidate)
        if score > best_score:
            best_score = score
            best = candidate

    lattice = []
    for row, count in enumerate((5, 4, 5, 4, 3)):
        if time.monotonic() >= deadline:
            break
        offset = 1.0 if row % 2 else 0.0
        if row == 4:
            offset = 2.0
        for col in range(count):
            if time.monotonic() >= deadline:
                break
            lattice.append([2.0*col + offset, np.sqrt(3.0)*row, 1.0])
    if len(lattice) != 21:
        return init
    lattice = np.asarray(lattice)
    try:
        consider(lattice)
        if init is not None:
            consider(np.asarray(init['circles'], dtype=float))
    except (Deadline, ValueError, TypeError, KeyError):
        return best

    # Variables are x[21], y[21], r[21], and rectangle width.
    wall_jac = np.zeros((84, 64))
    wall_jac[ids, ids] = -1
    wall_jac[ids, 42+ids] = -1
    wall_jac[ids, 63] = 0.5
    wall_jac[21+ids, ids] = 1
    wall_jac[21+ids, 42+ids] = -1
    wall_jac[21+ids, 63] = 0.5
    wall_jac[42+ids, 21+ids] = -1
    wall_jac[42+ids, 42+ids] = -1
    wall_jac[42+ids, 63] = -0.5
    wall_jac[63+ids, 21+ids] = 1
    wall_jac[63+ids, 42+ids] = -1
    wall_jac[63+ids, 63] = -0.5
    wall_offset = np.concatenate((np.zeros(42), np.ones(42)))
    objective_jac = np.zeros(64)
    objective_jac[42:63] = -1
    bounds = [(-1.0, 1.0)]*42 + [(1e-8, 0.5)]*21 + [(0.10, 1.90)]
    cache_z = None
    cache_g = None
    cache_j = None

    def calculate(z):
        nonlocal cache_z, cache_g, cache_j
        check()
        if cache_z is not None and np.array_equal(z, cache_z):
            return
        dx = z[ii] - z[jj]
        dy = z[21+ii] - z[21+jj]
        d = np.sqrt(dx*dx + dy*dy)
        inv = 1.0 / np.maximum(d, 1e-15)
        g = np.empty(294)
        g[:210] = d - z[42+ii] - z[42+jj]
        g[210:] = wall_jac @ z + wall_offset
        jac = np.zeros((294, 64))
        jac[rows, ii] = dx*inv
        jac[rows, jj] = -dx*inv
        jac[rows, 21+ii] = dy*inv
        jac[rows, 21+jj] = -dy*inv
        jac[rows, 42+ii] = -1
        jac[rows, 42+jj] = -1
        jac[210:] = wall_jac
        cache_z = z.copy()
        cache_g = g
        cache_j = jac

    def constraints(z):
        calculate(z)
        return cache_g

    def jacobian(z):
        calculate(z)
        return cache_j

    def objective(z):
        check()
        return -float(np.sum(z[42:63]))

    def gradient(z):
        check()
        return objective_jac

    def unpack(z):
        return np.column_stack((z[:21], z[21:42], z[42:63]))

    def callback(z):
        check()
        consider(unpack(z))

    def pack(a):
        a = np.asarray(a, dtype=float).copy()
        a[:, 2] = np.maximum(a[:, 2], 1e-6)
        lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
        hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
        ext = hi-lo
        a[:, :2] -= (hi+lo)/2
        factor = 2.0 / float(np.sum(ext))
        a *= factor
        return np.concatenate((a[:, 0], a[:, 1], a[:, 2], [ext[0]*factor]))

    try:
        first = np.asarray(init['circles'], dtype=float) if init is not None else lattice
        if first.shape != (21, 3) or not np.isfinite(first).all():
            first = lattice
    except (ValueError, TypeError, KeyError):
        first = lattice

    attempt = 0
    try:
        while time.monotonic() < deadline:
            check()
            if attempt == 0:
                a = first.copy()
            elif attempt == 1 and best_score < 2.30:
                a = lattice.copy()
            else:
                a = np.asarray(best['circles'], dtype=float).copy()
                displacement = params['restart_displacement'] * (1.0 + 0.35 * (attempt-1))
                a[:, :2] += rng.normal(0.0, displacement, (21, 2))
                a[:, 2] *= 0.97
            z = pack(a)
            cache_z = None
            result = minimize(objective, z, jac=gradient, method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                              callback=callback,
                              options={'maxiter': params['iterations'], 'ftol': params['ftol'], 'disp': False})
            consider(unpack(result.x))
            attempt += 1
    except Deadline:
        pass
    return best
