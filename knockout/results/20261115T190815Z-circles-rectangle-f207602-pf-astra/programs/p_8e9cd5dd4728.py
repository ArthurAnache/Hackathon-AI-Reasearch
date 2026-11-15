import time
import warnings
import numpy as np
from fractions import Fraction
from scipy.optimize import linprog, minimize

PARAMS = {
    'lp_steps': (42, 15, 90),
    'radius_floor': (0.002, 0.00001, 0.015, 'log'),
    'perturbation': (0.035, 0.005, 0.15),
    'polish_seconds': (0.16, 0.03, 0.35)
}

class OutOfTime(Exception):
    pass

def construct(params: dict, seed: int, budget_s: float, init=None):
    start = time.monotonic()
    stop = start + 0.70 * budget_s
    hard = start + 0.90 * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    rows = np.arange(m)
    objective = np.zeros(64)
    objective[42:63] = -1.0
    floor = float(params['radius_floor'])
    bounds = [(0.0, 2.0)] * 42 + [(floor, 0.5)] * 21 + [(0.35, 1.65)]
    wall = np.zeros((84, 64))
    wb = np.zeros(84)
    k = np.arange(n)
    wall[k, k] = -1
    wall[k, 42+k] = 1
    wall[21+k, k] = 1
    wall[21+k, 42+k] = 1
    wall[21+k, 63] = -1
    wall[42+k, 21+k] = -1
    wall[42+k, 42+k] = 1
    wall[63+k, 21+k] = 1
    wall[63+k, 42+k] = 1
    wall[63+k, 63] = 1
    wb[63:] = 2
    matrix = np.zeros((m+84, 64))
    matrix[m:] = wall
    matrix[rows, 42+ii] = 1
    matrix[rows, 42+jj] = 1
    rhs = np.concatenate((np.zeros(m), wb))
    best_obj = None
    best_c = None
    best_score = -float('inf')

    def decode(obj):
        if 'circles' in obj:
            return np.asarray(obj['circles'], dtype=float)
        return np.asarray(obj['coordinate_rows'], dtype=float).T

    def exact(c):
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return -float('inf')
        xp, xm, yp, ym = [], [], [], []
        for x, y, r in c:
            if time.monotonic() >= hard:
                return -float('inf')
            x, y, r = Fraction(float(x)), Fraction(float(y)), Fraction(float(r))
            xp.append(x+r)
            xm.append(x-r)
            yp.append(y+r)
            ym.append(y-r)
        if max(xp)-min(xm)+max(yp)-min(ym) > 2:
            return -float('inf')
        for a, b in zip(ii, jj):
            if time.monotonic() >= hard:
                return -float('inf')
            u, v = c[a], c[b]
            d = np.sqrt((u[0]-v[0])**2 + (u[1]-v[1])**2)
            if d < u[2]+v[2]:
                return -float('inf')
        return float(np.sum(c[:, 2]))

    def consider(c, original=None):
        nonlocal best_obj, best_c, best_score
        if time.monotonic() >= stop:
            return
        c = np.asarray(c, dtype=float)
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return
        if original is not None:
            s = exact(c)
            if s > best_score:
                best_score, best_c, best_obj = s, c.copy(), original
            return
        c = c.copy()
        dx = c[ii, 0]-c[jj, 0]
        dy = c[ii, 1]-c[jj, 1]
        dist = np.sqrt(dx*dx+dy*dy)
        ratio = float(np.min(dist/(c[ii, 2]+c[jj, 2])))
        if ratio <= 0:
            return
        c[:, 2] *= min(1.0, ratio) * (1.0-3e-13)
        lower = np.min(c[:, :2]-c[:, 2, None], axis=0)
        c[:, :2] -= lower
        ext = np.max(c[:, :2]+c[:, 2, None], axis=0)
        c *= min(1.0, 2.0/float(np.sum(ext))) * (1.0-3e-13)
        if float(np.sum(c[:, 2])) <= best_score:
            return
        s = exact(c)
        if s > best_score:
            best_score = s
            best_c = c.copy()
            best_obj = {'coordinate_rows': c.T.tolist()}

    if init is not None:
        try:
            consider(decode(init), init)
        except (ValueError, TypeError, KeyError):
            pass

    def vector(c):
        c = c.copy()
        low = np.min(c[:, :2]-c[:, 2, None], axis=0)
        c[:, :2] -= low
        extent = np.max(c[:, :2]+c[:, 2, None], axis=0)
        c *= 2.0 / float(np.sum(extent))
        z = np.empty(64)
        z[:21] = c[:, 0]
        z[21:42] = c[:, 1]
        z[42:63] = c[:, 2]
        z[63] = float(np.max(c[:, 0]+c[:, 2]))
        return z

    def circles(z):
        return z[:63].reshape(3, 21).T

    def lattice(counts):
        data = []
        for row, count in enumerate(counts):
            if time.monotonic() >= stop:
                raise OutOfTime
            x = 2*np.arange(count, dtype=float)-(count-1)
            y = np.full(count, row*np.sqrt(3.0))
            data.append(np.column_stack((x, y, np.ones(count))))
        return vector(np.concatenate(data, axis=0))

    try:
        initial = lattice([4, 3, 4, 3, 4, 3])
        consider(circles(initial))
    except OutOfTime:
        return best_obj if best_obj is not None else init

    def lp_optimize(z):
        previous = -1e100
        stable = 0
        for iteration in range(int(params['lp_steps'])):
            if time.monotonic() >= stop-0.015:
                break
            dx = z[ii]-z[jj]
            dy = z[21+ii]-z[21+jj]
            d = np.maximum(np.hypot(dx, dy), 1e-15)
            ux, uy = dx/d, dy/d
            matrix[rows, ii] = -ux
            matrix[rows, jj] = ux
            matrix[rows, 21+ii] = -uy
            matrix[rows, 21+jj] = uy
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result = linprog(objective, A_ub=matrix, b_ub=rhs,
                    bounds=bounds, method='highs', options={
                        'threads': 1,
                        'primal_feasibility_tolerance': 1e-9,
                        'dual_feasibility_tolerance': 1e-9,
                        'time_limit': max(0.001, stop-time.monotonic()-0.008)})
            if not result.success:
                break
            z = result.x
            value = float(np.sum(z[42:63]))
            consider(circles(z))
            stable = stable+1 if value-previous < 2e-10 else 0
            previous = value
            if stable >= 2:
                break
        return z

    def polish(z):
        end = min(stop-0.01, time.monotonic()+float(params['polish_seconds']))
        last = [z.copy()]
        def check():
            if time.monotonic() >= end:
                raise OutOfTime
        def fun(v):
            check()
            return -np.sum(v[42:63])
        def jac(v):
            check()
            return objective
        def constraints(v):
            check()
            dx = v[ii]-v[jj]
            dy = v[21+ii]-v[21+jj]
            return np.concatenate((np.hypot(dx, dy)-v[42+ii]-v[42+jj], wb-wall@v))
        def derivative(v):
            check()
            dx = v[ii]-v[jj]
            dy = v[21+ii]-v[21+jj]
            d = np.maximum(np.hypot(dx, dy), 1e-15)
            a = np.zeros((m+84, 64))
            a[rows, ii], a[rows, jj] = dx/d, -dx/d
            a[rows, 21+ii], a[rows, 21+jj] = dy/d, -dy/d
            a[rows, 42+ii], a[rows, 42+jj] = -1, -1
            a[m:] = -wall
            return a
        def callback(v):
            check()
            last[0] = v.copy()
            consider(circles(v))
        try:
            result = minimize(fun, z, jac=jac, method='SLSQP', bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraints, 'jac': derivative},
                callback=callback, options={'maxiter': 35, 'ftol': 2e-12, 'disp': False})
            last[0] = result.x
        except OutOfTime:
            pass
        consider(circles(last[0]))
        return last[0]

    patterns = [[4,3,4,3,4,3], [4,5,3,5,4], [4,4,5,4,4]]
    for attempt in range(20):
        if time.monotonic() >= stop-0.035:
            break
        try:
            if attempt == 0:
                z = initial.copy()
            elif attempt < len(patterns):
                z = lattice(patterns[attempt])
                z[:42] += rng.normal(0, 0.002, 42)
            elif attempt == 3 and init is not None:
                c = decode(init)
                z = vector(c) if c.shape == (21, 3) else vector(best_c)
            else:
                z = vector(best_c)
                amplitude = float(params['perturbation']) * (1.0 if attempt % 3 else 2.0)
                z[:42] += rng.normal(0, amplitude, 42)
            z = lp_optimize(z)
            if time.monotonic() < stop-0.065 and (attempt == 0 or np.sum(z[42:63]) > best_score-0.003):
                polish(z)
        except (OutOfTime, ValueError, FloatingPointError):
            break
    return best_obj if best_obj is not None else init
