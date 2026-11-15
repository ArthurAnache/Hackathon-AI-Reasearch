import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import lstsq

PARAMS = {
    'active_tolerance': (2e-7, 1e-10, 1e-4, 'log'),
    'polish_iterations': (100, 20, 200),
    'perturbation': (0.0003, 0.000001, 0.01, 'log'),
}


class Deadline(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -float('inf')
    ii, jj = np.triu_indices(21, 1)
    ids = np.arange(21)

    def check():
        if time.monotonic() >= end:
            raise Deadline()

    def exact_score(c):
        check()
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return -float('inf')
        d = c[ii, :2] - c[jj, :2]
        dist = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
        if np.any(dist < c[ii, 2] + c[jj, 2]):
            return -float('inf')
        xl, xr, yl, yr = [], [], [], []
        for x, y, r in c:
            check()
            x, y, r = Fraction(float(x)), Fraction(float(y)), Fraction(float(r))
            xl.append(x-r)
            xr.append(x+r)
            yl.append(y-r)
            yr.append(y+r)
        if max(xr)-min(xl)+max(yr)-min(yl) > 2:
            return -float('inf')
        return float(np.sum(c[:, 2]))

    def retain(c):
        nonlocal best, best_score
        check()
        val = exact_score(c)
        if val > best_score:
            best_score = val
            best = {'circles': c.tolist()}

    def repair(z):
        check()
        c = np.column_stack((z[:21], z[21:42], z[42:63])).copy()
        if not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return
        # Translation keeps the arithmetic well-conditioned.
        c[:, :2] -= np.mean(c[:, :2], axis=0)
        delta = c[ii, :2] - c[jj, :2]
        distances = np.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2)
        ratio = float(np.min(distances / (c[ii, 2] + c[jj, 2])))
        c[:, 2] *= min(1.0, ratio) * (1.0 - 5e-16)
        extent = np.ptp(c[:, 0, None] + np.array([-1., 1.])[None, :] * c[:, 2, None])
        extent += np.ptp(c[:, 1, None] + np.array([-1., 1.])[None, :] * c[:, 2, None])
        c *= min(1.0, 2.0 / extent) * (1.0 - 5e-16)
        for _ in range(5):
            check()
            value = exact_score(c)
            if np.isfinite(value):
                retain(c)
                return
            c[:, 2] *= 1.0 - 1e-15

    def encode(c):
        c = np.array(c, dtype=float, copy=True)
        lo = np.min(c[:, :2] - c[:, 2, None], axis=0)
        hi = np.max(c[:, :2] + c[:, 2, None], axis=0)
        c[:, :2] -= lo
        scale = 2.0 / np.sum(hi-lo)
        c *= scale
        return np.r_[c[:, 0], c[:, 1], c[:, 2], (hi[0]-lo[0])*scale]

    def constraints(z):
        check()
        x, y, r, w = z[:21], z[21:42], z[42:63], z[63]
        dx, dy = x[ii]-x[jj], y[ii]-y[jj]
        d = np.sqrt(dx*dx + dy*dy)
        return np.r_[x-r, w-x-r, y-r, 2.0-w-y-r, d-r[ii]-r[jj]]

    wall_jac = np.zeros((84, 64))
    wall_jac[ids, ids] = 1
    wall_jac[ids, 42+ids] = -1
    wall_jac[21+ids, ids] = -1
    wall_jac[21+ids, 42+ids] = -1
    wall_jac[21+ids, 63] = 1
    wall_jac[42+ids, 21+ids] = 1
    wall_jac[42+ids, 42+ids] = -1
    wall_jac[63+ids, 21+ids] = -1
    wall_jac[63+ids, 42+ids] = -1
    wall_jac[63+ids, 63] = -1
    rows = np.arange(len(ii)) + 84

    def jacobian(z):
        check()
        dx, dy = z[ii]-z[jj], z[21+ii]-z[21+jj]
        d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-100)
        a = np.zeros((294, 64))
        a[:84] = wall_jac
        a[rows, ii] = dx/d
        a[rows, jj] = -dx/d
        a[rows, 21+ii] = dy/d
        a[rows, 21+jj] = -dy/d
        a[rows, 42+ii] = -1
        a[rows, 42+jj] = -1
        return a

    gradient = np.r_[np.zeros(42), -np.ones(21), 0.0]

    def objective(z):
        check()
        return -float(np.sum(z[42:63]))

    def callback(z):
        check()
        if np.min(constraints(z)) > -1e-6:
            repair(z)

    def contact_polish(z, tolerance):
        z = z.copy()
        active = np.flatnonzero(constraints(z) < tolerance)
        if len(active) < 55:
            return z
        for _ in range(9):
            check()
            f = constraints(z)[active]
            a = jacobian(z)[active]
            step = lstsq(a, -f, cond=1e-12, lapack_driver='gelsy', check_finite=False)[0]
            if np.max(np.abs(step)) > 0.01:
                break
            z += step
            repair(z)
            if np.max(np.abs(step)) < 2e-16:
                break
        return z

    try:
        if init is not None:
            original = np.array(init['circles'], dtype=float)
            best_score = exact_score(original)
            z0 = encode(original)
        else:
            circles = []
            for row, count in enumerate([4, 5, 4, 5, 3]):
                check()
                for col in range(count):
                    check()
                    circles.append([(col-(count-1)/2)*0.22, row*0.19, 0.105])
            z0 = encode(circles)
        repair(z0)
        for tolerance in [params['active_tolerance'], 1e-8, 1e-6]:
            check()
            contact_polish(z0, tolerance)
        bounds = [(None, None)]*42 + [(1e-7, 0.5)]*21 + [(0.1, 1.9)]
        current = z0.copy()
        attempt = 0
        while time.monotonic() < end:
            check()
            result = minimize(objective, current, jac=lambda z: gradient,
                              method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                              callback=callback,
                              options={'maxiter': int(params['polish_iterations']),
                                       'ftol': 2e-14, 'disp': False})
            check()
            repair(result.x)
            contact_polish(result.x, params['active_tolerance'])
            attempt += 1
            if best is not None:
                current = encode(best['circles'])
            else:
                current = result.x.copy()
            amplitude = params['perturbation'] * (1.0 if attempt % 3 else 8.0)
            current[:42] += rng.normal(0.0, amplitude, 42)
            current[42:63] *= 1.0 - amplitude
    except Deadline:
        pass
    return best if best is not None else init
