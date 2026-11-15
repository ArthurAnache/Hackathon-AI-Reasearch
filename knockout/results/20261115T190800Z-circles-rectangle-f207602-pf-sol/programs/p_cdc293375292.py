import time
import numpy as np
from fractions import Fraction
from scipy.optimize import minimize

PARAMS = {
    'active_tolerance': (1e-7, 1e-9, 1e-4, 'log'),
    'polish_iterations': (110, 20, 200),
    'perturbation': (0.008, 0.0001, 0.08, 'log')
}

class OutOfTime(Exception):
    pass


def exact_score(data):
    try:
        a = np.asarray(data['circles'], dtype=np.float64)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        xs = [Fraction(float(x)) for x in a[:, 0]]
        ys = [Fraction(float(y)) for y in a[:, 1]]
        rs = [Fraction(float(r)) for r in a[:, 2]]
        p = (max(x+r for x, r in zip(xs, rs)) - min(x-r for x, r in zip(xs, rs))
             + max(y+r for y, r in zip(ys, rs)) - min(y-r for y, r in zip(ys, rs)))
        if p > 2:
            return -float('inf')
        for i in range(21):
            for j in range(i):
                d = np.sqrt((a[i, 0]-a[j, 0])**2 + (a[i, 1]-a[j, 1])**2)
                if d < a[i, 2]+a[j, 2]:
                    return -float('inf')
        return float(np.sum(a[:, 2]))
    except (ValueError, KeyError, OverflowError):
        return -float('inf')


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    best = init
    best_score = exact_score(init) if init is not None else -float('inf')

    def check():
        if time.monotonic() >= deadline:
            raise OutOfTime

    def perimeter(a):
        xs = [Fraction(float(x)) for x in a[:, 0]]
        ys = [Fraction(float(y)) for y in a[:, 1]]
        rs = [Fraction(float(r)) for r in a[:, 2]]
        return (max(x+r for x, r in zip(xs, rs)) - min(x-r for x, r in zip(xs, rs))
                + max(y+r for y, r in zip(ys, rs)) - min(y-r for y, r in zip(ys, rs)))

    def retain(a):
        nonlocal best, best_score
        check()
        a = np.array(a, dtype=float, copy=True)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        a[:, :2] -= (lo+hi)*0.5
        for k in range(5):
            check()
            delta = a[ii, :2]-a[jj, :2]
            dist = np.sqrt(delta[:, 0]**2+delta[:, 1]**2)
            sums = a[ii, 2]+a[jj, 2]
            ratio = float(np.min(dist/sums))
            if ratio < 1.0:
                a[:, 2] *= np.nextafter(ratio, 0.0)
            p = perimeter(a)
            factor = np.nextafter(np.nextafter(2.0/float(p), 0.0), 0.0)
            a *= factor
            if perimeter(a) > 2:
                a *= 1.0-8e-16
            candidate = {'circles': a.tolist()}
            s = exact_score(candidate)
            if s > best_score:
                best_score, best = s, candidate
            if s > -float('inf'):
                return
            a[:, 2] = np.nextafter(a[:, 2], 0.0)

    if init is None:
        a = np.array([[(i % 7 + 0.5)*0.2, (i // 7 + 0.5)*0.2, 0.1] for i in range(21)])
        try:
            retain(a)
        except OutOfTime:
            return best if best is not None else {'circles': a.tolist()}
    else:
        a = np.array(init['circles'], dtype=float)

    low = np.min(a[:, :2]-a[:, 2, None], axis=0)
    high = np.max(a[:, :2]+a[:, 2, None], axis=0)
    a[:, :2] -= (low+high)*0.5
    dimensions = high-low
    z0 = np.r_[a.ravel(), dimensions]
    pair_count = len(ii)
    ncon = pair_count+84
    row = np.arange(pair_count)
    wallrows = pair_count+np.arange(21)*4
    objective = np.zeros(65)
    objective[2:63:3] = -1.0

    def constraints(z):
        check()
        c = z[:63].reshape(21, 3)
        d = c[ii, :2]-c[jj, :2]
        distance = np.sqrt(np.sum(d*d, axis=1))
        walls = np.empty((21, 4))
        walls[:, 0] = z[63]*0.5+c[:, 0]-c[:, 2]
        walls[:, 1] = z[63]*0.5-c[:, 0]-c[:, 2]
        walls[:, 2] = z[64]*0.5+c[:, 1]-c[:, 2]
        walls[:, 3] = z[64]*0.5-c[:, 1]-c[:, 2]
        return np.r_[distance-c[ii, 2]-c[jj, 2], walls.ravel()]

    def jacobian(z):
        check()
        c = z[:63].reshape(21, 3)
        d = c[ii, :2]-c[jj, :2]
        length = np.maximum(np.sqrt(np.sum(d*d, axis=1)), 1e-100)
        u = d/length[:, None]
        j = np.zeros((ncon, 65))
        j[row, 3*ii] = u[:, 0]
        j[row, 3*ii+1] = u[:, 1]
        j[row, 3*jj] = -u[:, 0]
        j[row, 3*jj+1] = -u[:, 1]
        j[row, 3*ii+2] = -1
        j[row, 3*jj+2] = -1
        ids = np.arange(21)
        for k in range(4):
            check()
            rr = wallrows+k
            j[rr, 3*ids+2] = -1
            j[rr, 63+(k//2)] = 0.5
            j[rr, 3*ids+(k//2)] = 1 if k % 2 == 0 else -1
        return j

    equality_jac = np.zeros(65)
    equality_jac[63:] = 1

    def polish_contacts(z, tolerance):
        z = z.copy()
        active = np.flatnonzero(constraints(z) < tolerance)
        if len(active) < 58:
            return z
        for iteration in range(9):
            check()
            f = np.r_[constraints(z)[active], z[63]+z[64]-2]
            j = np.vstack((jacobian(z)[active], equality_jac))
            step = np.linalg.lstsq(j, -f, rcond=1e-12)[0]
            if np.max(np.abs(step)) > 0.015:
                break
            z += step
            if iteration >= 1:
                retain(z[:63].reshape(21, 3))
            if np.max(np.abs(step)) < 2e-16:
                break
        return z

    latest = z0.copy()
    try:
        retain(a)
        latest = polish_contacts(z0, params['active_tolerance'])
        retain(latest[:63].reshape(21, 3))

        def fun(z):
            check()
            return float(objective @ z)

        def callback(z):
            nonlocal latest
            check()
            latest = z.copy()
            if np.min(constraints(z)) > -2e-8:
                retain(z[:63].reshape(21, 3))

        cons = [
            {'type': 'ineq', 'fun': constraints, 'jac': jacobian},
            {'type': 'eq', 'fun': lambda z: z[63]+z[64]-2,
             'jac': lambda z: equality_jac}
        ]
        bounds = []
        for i in range(21):
            check()
            bounds.extend([(None, None), (None, None), (1e-7, 1.0)])
        bounds.extend([(0.05, 1.95), (0.05, 1.95)])
        res = minimize(fun, latest, jac=lambda z: objective, method='SLSQP',
                       bounds=bounds, constraints=cons, callback=callback,
                       options={'ftol': 2e-14, 'maxiter': int(params['polish_iterations'])})
        latest = res.x.copy()
        retain(latest[:63].reshape(21, 3))
        latest = polish_contacts(latest, 3e-7)
        retain(latest[:63].reshape(21, 3))

        # Use otherwise idle time for unrestricted, very small basin tests.
        for attempt in range(8):
            check()
            if deadline-time.monotonic() < 0.12:
                break
            base = latest.copy()
            magnitude = params['perturbation']*(1+attempt*0.5)
            base[:63] += rng.normal(0, magnitude, 63)
            base[2:63:3] = np.maximum(base[2:63:3], 0.02)
            res = minimize(fun, base, jac=lambda z: objective, method='SLSQP',
                           bounds=bounds, constraints=cons, callback=callback,
                           options={'ftol': 1e-13, 'maxiter': 45})
            retain(res.x[:63].reshape(21, 3))
            polish_contacts(res.x, 1e-7)
    except (OutOfTime, np.linalg.LinAlgError, ValueError, FloatingPointError):
        pass
    return best
