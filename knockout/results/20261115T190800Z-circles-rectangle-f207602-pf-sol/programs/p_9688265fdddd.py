import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import lstsq

PARAMS = {
    'active_tol': (2e-7, 1e-10, 1e-4, 'log'),
    'ftol': (2e-14, 1e-15, 1e-10, 'log'),
    'perturb': (0.012, 0.001, 0.08, 'log')
}


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    rows = np.arange(m)

    class Stop(Exception):
        pass

    def check():
        if time.monotonic() >= deadline:
            raise Stop()

    def array_of(obj):
        if 'circles' in obj:
            return np.asarray(obj['circles'], dtype=float).copy()
        a = np.asarray(obj['state'], dtype=float)
        return np.column_stack((a[:21], a[21:42], a[42:63]))

    def exact_score(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        xs = list(map(lambda x: Fraction(float(x)), a[:, 0]))
        ys = list(map(lambda x: Fraction(float(x)), a[:, 1]))
        rs = list(map(lambda x: Fraction(float(x)), a[:, 2]))
        width = max(x+r for x, r in zip(xs, rs)) - min(x-r for x, r in zip(xs, rs))
        height = max(y+r for y, r in zip(ys, rs)) - min(y-r for y, r in zip(ys, rs))
        if width + height > 2:
            return -float('inf')
        for i, j in zip(ii, jj):
            time.monotonic()
            dx = a[i, 0] - a[j, 0]
            dy = a[i, 1] - a[j, 1]
            if np.sqrt(dx**2 + dy**2) < a[i, 2] + a[j, 2]:
                return -float('inf')
        return float(np.sum(a[:, 2]))

    if init is None:
        a = np.array([[0.2*i, 0.2*j, 0.09999999999999]
                      for j in range(7) for i in range(3)], dtype=float)
        original = {'circles': a.tolist()}
    else:
        original = init
        a = array_of(init)
    best_obj = original
    best_score = exact_score(a)
    best_a = a.copy()

    def repair(a):
        a = np.asarray(a, dtype=float).copy()
        if not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return None
        dx = a[ii, 0] - a[jj, 0]
        dy = a[ii, 1] - a[jj, 1]
        ratios = np.sqrt(dx*dx + dy*dy) / (a[ii, 2] + a[jj, 2])
        factor = min(1.0, float(np.min(ratios)))
        a[:, 2] *= factor * (1.0 - 5e-16)
        for k in range(4):
            time.monotonic()
            xs = [Fraction(float(v)) for v in a[:, 0]]
            ys = [Fraction(float(v)) for v in a[:, 1]]
            rs = [Fraction(float(v)) for v in a[:, 2]]
            per = (max(x+r for x, r in zip(xs, rs)) - min(x-r for x, r in zip(xs, rs))
                   + max(y+r for y, r in zip(ys, rs)) - min(y-r for y, r in zip(ys, rs)))
            if per > 2:
                a *= float(Fraction(2, 1) / per) * (1.0 - 5e-16)
            if exact_score(a) > -float('inf'):
                return a
            a[:, 2] *= 1.0 - 2e-15
        return None

    def consider(z):
        nonlocal best_score, best_obj, best_a
        b = np.column_stack((z[:21], z[21:42], z[42:63]))
        b = repair(b)
        if b is not None:
            s = exact_score(b)
            if s > best_score:
                best_score = s
                best_a = b.copy()
                best_obj = {'circles': b.tolist()}

    def pack(b):
        b = b.copy()
        lo = np.min(b[:, :2] - b[:, 2, None], axis=0)
        hi = np.max(b[:, :2] + b[:, 2, None], axis=0)
        b[:, :2] -= (lo + hi) / 2
        w = (hi[0] - lo[0]) + (2 - (hi[0]-lo[0]) - (hi[1]-lo[1])) / 2
        return np.r_[b[:, 0], b[:, 1], b[:, 2], w]

    def values(z):
        x, y, r, w = z[:21], z[21:42], z[42:63], z[63]
        dx, dy = x[ii]-x[jj], y[ii]-y[jj]
        d = np.sqrt(dx*dx+dy*dy)
        return np.r_[d-r[ii]-r[jj], w/2-x-r, w/2+x-r,
                     1-w/2-y-r, 1-w/2+y-r]

    def jacobian(z):
        x, y = z[:21], z[21:42]
        dx, dy = x[ii]-x[jj], y[ii]-y[jj]
        d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-100)
        J = np.zeros((m+84, 64))
        J[rows, ii] = dx/d
        J[rows, jj] = -dx/d
        J[rows, 21+ii] = dy/d
        J[rows, 21+jj] = -dy/d
        J[rows, 42+ii] = -1
        J[rows, 42+jj] = -1
        k = np.arange(21)
        J[m+k, k] = -1
        J[m+21+k, k] = 1
        J[m+42+k, 21+k] = -1
        J[m+63+k, 21+k] = 1
        J[m+k, 42+k] = -1
        J[m+21+k, 42+k] = -1
        J[m+42+k, 42+k] = -1
        J[m+63+k, 42+k] = -1
        J[m:m+42, 63] = 0.5
        J[m+42:, 63] = -0.5
        return J

    def contact_polish(z):
        z = z.copy()
        active = np.flatnonzero(values(z) < params['active_tol'])
        for iteration in range(12):
            check()
            v = values(z)[active]
            J = jacobian(z)[active]
            step = lstsq(J, -v, cond=1e-12, lapack_driver='gelsy')[0]
            if not np.isfinite(step).all() or np.linalg.norm(step) > 0.05:
                break
            z += step
            consider(z)
            if np.max(np.abs(step)) < 2e-16:
                break
        return z

    g = np.r_[np.zeros(42), -np.ones(21), 0.0]
    bounds = [(None, None)]*42 + [(1e-7, 0.5)]*21 + [(0.15, 1.85)]

    def solve(z, maxiter):
        check()
        def callback(v):
            check()
        result = minimize(lambda v: -float(np.sum(v[42:63])), z,
                          jac=lambda v: g, method='SLSQP', bounds=bounds,
                          constraints={'type': 'ineq', 'fun': values, 'jac': jacobian},
                          callback=callback,
                          options={'ftol': params['ftol'], 'maxiter': maxiter, 'disp': False})
        consider(result.x)
        return result.x

    try:
        z = pack(a)
        z = contact_polish(z)
        z = solve(z, 90)
        contact_polish(z)
        attempt = 0
        while time.monotonic() < deadline - 0.12:
            check()
            z = pack(best_a)
            # Coordinated, zero-mean position changes explore neighboring contact graphs.
            amp = params['perturb'] * (1.0 + 0.35*(attempt % 3))
            z[:42] += rng.normal(0, amp, 42)
            z[42:63] *= rng.uniform(0.97, 1.01, 21)
            z = solve(z, 65)
            contact_polish(z)
            attempt += 1
    except Stop:
        pass
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        pass
    return best_obj
