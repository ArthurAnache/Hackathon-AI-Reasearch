import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"iterations": (350, 100, 700), "restart_scale": (0.004, 0.001, 0.025)}


def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + 0.69 * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    rows = np.arange(m)

    def read(obj):
        if obj is None:
            return None
        try:
            if 'circles' in obj:
                a = np.asarray(obj['circles'], dtype=float)
            else:
                a = np.asarray(obj['state'], dtype=float).reshape(3, n).T
            if a.shape == (n, 3) and np.isfinite(a).all() and np.all(a[:, 2] > 0):
                return a.copy()
        except Exception:
            pass
        return None

    def exact_score(a):
        if a is None or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -np.inf
        d = a[ii, :2] - a[jj, :2]
        if np.any(np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2) < a[ii, 2] + a[jj, 2]):
            return -np.inf
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        extent = max(u + v for u, v in zip(x, r)) - min(u - v for u, v in zip(x, r))
        extent += max(u + v for u, v in zip(y, r)) - min(u - v for u, v in zip(y, r))
        if extent > 2:
            return -np.inf
        return float(np.sum(a[:, 2]))

    original = read(init)
    best = original
    best_score = exact_score(original)

    def repair(a):
        a = a.copy()
        if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return None
        lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
        hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
        a[:, :2] -= (lo + hi) / 2
        delta = a[ii, :2] - a[jj, :2]
        distances = np.sqrt(np.sum(delta * delta, axis=1))
        ratio = float(np.min(distances / (a[ii, 2] + a[jj, 2])))
        a[:, 2] *= min(1.0, ratio) * (1.0 - 3e-14)
        if np.any(a[:, 2] <= 0):
            return None
        lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
        hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
        perimeter = float(np.sum(hi - lo))
        if perimeter <= 0:
            return None
        a *= (2.0 / perimeter) * (1.0 - 3e-14)
        return a

    def consider(a):
        nonlocal best, best_score
        b = repair(a)
        s = exact_score(b)
        if s > best_score:
            best, best_score = b, s

    grid = np.array([[(i - 1) * .2, (j - 3) * .2, .1] for j in range(7) for i in range(3)])
    consider(grid)
    if original is not None:
        consider(original)
    start = repair(original) if original is not None else repair(grid)
    if start is None:
        start = best.copy()

    objective_gradient = np.zeros(64)
    objective_gradient[42:63] = -1.0
    fixed_jac = np.zeros((m + 84, 64))
    k = np.arange(n)
    fixed_jac[m + k, k] = -1
    fixed_jac[m + k, 42 + k] = -1
    fixed_jac[m + k, 63] = .5
    fixed_jac[m + 21 + k, k] = 1
    fixed_jac[m + 21 + k, 42 + k] = -1
    fixed_jac[m + 21 + k, 63] = .5
    fixed_jac[m + 42 + k, 21 + k] = -1
    fixed_jac[m + 42 + k, 42 + k] = -1
    fixed_jac[m + 42 + k, 63] = -.5
    fixed_jac[m + 63 + k, 21 + k] = 1
    fixed_jac[m + 63 + k, 42 + k] = -1
    fixed_jac[m + 63 + k, 63] = -.5

    class Stop(Exception):
        pass

    def check():
        if time.monotonic() >= deadline:
            raise Stop()

    def fun(z):
        check()
        return -float(np.sum(z[42:63]))

    def cons(z):
        check()
        x, y, r, w = z[:21], z[21:42], z[42:63], z[63]
        dx, dy, rr = x[ii] - x[jj], y[ii] - y[jj], r[ii] + r[jj]
        return np.concatenate(((dx * dx + dy * dy - rr * rr) * 4.0,
                               w / 2 - x - r, w / 2 + x - r,
                               1 - w / 2 - y - r, 1 - w / 2 + y - r))

    def jac(z):
        check()
        x, y, r = z[:21], z[21:42], z[42:63]
        dx, dy, rr = 8 * (x[ii] - x[jj]), 8 * (y[ii] - y[jj]), -8 * (r[ii] + r[jj])
        a = fixed_jac.copy()
        a[rows, ii] = dx
        a[rows, jj] = -dx
        a[rows, 21 + ii] = dy
        a[rows, 21 + jj] = -dy
        a[rows, 42 + ii] = rr
        a[rows, 42 + jj] = rr
        return a

    calls = 0
    last = None

    def callback(z):
        nonlocal calls, last
        check()
        last = z.copy()
        calls += 1
        if calls % 8 == 0:
            consider(z[:63].reshape(3, 21).T)

    bounds = [(-1.0, 1.0)] * 42 + [(1e-7, .5)] * 21 + [(.02, 1.98)]
    attempt = 0
    while time.monotonic() < deadline:
        a = start.copy() if attempt == 0 else best.copy()
        if attempt:
            a[:, :2] += rng.normal(0, params['restart_scale'], (21, 2))
        lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
        hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
        a[:, :2] -= (lo + hi) / 2
        w = np.clip((hi[0] - lo[0]) + (2 - np.sum(hi - lo)) / 2, .02, 1.98)
        z = np.concatenate((a.T.ravel(), [w]))
        last = None
        try:
            result = minimize(fun, z, jac=lambda z: objective_gradient,
                              bounds=bounds, constraints={'type': 'ineq', 'fun': cons, 'jac': jac},
                              method='SLSQP', callback=callback,
                              options={'maxiter': int(params['iterations']), 'ftol': 2e-12, 'disp': False})
            consider(result.x[:63].reshape(3, 21).T)
        except Stop:
            if last is not None:
                consider(last[:63].reshape(3, 21).T)
            break
        except (ValueError, FloatingPointError):
            break
        attempt += 1
    if best is original and init is not None:
        return init
    return {'circles': best.tolist()}
