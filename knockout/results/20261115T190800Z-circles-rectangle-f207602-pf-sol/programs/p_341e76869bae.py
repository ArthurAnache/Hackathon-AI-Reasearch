import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'first_iterations': (65, 20, 120),
    'restart_iterations': (40, 10, 90),
    'perturbation': (0.012, 0.001, 0.05, 'log'),
}


def construct(params: dict, seed: int, budget_s: float, init=None):
    started = time.monotonic()
    deadline = started + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    ar = np.arange(n)
    best = None
    best_score = -float('inf')
    best_array = None

    def exact_score(c):
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return -float('inf')
        dx = c[ii, 0] - c[jj, 0]
        dy = c[ii, 1] - c[jj, 1]
        if np.any(np.sqrt(dx * dx + dy * dy) < c[ii, 2] + c[jj, 2]):
            return -float('inf')
        xs = list(map(Fraction, map(float, c[:, 0])))
        ys = list(map(Fraction, map(float, c[:, 1])))
        rs = list(map(Fraction, map(float, c[:, 2])))
        w = max(map(lambda a, b: a + b, xs, rs)) - min(map(lambda a, b: a - b, xs, rs))
        h = max(map(lambda a, b: a + b, ys, rs)) - min(map(lambda a, b: a - b, ys, rs))
        if w + h > 2:
            return -float('inf')
        return float(np.sum(c[:, 2]))

    def accept(c, original=None):
        nonlocal best, best_score, best_array
        s = exact_score(c)
        if s > best_score:
            best_score = s
            best_array = c.copy()
            best = original if original is not None else {'circles': c.tolist()}

    def repair(c):
        if c.shape != (21, 3) or not np.isfinite(c).all():
            return
        c = c.copy()
        c[:, 2] = np.maximum(c[:, 2], 1e-10)
        d = np.sqrt(np.sum((c[ii, :2] - c[jj, :2]) ** 2, axis=1))
        f = min(1.0, float(np.min(d / (c[ii, 2] + c[jj, 2]))))
        if f <= 0:
            return
        c[:, 2] *= f * (1.0 - 3e-14)
        lo = np.min(c[:, :2] - c[:, 2, None], axis=0)
        hi = np.max(c[:, :2] + c[:, 2, None], axis=0)
        perimeter = float(np.sum(hi - lo))
        if perimeter <= 0:
            return
        c[:, :2] -= lo
        c *= (2.0 / perimeter) * (1.0 - 3e-14)
        accept(c)

    if init is not None:
        try:
            initial = np.asarray(init['circles'], dtype=float)
            accept(initial, init)
            repair(initial)
        except (ValueError, TypeError, KeyError, OverflowError):
            pass
    if best_array is None:
        grid = np.empty((21, 3))
        grid[:, 0] = (ar % 7) * 0.2 + 0.1
        grid[:, 1] = (ar // 7) * 0.2 + 0.1
        grid[:, 2] = 0.1
        repair(grid)

    def encode(c):
        c = c.copy()
        lo = np.min(c[:, :2] - c[:, 2, None], axis=0)
        hi = np.max(c[:, :2] + c[:, 2, None], axis=0)
        scale = 2.0 / np.sum(hi - lo)
        c[:, :2] -= lo
        c *= scale
        return np.concatenate((c[:, 0], c[:, 1], c[:, 2], [(hi[0] - lo[0]) * scale]))

    def decode(z):
        return np.column_stack((z[:21], z[21:42], z[42:63]))

    obj_jac = np.zeros(64)
    obj_jac[42:63] = -1.0
    base_jac = np.zeros((294, 64))
    rows = np.arange(210)
    base_jac[rows, 42 + ii] = -1
    base_jac[rows, 42 + jj] = -1
    base_jac[210 + ar, ar] = 1
    base_jac[210 + ar, 42 + ar] = -1
    base_jac[231 + ar, ar] = -1
    base_jac[231 + ar, 42 + ar] = -1
    base_jac[231 + ar, 63] = 1
    base_jac[252 + ar, 21 + ar] = 1
    base_jac[252 + ar, 42 + ar] = -1
    base_jac[273 + ar, 21 + ar] = -1
    base_jac[273 + ar, 42 + ar] = -1
    base_jac[273 + ar, 63] = -1

    class StopSearch(Exception):
        pass

    last = None
    attempts = 0
    bounds = [(None, None)] * 42 + [(1e-7, 0.6)] * 21 + [(0.2, 1.8)]
    while time.monotonic() < deadline:
        z0 = encode(best_array)
        if attempts:
            amplitude = float(params['perturbation']) * (1.0 if attempts % 3 else 2.5)
            z0[:42] += rng.normal(0, amplitude, 42)
            z0[42:63] *= np.exp(rng.normal(0, amplitude * 2, 21))
            z0[63] = np.clip(z0[63] + rng.normal(0, amplitude), 0.25, 1.75)
        cache_z = None
        cache_c = None
        cache_j = None
        iteration = 0
        last = z0.copy()

        def geometry(z):
            nonlocal cache_z, cache_c, cache_j
            if time.monotonic() >= deadline:
                raise StopSearch
            if cache_z is not None and np.array_equal(cache_z, z):
                return cache_c, cache_j
            x, y, r, w = z[:21], z[21:42], z[42:63], z[63]
            dx = x[ii] - x[jj]
            dy = y[ii] - y[jj]
            dist = np.maximum(np.sqrt(dx * dx + dy * dy), 1e-15)
            c = np.concatenate((dist - r[ii] - r[jj], x-r, w-x-r, y-r, 2-w-y-r))
            j = base_jac.copy()
            j[rows, ii] = dx / dist
            j[rows, jj] = -dx / dist
            j[rows, 21 + ii] = dy / dist
            j[rows, 21 + jj] = -dy / dist
            cache_z, cache_c, cache_j = z.copy(), c, j
            return c, j

        def callback(z):
            nonlocal last, iteration
            last = z.copy()
            iteration += 1
            if time.monotonic() >= deadline:
                raise StopSearch
            if iteration % 5 == 0:
                repair(decode(z))

        try:
            result = minimize(lambda z: -np.sum(z[42:63]), z0,
                              jac=lambda z: obj_jac, method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': lambda z: geometry(z)[0],
                                           'jac': lambda z: geometry(z)[1]},
                              callback=callback,
                              options={'maxiter': int(params['first_iterations'] if attempts == 0 else params['restart_iterations']),
                                       'ftol': 2e-12, 'disp': False})
            repair(decode(result.x))
        except StopSearch:
            repair(decode(last))
            break
        attempts += 1
    return best
