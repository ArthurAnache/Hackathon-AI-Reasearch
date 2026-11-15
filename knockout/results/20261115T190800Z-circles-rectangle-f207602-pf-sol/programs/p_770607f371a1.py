import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'first_fraction': (0.43, 0.25, 0.55),
    'restart_noise': (0.018, 0.001, 0.08, 'log'),
    'iterations': (160, 40, 300),
}


def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + 0.69 * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    rows = np.arange(len(ii))
    best = init
    best_score = -float('inf')
    best_array = None

    def exact_score(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        delta = a[ii, :2] - a[jj, :2]
        distances = np.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2)
        if np.any(distances < a[ii, 2] + a[jj, 2]):
            return -float('inf')
        xs = list(map(Fraction, map(float, a[:, 0])))
        ys = list(map(Fraction, map(float, a[:, 1])))
        rs = list(map(Fraction, map(float, a[:, 2])))
        width = max(map(lambda p: p[0] + p[1], zip(xs, rs))) - min(map(lambda p: p[0] - p[1], zip(xs, rs)))
        height = max(map(lambda p: p[0] + p[1], zip(ys, rs))) - min(map(lambda p: p[0] - p[1], zip(ys, rs)))
        if width + height > 2:
            return -float('inf')
        return float(np.sum(a[:, 2]))

    if init is not None:
        try:
            a0 = np.asarray(init['circles'], dtype=float).copy()
            best_score = exact_score(a0)
            if np.isfinite(best_score):
                best_array = a0.copy()
        except (ValueError, KeyError, TypeError, OverflowError):
            a0 = None
    else:
        a0 = None

    def offer(a):
        nonlocal best, best_score, best_array
        if time.monotonic() >= deadline:
            return
        a = np.asarray(a, dtype=float).copy()
        if a.shape != (21, 3) or not np.isfinite(a).all():
            return
        a[:, 2] = np.maximum(a[:, 2], 1e-8)
        dxy = a[ii, :2] - a[jj, :2]
        d = np.sqrt(dxy[:, 0] ** 2 + dxy[:, 1] ** 2)
        ratio = np.min(d / (a[ii, 2] + a[jj, 2]))
        if ratio <= 0:
            return
        a[:, 2] *= min(1.0, ratio) * (1.0 - 3e-12)
        lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
        hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
        perimeter = np.sum(hi - lo)
        if perimeter <= 0:
            return
        a[:, :2] -= lo
        a *= (2.0 / perimeter) * (1.0 - 3e-12)
        estimate = float(np.sum(a[:, 2]))
        if estimate <= best_score:
            return
        score = exact_score(a)
        if score > best_score:
            best_score = score
            best_array = a.copy()
            best = {'circles': a.tolist()}

    grid = np.array([[0.1 + 0.2*x, 0.1 + 0.2*y, 0.1]
                     for y in range(5) for x in range(5)
                     if (x, y) not in ((0, 0), (0, 4), (4, 0), (4, 4))], dtype=float)
    offer(grid)
    if a0 is not None and a0.shape == (21, 3) and np.isfinite(a0).all():
        offer(a0)
    else:
        a0 = grid.copy()

    # z = (all x, all y, all radii, rectangle width), height = 2-width.
    B = np.zeros((84, 64))
    k = np.arange(n)
    B[k, k] = 1
    B[k, 42+k] = -1
    B[21+k, k] = -1
    B[21+k, 42+k] = -1
    B[21+k, 63] = 1
    B[42+k, 21+k] = 1
    B[42+k, 42+k] = -1
    B[63+k, 21+k] = -1
    B[63+k, 42+k] = -1
    B[63+k, 63] = -1
    bc = np.zeros(84)
    bc[63:] = 2
    gradient = np.zeros(64)
    gradient[42:63] = -1
    bounds = [(None, None)] * 42 + [(1e-7, 0.6)] * 21 + [(0.15, 1.85)]

    class StopSearch(Exception):
        pass

    run_deadline = deadline

    def check():
        if time.monotonic() >= run_deadline:
            raise StopSearch()

    def unpack(z):
        return np.column_stack((z[:21], z[21:42], z[42:63]))

    def constraints(z):
        check()
        dx = z[ii] - z[jj]
        dy = z[21+ii] - z[21+jj]
        distance = np.sqrt(dx*dx + dy*dy)
        return np.concatenate((B @ z + bc, distance - z[42+ii] - z[42+jj]))

    def jacobian(z):
        check()
        dx = z[ii] - z[jj]
        dy = z[21+ii] - z[21+jj]
        inverse = 1.0 / np.maximum(np.sqrt(dx*dx + dy*dy), 1e-15)
        J = np.zeros((len(ii), 64))
        J[rows, ii] = dx * inverse
        J[rows, jj] = -dx * inverse
        J[rows, 21+ii] = dy * inverse
        J[rows, 21+jj] = -dy * inverse
        J[rows, 42+ii] = -1
        J[rows, 42+jj] = -1
        return np.vstack((B, J))

    def objective(z):
        check()
        return -float(np.sum(z[42:63]))

    def callback(z):
        check()
        offer(unpack(z))

    def optimize(a, seconds):
        nonlocal run_deadline
        if time.monotonic() >= deadline - 0.015:
            return
        run_deadline = min(deadline, time.monotonic() + seconds)
        a = a.copy()
        lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
        hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
        dimensions = hi - lo
        scale = 2 / max(float(np.sum(dimensions)), 1e-12)
        a[:, :2] = (a[:, :2] - lo) * scale
        a[:, 2] = np.clip(a[:, 2] * scale, 1e-6, 0.59)
        z = np.concatenate((a[:, 0], a[:, 1], a[:, 2], [np.clip(dimensions[0]*scale, 0.15, 1.85)]))
        try:
            result = minimize(objective, z, jac=lambda z: gradient,
                              constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                              bounds=bounds, method='SLSQP', callback=callback,
                              options={'maxiter': int(params['iterations']), 'ftol': 2e-12, 'disp': False})
            offer(unpack(result.x))
        except StopSearch:
            pass
        except (ValueError, FloatingPointError):
            pass

    optimize(a0, float(params['first_fraction']) * budget_s)
    attempt = 0
    while time.monotonic() < deadline - 0.035:
        if best_score < 2.33 and attempt == 0:
            trial = grid.copy()
        elif best_array is not None:
            trial = best_array.copy()
            amplitude = float(params['restart_noise']) * (1 + 0.5 * (attempt % 3))
            trial[:, :2] += rng.normal(0, amplitude, (21, 2))
            trial[:, 2] *= rng.uniform(0.96, 1.02, 21)
        else:
            trial = grid.copy()
        optimize(trial, min(0.31 * budget_s, deadline - time.monotonic()))
        attempt += 1
    return best
