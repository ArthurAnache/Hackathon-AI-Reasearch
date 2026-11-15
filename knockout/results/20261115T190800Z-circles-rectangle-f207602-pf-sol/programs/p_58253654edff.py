import time
import numpy as np
from fractions import Fraction
from scipy.optimize import minimize

PARAMS = {
    'fraction': (0.70, 0.5, 0.75),
    'ftol': (2e-12, 1e-14, 1e-7, 'log'),
    'restart_noise': (0.007, 0.001, 0.04, 'log')
}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + params['fraction'] * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    rows = np.arange(m)
    best = init
    best_score = -float('inf')

    def check_time():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def exact_score(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return -float('inf')
        dx = a[ii, 0] - a[jj, 0]
        dy = a[ii, 1] - a[jj, 1]
        if np.any(np.sqrt(dx * dx + dy * dy) < a[ii, 2] + a[jj, 2]):
            return -float('inf')
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        width = max(u + v for u, v in zip(x, r)) - min(u - v for u, v in zip(x, r))
        height = max(u + v for u, v in zip(y, r)) - min(u - v for u, v in zip(y, r))
        if width + height > 2:
            return -float('inf')
        return float(np.sum(a[:, 2]))

    def keep(a, repair=True):
        nonlocal best, best_score
        if not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return
        a = a.copy()
        if repair:
            distances = np.sqrt(np.sum((a[ii, :2] - a[jj, :2]) ** 2, axis=1))
            factor = min(1.0, float(np.min(distances / (a[ii, 2] + a[jj, 2]))))
            if factor <= 0:
                return
            a[:, 2] *= factor * (1.0 - 4e-14)
            lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
            hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
            perimeter = float(np.sum(hi - lo))
            if perimeter <= 0:
                return
            a[:, :2] -= (hi + lo) / 2
            a *= (2.0 / perimeter) * (1.0 - 4e-14)
        score = exact_score(a)
        if score > best_score:
            best_score = score
            best = {'circles': a.tolist()}

    if init is not None:
        if 'circles' in init:
            initial = np.asarray(init['circles'], dtype=float).copy()
        else:
            v = np.asarray(init['state'], dtype=float)
            initial = np.column_stack((v[:21], v[21:42], v[42:63]))
        best_score = exact_score(initial)
        keep(initial)
    else:
        initial = np.zeros((21, 3))
        for k in range(21):
            check_time()
            initial[k] = [(k % 5) * 0.2, (k // 5) * 0.2, 0.09]
        keep(initial)

    lo = np.min(initial[:, :2] - initial[:, 2, None], axis=0)
    hi = np.max(initial[:, :2] + initial[:, 2, None], axis=0)
    initial[:, :2] -= (lo + hi) / 2
    scale = 2.0 / float(np.sum(hi - lo))
    initial *= scale
    z = np.r_[initial[:, 0], initial[:, 1], initial[:, 2], (hi[0] - lo[0]) * scale]
    z[42:63] = np.maximum(z[42:63], 1e-7)

    wall_jac = np.zeros((84, 64))
    q = np.arange(21)
    wall_jac[q, q] = 1
    wall_jac[q, q + 42] = -1
    wall_jac[q, 63] = 0.5
    wall_jac[q + 21, q] = -1
    wall_jac[q + 21, q + 42] = -1
    wall_jac[q + 21, 63] = 0.5
    wall_jac[q + 42, q + 21] = 1
    wall_jac[q + 42, q + 42] = -1
    wall_jac[q + 42, 63] = -0.5
    wall_jac[q + 63, q + 21] = -1
    wall_jac[q + 63, q + 42] = -1
    wall_jac[q + 63, 63] = -0.5
    objective_jac = np.zeros(64)
    objective_jac[42:63] = -1

    def objective(v):
        check_time()
        return -float(np.sum(v[42:63]))

    def constraints(v):
        check_time()
        x, y, r, w = v[:21], v[21:42], v[42:63], v[63]
        dx, dy, rr = x[ii] - x[jj], y[ii] - y[jj], r[ii] + r[jj]
        return np.concatenate((dx * dx + dy * dy - rr * rr,
                               x - r + w / 2, -x - r + w / 2,
                               y - r + 1 - w / 2, -y - r + 1 - w / 2))

    def jacobian(v):
        check_time()
        x, y, r = v[:21], v[21:42], v[42:63]
        dx, dy, rr = 2 * (x[ii] - x[jj]), 2 * (y[ii] - y[jj]), -2 * (r[ii] + r[jj])
        jac = np.zeros((m + 84, 64))
        jac[rows, ii] = dx
        jac[rows, jj] = -dx
        jac[rows, ii + 21] = dy
        jac[rows, jj + 21] = -dy
        jac[rows, ii + 42] = rr
        jac[rows, jj + 42] = rr
        jac[m:] = wall_jac
        return jac

    last = z.copy()
    def callback(v):
        nonlocal last
        check_time()
        last = v.copy()
        keep(np.column_stack((v[:21], v[21:42], v[42:63])))

    bounds = [(None, None)] * 42 + [(1e-7, 0.5)] * 21 + [(0.05, 1.95)]
    try:
        for attempt in range(4):
            check_time()
            result = minimize(objective, z, method='SLSQP', jac=lambda v: objective_jac,
                              bounds=bounds,
                              constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                              callback=callback,
                              options={'maxiter': 350, 'ftol': params['ftol'], 'disp': False})
            check_time()
            callback(result.x)
            if time.monotonic() - started > 0.5 * params['fraction'] * budget_s:
                break
            if best is not None and 'circles' in best:
                a = np.asarray(best['circles'])
                width = np.max(a[:, 0] + a[:, 2]) - np.min(a[:, 0] - a[:, 2])
                z = np.r_[a[:, 0], a[:, 1], a[:, 2], width]
            else:
                z = last.copy()
            z[:42] += rng.normal(0, params['restart_noise'], 42)
    except StopSearch:
        pass
    return best
