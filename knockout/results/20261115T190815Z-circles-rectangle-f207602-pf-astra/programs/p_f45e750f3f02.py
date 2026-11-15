import time
import numpy as np
from fractions import Fraction
from scipy.optimize import minimize

PARAMS = {
    'first_fraction': (0.73, 0.4, 0.9),
    'restart_displacement': (0.012, 0.001, 0.06),
    'maxiter': (130, 30, 250),
    'ftol': (2e-12, 1e-14, 1e-8, 'log')
}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    begun = time.monotonic()
    end = begun + 0.69 * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    idx = np.arange(n)
    rows = np.arange(m)
    best = None
    best_score = -float('inf')

    def exact_score(data):
        try:
            a = np.asarray(data['circles'], dtype=np.float64)
            if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
                return -float('inf')
            dx = a[ii, 0] - a[jj, 0]
            dy = a[ii, 1] - a[jj, 1]
            if np.any(np.sqrt(dx**2 + dy**2) < a[ii, 2] + a[jj, 2]):
                return -float('inf')
            x = [Fraction(float(v)) for v in a[:, 0]]
            y = [Fraction(float(v)) for v in a[:, 1]]
            r = [Fraction(float(v)) for v in a[:, 2]]
            width = max(u + v for u, v in zip(x, r)) - min(u - v for u, v in zip(x, r))
            height = max(u + v for u, v in zip(y, r)) - min(u - v for u, v in zip(y, r))
            if width + height > 2:
                return -float('inf')
            return float(np.sum(a[:, 2]))
        except (ValueError, TypeError, KeyError, OverflowError):
            return -float('inf')

    if init is not None:
        best_score = exact_score(init)
        if np.isfinite(best_score):
            best = init

    def record(a):
        nonlocal best, best_score
        a = np.array(a, dtype=float, copy=True)
        if a.shape != (21, 3) or not np.isfinite(a).all():
            return
        a[:, 2] = np.maximum(a[:, 2], 1e-10)
        a[:, :2] -= (np.max(a[:, :2], axis=0) + np.min(a[:, :2], axis=0)) / 2
        dx = a[ii, 0] - a[jj, 0]
        dy = a[ii, 1] - a[jj, 1]
        distances = np.sqrt(dx**2 + dy**2)
        ratio = float(np.min(distances / (a[ii, 2] + a[jj, 2])))
        if ratio <= 0:
            return
        a[:, 2] *= min(1.0, ratio) * (1 - 4e-14)
        lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
        hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
        extent = float(np.sum(hi - lo))
        if extent <= 0:
            return
        a[:, :2] -= (lo + hi) / 2
        a *= (2.0 / extent) * (1 - 4e-14)
        if float(np.sum(a[:, 2])) <= best_score:
            return
        data = {'circles': a.tolist()}
        value = exact_score(data)
        if value > best_score:
            best_score, best = value, data

    lattice = []
    for row, count in enumerate((5, 4, 5, 4, 3)):
        if time.monotonic() >= end:
            break
        for col in range(count):
            if time.monotonic() >= end:
                break
            lattice.append([col - (count - 1) / 2, row * np.sqrt(3) / 2, 0.5])
    fallback = np.asarray(lattice, dtype=float)
    if fallback.shape == (21, 3):
        record(fallback)
    try:
        initial = np.array(init['circles'], dtype=float)
        if initial.shape != (21, 3) or not np.isfinite(initial).all():
            initial = fallback.copy()
    except (TypeError, KeyError, ValueError):
        initial = fallback.copy()
    if initial.shape != (21, 3):
        return best if best is not None else init
    record(initial)

    constant_jac = np.zeros((m + 84, 64))
    constant_jac[rows, 42 + ii] = -1
    constant_jac[rows, 42 + jj] = -1
    for block in range(4):
        if time.monotonic() >= end:
            return best if best is not None else init
        rr = m + block * n + idx
        constant_jac[rr, 42 + idx] = -1
        constant_jac[rr, 63] = 0.5 if block < 2 else -0.5
        constant_jac[rr, idx + (0 if block < 2 else 21)] = -1 if block % 2 == 0 else 1
    grad = np.zeros(64)
    grad[42:63] = -1
    bounds = [(-1.0, 1.0)] * 42 + [(1e-8, 0.6)] * 21 + [(0.15, 1.85)]

    def unpack(z):
        return np.column_stack((z[:21], z[21:42], z[42:63]))

    def pack(a):
        a = a.copy()
        a[:, 2] = np.maximum(a[:, 2], 1e-7)
        lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
        hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
        a[:, :2] -= (hi + lo) / 2
        scale = 2 / max(float(np.sum(hi - lo)), 1e-8)
        a *= scale
        width = float((hi[0] - lo[0]) * scale)
        return np.r_[a[:, 0], a[:, 1], a[:, 2], np.clip(width, 0.15, 1.85)]

    for attempt in range(3):
        now = time.monotonic()
        if now >= end - 0.035:
            break
        if attempt == 0:
            start = initial
            cutoff = min(end, now + params['first_fraction'] * max(0, end - now))
        else:
            if best is None:
                start = fallback.copy()
            elif attempt == 1 and best_score < 2.26:
                start = fallback.copy()
            else:
                start = np.array(best['circles'], dtype=float)
                noise = rng.normal(size=(21, 2))
                noise -= noise.mean(axis=0)
                start[:, :2] += params['restart_displacement'] * noise * (1.0 if attempt == 1 else 1.7)
            cutoff = end if attempt == 2 else now + 0.57 * (end - now)
        last = [pack(start)]
        count = [0]

        def guard():
            if time.monotonic() >= cutoff:
                raise StopSearch()

        def objective(z):
            guard()
            return -float(np.sum(z[42:63]))

        def objective_jac(z):
            guard()
            return grad

        def constraints(z):
            guard()
            x, y, r, w = z[:21], z[21:42], z[42:63], z[63]
            dx, dy = x[ii] - x[jj], y[ii] - y[jj]
            return np.r_[np.sqrt(dx**2 + dy**2) - r[ii] - r[jj], w/2-x-r, w/2+x-r, 1-w/2-y-r, 1-w/2+y-r]

        def jacobian(z):
            guard()
            dx = z[ii] - z[jj]
            dy = z[21 + ii] - z[21 + jj]
            d = np.maximum(np.sqrt(dx**2 + dy**2), 1e-15)
            j = constant_jac.copy()
            j[rows, ii] = dx/d
            j[rows, jj] = -dx/d
            j[rows, 21 + ii] = dy/d
            j[rows, 21 + jj] = -dy/d
            return j

        def callback(z):
            last[0] = z.copy()
            count[0] += 1
            guard()
            if count[0] % 3 == 0:
                record(unpack(z))

        try:
            result = minimize(objective, last[0], jac=objective_jac, method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                              callback=callback,
                              options={'maxiter': params['maxiter'], 'ftol': params['ftol'], 'disp': False})
            last[0] = result.x
        except StopSearch:
            pass
        record(unpack(last[0]))
    return best if best is not None else init
