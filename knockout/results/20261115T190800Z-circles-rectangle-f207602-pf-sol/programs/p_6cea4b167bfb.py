import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"iterations": (180, 60, 400), "polish_fraction": (0.24, 0.1, 0.4)}


def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + 0.70 * budget_s
    n = 21
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    rows = np.arange(m)
    rng = np.random.default_rng(seed)

    def read(obj):
        try:
            if obj is None:
                return None
            if 'circles' in obj:
                a = np.asarray(obj['circles'], dtype=float)
            else:
                a = np.asarray(obj['state'], dtype=float).reshape(3, n).T
            if a.shape == (21, 3) and np.isfinite(a).all():
                return a.copy()
        except Exception:
            pass
        return None

    def score(a):
        if a is None or a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -np.inf
        d = a[ii, :2] - a[jj, :2]
        if np.any(np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2) < a[ii, 2] + a[jj, 2]):
            return -np.inf
        x = [Fraction(float(t)) for t in a[:, 0]]
        y = [Fraction(float(t)) for t in a[:, 1]]
        r = [Fraction(float(t)) for t in a[:, 2]]
        extent = max(u + v for u, v in zip(x, r)) - min(u - v for u, v in zip(x, r))
        extent += max(u + v for u, v in zip(y, r)) - min(u - v for u, v in zip(y, r))
        return float(np.sum(a[:, 2])) if extent <= 2 else -np.inf

    original = read(init)
    best = original
    best_score = score(original)

    def repair(a):
        a = np.asarray(a, dtype=float).copy()
        if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return None
        lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
        hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
        a[:, :2] -= (lo + hi) / 2
        d = a[ii, :2] - a[jj, :2]
        ratio = np.min(np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2) / (a[ii, 2] + a[jj, 2]))
        a[:, 2] *= min(1.0, float(ratio)) * (1 - 5e-14)
        if np.any(a[:, 2] <= 0):
            return None
        lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
        hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
        a *= 2 / np.sum(hi - lo) * (1 - 5e-14)
        return a

    def consider(a):
        nonlocal best, best_score
        a = repair(a)
        value = score(a)
        if value > best_score:
            best, best_score = a, value

    # Three axis circles and nine pairs, always ordered identically.
    def lattice(counts):
        singles, pairs = [], []
        for row, count in enumerate(counts):
            if time.monotonic() >= deadline:
                break
            y = (row - (len(counts) - 1) / 2) * np.sqrt(3)
            if count % 2:
                singles.append([0., y, 1.])
            for k in range(count // 2):
                x = 2 * k + (1 if count % 2 == 0 else 2)
                pairs.append([float(x), y, 1.])
        if len(singles) != 3 or len(pairs) != 9:
            return None
        a = np.array(singles + pairs + [[-x, y, r] for x, y, r in pairs])
        return repair(a)

    a0 = lattice([3, 4, 3, 4, 3, 4])
    if a0 is None:
        return init
    consider(a0)
    if original is not None:
        consider(original)

    fixed = np.zeros((m + 84, 64))
    k = np.arange(n)
    fixed[m + k, k] = -1
    fixed[m + k, 42 + k] = -1
    fixed[m + k, 63] = .5
    fixed[m + 21 + k, k] = 1
    fixed[m + 21 + k, 42 + k] = -1
    fixed[m + 21 + k, 63] = .5
    fixed[m + 42 + k, 21 + k] = -1
    fixed[m + 42 + k, 42 + k] = -1
    fixed[m + 42 + k, 63] = -.5
    fixed[m + 63 + k, 21 + k] = 1
    fixed[m + 63 + k, 42 + k] = -1
    fixed[m + 63 + k, 63] = -.5
    gradient = np.zeros(64)
    gradient[42:63] = -1

    T = np.zeros((64, 34))
    for i in range(3):
        T[21 + i, 2 * i] = 1
        T[42 + i, 2 * i + 1] = 1
    for p in range(9):
        i, j, q = 3 + p, 12 + p, 6 + 3 * p
        T[i, q], T[j, q] = 1, -1
        T[21 + i, q + 1] = T[21 + j, q + 1] = 1
        T[42 + i, q + 2] = T[42 + j, q + 2] = 1
    T[63, 33] = 1
    # Removing constraints duplicated by reflection saves QP work.
    partner = np.array([0, 1, 2] + list(range(12, 21)) + list(range(3, 12)))
    keep = []
    for q, (i, j) in enumerate(zip(ii, jj)):
        if time.monotonic() >= deadline:
            break
        reflected = tuple(sorted((int(partner[i]), int(partner[j]))))
        if (int(i), int(j)) <= reflected:
            keep.append(q)
    keep.extend((m + np.arange(12)).tolist())
    keep.extend((m + 42 + np.arange(12)).tolist())
    keep.extend((m + 63 + np.arange(12)).tolist())
    keep = np.array(keep, dtype=int)

    class Stop(Exception):
        pass

    def solve(a, symmetric, until):
        lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
        hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
        z = np.r_[a.T.ravel(), hi[0] - lo[0]]
        last = z.copy()
        tick = 0
        if symmetric:
            u = np.zeros(34)
            u[:6] = a[:3, 1:].ravel()
            u[6:33] = a[3:12].ravel()
            u[33] = z[63]
            bounds = [(-1, 1), (1e-7, .5)] * 3 + [(1e-7, 1), (-1, 1), (1e-7, .5)] * 9 + [(.02, 1.98)]
            grad = T.T @ gradient
        else:
            u = z
            bounds = [(-1, 1)] * 42 + [(1e-7, .5)] * 21 + [(.02, 1.98)]
            grad = gradient

        def expand(u):
            return T @ u if symmetric else u

        def check():
            if time.monotonic() >= min(until, deadline):
                raise Stop()

        def fun(u):
            check()
            return float(grad @ u)

        def cons(u):
            check()
            v = expand(u)
            x, y, r, w = v[:21], v[21:42], v[42:63], v[63]
            dx, dy, rr = x[ii] - x[jj], y[ii] - y[jj], r[ii] + r[jj]
            c = np.r_[4 * (dx * dx + dy * dy - rr * rr), w / 2 - x - r, w / 2 + x - r, 1 - w / 2 - y - r, 1 - w / 2 + y - r]
            return c[keep] if symmetric else c

        def jac(u):
            check()
            v = expand(u)
            x, y, r = v[:21], v[21:42], v[42:63]
            dx, dy, rr = 8 * (x[ii] - x[jj]), 8 * (y[ii] - y[jj]), -8 * (r[ii] + r[jj])
            J = fixed.copy()
            J[rows, ii], J[rows, jj] = dx, -dx
            J[rows, 21 + ii], J[rows, 21 + jj] = dy, -dy
            J[rows, 42 + ii], J[rows, 42 + jj] = rr, rr
            return J[keep] @ T if symmetric else J

        def callback(u):
            nonlocal last, tick
            last = expand(u).copy()
            tick += 1
            if tick % 10 == 0:
                consider(last[:63].reshape(3, 21).T)
            check()

        try:
            result = minimize(fun, u, jac=lambda u: grad, method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': cons, 'jac': jac}, callback=callback,
                              options={'maxiter': int(params['iterations']), 'ftol': 1e-12, 'disp': False})
            last = expand(result.x)
        except (Stop, ValueError, FloatingPointError):
            pass
        consider(last[:63].reshape(3, 21).T)

    search_end = started + (0.70 - params['polish_fraction']) * budget_s
    solve(a0, True, started + .28 * budget_s)
    if time.monotonic() < search_end:
        a1 = lattice([2, 3, 4, 3, 4, 3, 2])
        if a1 is not None:
            solve(a1, True, search_end)
    if time.monotonic() < deadline:
        solve(best.copy(), False, deadline)
    if best is original and init is not None:
        return init
    return {'state': best.T.ravel().tolist()}
