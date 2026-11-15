import time
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog

PARAMS = {
    'trust': (0.16, 0.03, 0.4),
    'first_share': (0.48, 0.3, 0.65),
    'symmetric_share': (0.27, 0.15, 0.4),
    'radius_floor': (0.001, 0.00001, 0.03, 'log'),
}

N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
K = np.arange(N)


def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    r = a[:, 2]
    d = np.sqrt((a[I, 0] - a[J, 0]) ** 2 + (a[I, 1] - a[J, 1]) ** 2)
    if np.any(d < r[I] + r[J]):
        return -np.inf
    x = list(map(lambda v: Fraction(float(v)), a[:, 0]))
    y = list(map(lambda v: Fraction(float(v)), a[:, 1]))
    q = list(map(lambda v: Fraction(float(v)), r))
    w = max(map(lambda t: t[0] + t[1], zip(x, q))) - min(map(lambda t: t[0] - t[1], zip(x, q)))
    h = max(map(lambda t: t[0] + t[1], zip(y, q))) - min(map(lambda t: t[0] - t[1], zip(y, q)))
    return float(np.sum(r)) if w + h <= 2 else -np.inf


def repaired(z):
    a = np.column_stack((z[:21], z[21:42], z[42:63])).copy()
    if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return None
    d = np.sqrt((a[I, 0] - a[J, 0]) ** 2 + (a[I, 1] - a[J, 1]) ** 2)
    factor = min(1.0, float(np.min(d / (a[I, 2] + a[J, 2]))))
    if factor <= 0:
        return None
    a[:, 2] *= factor * (1.0 - 3e-13)
    lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
    hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
    a[:, :2] -= (hi + lo) * 0.5
    a *= min(1.0, 2.0 / float(np.sum(hi - lo))) * (1.0 - 3e-13)
    return a


def normalized(a):
    p = a[:, :2].copy()
    r = np.maximum(np.abs(a[:, 2]), 1e-5)
    lo = np.min(p - r[:, None], axis=0)
    hi = np.max(p + r[:, None], axis=0)
    sides = hi - lo
    scale = 2.0 / max(float(np.sum(sides)), 1e-9)
    p = (p - (hi + lo) * 0.5) * scale
    return np.r_[p[:, 0], p[:, 1], r * scale, sides[0] * scale]


def symmetric_start():
    # Three axial circles and nine representative circles above the axis.
    x = np.r_[[-0.4, 0.0, 0.4], [-0.3, -0.1, 0.1, 0.3], [-0.4, -0.2, 0.0, 0.2, 0.4]]
    y = np.r_[np.full(4, 0.175), np.full(5, 0.35)]
    u = np.r_[x, y, np.full(12, 0.095), 1.0]
    t = np.zeros((64, 34))
    t[np.arange(12), np.arange(12)] = 1.0
    t[np.arange(12, 21), np.arange(3, 12)] = 1.0
    t[21 + np.arange(3, 12), 12 + np.arange(9)] = 1.0
    t[21 + np.arange(12, 21), 12 + np.arange(9)] = -1.0
    t[42 + np.arange(12), 21 + np.arange(12)] = 1.0
    t[42 + np.arange(12, 21), 21 + np.arange(3, 12)] = 1.0
    t[63, 33] = 1.0
    return t @ u, t, u


def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    end = started + 0.70 * budget_s
    best = None
    best_score = -np.inf
    best_z = None

    # A valid fallback is available even when the supplied start is invalid.
    ids = np.arange(21)
    fallback = np.column_stack(((ids % 7 - 3) * 0.2, (ids // 7 - 1) * 0.2, np.full(21, 0.1 - 1e-13)))
    fs = exact_score(fallback)
    best = {'circles': fallback.tolist()}
    best_score = fs
    best_z = normalized(fallback)
    initial_z = best_z.copy()
    if init is not None:
        try:
            a = np.asarray(init['circles'], dtype=float)
            if a.shape == (21, 3) and np.isfinite(a).all():
                val = exact_score(a)
                if val > best_score:
                    best, best_score = init, val
                    best_z = normalized(a)
                initial_z = normalized(a)
        except (ValueError, TypeError, KeyError, OverflowError):
            pass

    def consider(z):
        nonlocal best, best_score, best_z
        if time.monotonic() >= end:
            return
        a = repaired(z)
        if a is None or float(np.sum(a[:, 2])) <= best_score:
            return
        val = exact_score(a)
        if val > best_score:
            best_score = val
            best = {'circles': a.tolist()}
            best_z = normalized(a)

    base = np.zeros((M + 84, 64))
    rows = np.arange(M)
    base[rows, 42 + I] = 1.0
    base[rows, 42 + J] = 1.0
    rhs = np.zeros(M + 84)
    base[M + K, K] = 1.0
    base[M + K, 42 + K] = 1.0
    base[M + K, 63] = -0.5
    base[M + 21 + K, K] = -1.0
    base[M + 21 + K, 42 + K] = 1.0
    base[M + 21 + K, 63] = -0.5
    base[M + 42 + K, 21 + K] = 1.0
    base[M + 42 + K, 42 + K] = 1.0
    base[M + 42 + K, 63] = 0.5
    base[M + 63 + K, 21 + K] = -1.0
    base[M + 63 + K, 42 + K] = 1.0
    base[M + 63 + K, 63] = 0.5
    rhs[M + 42:] = 1.0
    objective = np.r_[np.zeros(42), -np.ones(21), 0.0]
    floor = float(params['radius_floor'])
    trust = float(params['trust'])

    def search(z, stop, transform=None, u=None):
        stop = min(stop, end)
        c = objective if transform is None else objective @ transform
        previous = float(np.sum(z[42:63]))
        stalled = 0
        iteration = 0
        while time.monotonic() < stop - 0.012:
            now = time.monotonic()
            dx = z[I] - z[J]
            dy = z[21 + I] - z[21 + J]
            dist = np.maximum(np.sqrt(dx * dx + dy * dy), 1e-12)
            nx, ny = dx / dist, dy / dist
            a = base.copy()
            a[rows, I] = -nx
            a[rows, J] = nx
            a[rows, 21 + I] = -ny
            a[rows, 21 + J] = ny
            delta = trust if iteration < 12 else trust * 0.6
            if transform is None:
                bounds = np.empty((64, 2))
                bounds[:42, 0] = z[:42] - delta
                bounds[:42, 1] = z[:42] + delta
                bounds[42:63, 0] = floor
                bounds[42:63, 1] = 0.5
                bounds[63] = [0.25, 1.75]
                matrix = a
            else:
                bounds = np.empty((34, 2))
                bounds[:21, 0] = u[:21] - delta
                bounds[:21, 1] = u[:21] + delta
                bounds[12:21, 0] = np.maximum(bounds[12:21, 0], 0.001)
                bounds[21:33, 0] = floor
                bounds[21:33, 1] = 0.5
                bounds[33] = [0.25, 1.75]
                matrix = a @ transform
            remaining = stop - time.monotonic()
            if remaining < 0.012:
                break
            result = linprog(c, A_ub=matrix, b_ub=rhs, bounds=bounds, method='highs', options={
                'time_limit': max(0.001, remaining - 0.006),
                'primal_feasibility_tolerance': 1e-9,
                'dual_feasibility_tolerance': 1e-9,
            })
            if result.x is None or not result.success:
                break
            if transform is None:
                z = result.x
            else:
                u = result.x
                z = transform @ u
            consider(z)
            value = float(np.sum(z[42:63]))
            stalled = stalled + 1 if value - previous < 2e-10 else 0
            previous = value
            iteration += 1
            if stalled >= 3:
                break
        return z

    available = max(0.0, end - time.monotonic())
    first_end = time.monotonic() + available * float(params['first_share'])
    z = search(initial_z, first_end)
    consider(z)
    if time.monotonic() < end - 0.025:
        sym, transform, u = symmetric_start()
        sym_end = min(end, first_end + available * float(params['symmetric_share']))
        sym = search(sym, sym_end, transform, u)
        consider(sym)
    if time.monotonic() < end - 0.025:
        search(best_z.copy(), end)
    return best
