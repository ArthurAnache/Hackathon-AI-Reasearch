import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'contact_threshold': (2e-7, 1e-10, 1e-4, 'log'),
    'restart_amplitude': (0.055, 0.005, 0.15),
    'cluster_size': (5, 3, 9),
    'polish_share': (0.30, 0.15, 0.6),
}

N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
K = np.arange(N)
ROWS = np.arange(M)

class Deadline(Exception):
    pass


def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0] - a[J, 0])**2 + (a[I, 1] - a[J, 1])**2)
    if np.any(d < a[I, 2] + a[J, 2]):
        return -np.inf
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    w = max(xi + ri for xi, ri in zip(x, r)) - min(xi - ri for xi, ri in zip(x, r))
    h = max(yi + ri for yi, ri in zip(y, r)) - min(yi - ri for yi, ri in zip(y, r))
    return float(np.sum(a[:, 2])) if w + h <= 2 else -np.inf


def encode(a):
    lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
    hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
    scale = 2.0 / np.sum(hi - lo)
    p = (a[:, :2] - (hi + lo) * 0.5) * scale
    return np.r_[p[:, 0], p[:, 1], a[:, 2] * scale, (hi[0] - lo[0]) * scale]


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    ids = np.arange(N)
    fallback = np.column_stack(((ids % 7 - 3) * 0.2, (ids // 7 - 1) * 0.2, np.full(N, 0.1 - 1e-14)))
    best = {'circles': fallback.tolist()}
    best_score = exact_score(fallback)
    best_z = encode(fallback)
    if init is not None:
        try:
            a = np.asarray(init['circles'], dtype=float)
            score = exact_score(a)
            if score >= best_score:
                best, best_score, best_z = init, score, encode(a)
        except (ValueError, TypeError, KeyError, OverflowError):
            pass
    original = best_z.copy()

    def consider(z):
        nonlocal best, best_score, best_z
        if time.monotonic() >= end:
            return
        if not np.isfinite(z).all() or np.any(z[42:63] <= 0):
            return
        a = np.column_stack((z[:21], z[21:42], z[42:63]))
        d = np.sqrt((a[I, 0] - a[J, 0])**2 + (a[I, 1] - a[J, 1])**2)
        shrink = min(1.0, float(np.min(d / (a[I, 2] + a[J, 2]))))
        if shrink <= 0:
            return
        a[:, 2] *= shrink * (1.0 - 3e-15)
        lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
        hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
        a[:, :2] -= (hi + lo) * 0.5
        a *= (2.0 / np.sum(hi - lo)) * (1.0 - 4e-15)
        if float(np.sum(a[:, 2])) <= best_score:
            return
        score = exact_score(a)
        if score > best_score:
            best = {'circles': a.tolist()}
            best_score = score
            best_z = encode(a)

    wall = np.zeros((84, 64))
    wall[K, K] = -1
    wall[K, 42 + K] = -1
    wall[K, 63] = 0.5
    wall[21 + K, K] = 1
    wall[21 + K, 42 + K] = -1
    wall[21 + K, 63] = 0.5
    wall[42 + K, 21 + K] = -1
    wall[42 + K, 42 + K] = -1
    wall[42 + K, 63] = -0.5
    wall[63 + K, 21 + K] = 1
    wall[63 + K, 42 + K] = -1
    wall[63 + K, 63] = -0.5
    offset = np.r_[np.zeros(42), np.ones(42)]
    basejac = np.zeros((M + 84, 64))
    basejac[ROWS, 42 + I] = -1
    basejac[ROWS, 42 + J] = -1
    basejac[M:] = wall
    objective = np.r_[np.zeros(42), -np.ones(21), 0.0]

    def geometry(z):
        dx = z[I] - z[J]
        dy = z[21 + I] - z[21 + J]
        dist = np.maximum(np.sqrt(dx * dx + dy * dy), 1e-30)
        values = np.r_[dist - z[42 + I] - z[42 + J], wall @ z + offset]
        jac = basejac.copy()
        jac[ROWS, I] = dx / dist
        jac[ROWS, J] = -dx / dist
        jac[ROWS, 21 + I] = dy / dist
        jac[ROWS, 21 + J] = -dy / dist
        return values, jac

    def contact_refine(z, stop):
        z = z.copy()
        values, _ = geometry(z)
        active = np.flatnonzero(np.abs(values) < float(params['contact_threshold']))
        if len(active) < 60:
            return z
        for iteration in range(7):
            if time.monotonic() >= min(stop, end) - 0.006:
                break
            values, jac = geometry(z)
            step = np.linalg.lstsq(jac[active], -values[active], rcond=1e-12)[0]
            if not np.isfinite(step).all() or np.max(np.abs(step)) > 0.03:
                break
            z += step
            consider(z)
            if np.max(np.abs(step)) < 2e-15:
                break
        return z

    bounds = [(None, None)] * 42 + [(1e-5, 0.5)] * 21 + [(0.2, 1.8)]

    def polish(z, stop, maxiter):
        stop = min(stop, end - 0.008)
        last = z.copy()
        cached_z = None
        cached = None
        def check():
            if time.monotonic() >= stop:
                raise Deadline()
        def calc(v):
            nonlocal cached_z, cached
            check()
            if cached_z is None or not np.array_equal(v, cached_z):
                cached_z = v.copy()
                cached = geometry(v)
            return cached
        def fun(v):
            check()
            return float(objective @ v)
        def callback(v):
            nonlocal last
            check()
            last = v.copy()
            consider(v)
        try:
            result = minimize(fun, z, jac=lambda v: objective, method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': lambda v: calc(v)[0], 'jac': lambda v: calc(v)[1]},
                callback=callback,
                options={'ftol': 2e-14, 'maxiter': maxiter, 'disp': False})
            last = result.x
        except Deadline:
            pass
        consider(last)
        return last

    if time.monotonic() < end - 0.03:
        original = contact_refine(original, min(end, start + 0.12 * budget_s))
        first_stop = start + float(params['polish_share']) * 0.70 * budget_s
        z = polish(original, first_stop, 55)
        contact_refine(z, min(end, first_stop + 0.04))

    attempt = 0
    while time.monotonic() < end - 0.08:
        now = time.monotonic()
        z = best_z.copy()
        points = np.column_stack((z[:21], z[21:42]))
        pivot = int(rng.integers(N))
        distances = np.sum((points - points[pivot])**2, axis=1)
        cluster = np.argsort(distances)[:int(params['cluster_size'])]
        amplitude = float(params['restart_amplitude']) * (1.0 if attempt % 2 == 0 else 1.65)
        angle = rng.uniform(-np.pi, np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)])
        relative = points[cluster] - points[cluster].mean(axis=0)
        tangent = np.column_stack((-relative[:, 1], relative[:, 0]))
        displacement = amplitude * (direction + 3.0 * tangent + rng.normal(0, 0.35, (len(cluster), 2)))
        z[cluster] += displacement[:, 0]
        z[21 + cluster] += displacement[:, 1]
        z[42 + cluster] *= 0.90
        local_stop = min(end - 0.018, now + 0.38)
        result = polish(z, local_stop, 65)
        if time.monotonic() < end - 0.02:
            contact_refine(result, min(end - 0.006, local_stop + 0.02))
        attempt += 1
    return best
