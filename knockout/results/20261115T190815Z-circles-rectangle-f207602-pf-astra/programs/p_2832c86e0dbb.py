import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'strain': (0.34, 0.05, 0.8),
    'strain_growth': (1.65, 1.1, 2.5),
    'forced_iterations': (38, 10, 100),
    'free_iterations': (65, 15, 150),
    'attempts': (6, 2, 12)
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class Timeout(Exception):
    pass


def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or not (a[:, 2] > 0).all():
        return -np.inf
    dx = a[I, 0] - a[J, 0]
    dy = a[I, 1] - a[J, 1]
    if np.any(np.sqrt(dx * dx + dy * dy) < a[I, 2] + a[J, 2]):
        return -np.inf
    x = list(map(lambda v: Fraction(float(v)), a[:, 0]))
    y = list(map(lambda v: Fraction(float(v)), a[:, 1]))
    r = list(map(lambda v: Fraction(float(v)), a[:, 2]))
    perimeter = (max(map(lambda t: t[0] + t[1], zip(x, r)))
                 - min(map(lambda t: t[0] - t[1], zip(x, r)))
                 + max(map(lambda t: t[0] + t[1], zip(y, r)))
                 - min(map(lambda t: t[0] - t[1], zip(y, r))))
    if perimeter > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    best = init
    best_score = -np.inf
    if init is not None:
        original = np.asarray(init['circles'], dtype=float)
        best_score = exact_score(original)
    else:
        k = np.arange(N)
        original = np.column_stack(((k % 5 + .5) / 5, (k // 5 + .5) / 5, np.full(N, .095)))
        best = {'circles': original.tolist()}
        best_score = exact_score(original)

    def check():
        if time.monotonic() >= deadline:
            raise Timeout()

    def consider(z):
        check()
        r = z[42:63].copy()
        p = np.column_stack((z[:21], z[21:42]))
        if not np.isfinite(z).all() or np.min(r) <= 0:
            return
        d = np.sqrt(np.sum((p[I] - p[J]) ** 2, axis=1))
        contraction = min(1.0, float(np.min(d / (r[I] + r[J]))))
        if contraction <= 0:
            return
        r *= contraction * (1 - 3e-14)
        lo = np.min(p - r[:, None], axis=0)
        side = np.max(p + r[:, None], axis=0) - lo
        scale = 2.0 / float(np.sum(side)) * (1 - 3e-14)
        if float(np.sum(r)) * scale <= best_score:
            return
        a = np.column_stack(((p - lo) * scale, r * scale))
        score = exact_score(a)
        nonlocal_dummy = None
        update(a, score)

    def update(a, score):
        nonlocal best, best_score
        if score > best_score:
            best_score = score
            best = {'circles': a.tolist()}

    radius_sum = float(np.sum(original[:, 2]))
    p = original[:, :2] / radius_sum
    r = original[:, 2] / radius_sum
    lo = np.min(p - r[:, None], axis=0)
    side = np.max(p + r[:, None], axis=0) - lo
    p -= lo
    base = np.concatenate((p[:, 0], p[:, 1], r, side))
    base_aspect = float(side[0] / side[1])
    gradient = np.zeros(65)
    gradient[63:] = 1
    sum_jac = np.zeros(65)
    sum_jac[42:63] = 1
    rows = np.arange(M)
    idx = np.arange(N)
    bounds = [(None, None)] * 42 + [(1e-7, 1.0)] * 21 + [(1e-5, 2.0)] * 2

    def objective(z):
        check()
        return z[63] + z[64]

    cached_z = None
    cached_values = None
    cached_jac = None

    def compute(z):
        nonlocal cached_z, cached_values, cached_jac
        check()
        if cached_z is not None and np.array_equal(z, cached_z):
            return
        x, y, rr = z[:21], z[21:42], z[42:63]
        dx, dy = x[I] - x[J], y[I] - y[J]
        d = np.sqrt(dx * dx + dy * dy)
        safe = np.maximum(d, 1e-20)
        cached_values = np.concatenate((d - rr[I] - rr[J], x - rr, z[63] - x - rr, y - rr, z[64] - y - rr))
        a = np.zeros((M + 84, 65))
        a[rows, I] = dx / safe
        a[rows, J] = -dx / safe
        a[rows, 21 + I] = dy / safe
        a[rows, 21 + J] = -dy / safe
        a[rows, 42 + I] = -1
        a[rows, 42 + J] = -1
        a[M + idx, idx] = 1
        a[M + idx, 42 + idx] = -1
        a[M + 21 + idx, idx] = -1
        a[M + 21 + idx, 42 + idx] = -1
        a[M + 21 + idx, 63] = 1
        a[M + 42 + idx, 21 + idx] = 1
        a[M + 42 + idx, 42 + idx] = -1
        a[M + 63 + idx, 21 + idx] = -1
        a[M + 63 + idx, 42 + idx] = -1
        a[M + 63 + idx, 64] = 1
        cached_z = z.copy()
        cached_jac = a

    def constraints(z):
        compute(z)
        return cached_values

    def jacobian(z):
        compute(z)
        return cached_jac

    common = [
        {'type': 'ineq', 'fun': constraints, 'jac': jacobian},
        {'type': 'eq', 'fun': lambda z: np.sum(z[42:63]) - 1, 'jac': lambda z: sum_jac}
    ]
    ticks = 0

    def callback(z):
        nonlocal ticks
        check()
        ticks += 1
        if ticks % 4 == 0:
            consider(z)

    try:
        for trial in range(int(params['attempts'])):
            check()
            magnitude = float(params['strain']) * float(params['strain_growth']) ** (trial // 2)
            sign = 1 if (trial + seed) % 2 == 0 else -1
            aspect = base_aspect * np.exp(sign * magnitude)
            z = base.copy()
            total = z[63] + z[64]
            target_w = total * aspect / (1 + aspect)
            target_h = total / (1 + aspect)
            z[:21] *= target_w / z[63]
            z[21:42] *= target_h / z[64]
            z[63], z[64] = target_w, target_h
            aspect_jac = np.zeros(65)
            aspect_jac[63] = 1
            aspect_jac[64] = -aspect
            forced = common + [{'type': 'eq', 'fun': lambda q: q[63] - aspect * q[64], 'jac': lambda q: aspect_jac}]
            result = minimize(objective, z, jac=lambda q: gradient, bounds=bounds,
                              constraints=forced, method='SLSQP', callback=callback,
                              options={'maxiter': int(params['forced_iterations']), 'ftol': 2e-10})
            check()
            consider(result.x)
            result = minimize(objective, result.x, jac=lambda q: gradient, bounds=bounds,
                              constraints=common, method='SLSQP', callback=callback,
                              options={'maxiter': int(params['free_iterations']), 'ftol': 2e-13})
            consider(result.x)
    except Timeout:
        pass
    return best
