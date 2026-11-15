import time
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog

PARAMS = {
    'trust': (0.085, 0.02, 0.25),
    'loss': (0.008, 0.001, 0.06, 'log'),
    'push_steps': (3, 1, 6),
    'recover_steps': (14, 5, 30),
}

N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
K = np.arange(N)
ROWS = np.arange(M)


def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    r = a[:, 2]
    d = np.sqrt((a[I, 0] - a[J, 0]) ** 2 + (a[I, 1] - a[J, 1]) ** 2)
    if np.any(d < r[I] + r[J]):
        return -np.inf
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    q = [Fraction(float(v)) for v in r]
    w = max(v + t for v, t in zip(x, q)) - min(v - t for v, t in zip(x, q))
    h = max(v + t for v, t in zip(y, q)) - min(v - t for v, t in zip(y, q))
    return float(np.sum(r)) if w + h <= 2 else -np.inf


def normalize(a):
    lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
    hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
    scale = 2.0 / np.sum(hi - lo)
    p = (a[:, :2] - 0.5 * (lo + hi)) * scale
    return np.r_[p[:, 0], p[:, 1], a[:, 2] * scale, (hi[0] - lo[0]) * scale]


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    ids = np.arange(21)
    fallback = np.column_stack(((ids % 7 - 3) * 0.2, (ids // 7 - 1) * 0.2, np.full(21, 0.1 - 1e-14)))
    best = {'circles': fallback.tolist()}
    score = exact_score(fallback)
    best_z = normalize(fallback)
    if init is not None:
        try:
            a = np.asarray(init['circles'], dtype=float)
            s = exact_score(a)
            if s >= score:
                best, score, best_z = init, s, normalize(a)
        except (ValueError, TypeError, KeyError, OverflowError):
            pass

    base = np.zeros((M + 84, 64))
    b = np.zeros(M + 84)
    base[ROWS, 42 + I] = 1
    base[ROWS, 42 + J] = 1
    base[M + K, K] = 1
    base[M + K, 42 + K] = 1
    base[M + K, 63] = -0.5
    base[M + 21 + K, K] = -1
    base[M + 21 + K, 42 + K] = 1
    base[M + 21 + K, 63] = -0.5
    base[M + 42 + K, 21 + K] = 1
    base[M + 42 + K, 42 + K] = 1
    base[M + 42 + K, 63] = 0.5
    base[M + 63 + K, 21 + K] = -1
    base[M + 63 + K, 42 + K] = 1
    base[M + 63 + K, 63] = 0.5
    b[M + 42:] = 1
    objective = np.r_[np.zeros(42), -np.ones(21), 0.0]

    def matrix(z):
        dx = z[I] - z[J]
        dy = z[21 + I] - z[21 + J]
        d = np.maximum(np.sqrt(dx * dx + dy * dy), 1e-15)
        a = base.copy()
        a[ROWS, I] = -dx / d
        a[ROWS, J] = dx / d
        a[ROWS, 21 + I] = -dy / d
        a[ROWS, 21 + J] = dy / d
        return a

    def consider(z):
        nonlocal best, score, best_z
        if time.monotonic() >= end or not np.isfinite(z).all():
            return
        a = np.column_stack((z[:21], z[21:42], z[42:63]))
        if np.any(a[:, 2] <= 0):
            return
        d = np.sqrt((a[I, 0] - a[J, 0]) ** 2 + (a[I, 1] - a[J, 1]) ** 2)
        f = min(1.0, float(np.min(d / (a[I, 2] + a[J, 2]))))
        a[:, 2] *= f * (1 - 8e-15)
        lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
        hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
        a[:, :2] -= 0.5 * (hi + lo)
        a *= (2.0 / np.sum(hi - lo)) * (1 - 8e-15)
        if float(np.sum(a[:, 2])) <= score:
            return
        s = exact_score(a)
        if s > score:
            best = {'circles': a.tolist()}
            score = s
            best_z = normalize(a)

    def polish(z):
        q = z.copy()
        for _ in range(4):
            if time.monotonic() >= end - 0.018:
                break
            a = matrix(q)
            residual = b - a @ q
            active = np.flatnonzero(np.abs(residual) < 2e-7)
            if len(active) < 60:
                break
            step = np.linalg.lstsq(a[active], residual[active], rcond=1e-11)[0]
            if np.max(np.abs(step)) > 1e-4:
                break
            q += step
            consider(q)
            if np.max(np.abs(step)) < 2e-15:
                break
        return q

    def lp(z, c, floor=None):
        remaining = end - time.monotonic()
        if remaining < 0.025:
            return None
        a = matrix(z)
        rhs = b
        if floor is not None:
            a = np.vstack((a, objective))
            rhs = np.r_[b, -floor]
        bounds = np.empty((64, 2))
        trust = float(params['trust'])
        bounds[:42, 0] = z[:42] - trust
        bounds[:42, 1] = z[:42] + trust
        bounds[42:63, 0] = 0.005
        bounds[42:63, 1] = 0.4
        bounds[63] = [0.45, 1.55]
        result = linprog(c, A_ub=a, b_ub=rhs, bounds=bounds, method='highs', options={
            'time_limit': max(0.001, remaining - 0.018),
            'primal_feasibility_tolerance': 1e-10,
            'dual_feasibility_tolerance': 1e-10,
        })
        if not result.success or result.x is None:
            return None
        consider(result.x)
        return result.x

    def recover(z, count):
        stalled = 0
        for _ in range(count):
            if time.monotonic() >= end - 0.025:
                break
            q = lp(z, objective)
            if q is None:
                break
            gain = np.sum(q[42:63]) - np.sum(z[42:63])
            z = q
            stalled = stalled + 1 if gain < 1e-10 else 0
            if stalled >= 2:
                polish(z)
                break
        return z

    recover(best_z.copy(), 5)
    attempt = 0
    while time.monotonic() < end - 0.06:
        z = best_z.copy()
        x, y = z[:21], z[21:42]
        anchor = int(rng.integers(21))
        weight = np.exp(-((x - x[anchor]) ** 2 + (y - y[anchor]) ** 2) / 0.065)
        angle = rng.uniform(0, 2 * np.pi)
        mode = attempt % 3
        if mode == 0:
            vx = weight * (y - y[anchor])
            vy = -weight * (x - x[anchor])
        elif mode == 1:
            vx = weight * np.cos(angle)
            vy = weight * np.sin(angle)
        else:
            vx = np.cos(angle) * x + np.sin(angle) * y
            vy = np.sin(angle) * x - np.cos(angle) * y
        vx = vx - np.mean(vx)
        vy = vy - np.mean(vy)
        magnitude = max(float(np.max(np.abs(np.r_[vx, vy]))), 1e-9)
        c = np.r_[-vx / magnitude, -vy / magnitude, np.full(21, -0.15), 0.0]
        loss = float(params['loss']) * (1.0, 2.5, 5.0, 0.4)[attempt % 4]
        floor = score - loss
        for _ in range(int(params['push_steps'])):
            if time.monotonic() >= end - 0.025:
                break
            q = lp(z, c, floor)
            if q is None:
                break
            z = q
        if time.monotonic() < end - 0.025:
            recover(z, int(params['recover_steps']))
        attempt += 1
    return best
