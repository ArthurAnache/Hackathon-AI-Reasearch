import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import lstsq

PARAMS = {
    'mutation': (0.012, 0.0001, 0.08, 'log'),
    'iterations': (110, 20, 250),
    'contact_tol': (2e-7, 1e-10, 1e-4, 'log'),
}
N = 21
I, J = np.triu_indices(N, 1)

class StopSearch(Exception):
    pass


def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -float('inf')
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    perimeter = (max(u + v for u, v in zip(x, r)) - min(u - v for u, v in zip(x, r))
                 + max(u + v for u, v in zip(y, r)) - min(u - v for u, v in zip(y, r)))
    if perimeter > 2:
        return -float('inf')
    d = np.sqrt((a[I, 0] - a[J, 0]) ** 2 + (a[I, 1] - a[J, 1]) ** 2)
    if np.any(d < a[I, 2] + a[J, 2]):
        return -float('inf')
    return float(np.sum(a[:, 2]))


def pack(a):
    lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
    hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
    scale = 2.0 / np.sum(hi - lo)
    p = (a[:, :2] - lo) * scale
    return np.r_[p[:, 0], p[:, 1], a[:, 2] * scale, (hi[0] - lo[0]) * scale]


def constraints(z, jac=False):
    x, y, r, w = z[:21], z[21:42], z[42:63], z[63]
    dx, dy = x[I] - x[J], y[I] - y[J]
    dist = np.maximum(np.sqrt(dx * dx + dy * dy), 1e-100)
    c = np.r_[dist - r[I] - r[J], x-r, w-x-r, y-r, 2-w-y-r]
    if not jac:
        return c
    a = np.zeros((294, 64))
    k = np.arange(210)
    a[k, I] = dx / dist
    a[k, J] = -dx / dist
    a[k, 21 + I] = dy / dist
    a[k, 21 + J] = -dy / dist
    a[k, 42 + I] = -1
    a[k, 42 + J] = -1
    q = np.arange(21)
    a[210+q, q] = 1
    a[231+q, q] = -1
    a[231+q, 63] = 1
    a[252+q, 21+q] = 1
    a[273+q, 21+q] = -1
    a[273+q, 63] = -1
    for k0 in (210, 231, 252, 273):
        a[k0+q, 42+q] = -1
    return a


def repair(z, deadline):
    a = np.column_stack((z[:21], z[21:42], z[42:63])).copy()
    if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return None
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    ratio = min(1.0, float(np.min(d / (a[I, 2]+a[J, 2]))))
    a[:, 2] *= ratio * (1 - 3e-15)
    for _ in range(5):
        if time.monotonic() >= deadline:
            return None
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        perimeter = (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
                     +max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)))
        if perimeter <= 2:
            break
        a[:, 2] -= float(perimeter - 2) / 4 + 2e-16
    return a


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a = np.array([[(i % 5 + .5) / 5, (i // 5 + .5) / 5, .07] for i in range(21)])
        best = {'circles': a.tolist()}
        best_score = exact_score(a)
    else:
        a = np.array(init['circles'], dtype=float)
        best = init
        best_score = exact_score(a)
    zbase = pack(a)
    objective_grad = np.r_[np.zeros(42), -np.ones(21), 0.0]
    bounds = [(None, None)] * 42 + [(1e-7, .6)] * 21 + [(.1, 1.9)]
    last = zbase.copy()

    def guard():
        if time.monotonic() >= deadline - .012:
            raise StopSearch()

    def fun(z):
        guard()
        return -float(np.sum(z[42:63]))

    def con(z):
        guard()
        return constraints(z)

    def jac(z):
        guard()
        return constraints(z, True)

    def callback(z):
        nonlocal last
        last = z.copy()
        guard()

    def consider(z):
        nonlocal best, best_score, zbase
        if time.monotonic() >= deadline - .005:
            return
        candidate = repair(z, deadline)
        if candidate is not None:
            score = exact_score(candidate)
            if score > best_score:
                best_score = score
                best = {'circles': candidate.tolist()}
                zbase = pack(candidate)

    def polish(z):
        active = np.flatnonzero(constraints(z) < float(params['contact_tol']))
        if len(active) < 64 or len(active) > 70:
            return z
        v = z.copy()
        for _ in range(5):
            guard()
            c = constraints(v)[active]
            j = constraints(v, True)[active]
            step, _, rank, _ = lstsq(j, -c, cond=1e-11, lapack_driver='gelsy')
            if rank < 64 or np.max(np.abs(step)) > .003:
                break
            v += step
            if np.max(np.abs(step)) < 1e-15:
                break
        return v

    attempt = 0
    while time.monotonic() < deadline - .035:
        z = zbase.copy()
        if attempt:
            p = np.column_stack((z[:21], z[21:42]))
            center = int(rng.integers(21))
            dd = np.sum((p - p[center])**2, axis=1)
            ids = np.argsort(dd)[:3]
            amp = float(params['mutation']) * (1.0 if attempt % 3 else 2.5)
            p[ids] += rng.normal(0, amp, (3, 2))
            z[:21], z[21:42] = p[:, 0], p[:, 1]
            z[42+ids] *= rng.uniform(.91, 1.04, 3)
        last = z.copy()
        try:
            if attempt == 0:
                consider(polish(z))
            result = minimize(fun, z, jac=lambda v: objective_grad,
                              constraints={'type': 'ineq', 'fun': con, 'jac': jac},
                              bounds=bounds, method='SLSQP', callback=callback,
                              options={'maxiter': int(params['iterations']), 'ftol': 2e-14, 'disp': False})
            last = result.x
            consider(last)
            consider(polish(last))
        except StopSearch:
            break
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            pass
        attempt += 1
    return best
