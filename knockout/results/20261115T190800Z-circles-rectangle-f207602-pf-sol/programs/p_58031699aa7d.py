import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    "perturbation": (0.024, 0.001, 0.1),
    "iterations": (65, 15, 150),
    "time_fraction": (0.70, 0.3, 0.75),
}


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + float(params['time_fraction']) * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    best = init
    best_score = -float('inf')

    class Stop(Exception):
        pass

    def check():
        if time.monotonic() >= deadline:
            raise Stop()

    def exact_score(data):
        try:
            a = np.asarray(data['circles'], dtype=float)
            if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
                return -float('inf')
            xs = [Fraction(float(v)) for v in a[:, 0]]
            ys = [Fraction(float(v)) for v in a[:, 1]]
            rs = [Fraction(float(v)) for v in a[:, 2]]
            width = max(x+r for x, r in zip(xs, rs))-min(x-r for x, r in zip(xs, rs))
            height = max(y+r for y, r in zip(ys, rs))-min(y-r for y, r in zip(ys, rs))
            if width + height > 2:
                return -float('inf')
            for i, j in zip(ii, jj):
                if time.monotonic() >= deadline:
                    return -float('inf')
                d = np.sqrt((a[i, 0]-a[j, 0])**2 + (a[i, 1]-a[j, 1])**2)
                if d < a[i, 2]+a[j, 2]:
                    return -float('inf')
            return float(np.sum(a[:, 2]))
        except (ValueError, TypeError, KeyError, OverflowError):
            return -float('inf')

    if init is not None:
        best_score = exact_score(init)

    def save(a):
        nonlocal best, best_score
        check()
        a = np.asarray(a, dtype=float).copy()
        if a.shape != (21, 3) or not np.isfinite(a).all():
            return
        a[:, 2] = np.maximum(a[:, 2], 1e-9)
        d = np.sqrt(np.sum((a[ii, :2]-a[jj, :2])**2, axis=1))
        ratio = float(np.min(d/(a[ii, 2]+a[jj, 2])))
        a[:, 2] *= min(1.0, ratio) * (1.0-3e-14)
        if np.any(a[:, 2] <= 0):
            return
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        perimeter = float(np.sum(hi-lo))
        if perimeter <= 0:
            return
        a[:, :2] -= (hi+lo)/2
        a *= (2.0/perimeter)*(1.0-5e-14)
        if float(a[:, 2].sum()) <= best_score:
            return
        candidate = {'circles': a.tolist()}
        score = exact_score(candidate)
        if score > best_score:
            best_score, best = score, candidate

    if init is not None:
        try:
            base = np.asarray(init['circles'], dtype=float).copy()
            if base.shape != (21, 3) or not np.isfinite(base).all():
                raise ValueError()
        except (ValueError, KeyError, TypeError):
            base = None
    else:
        base = None
    if base is None:
        base = np.array([[0.18*(i % 5), 0.18*(i//5), 0.085] for i in range(21)], float)
    try:
        save(base)
    except Stop:
        return best

    def encode(a):
        a = a.copy()
        low = np.min(a[:, :2]-a[:, 2, None], axis=0)
        high = np.max(a[:, :2]+a[:, 2, None], axis=0)
        scale = 2.0/float(np.sum(high-low))
        a[:, :2] = (a[:, :2]-low)*scale
        a[:, 2] *= scale
        return np.r_[a[:, 0], a[:, 1], a[:, 2], (high[0]-low[0])*scale]

    def decode(z):
        return np.column_stack((z[:21], z[21:42], z[42:63]))

    objective_jac = np.zeros(64)
    objective_jac[42:63] = -1
    bounds = [(None, None)]*42 + [(1e-7, 0.6)]*21 + [(0.15, 1.85)]
    linear_jac = np.zeros((84, 64))
    ids = np.arange(21)
    linear_jac[ids, ids] = 1
    linear_jac[ids, 42+ids] = -1
    linear_jac[21+ids, ids] = -1
    linear_jac[21+ids, 42+ids] = -1
    linear_jac[21+ids, 63] = 1
    linear_jac[42+ids, 21+ids] = 1
    linear_jac[42+ids, 42+ids] = -1
    linear_jac[63+ids, 21+ids] = -1
    linear_jac[63+ids, 42+ids] = -1
    linear_jac[63+ids, 63] = -1
    rows = np.arange(m)
    cached_z = None
    cached_c = None
    cached_j = None

    def calculate(z):
        nonlocal cached_z, cached_c, cached_j
        check()
        if cached_z is not None and np.array_equal(z, cached_z):
            return
        x, y, r, w = z[:21], z[21:42], z[42:63], z[63]
        dx, dy = x[ii]-x[jj], y[ii]-y[jj]
        dist = np.sqrt(dx*dx+dy*dy)
        safe = np.maximum(dist, 1e-15)
        c = np.r_[dist-r[ii]-r[jj], x-r, w-x-r, y-r, 2-w-y-r]
        jac = np.zeros((m+84, 64))
        jac[rows, ii] = dx/safe
        jac[rows, jj] = -dx/safe
        jac[rows, 21+ii] = dy/safe
        jac[rows, 21+jj] = -dy/safe
        jac[rows, 42+ii] = -1
        jac[rows, 42+jj] = -1
        jac[m:] = linear_jac
        cached_z, cached_c, cached_j = z.copy(), c, jac

    def con(z):
        calculate(z)
        return cached_c

    def jac(z):
        calculate(z)
        return cached_j

    def objective(z):
        check()
        return -float(np.sum(z[42:63]))

    def callback(z):
        check()
        save(decode(z))

    attempt = 0
    try:
        while time.monotonic() < deadline:
            check()
            if attempt == 0:
                a = base.copy()
            else:
                a = np.asarray(best['circles'], dtype=float).copy() if best is not None else base.copy()
                anchor = int(rng.integers(21))
                distance = np.sum((a[:, :2]-a[anchor, :2])**2, axis=1)
                count = int(rng.integers(3, 8))
                chosen = np.argsort(distance)[:count]
                magnitude = float(params['perturbation']) * (0.65 + 0.55*(attempt % 4))
                a[chosen, :2] += rng.normal(0, magnitude, (count, 2))
                a[chosen, 2] *= np.exp(rng.normal(0, 0.06, count))
            z = encode(a)
            cached_z = None
            result = minimize(objective, z, jac=lambda z: objective_jac,
                method='SLSQP', bounds=bounds,
                constraints={'type': 'ineq', 'fun': con, 'jac': jac},
                callback=callback,
                options={'maxiter': int(params['iterations']), 'ftol': 2e-12, 'disp': False})
            save(decode(result.x))
            attempt += 1
    except Stop:
        pass
    return best
