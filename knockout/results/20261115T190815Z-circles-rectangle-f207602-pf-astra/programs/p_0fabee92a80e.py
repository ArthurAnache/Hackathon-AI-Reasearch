import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'first_share': (0.64, 0.4, 0.85),
    'jitter': (0.014, 0.002, 0.06),
    'maxiter': (150, 50, 250)
}

class SearchStop(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    end = started + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    rows = np.arange(len(ii))
    best = init
    best_score = -float('inf')

    def exact(data):
        try:
            a = np.asarray(data['circles'], dtype=np.float64)
            if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
                return -float('inf')
            xs, ys, rs = [], [], []
            for x, y, r in a:
                if time.monotonic() >= end:
                    return -float('inf')
                xs.append(Fraction(float(x)))
                ys.append(Fraction(float(y)))
                rs.append(Fraction(float(r)))
            width = max(x+r for x, r in zip(xs, rs)) - min(x-r for x, r in zip(xs, rs))
            height = max(y+r for y, r in zip(ys, rs)) - min(y-r for y, r in zip(ys, rs))
            if width + height > 2:
                return -float('inf')
            for p, q in itertools.combinations(a, 2):
                if time.monotonic() >= end:
                    return -float('inf')
                dist = np.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2)
                if dist < p[2] + q[2]:
                    return -float('inf')
            return float(np.sum(a[:, 2]))
        except (ValueError, TypeError, KeyError, OverflowError):
            return -float('inf')

    if init is not None:
        best_score = exact(init)

    def consider(a):
        nonlocal best, best_score
        if time.monotonic() >= end:
            return
        a = np.array(a, dtype=float, copy=True)
        if a.shape != (21, 3) or not np.isfinite(a).all():
            return
        a[:, 2] = np.maximum(a[:, 2], 1e-10)
        d = np.sqrt(np.sum((a[ii, :2]-a[jj, :2])**2, axis=1))
        shrink = min(1.0, float(np.min(d / (a[ii, 2]+a[jj, 2]))))
        if shrink <= 0:
            return
        a[:, 2] *= shrink * (1.0-2e-14)
        lower = np.min(a[:, :2]-a[:, 2, None], axis=0)
        upper = np.max(a[:, :2]+a[:, 2, None], axis=0)
        a[:, :2] -= (lower+upper)*0.5
        perimeter = float(np.sum(upper-lower))
        if perimeter <= 0:
            return
        a *= (2.0-8e-14)/perimeter
        if float(np.sum(a[:, 2])) <= best_score:
            return
        data = {'circles': a.tolist()}
        value = exact(data)
        if value > best_score:
            best, best_score = data, value

    lattice = []
    for row, count in enumerate((5, 4, 5, 4, 3)):
        if time.monotonic() >= end:
            return best
        for col in range(count):
            if time.monotonic() >= end:
                return best
            lattice.append([2.0*col-(count-1), np.sqrt(3.0)*row, 1.0])
    lattice = np.asarray(lattice)
    consider(lattice)

    def encode(a):
        a = np.array(a, dtype=float, copy=True)
        a[:, 2] = np.maximum(a[:, 2], 1e-6)
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        size = hi-lo
        a[:, :2] -= (lo+hi)*0.5
        factor = 1.999999 / np.sum(size)
        a *= factor
        w = float(size[0]*factor + 0.0000005)
        return np.concatenate((a[:, 0], a[:, 1], a[:, 2], [w]))

    def decode(z):
        return np.column_stack((z[:21], z[21:42], z[42:63]))

    gradient = np.zeros(64)
    gradient[42:63] = -1.0
    static = np.zeros((294, 64))
    idx = np.arange(21)
    for block, offset, sign, ws in ((0, 0, 1, 0.5), (1, 0, -1, 0.5), (2, 21, 1, -0.5), (3, 21, -1, -0.5)):
        if time.monotonic() >= end:
            return best
        rr = 210 + block*21 + idx
        static[rr, offset+idx] = sign
        static[rr, 42+idx] = -1
        static[rr, 63] = ws
    static[rows, 42+ii] = -1
    static[rows, 42+jj] = -1
    bounds = [(None, None)]*42 + [(1e-7, 0.5)]*21 + [(0.12, 1.88)]

    def optimize(a, deadline):
        z0 = encode(a)
        last = z0.copy()
        iterations = 0
        def check():
            if time.monotonic() >= deadline:
                raise SearchStop()
        def fun(z):
            nonlocal last
            check()
            last = z.copy()
            return -float(np.sum(z[42:63]))
        def jac(z):
            check()
            return gradient
        def constraints(z):
            check()
            x, y, r, w = z[:21], z[21:42], z[42:63], z[63]
            dx, dy = x[ii]-x[jj], y[ii]-y[jj]
            d = np.sqrt(dx*dx+dy*dy)
            return np.concatenate((d-r[ii]-r[jj], x+w/2-r, -x+w/2-r, y+1-w/2-r, -y+1-w/2-r))
        def cjac(z):
            check()
            dx = z[ii]-z[jj]
            dy = z[21+ii]-z[21+jj]
            d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-15)
            out = static.copy()
            out[rows, ii] = dx/d
            out[rows, jj] = -dx/d
            out[rows, 21+ii] = dy/d
            out[rows, 21+jj] = -dy/d
            return out
        def callback(z):
            nonlocal iterations, last
            check()
            last = z.copy()
            iterations += 1
            if iterations % 7 == 0:
                consider(decode(z))
        try:
            result = minimize(fun, z0, jac=jac, method='SLSQP', bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraints, 'jac': cjac},
                callback=callback,
                options={'maxiter': int(params['maxiter']), 'ftol': 2e-12, 'disp': False})
            last = result.x
        except SearchStop:
            pass
        except (ValueError, FloatingPointError):
            pass
        consider(decode(last))

    initial = lattice
    if init is not None:
        try:
            trial = np.asarray(init['circles'], dtype=float)
            if trial.shape == (21, 3) and np.isfinite(trial).all():
                initial = trial
        except (ValueError, TypeError, KeyError):
            pass
    remaining = max(0.0, end-time.monotonic()-0.025)
    first_end = time.monotonic() + float(params['first_share'])*remaining
    if time.monotonic() < first_end:
        optimize(initial, first_end)
    if time.monotonic() < end-0.05:
        if best_score < 2.32:
            restart = decode(encode(lattice))
            restart[:, :2] += rng.normal(0, 0.002, (21, 2))
        else:
            restart = np.asarray(best['circles'], dtype=float).copy()
            restart[:, :2] += rng.normal(0, float(params['jitter']), (21, 2))
            restart[:, 2] *= 0.97
        optimize(restart, end-0.025)
    return best
