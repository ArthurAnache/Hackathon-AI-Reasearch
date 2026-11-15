import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'time_fraction': (0.69, 0.5, 0.75),
    'iterations': (110, 30, 250),
    'perturbation': (0.026, 0.001, 0.10, 'log'),
}

class StopSearch(Exception):
    pass

def construct(params: dict, seed: int, budget_s: float, init=None):
    start = time.monotonic()
    deadline = start + float(params['time_fraction']) * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    ar = np.arange(n)
    rows = np.arange(m)
    best = None
    best_score = -float('inf')
    best_array = None

    def exact(data):
        try:
            c = np.asarray(data['circles'], dtype=np.float64)
            if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
                return -float('inf')
            dx = c[ii, 0] - c[jj, 0]
            dy = c[ii, 1] - c[jj, 1]
            if np.any(np.sqrt(dx*dx + dy*dy) < c[ii, 2] + c[jj, 2]):
                return -float('inf')
            xl, xr, yl, yr = [], [], [], []
            for x, y, r in c:
                if time.monotonic() > start + 0.90 * budget_s:
                    return -float('inf')
                x, y, r = Fraction(float(x)), Fraction(float(y)), Fraction(float(r))
                xl.append(x-r)
                xr.append(x+r)
                yl.append(y-r)
                yr.append(y+r)
            if max(xr)-min(xl)+max(yr)-min(yl) > 2:
                return -float('inf')
            return float(np.sum(c[:, 2]))
        except (ValueError, TypeError, KeyError, OverflowError):
            return -float('inf')

    if init is not None:
        best_score = exact(init)
        if np.isfinite(best_score):
            best = init
            best_array = np.asarray(init['circles'], dtype=float).copy()

    def consider(c):
        nonlocal best, best_score, best_array
        if time.monotonic() >= deadline:
            return
        c = np.asarray(c, dtype=float).copy()
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return
        d = np.hypot(c[ii, 0]-c[jj, 0], c[ii, 1]-c[jj, 1])
        ratio = float(np.min(d/(c[ii, 2]+c[jj, 2])))
        c[:, 2] *= min(1.0, ratio) * (1.0-2e-14)
        lo = np.min(c[:, :2]-c[:, 2, None], axis=0)
        hi = np.max(c[:, :2]+c[:, 2, None], axis=0)
        perimeter = float(np.sum(hi-lo))
        if perimeter <= 0 or not np.isfinite(perimeter):
            return
        c[:, :2] -= lo
        c *= (2.0/perimeter)*(1.0-2e-14)
        if float(np.sum(c[:, 2])) <= best_score:
            return
        candidate = {'circles': c.tolist()}
        score = exact(candidate)
        if score > best_score:
            best, best_score, best_array = candidate, score, c.copy()

    if best_array is None:
        fallback = np.empty((n, 3))
        for k in range(n):
            if time.monotonic() >= deadline:
                break
            fallback[k] = [0.1+0.2*(k % 7), 0.1+0.2*(k // 7), 0.1]
        consider(fallback)
    if best_array is None:
        return init

    def encode(c):
        lo = np.min(c[:, :2]-c[:, 2, None], axis=0)
        hi = np.max(c[:, :2]+c[:, 2, None], axis=0)
        extent = hi-lo
        scale = 2.0/float(np.sum(extent))
        z = np.empty(64)
        z[:21] = (c[:, 0]-lo[0])*scale
        z[21:42] = (c[:, 1]-lo[1])*scale
        z[42:63] = c[:, 2]*scale
        z[63] = extent[0]*scale
        return z

    def decode(z):
        return np.column_stack((z[:21], z[21:42], z[42:63]))

    objective_gradient = np.zeros(64)
    objective_gradient[42:63] = -1.0

    def check_time():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def objective(z):
        check_time()
        return -float(np.sum(z[42:63]))

    def gradient(z):
        check_time()
        return objective_gradient

    def constraints(z):
        check_time()
        x, y, r, w = z[:21], z[21:42], z[42:63], z[63]
        d = np.sqrt((x[ii]-x[jj])**2+(y[ii]-y[jj])**2)
        return np.concatenate((d-r[ii]-r[jj], x-r, w-x-r, y-r, 2.0-w-y-r))

    def jacobian(z):
        check_time()
        x, y = z[:21], z[21:42]
        dx, dy = x[ii]-x[jj], y[ii]-y[jj]
        d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-15)
        jac = np.zeros((m+84, 64))
        jac[rows, ii] = dx/d
        jac[rows, jj] = -dx/d
        jac[rows, 21+ii] = dy/d
        jac[rows, 21+jj] = -dy/d
        jac[rows, 42+ii] = -1.0
        jac[rows, 42+jj] = -1.0
        jac[m+ar, ar] = 1.0
        jac[m+ar, 42+ar] = -1.0
        jac[m+21+ar, ar] = -1.0
        jac[m+21+ar, 42+ar] = -1.0
        jac[m+21+ar, 63] = 1.0
        jac[m+42+ar, 21+ar] = 1.0
        jac[m+42+ar, 42+ar] = -1.0
        jac[m+63+ar, 21+ar] = -1.0
        jac[m+63+ar, 42+ar] = -1.0
        jac[m+63+ar, 63] = -1.0
        return jac

    bounds = [(0.0, 2.0)]*42 + [(1e-6, 0.7)]*21 + [(0.15, 1.85)]
    original = best_array.copy()
    trial = 0
    while time.monotonic() < deadline:
        base = best_array if trial % 4 != 3 else original
        z = encode(base)
        if trial:
            amplitude = float(params['perturbation']) * (0.5, 1.0, 1.8, 2.7)[(trial-1) % 4]
            z[:42] += rng.normal(0.0, amplitude, 42)
            z[42:63] *= np.exp(rng.normal(0.0, amplitude*1.4, 21))
            z[63] += rng.normal(0.0, amplitude)
            z[:42] = np.clip(z[:42], 1e-5, 1.99999)
            z[42:63] = np.clip(z[42:63], 1e-6, 0.699)
            z[63] = np.clip(z[63], 0.151, 1.849)
        last = [z.copy()]
        def callback(v):
            check_time()
            last[0] = v.copy()
            consider(decode(v))
        try:
            result = minimize(objective, z, jac=gradient, method='SLSQP',
                              bounds=bounds,
                              constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                              callback=callback,
                              options={'maxiter': int(params['iterations']), 'ftol': 2e-11, 'disp': False})
            consider(decode(result.x))
        except StopSearch:
            break
        except (ValueError, FloatingPointError):
            if time.monotonic() >= deadline:
                break
        trial += 1
    return best
