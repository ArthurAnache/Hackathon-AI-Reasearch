import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'first_iterations': (100, 40, 180),
    'restart_iterations': (55, 20, 120),
    'perturbation': (0.014, 0.001, 0.08, 'log'),
    'lattice_threshold': (2.3657, 2.30, 2.366),
}


class StopSearch(Exception):
    pass


def construct(params: dict, seed: int, budget_s: float, init=None):
    deadline = time.monotonic() + 0.69 * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    ar = np.arange(n)
    pr = np.arange(len(ii))
    best = None
    best_array = None
    best_score = -float('inf')
    pending = None
    pending_score = -float('inf')

    def score(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        dx = a[ii, 0] - a[jj, 0]
        dy = a[ii, 1] - a[jj, 1]
        if np.any(np.sqrt(dx*dx + dy*dy) < a[ii, 2] + a[jj, 2]):
            return -float('inf')
        xs = [Fraction(float(v)) for v in a[:, 0]]
        ys = [Fraction(float(v)) for v in a[:, 1]]
        rs = [Fraction(float(v)) for v in a[:, 2]]
        w = max(x+r for x, r in zip(xs, rs)) - min(x-r for x, r in zip(xs, rs))
        h = max(y+r for y, r in zip(ys, rs)) - min(y-r for y, r in zip(ys, rs))
        if w+h > 2:
            return -float('inf')
        return float(np.sum(a[:, 2]))

    def accept(a, original=None):
        nonlocal best, best_array, best_score
        s = score(a)
        if s > best_score:
            best_score = s
            best_array = a.copy()
            best = original if original is not None else {'circles': a.tolist()}

    def repaired(a):
        if a.shape != (21, 3) or not np.isfinite(a).all():
            return None
        a = a.copy()
        a[:, 2] = np.maximum(a[:, 2], 1e-10)
        dx = a[ii, 0]-a[jj, 0]
        dy = a[ii, 1]-a[jj, 1]
        d = np.sqrt(dx*dx+dy*dy)
        f = min(1.0, float(np.min(d/(a[ii, 2]+a[jj, 2]))))
        if f <= 0:
            return None
        a[:, 2] *= f*(1-5e-14)
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        extent = float(np.sum(hi-lo))
        if extent <= 0:
            return None
        a[:, :2] -= lo
        a *= (2/extent)*(1-5e-14)
        return a

    def consider(a):
        nonlocal pending, pending_score
        a = repaired(a)
        if a is not None:
            s = float(np.sum(a[:, 2]))
            if s > max(best_score, pending_score):
                pending, pending_score = a, s

    def flush():
        nonlocal pending, pending_score
        if pending is not None:
            accept(pending)
            pending = None
        pending_score = best_score

    if init is not None:
        try:
            a = np.asarray(init['circles'], dtype=float)
            accept(a, init)
            consider(a)
            flush()
        except (KeyError, TypeError, ValueError, OverflowError):
            pass

    def lattice(pattern):
        points = []
        for row, count in enumerate(pattern):
            if time.monotonic() >= deadline:
                return None
            offset = 1.0 if row % 2 == 0 else 2.0
            if count == 3:
                offset += 2
            for k in range(count):
                if time.monotonic() >= deadline:
                    return None
                points.append([offset+2*k, 1+np.sqrt(3)*row, 1.0])
        return repaired(np.asarray(points))

    if best_array is None:
        a = np.column_stack(((ar % 7)*0.2+0.1, (ar // 7)*0.2+0.1, np.full(n, 0.1)))
        a = repaired(a)
        accept(a)

    def encode(a):
        a = a.copy()
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        scale = 2/float(np.sum(hi-lo))
        a[:, :2] -= lo
        a *= scale
        return np.r_[a[:, 0], a[:, 1], a[:, 2], (hi[0]-lo[0])*scale]

    def decode(z):
        return np.column_stack((z[:21], z[21:42], z[42:63]))

    base = np.zeros((294, 64))
    base[pr, 42+ii] = -1
    base[pr, 42+jj] = -1
    base[210+ar, ar] = 1
    base[210+ar, 42+ar] = -1
    base[231+ar, ar] = -1
    base[231+ar, 42+ar] = -1
    base[231+ar, 63] = 1
    base[252+ar, 21+ar] = 1
    base[252+ar, 42+ar] = -1
    base[273+ar, 21+ar] = -1
    base[273+ar, 42+ar] = -1
    base[273+ar, 63] = -1
    grad = np.zeros(64)
    grad[42:63] = -1
    bounds = [(None, None)]*42 + [(1e-8, 0.65)]*21 + [(0.15, 1.85)]
    attempt = 0
    used_lattice = False
    patterns = [[5, 4, 5, 4, 3], [4, 4, 5, 4, 4], [6, 5, 5, 5]]

    while time.monotonic() < deadline:
        use_lattice = attempt > 0 and not used_lattice and best_score < float(params['lattice_threshold'])
        if use_lattice:
            a = lattice(patterns[0])
            used_lattice = True
            if a is None:
                break
            z0 = encode(a)
        else:
            z0 = encode(best_array)
            if attempt:
                amp = float(params['perturbation'])*(1 if attempt % 3 else 2.5)
                z0[:42] += rng.normal(0, amp, 42)
                z0[42:63] *= np.exp(rng.normal(0, amp, 21))
                z0[63] = np.clip(z0[63]+rng.normal(0, amp), 0.2, 1.8)
        cached_z = None
        cached_c = None
        cached_j = None
        last = z0.copy()
        iteration = 0

        def geometry(z):
            nonlocal cached_z, cached_c, cached_j
            if time.monotonic() >= deadline:
                raise StopSearch()
            if cached_z is not None and np.array_equal(z, cached_z):
                return cached_c, cached_j
            x, y, r, w = z[:21], z[21:42], z[42:63], z[63]
            dx, dy = x[ii]-x[jj], y[ii]-y[jj]
            d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-15)
            c = np.r_[d-r[ii]-r[jj], x-r, w-x-r, y-r, 2-w-y-r]
            j = base.copy()
            j[pr, ii] = dx/d
            j[pr, jj] = -dx/d
            j[pr, 21+ii] = dy/d
            j[pr, 21+jj] = -dy/d
            cached_z, cached_c, cached_j = z.copy(), c, j
            return c, j

        def callback(z):
            nonlocal last, iteration
            last = z.copy()
            iteration += 1
            if time.monotonic() >= deadline:
                raise StopSearch()
            if iteration % 3 == 0:
                consider(decode(z))

        try:
            result = minimize(lambda z: -np.sum(z[42:63]), z0,
                              jac=lambda z: grad, method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': lambda z: geometry(z)[0],
                                           'jac': lambda z: geometry(z)[1]},
                              callback=callback,
                              options={'maxiter': int(params['first_iterations'] if attempt == 0 else params['restart_iterations']),
                                       'ftol': 8e-13, 'disp': False})
            consider(decode(result.x))
        except StopSearch:
            consider(decode(last))
            flush()
            break
        except (ValueError, FloatingPointError):
            consider(decode(last))
        flush()
        attempt += 1
    flush()
    return best
