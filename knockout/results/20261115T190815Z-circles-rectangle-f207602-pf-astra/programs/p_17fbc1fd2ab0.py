import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'aspect_excursion': (1.18, 1.06, 1.36),
    'shear': (0.024, 0.0, 0.10),
    'strained_iterations': (32, 12, 65),
    'released_iterations': (65, 25, 110),
    'time_fraction': (0.70, 0.50, 0.75)
}

class Deadline(Exception):
    pass

def construct(params: dict, seed: int, budget_s: float, init=None):
    start = time.monotonic()
    deadline = start + float(params['time_fraction']) * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    ids = np.arange(n)
    rows = np.arange(m)

    def check():
        if time.monotonic() >= deadline:
            raise Deadline()

    def exact_score(data):
        check()
        a = np.asarray(data['circles'], dtype=np.float64)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        xs, ys, rs = [], [], []
        for x, y, r in a:
            check()
            xs.append(Fraction(float(x)))
            ys.append(Fraction(float(y)))
            rs.append(Fraction(float(r)))
        width = max(x+r for x, r in zip(xs, rs)) - min(x-r for x, r in zip(xs, rs))
        height = max(y+r for y, r in zip(ys, rs)) - min(y-r for y, r in zip(ys, rs))
        if width + height > 2:
            return -float('inf')
        for i, j in zip(ii, jj):
            check()
            c1, c2 = a[i], a[j]
            dist = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
            if dist < c1[2] + c2[2]:
                return -float('inf')
        return float(np.sum(a[:, 2]))

    if init is None:
        a = np.empty((21, 3), dtype=float)
        for k in range(21):
            if time.monotonic() >= deadline:
                break
            a[k] = [(k % 7 + 0.5)*0.2, (k // 7 + 0.5)*0.2, 0.09999999999999]
        best = {'circles': a.tolist()}
    else:
        best = init
        a = np.asarray(init['circles'], dtype=float).copy()
    best_score = -float('inf')

    try:
        best_score = exact_score(best)
        lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
        hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
        a[:, :2] -= lo
        dims = hi-lo
        if dims[0] < dims[1]:
            a[:, :2] = a[:, [1, 0]]
            dims = dims[::-1]
        a *= 2.0 / np.sum(dims)
        dims *= 2.0 / np.sum(dims)
        base = np.concatenate((a[:, 0], a[:, 1], a[:, 2], [dims[0]]))
        grad = np.zeros(64)
        grad[42:63] = -1.0

        def objective(z):
            check()
            return -float(np.sum(z[42:63]))

        def objective_jac(z):
            check()
            return grad

        wall_jac = np.zeros((84, 64))
        wall_jac[ids, ids] = 1
        wall_jac[ids, 42+ids] = -1
        wall_jac[21+ids, ids] = -1
        wall_jac[21+ids, 42+ids] = -1
        wall_jac[21+ids, 63] = 1
        wall_jac[42+ids, 21+ids] = 1
        wall_jac[42+ids, 42+ids] = -1
        wall_jac[63+ids, 21+ids] = -1
        wall_jac[63+ids, 42+ids] = -1
        wall_jac[63+ids, 63] = -1
        cached_z = None
        cached_c = None
        cached_j = None

        def calculate(z):
            nonlocal cached_z, cached_c, cached_j
            check()
            if cached_z is not None and np.array_equal(z, cached_z):
                return cached_c, cached_j
            x, y, r, w = z[:21], z[21:42], z[42:63], z[63]
            dx, dy = x[ii]-x[jj], y[ii]-y[jj]
            d = np.sqrt(dx*dx + dy*dy)
            c = np.concatenate((d-r[ii]-r[jj], x-r, w-x-r, y-r, 2-w-y-r))
            j = np.zeros((m+84, 64))
            inv = 1.0 / np.maximum(d, 1e-100)
            ux, uy = dx*inv, dy*inv
            j[rows, ii] = ux
            j[rows, jj] = -ux
            j[rows, 21+ii] = uy
            j[rows, 21+jj] = -uy
            j[rows, 42+ii] = -1
            j[rows, 42+jj] = -1
            j[m:] = wall_jac
            cached_z, cached_c, cached_j = z.copy(), c, j
            return c, j

        def consider(z):
            nonlocal best, best_score
            check()
            b = np.column_stack((z[:21], z[21:42], z[42:63])).copy()
            if not np.isfinite(b).all() or np.any(b[:, 2] <= 0):
                return
            dx = b[ii, 0]-b[jj, 0]
            dy = b[ii, 1]-b[jj, 1]
            d = np.sqrt(dx*dx + dy*dy)
            ratios = d / (b[ii, 2]+b[jj, 2])
            shrink = min(1.0, float(np.min(ratios)))
            if shrink <= 0:
                return
            b[:, 2] *= shrink * (1.0-8e-15)
            lower = np.min(b[:, :2]-b[:, 2, None], axis=0)
            upper = np.max(b[:, :2]+b[:, 2, None], axis=0)
            perimeter = float(np.sum(upper-lower))
            b[:, :2] -= lower
            b *= (2.0/perimeter)*(1.0-8e-15)
            if float(np.sum(b[:, 2])) <= best_score:
                return
            candidate = {'circles': b.tolist()}
            score = exact_score(candidate)
            if score > best_score:
                best, best_score = candidate, score

        constraints = {'type': 'ineq', 'fun': lambda z: calculate(z)[0], 'jac': lambda z: calculate(z)[1]}
        bounds_free = [(0.0, 2.0)]*42 + [(1e-7, 1.0)]*21 + [(0.35, 1.65)]

        def optimize(z, width, count):
            check()
            bounds = list(bounds_free)
            if width is not None:
                bounds[63] = (width, width)
            result = minimize(objective, z, jac=objective_jac, method='SLSQP',
                              bounds=bounds, constraints=constraints,
                              callback=consider,
                              options={'maxiter': int(count), 'ftol': 3e-12, 'disp': False})
            consider(result.x)
            return result.x

        target = float(params['aspect_excursion'])
        targets = (target, 2.0-target, min(1.42, target+0.10))
        for cycle, width in enumerate(targets):
            check()
            z = base.copy()
            old_w = z[63]
            z[:21] *= width/old_w
            z[21:42] *= (2.0-width)/(2.0-old_w)
            sign = 1.0 if rng.random() < 0.5 else -1.0
            z[:21] += sign*float(params['shear'])*(z[21:42]-(2.0-width)/2.0)
            z[42:63] *= min(width/old_w, (2.0-width)/(2.0-old_w))
            z[63] = width
            strained = optimize(z, width, params['strained_iterations'])
            check()
            optimize(strained, None, params['released_iterations'])
    except Deadline:
        pass
    return best
