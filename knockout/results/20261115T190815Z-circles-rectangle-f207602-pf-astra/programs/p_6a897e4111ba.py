import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'excursion': (0.17, 0.06, 0.32),
    'forced_fraction': (0.14, 0.06, 0.24),
    'release_fraction': (0.18, 0.08, 0.30),
    'iterations': (90, 30, 180)
}

class Deadline(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    npairs = len(ii)
    rows = np.arange(npairs)
    ids = np.arange(n)

    def exact_score(data):
        if time.monotonic() >= deadline:
            return -float('inf')
        try:
            a = np.asarray(data['circles'], dtype=np.float64)
            if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
                return -float('inf')
            xx, yy, rr = [], [], []
            for x, y, r in a:
                if time.monotonic() >= deadline:
                    return -float('inf')
                xx.append(Fraction(float(x)))
                yy.append(Fraction(float(y)))
                rr.append(Fraction(float(r)))
            width = max(x+r for x, r in zip(xx, rr)) - min(x-r for x, r in zip(xx, rr))
            height = max(y+r for y, r in zip(yy, rr)) - min(y-r for y, r in zip(yy, rr))
            if width + height > 2:
                return -float('inf')
            for p, q in itertools.combinations(a, 2):
                if time.monotonic() >= deadline:
                    return -float('inf')
                dist = np.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2)
                if dist < p[2] + q[2]:
                    return -float('inf')
            return float(np.sum(a[:, 2]))
        except (ValueError, TypeError, KeyError, OverflowError):
            return -float('inf')

    if init is None:
        a = np.array([[0.1 + 0.2*(k % 5), 0.1 + 0.2*(k // 5), 0.1*(1-1e-12)] for k in range(21)])
        best = {'circles': a.tolist()}
    else:
        best = init
        a = np.asarray(init['circles'], dtype=float).copy()
    best_score = exact_score(best)
    if time.monotonic() >= deadline:
        return best

    lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
    hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
    spans = hi-lo
    a[:, :2] -= lo
    scale = 2.0 / np.sum(spans)
    a *= scale
    base = np.r_[a[:, 0], a[:, 1], a[:, 2], spans[0]*scale]

    def consider(z):
        nonlocal best, best_score
        if time.monotonic() >= deadline - 0.008:
            return
        c = np.column_stack((z[:n], z[n:2*n], z[2*n:3*n])).copy()
        if not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return
        d = np.sqrt((c[ii, 0]-c[jj, 0])**2 + (c[ii, 1]-c[jj, 1])**2)
        factor = min(1.0, float(np.min(d/(c[ii, 2]+c[jj, 2]))))
        c[:, 2] *= factor * (1.0-2e-14)
        lower = np.min(c[:, :2]-c[:, 2, None], axis=0)
        upper = np.max(c[:, :2]+c[:, 2, None], axis=0)
        extent = float(np.sum(upper-lower))
        if extent <= 0:
            return
        estimated = float(np.sum(c[:, 2]))*2.0/extent
        if estimated <= best_score:
            return
        c[:, :2] -= lower
        c *= (2.0/extent)*(1.0-2e-14)
        candidate = {'circles': c.tolist()}
        score = exact_score(candidate)
        if score > best_score:
            best_score, best = score, candidate

    # Constant Jacobian of the four rectangle-wall inequalities.
    wall_jac = np.zeros((4*n, 3*n+1))
    wall_jac[ids, ids] = 1
    wall_jac[ids, 2*n+ids] = -1
    wall_jac[n+ids, ids] = -1
    wall_jac[n+ids, 2*n+ids] = -1
    wall_jac[n+ids, -1] = 1
    wall_jac[2*n+ids, n+ids] = 1
    wall_jac[2*n+ids, 2*n+ids] = -1
    wall_jac[3*n+ids, n+ids] = -1
    wall_jac[3*n+ids, 2*n+ids] = -1
    wall_jac[3*n+ids, -1] = -1
    gradient = np.r_[np.zeros(2*n), -np.ones(n), 0.0]
    phase_end = deadline
    cache_z = None
    cache_values = None
    cache_jac = None

    def check():
        if time.monotonic() >= min(phase_end, deadline-0.012):
            raise Deadline()

    def calculate(z):
        nonlocal cache_z, cache_values, cache_jac
        check()
        if cache_z is not None and np.array_equal(z, cache_z):
            return
        x, y, r, w = z[:n], z[n:2*n], z[2*n:3*n], z[-1]
        dx, dy = x[ii]-x[jj], y[ii]-y[jj]
        d = np.sqrt(dx*dx+dy*dy)
        safe = np.maximum(d, 1e-15)
        j = np.zeros((npairs+4*n, 3*n+1))
        j[rows, ii] = dx/safe
        j[rows, jj] = -dx/safe
        j[rows, n+ii] = dy/safe
        j[rows, n+jj] = -dy/safe
        j[rows, 2*n+ii] = -1
        j[rows, 2*n+jj] = -1
        j[npairs:] = wall_jac
        cache_values = np.r_[d-r[ii]-r[jj], x-r, w-x-r, y-r, 2-w-y-r]
        cache_jac = j
        cache_z = z.copy()

    def constraints(z):
        calculate(z)
        return cache_values

    def jacobian(z):
        calculate(z)
        return cache_jac

    def objective(z):
        check()
        return -float(np.sum(z[2*n:3*n]))

    def optimize(z, fixed_width, seconds):
        nonlocal phase_end, cache_z
        phase_end = min(deadline-0.015, time.monotonic()+seconds)
        cache_z = None
        last = z.copy()
        calls = 0
        bounds = [(0.0, 2.0)]*(2*n) + [(1e-5, 0.45)]*n
        bounds += [(0.45, 1.55) if fixed_width is None else (fixed_width, fixed_width)]
        def callback(v):
            nonlocal last, calls
            check()
            last = v.copy()
            calls += 1
            if calls % 5 == 0:
                consider(v)
        try:
            result = minimize(objective, z, jac=lambda v: gradient, method='SLSQP',
                              bounds=bounds,
                              constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                              callback=callback,
                              options={'maxiter': int(params['iterations']), 'ftol': 3e-12 if fixed_width is None else 2e-9, 'disp': False})
            last = result.x.copy()
        except Deadline:
            pass
        consider(last)
        return last

    # Opposite finite aspect excursions need not land in equivalent contact graphs.
    trial = 0
    while time.monotonic() < deadline-0.065:
        direction = -1.0 if trial % 2 == 0 else 1.0
        magnitude = float(params['excursion']) * (1.0 + 0.4*(trial//2))
        target = float(np.clip(base[-1]+direction*magnitude, 0.57, 1.43))
        z = base.copy()
        sx = target/base[-1]
        sy = (2-target)/(2-base[-1])
        z[:n] *= sx
        z[n:2*n] *= sy
        z[2*n:3*n] *= min(sx, sy)
        z[-1] = target
        if trial >= 2:
            chosen = rng.choice(n, size=4, replace=False)
            z[chosen] += rng.normal(0, 0.008, len(chosen))
            z[n+chosen] += rng.normal(0, 0.008, len(chosen))
        z = optimize(z, target, float(params['forced_fraction'])*budget_s)
        if time.monotonic() >= deadline-0.035:
            break
        z = optimize(z, None, float(params['release_fraction'])*budget_s)
        trial += 1
    return best
