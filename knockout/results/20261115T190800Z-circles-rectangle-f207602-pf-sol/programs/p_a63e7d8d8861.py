import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'first_iterations': (110, 50, 200),
    'restart_noise': (0.018, 0.002, 0.08),
    'tolerance': (2e-13, 1e-14, 1e-8, 'log')
}

class StopSearch(Exception):
    pass

def exact_score(data):
    try:
        c = np.asarray(data['circles'], dtype=float)
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return -float('inf')
        f = [[Fraction(float(v)) for v in row] for row in c]
        wh = sum(max(row[k] + row[2] for row in f) - min(row[k] - row[2] for row in f) for k in (0, 1))
        if wh > 2:
            return -float('inf')
        for i, j in itertools.combinations(range(21), 2):
            dx = c[i, 0] - c[j, 0]
            dy = c[i, 1] - c[j, 1]
            if np.sqrt(dx * dx + dy * dy) < c[i, 2] + c[j, 2]:
                return -float('inf')
        return float(np.sum(c[:, 2]))
    except Exception:
        return -float('inf')

def construct(params, seed, budget_s, init=None):
    began = time.monotonic()
    deadline = began + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    bestscore = exact_score(init)
    bestz = None
    ii, jj = np.triu_indices(21, 1)

    def repair(c):
        nonlocal best, bestscore
        if time.monotonic() >= deadline:
            return
        c = np.asarray(c, dtype=float).copy()
        if not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return
        d = c[ii, :2] - c[jj, :2]
        dist = np.sqrt(np.sum(d*d, axis=1))
        t = min(1.0, float(np.min(dist / (c[ii, 2] + c[jj, 2]))))
        if t <= 0:
            return
        c[:, 2] *= t * (1.0 - 8e-15)
        lo = np.min(c[:, :2] - c[:, 2, None], axis=0)
        hi = np.max(c[:, :2] + c[:, 2, None], axis=0)
        extent = float(np.sum(hi-lo))
        if extent <= 0:
            return
        c[:, :2] -= (lo + hi) / 2
        c *= (2.0 / extent) * (1.0 - 8e-15)
        d = c[ii, :2] - c[jj, :2]
        dist = np.sqrt(np.sum(d*d, axis=1))
        t = min(1.0, float(np.min(dist / (c[ii, 2] + c[jj, 2]))))
        c[:, 2] *= t * (1.0 - 8e-15)
        if float(np.sum(c[:, 2])) <= bestscore:
            return
        data = {'circles': c.tolist()}
        s = exact_score(data)
        if s > bestscore:
            bestscore, best = s, data

    fallback = np.zeros((21, 3))
    fallback[:, 0] = np.arange(21) * 2.0
    fallback[:, 2] = 0.999999999999
    repair(fallback)

    # Reduced circle order: three axis circles, then nine upper circles.
    xidx = np.r_[np.arange(3), np.arange(6, 15)]
    ridx = np.r_[np.arange(3, 6), np.arange(24, 33)]
    yidx = np.r_[np.full(3, -1), np.arange(15, 24)]
    a, b = np.triu_indices(12, 1)
    m = len(a)
    rr = np.arange(m)
    lin = []
    const = []
    for k in range(12):
        v = np.zeros(34); v[xidx[k]] = 1; v[ridx[k]] = -1
        lin.append(v); const.append(0.)
        v = np.zeros(34); v[xidx[k]] = -1; v[ridx[k]] = -1; v[33] = -2
        lin.append(v); const.append(2.)
        v = np.zeros(34); v[ridx[k]] = -1; v[33] = 1
        if k >= 3:
            v[yidx[k]] = -1
        lin.append(v); const.append(0.)
        if k >= 3:
            v = np.zeros(34); v[yidx[k]] = 1; v[ridx[k]] = -1
            lin.append(v); const.append(0.)
    L = np.asarray(lin)
    C = np.asarray(const)
    g = np.zeros(34); g[3:6] = -1; g[24:33] = -2
    bounds = [(0., 2.)] * 34
    for k in ridx:
        bounds[k] = (1e-6, 0.8)
    bounds[33] = (0.12, 0.88)

    def unpack(z):
        x = z[xidx]
        y = np.r_[np.zeros(3), z[15:24]]
        r = z[ridx]
        return x, y, r

    def expand(z):
        x, y, r = unpack(z)
        c = np.column_stack((x, y, r))
        return np.vstack((c, np.column_stack((x[3:], -y[3:], r[3:]))))

    current_end = deadline
    lastz = None
    calls = 0
    def check():
        if time.monotonic() >= current_end:
            raise StopSearch()

    def fun(z):
        check()
        return float(g @ z)

    def constraints(z):
        check()
        x, y, r = unpack(z)
        dx = x[a] - x[b]; dy = y[a] - y[b]
        return np.r_[np.sqrt(dx*dx + dy*dy) - r[a] - r[b], L @ z + C]

    def jac(z):
        check()
        x, y, r = unpack(z)
        dx = x[a] - x[b]; dy = y[a] - y[b]
        d = np.maximum(np.sqrt(dx*dx + dy*dy), 1e-16)
        J = np.zeros((m, 34))
        J[rr, xidx[a]] += dx/d
        J[rr, xidx[b]] -= dx/d
        J[rr, ridx[a]] -= 1
        J[rr, ridx[b]] -= 1
        mask = a >= 3
        J[rr[mask], yidx[a[mask]]] += (dy/d)[mask]
        mask = b >= 3
        J[rr[mask], yidx[b[mask]]] -= (dy/d)[mask]
        return np.vstack((J, L))

    def callback(z):
        nonlocal lastz, calls, bestz
        check()
        lastz = z.copy()
        calls += 1
        if calls % 5 == 0:
            old = bestscore
            repair(expand(z))
            if bestscore > old:
                bestz = z.copy()

    def lattice(inner, outer, phase=0):
        z = np.zeros(34)
        z[:3] = [0.12, 0.5, 0.88]
        z[3:6] = 0.095
        z[6:15] = np.r_[np.linspace(0.10, 0.90, inner), np.linspace(0.12, 0.88, outer)]
        z[15:24] = np.r_[np.full(inner, 0.20), np.full(outer, 0.39)]
        z[24:33] = 0.095
        z[33] = 0.5
        if phase:
            z[:3] = [0.2, 0.5, 0.8]
        return z

    starts = [lattice(5, 4), lattice(4, 5), lattice(5, 4, 1)]
    run = 0
    while time.monotonic() < deadline - 0.025:
        if run < len(starts):
            z = starts[run]
        elif bestz is not None:
            z = bestz.copy()
            z[:3] += rng.normal(0, params['restart_noise'], 3)
            z[6:24] += rng.normal(0, params['restart_noise'], 18)
            z = np.maximum(z, 1e-5)
        else:
            z = starts[0].copy()
        now = time.monotonic()
        remaining = deadline - now
        allowance = min(remaining, (0.52 if run == 0 else 0.40) * budget_s / 2.0)
        if run >= 2:
            allowance = remaining
        current_end = min(deadline - 0.008, now + allowance)
        lastz = z.copy()
        calls = 0
        try:
            result = minimize(fun, z, jac=lambda z: g, bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraints, 'jac': jac},
                method='SLSQP', callback=callback,
                options={'maxiter': int(params['first_iterations']), 'ftol': params['tolerance'], 'disp': False})
            lastz = result.x
        except StopSearch:
            pass
        if lastz is not None and time.monotonic() < deadline:
            old = bestscore
            repair(expand(lastz))
            if bestscore > old:
                bestz = lastz.copy()
        run += 1
    return best
