import time
import math
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'solve_slice': (0.20, 0.05, 0.40),
    'iterations': (120, 40, 240),
    'polish_fraction': (0.18, 0.05, 0.35),
}

class StopSearch(Exception):
    pass


def construct(params: dict, seed: int, budget_s: float, init=None):
    started = time.monotonic()
    deadline = started + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best = None
    best_score = -float('inf')
    best_state = None

    def exact_score(obj):
        try:
            if 'circles' in obj:
                c = np.asarray(obj['circles'], dtype=float)
            else:
                c = np.column_stack((obj['centers'], obj['radii']))
            if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
                return -float('inf')
            ii, jj = np.triu_indices(21, 1)
            dx = c[ii, 0] - c[jj, 0]
            dy = c[ii, 1] - c[jj, 1]
            if np.any(np.sqrt(dx * dx + dy * dy) < c[ii, 2] + c[jj, 2]):
                return -float('inf')
            xx = list(map(lambda z: Fraction(float(z)), c[:, 0]))
            yy = list(map(lambda z: Fraction(float(z)), c[:, 1]))
            rr = list(map(lambda z: Fraction(float(z)), c[:, 2]))
            width = max(x+r for x, r in zip(xx, rr)) - min(x-r for x, r in zip(xx, rr))
            height = max(y+r for y, r in zip(yy, rr)) - min(y-r for y, r in zip(yy, rr))
            if width + height > 2:
                return -float('inf')
            return float(np.sum(c[:, 2]))
        except (ValueError, TypeError, KeyError, OverflowError):
            return -float('inf')

    if init is not None:
        best_score = exact_score(init)
        if np.isfinite(best_score):
            best = init

    def repair(c):
        c = np.array(c, dtype=float, copy=True)
        if c.shape != (21, 3) or not np.isfinite(c).all():
            return None
        c[:, 2] = np.maximum(c[:, 2], 1e-12)
        ii, jj = np.triu_indices(21, 1)
        d = c[ii, :2] - c[jj, :2]
        dd = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
        ratio = float(np.min(dd / (c[ii, 2] + c[jj, 2])))
        if ratio <= 0:
            return None
        c[:, 2] *= min(1.0, ratio) * (1.0 - 3e-13)
        lo = np.min(c[:, :2] - c[:, 2, None], axis=0)
        hi = np.max(c[:, :2] + c[:, 2, None], axis=0)
        per = float(np.sum(hi - lo))
        if not per > 0:
            return None
        c[:, :2] -= (lo + hi) * 0.5
        c *= (2.0 / per) * (1.0 - 3e-13)
        return {'centers': c[:, :2].tolist(), 'radii': c[:, 2].tolist()}

    def consider(c, state=None):
        nonlocal best, best_score, best_state
        if time.monotonic() >= deadline:
            return
        obj = repair(c)
        if obj is None:
            return
        score = exact_score(obj)
        if score > best_score:
            best_score = score
            best = obj
            if state is not None:
                best_state = state

    # A safe fallback, independent of the supplied small-circle start.
    fallback = np.array([[float(x), float(y), 0.5] for y in range(3) for x in range(7)])
    consider(fallback)

    def make_problem(a, points):
        # Unique circles: a on y=0, then m circles in y>0.
        m = len(points) - a
        k = a + m
        n = 2*k + m + 1
        yi = np.full(k, -1, dtype=int)
        yi[a:] = np.arange(k, k+m)
        ri = k+m + np.arange(k)
        wi = n-1
        pu, pv, su, sv = [], [], [], []
        for u in range(k):
            if time.monotonic() >= deadline:
                raise StopSearch
            for v in range(u+1, k):
                if time.monotonic() >= deadline:
                    raise StopSearch
                pu.append(u); pv.append(v)
                su.append(0 if u < a else 1)
                sv.append(0 if v < a else 1)
                if u >= a:
                    pu.append(u); pv.append(v)
                    su.append(1); sv.append(-1)
        pu = np.asarray(pu, dtype=int)
        pv = np.asarray(pv, dtype=int)
        su = np.asarray(su, dtype=float)
        sv = np.asarray(sv, dtype=float)
        nc = len(pu)
        DX = np.zeros((nc, n))
        DY = np.zeros((nc, n))
        DR = np.zeros((nc, n))
        rows = np.arange(nc)
        DX[rows, pu] = 1
        DX[rows, pv] -= 1
        good = pu >= a
        DY[rows[good], yi[pu[good]]] = su[good]
        good = pv >= a
        DY[rows[good], yi[pv[good]]] -= sv[good]
        DR[rows, ri[pu]] = 1
        DR[rows, ri[pv]] += 1
        # Linear inequalities: left/right wall, upper wall, and own reflection.
        L = np.zeros((3*k+m, n))
        b = np.zeros(3*k+m)
        for u in range(k):
            if time.monotonic() >= deadline:
                raise StopSearch
            L[u, u] = 1; L[u, ri[u]] = -1
            L[k+u, wi] = 1; L[k+u, u] = -1; L[k+u, ri[u]] = -1
            L[2*k+u, wi] = -0.5; L[2*k+u, ri[u]] = -1
            b[2*k+u] = 1
            if u >= a:
                L[2*k+u, yi[u]] = -1
                L[3*k+u-a, yi[u]] = 1
                L[3*k+u-a, ri[u]] = -1
        objective = np.zeros(n)
        objective[ri[:a]] = -1
        objective[ri[a:]] = -2
        lower = np.full(n, -np.inf)
        upper = np.full(n, np.inf)
        lower[ri] = 1e-8
        lower[k:k+m] = 0
        lower[wi] = 0.08
        upper[wi] = 1.92
        bounds = list(zip(lower, upper))
        cache_x = None
        cache_v = None
        cache_j = None

        def constraints(z):
            nonlocal cache_x, cache_v, cache_j
            if cache_x is None or not np.array_equal(cache_x, z):
                dx = z[pu] - z[pv]
                y = np.zeros(k)
                y[a:] = z[k:k+m]
                dy = su*y[pu] - sv*y[pv]
                dist = np.sqrt(dx*dx + dy*dy + 1e-30)
                cache_v = np.concatenate((dist-z[ri[pu]]-z[ri[pv]], L @ z + b))
                jac = (dx/dist)[:, None]*DX + (dy/dist)[:, None]*DY - DR
                cache_j = np.vstack((jac, L))
                cache_x = z.copy()
            return cache_v

        def jacobian(z):
            constraints(z)
            return cache_j

        def unpack(z):
            out = np.empty((21, 3))
            out[:k, 0] = z[:k]
            out[:a, 1] = 0
            out[a:k, 1] = z[k:k+m]
            out[:k, 2] = z[ri]
            out[k:] = out[a:k]
            out[k:, 1] *= -1
            return out

        p = np.asarray(points, dtype=float)
        full = np.vstack((p, p[a:]*np.array([1., -1., 1.])))
        low = np.min(full[:, :2]-full[:, 2, None], axis=0)
        high = np.max(full[:, :2]+full[:, 2, None], axis=0)
        scale = 2 / np.sum(high-low)
        z = np.zeros(n)
        z[:k] = (p[:, 0]-low[0])*scale
        z[k:k+m] = p[a:, 1]*scale
        z[ri] = p[:, 2]*scale
        z[wi] = (high[0]-low[0])*scale
        return z, objective, bounds, constraints, jacobian, unpack

    def layout(a, counts, phase=0, hole=False):
        p = [[j-(a-1)/2, 0., .5] for j in range(a)]
        for level, count in enumerate(counts, 1):
            if time.monotonic() >= deadline:
                raise StopSearch
            shift = (0.5 if level % 2 else 0.0) if phase else 0.0
            for j in range(count):
                if time.monotonic() >= deadline:
                    raise StopSearch
                p.append([j-(count-1)/2+shift, level*math.sqrt(3)/2, .5])
        if hole:
            p.append([0.5, math.sqrt(3)/6, (2/math.sqrt(3)-1)*.5])
        return p

    def run(problem, z, limit):
        nonlocal best_state
        base, obj, bounds, fun, jac, unpack = problem
        local_deadline = min(deadline, limit)
        last = z.copy()
        calls = 0
        def cb(v):
            nonlocal last, calls
            last = v.copy()
            calls += 1
            if time.monotonic() >= local_deadline:
                raise StopSearch
            if calls % 9 == 0:
                consider(unpack(v), (problem, v.copy()))
        try:
            result = minimize(lambda v: float(obj @ v), z, jac=lambda v: obj,
                              method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': fun, 'jac': jac},
                              callback=cb,
                              options={'maxiter': int(params['iterations']), 'ftol': 3e-12, 'disp': False})
            last = result.x
        except StopSearch:
            pass
        except (ValueError, FloatingPointError):
            pass
        if time.monotonic() < deadline:
            consider(unpack(last), (problem, last.copy()))

    # Different row populations cover compact hexagonal, rectangular,
    # elongated, and small-interstitial basins.
    specs = [(3, [4,3,2], 0, False),
             (5, [4,4], 0, False),
             (3, [3,3,3], 1, False),
             (5, [4,3], 0, True),
             (7, [4,3], 0, False),
             (3, [4,5], 0, False),
             (5, [4,3,1], 0, False),
             (3, [3,3,3], 0, False)]
    portfolio_end = deadline - params['polish_fraction'] * (deadline-started)
    for a, counts, phase, hole in specs:
        if time.monotonic() >= portfolio_end:
            break
        try:
            p = layout(a, counts, phase, hole)
            problem = make_problem(a, p)
            run(problem, problem[0], min(portfolio_end, time.monotonic()+params['solve_slice']))
        except StopSearch:
            break

    if best_state is not None and time.monotonic() < deadline - 0.015:
        problem, z = best_state
        run(problem, z, deadline-0.008)

    # init is returned unchanged unless an exactly verified improvement won.
    if best is not None:
        return best
    # This branch is only relevant to vanishingly small budgets.
    r = 0.0999999999999
    return {'centers': [[(x-3)*0.2, (y-1)*0.2] for y in range(3) for x in range(7)],
            'radii': [r]*21}
