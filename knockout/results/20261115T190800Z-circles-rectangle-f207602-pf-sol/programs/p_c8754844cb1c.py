import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'iterations': (300, 100, 600),
    'ftol': (2e-12, 1e-14, 1e-8, 'log'),
    'time_fraction': (0.69, 0.5, 0.72)
}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + float(params['time_fraction']) * budget_s
    n = 21
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    rows = np.arange(m)
    ids = np.arange(n)

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def canonical(obj):
        if 'circles' in obj:
            return np.array(obj['circles'], dtype=float)
        v = np.asarray(obj['state'], dtype=float)
        return np.column_stack((v[:21], v[21:42], v[42:63]))

    def exact_score(c):
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return -float('inf')
        d = c[ii, :2] - c[jj, :2]
        dist = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
        if np.any(dist < c[ii, 2] + c[jj, 2]):
            return -float('inf')
        lo_x = []; hi_x = []; lo_y = []; hi_y = []
        for x, y, r in c:
            check()
            x, y, r = Fraction(float(x)), Fraction(float(y)), Fraction(float(r))
            lo_x.append(x-r); hi_x.append(x+r)
            lo_y.append(y-r); hi_y.append(y+r)
        if max(hi_x)-min(lo_x)+max(hi_y)-min(lo_y) > 2:
            return -float('inf')
        return float(np.sum(c[:, 2]))

    # A strictly feasible fallback also protects against malformed warm starts.
    fallback = np.array([[(i % 5 - 2)*0.2, (i // 5 - 2)*0.2, 0.1*(1-2e-13)] for i in range(21)])
    best = {'circles': fallback.tolist()}
    best_score = exact_score(fallback)
    base = fallback.copy()
    if init is not None:
        try:
            c = canonical(init)
            score = exact_score(c)
            if score >= best_score:
                best, best_score = init, score
            if c.shape == (21, 3) and np.isfinite(c).all() and np.all(c[:, 2] > 0):
                base = c.copy()
        except StopSearch:
            return best
        except (ValueError, TypeError, KeyError, IndexError):
            pass

    def consider(v):
        nonlocal best, best_score
        check()
        c = np.column_stack((v[:21], v[21:42], v[42:63])).copy()
        if not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return
        delta = c[ii, :2] - c[jj, :2]
        ds = np.sqrt(delta[:, 0]**2 + delta[:, 1]**2)
        factor = min(1.0, float(np.min(ds / (c[ii, 2]+c[jj, 2]))))
        if factor <= 0:
            return
        c[:, 2] *= factor*(1-3e-14)
        lower = np.min(c[:, :2]-c[:, 2, None], axis=0)
        upper = np.max(c[:, :2]+c[:, 2, None], axis=0)
        c[:, :2] -= (lower+upper)/2
        perimeter = float(np.sum(upper-lower))
        c *= min(1.0, 2.0/perimeter)*(1-3e-14)
        if float(np.sum(c[:, 2])) <= best_score:
            return
        score = exact_score(c)
        if score > best_score:
            best_score = score
            best = {'circles': c.tolist()}

    lower = np.min(base[:, :2]-base[:, 2, None], axis=0)
    upper = np.max(base[:, :2]+base[:, 2, None], axis=0)
    base[:, :2] -= (lower+upper)/2
    span = upper-lower
    if np.sum(span) > 2:
        base *= (2-1e-12)/np.sum(span)
        span *= (2-1e-12)/np.sum(span)
    w = np.clip((span[0]+2-span[1])/2, 0.02, 1.98)
    v0 = np.concatenate((base[:, 0], base[:, 1], base[:, 2], [w]))
    objective_jac = np.zeros(64)
    objective_jac[42:63] = -1

    wall_jac = np.zeros((84, 64))
    wall_jac[ids, ids] = 1
    wall_jac[21+ids, ids] = -1
    wall_jac[42+ids, 21+ids] = 1
    wall_jac[63+ids, 21+ids] = -1
    wall_jac[ids, 42+ids] = -1
    wall_jac[21+ids, 42+ids] = -1
    wall_jac[42+ids, 42+ids] = -1
    wall_jac[63+ids, 42+ids] = -1
    wall_jac[:42, 63] = 0.5
    wall_jac[42:, 63] = -0.5

    def fun(v):
        check()
        return -float(np.sum(v[42:63]))

    def con(v):
        check()
        x, y, r, w = v[:21], v[21:42], v[42:63], v[63]
        dx, dy, sr = x[ii]-x[jj], y[ii]-y[jj], r[ii]+r[jj]
        return np.concatenate((dx*dx+dy*dy-sr*sr,
                               w/2+x-r, w/2-x-r,
                               1-w/2+y-r, 1-w/2-y-r))

    def jac(v):
        check()
        x, y, r = v[:21], v[21:42], v[42:63]
        dx, dy, sr = 2*(x[ii]-x[jj]), 2*(y[ii]-y[jj]), -2*(r[ii]+r[jj])
        a = np.zeros((m+84, 64))
        a[rows, ii] = dx
        a[rows, jj] = -dx
        a[rows, 21+ii] = dy
        a[rows, 21+jj] = -dy
        a[rows, 42+ii] = sr
        a[rows, 42+jj] = sr
        a[m:] = wall_jac
        return a

    bounds = [(-1.0, 1.0)]*42 + [(1e-7, 0.5)]*21 + [(0.02, 1.98)]
    try:
        consider(v0)
        result = minimize(fun, v0, jac=lambda v: objective_jac,
                          method='SLSQP', bounds=bounds,
                          constraints={'type': 'ineq', 'fun': con, 'jac': jac},
                          callback=consider,
                          options={'maxiter': int(params['iterations']),
                                   'ftol': float(params['ftol']), 'disp': False})
        consider(result.x)
    except StopSearch:
        pass
    return best
