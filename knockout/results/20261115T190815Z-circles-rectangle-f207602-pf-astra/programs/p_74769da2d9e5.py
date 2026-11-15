import time
import math
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'screen_share': (0.42, 0.25, 0.55),
    'contact_margin': (0.12, 0.05, 0.3),
    'penalty': (900.0, 200.0, 3000.0, 'log'),
    'jitter': (0.002, 0.0, 0.015)
}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.69 * budget_s
    stage_end = deadline
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    ar = np.arange(21)
    best = init
    best_score = -float('inf')
    grad = np.zeros(64)
    grad[2:63:3] = -1.0
    bounds = [(None, None), (None, None), (1e-6, 0.6)] * 21 + [(0.15, 1.85)]

    def guard():
        if time.monotonic() >= stage_end:
            raise StopSearch()

    def exact(c):
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return -float('inf')
        d = c[ii, :2] - c[jj, :2]
        if np.any(np.sqrt(d[:, 0]**2 + d[:, 1]**2) < c[ii, 2] + c[jj, 2]):
            return -float('inf')
        left, right, bottom, top = [], [], [], []
        for x, y, r in c:
            if time.monotonic() >= deadline:
                return -float('inf')
            x, y, r = Fraction(float(x)), Fraction(float(y)), Fraction(float(r))
            left.append(x-r)
            right.append(x+r)
            bottom.append(y-r)
            top.append(y+r)
        if max(right)-min(left)+max(top)-min(bottom) > 2:
            return -float('inf')
        return float(np.sum(c[:, 2]))

    if init is not None:
        best_score = exact(np.asarray(init['circles'], dtype=float))

    def repair(z):
        c = np.array(z[:63], dtype=float).reshape(21, 3).copy()
        if not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return None
        d = c[ii, :2] - c[jj, :2]
        ds = np.sqrt(d[:, 0]**2 + d[:, 1]**2)
        f = min(1.0, float(np.min(ds / (c[ii, 2] + c[jj, 2]))))
        if f <= 0:
            return None
        c[:, 2] *= f * (1.0 - 3e-15)
        lo = np.min(c[:, :2] - c[:, 2, None], axis=0)
        hi = np.max(c[:, :2] + c[:, 2, None], axis=0)
        span = float(np.sum(hi-lo))
        if span <= 0:
            return None
        c[:, :2] -= lo
        c *= (2.0/span) * (1.0 - 6e-15)
        return c

    def consider(z):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return -float('inf')
        c = repair(z)
        if c is None:
            return -float('inf')
        score = float(np.sum(c[:, 2]))
        if score > best_score:
            verified = exact(c)
            if verified > best_score:
                best_score = verified
                best = {'circles': c.tolist()}
        return score

    def geometry(z, a=ii, b=jj, jac=False):
        guard()
        c = z[:63].reshape(21, 3)
        x, y, r = c.T
        dx, dy = x[a]-x[b], y[a]-y[b]
        ds = np.sqrt(dx*dx + dy*dy + 1e-30)
        values = np.r_[ds-r[a]-r[b], x-r, z[63]-x-r, y-r, 2-z[63]-y-r]
        if not jac:
            return values
        m = len(a)
        rows = np.arange(m)
        j = np.zeros((m+84, 64))
        j[rows, 3*a] = dx/ds
        j[rows, 3*b] = -dx/ds
        j[rows, 3*a+1] = dy/ds
        j[rows, 3*b+1] = -dy/ds
        j[rows, 3*a+2] = -1
        j[rows, 3*b+2] = -1
        j[m+ar, 3*ar] = 1
        j[m+ar, 3*ar+2] = -1
        j[m+21+ar, 3*ar] = -1
        j[m+21+ar, 3*ar+2] = -1
        j[m+21+ar, 63] = 1
        j[m+42+ar, 3*ar+1] = 1
        j[m+42+ar, 3*ar+2] = -1
        j[m+63+ar, 3*ar+1] = -1
        j[m+63+ar, 3*ar+2] = -1
        j[m+63+ar, 63] = -1
        return values, j

    def make_seed(kind):
        guard()
        if kind in (0, 1):
            xx, yy = np.meshgrid(np.arange(4), np.arange(4))
            base = np.column_stack((2*xx.ravel()+1, 2*yy.ravel()+1, np.ones(16)))
            holes = np.array([(0,1),(1,0),(1,1),(1,2),(2,1)] if kind == 0 else [(0,0),(0,2),(1,1),(2,0),(2,2)])
            extra = np.column_stack((2*holes+2, np.full(5, math.sqrt(2)-1)))
            c = np.vstack((base, extra))
        elif kind == 2:
            rows = np.repeat(np.arange(5), [4,5,4,4,4])
            cols = np.r_[np.arange(4), np.arange(5), np.arange(4), np.arange(4), np.arange(4)]
            c = np.column_stack((2*cols+(rows % 2 == 0), math.sqrt(3)*rows, np.ones(21)))
        else:
            xx, yy = np.meshgrid(np.arange(5), np.arange(5))
            mask = ~(((xx == 0) | (xx == 4)) & ((yy == 0) | (yy == 4)))
            c = np.column_stack((2*xx[mask], 2*yy[mask], np.ones(21)))
        lo = np.min(c[:, :2]-c[:, 2,None], axis=0)
        hi = np.max(c[:, :2]+c[:, 2,None], axis=0)
        c[:, :2] -= lo
        scale = 2.0/float(np.sum(hi-lo))
        c *= scale
        c[:, :2] += rng.normal(0, params['jitter'], (21,2))
        return np.r_[c.ravel(), (hi[0]-lo[0])*scale]

    def polish(z):
        latest = z.copy()
        try:
            guard()
            active = geometry(z)[:210] < params['contact_margin']
            for attempt in range(3):
                guard()
                a, b = ii[active], jj[active]
                def fun(v):
                    guard()
                    return float(grad @ v)
                def callback(v):
                    nonlocal latest
                    guard()
                    latest = v.copy()
                    consider(v)
                result = minimize(fun, latest, jac=lambda v: grad,
                    method='SLSQP', bounds=bounds,
                    constraints={'type':'ineq', 'fun':lambda v: geometry(v,a,b),
                                 'jac':lambda v: geometry(v,a,b,True)[1]},
                    callback=callback,
                    options={'maxiter':160, 'ftol':2e-12, 'disp':False})
                latest = result.x
                consider(latest)
                missed = geometry(latest)[:210] < 1e-7
                if not np.any(missed & ~active):
                    break
                active |= missed
        except StopSearch:
            pass
        return latest

    # First protect any inexpensive local gain, then devote the budget to new basins.
    if init is not None:
        c = np.asarray(init['circles'], dtype=float).copy()
        lo = np.min(c[:,:2]-c[:,2,None], axis=0)
        hi = np.max(c[:,:2]+c[:,2,None], axis=0)
        c[:,:2] -= lo
        stage_end = min(deadline, start + 0.075*budget_s)
        polish(np.r_[c.ravel(), hi[0]-lo[0]])

    pool = []
    screen_end = min(deadline, start + params['screen_share']*budget_s)
    for kind in (0, 2, 3, 1):
        if time.monotonic() >= screen_end:
            break
        stage_end = min(screen_end, time.monotonic() + 0.085*budget_s)
        latest = None
        try:
            z = make_seed(kind)
            latest = z.copy()
            consider(z)
            local_end = stage_end
            for multiplier in (0.035, 0.2, 1.0):
                guard()
                k = params['penalty']*multiplier
                def penalty(v):
                    guard()
                    g, j = geometry(v, jac=True)
                    neg = np.minimum(g, 0.0)
                    return float(grad @ v + k*(neg @ neg)), grad + 2*k*(neg @ j)
                def callback(v):
                    nonlocal latest
                    guard()
                    latest = v.copy()
                result = minimize(penalty, latest, jac=True, method='L-BFGS-B',
                    bounds=bounds, callback=callback,
                    options={'maxiter':65, 'ftol':1e-11, 'gtol':1e-6, 'maxls':20})
                latest = result.x
            stage_end = local_end
        except StopSearch:
            pass
        if latest is not None and time.monotonic() < deadline:
            score = consider(latest)
            pool.append((score, latest.copy()))

    pool.sort(key=lambda item: item[0], reverse=True)
    for index, (_, z) in enumerate(pool):
        now = time.monotonic()
        if now >= deadline-0.025:
            break
        remaining = deadline-now
        stage_end = deadline if index >= 1 else min(deadline, now + 0.62*remaining)
        polish(z)

    if best is None:
        r = 0.099999999999
        rows = np.repeat(np.arange(3), 7)
        cols = np.tile(np.arange(7), 3)
        c = np.column_stack(((2*cols+1)*r, (2*rows+1)*r, np.full(21,r)))
        best = {'circles': c.tolist()}
    return best
