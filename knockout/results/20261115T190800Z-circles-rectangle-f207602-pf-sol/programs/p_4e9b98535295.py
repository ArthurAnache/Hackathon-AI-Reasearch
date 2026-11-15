import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"seed_fraction": (0.23, 0.12, 0.32), "initial_fraction": (0.31, 0.15, 0.40), "iterations": (130, 50, 220), "tolerance": (2e-12, 1e-14, 1e-8, "log")}

class Deadline(Exception):
    pass

def canonical(obj):
    if obj is None:
        return None
    if 'circles' in obj:
        return obj
    rows = []
    for x, y, r, m in obj['orbits']:
        rows.append([x, y, r])
        if m == 2:
            rows.append([x, -y, r])
    return {'circles': rows}

def score(obj):
    try:
        a = np.asarray(canonical(obj)['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        i, j = np.triu_indices(21, 1)
        d = np.sqrt((a[i, 0]-a[j, 0])**2 + (a[i, 1]-a[j, 1])**2)
        if np.any(d < a[i, 2]+a[j, 2]):
            return -float('inf')
        x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
        w = max(u+v for u, v in zip(x, r))-min(u-v for u, v in zip(x, r))
        h = max(u+v for u, v in zip(y, r))-min(u-v for u, v in zip(y, r))
        if w+h > 2:
            return -float('inf')
        return float(np.sum(a[:, 2]))
    except (ValueError, TypeError, KeyError, OverflowError):
        return -float('inf')

def construct(params, seed, budget_s, init=None):
    began = time.monotonic()
    end = began + 0.70*budget_s
    best = init
    best_value = score(init)
    ii, jj = np.triu_indices(21, 1)
    def check(limit=end):
        if time.monotonic() >= min(limit, end):
            raise Deadline()
    def retain(a):
        nonlocal best, best_value
        check()
        a = np.asarray(a, float).copy()
        if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return
        d = np.sqrt((a[ii, 0]-a[jj, 0])**2+(a[ii, 1]-a[jj, 1])**2)
        a[:, 2] *= min(1., float(np.min(d/(a[ii, 2]+a[jj, 2]))))*(1.-2e-14)
        if np.any(a[:, 2] <= 0):
            return
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        a[:, :2] -= (lo+hi)/2
        a *= min(1., 2./float(np.sum(hi-lo)))*(1.-2e-14)
        if float(np.sum(a[:, 2])) <= best_value:
            return
        obj = {'circles': a.tolist()}
        val = score(obj)
        if val > best_value:
            best, best_value = obj, val
    fallback = np.array([[.2*(k % 7-3), .2*(k//7-1), .0999999999999] for k in range(21)])
    try:
        retain(fallback)
    except Deadline:
        return best

    def optimize(a, symmetric, limit, maxiter):
        check(limit)
        if symmetric:
            gids = np.r_[np.arange(3), np.repeat(np.arange(3, 12), 2)]
            signs = np.r_[np.zeros(3), np.tile([1., -1.], 9)]
            n = 34
            X = np.zeros((21, n)); Y = np.zeros((21, n)); R = np.zeros((21, n))
            X[np.arange(21), gids] = 1.
            R[np.arange(21), 21+gids] = 1.
            ind = np.flatnonzero(gids >= 3)
            Y[ind, 9+gids[ind]] = signs[ind]
            v = np.zeros(n)
            v[:12] = a[:, 0]
            v[12:21] = a[3:, 1]
            v[21:33] = a[:, 2]
            bounds = [(-1., 1.)]*12+[(0., 1.)]*9+[(1e-7, .5)]*12+[(.05, 1.95)]
            seen = set(); pairs = []
            for i, j in zip(ii, jj):
                check(limit)
                key = (int(gids[i]), int(gids[j]), float(signs[i]*signs[j]))
                if key not in seen:
                    seen.add(key); pairs.append((i, j))
            pi, pj = np.asarray(pairs).T
        else:
            n = 64
            X = np.zeros((21, n)); Y = np.zeros((21, n)); R = np.zeros((21, n))
            X[:, :21] = np.eye(21); Y[:, 21:42] = np.eye(21); R[:, 42:63] = np.eye(21)
            a = a.copy()
            lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
            hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
            a[:, :2] -= (lo+hi)/2
            v = np.r_[a[:, 0], a[:, 1], a[:, 2], 1.]
            bounds = [(-1., 1.)]*42+[(1e-7, .5)]*21+[(.05, 1.95)]
            pi, pj = ii, jj
        full = np.column_stack((X@v, Y@v, R@v))
        widths = np.max(full[:, :2]+full[:, 2, None], axis=0)-np.min(full[:, :2]-full[:, 2, None], axis=0)
        v[-1] = np.clip(widths[0], .05, 1.95)
        DX = X[pi]-X[pj]; DY = Y[pi]-Y[pj]; RS = R[pi]+R[pj]
        width = np.zeros(n); width[-1] = .5
        W = np.vstack((X-R+width, -X-R+width, Y-R-width, -Y-R-width))
        off = np.r_[np.zeros(42), np.ones(42)]
        objective = -np.sum(R, axis=0)
        calls = [0]
        def fun(z):
            check(limit)
            return float(objective@z)
        def jac(z):
            check(limit)
            return objective
        def con(z):
            check(limit)
            dx = DX@z; dy = DY@z
            return np.r_[np.sqrt(dx*dx+dy*dy)-RS@z, W@z+off]
        def cj(z):
            check(limit)
            dx = DX@z; dy = DY@z
            d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-25)
            return np.vstack(((dx/d)[:, None]*DX+(dy/d)[:, None]*DY-RS, W))
        def callback(z):
            check(limit)
            calls[0] += 1
            if calls[0] % 4 == 0:
                retain(np.column_stack((X@z, Y@z, R@z)))
        result = minimize(fun, v, jac=jac, bounds=bounds, method='SLSQP', constraints={'type':'ineq', 'fun':con, 'jac':cj}, callback=callback, options={'maxiter':maxiter, 'ftol':float(params['tolerance']), 'disp':False})
        retain(np.column_stack((X@result.x, Y@result.x, R@result.x)))

    groups = np.zeros((12, 3))
    groups[:, 2] = .095
    groups[:3, 0] = [-.30, 0., .30]
    groups[3:7, 0] = np.linspace(-.45, .45, 4)
    groups[3:7, 1] = .18
    groups[7:, 0] = np.linspace(-.44, .44, 5)
    groups[7:, 1] = .37
    seed_end = min(end, began+float(params['seed_fraction'])*budget_s)
    try:
        optimize(groups, True, seed_end, int(params['iterations']))
    except (Deadline, ValueError, FloatingPointError):
        pass
    initial_end = min(end, began+(float(params['seed_fraction'])+float(params['initial_fraction']))*budget_s)
    try:
        raw = canonical(init)
        if raw is not None:
            a = np.asarray(raw['circles'], float)
            if a.shape == (21, 3) and np.isfinite(a).all() and np.all(a[:, 2] > 0):
                optimize(a, False, initial_end, int(params['iterations']))
    except (Deadline, ValueError, FloatingPointError):
        pass
    try:
        if best is not None:
            optimize(np.asarray(canonical(best)['circles'], float), False, end, int(params['iterations']))
    except (Deadline, ValueError, FloatingPointError):
        pass
    return best
