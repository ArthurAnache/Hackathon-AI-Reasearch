import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import maximum_filter

PARAMS = {
    'grid': (57, 31, 91),
    'cluster_size': (3, 1, 4),
    'trial_seconds': (0.38, 0.12, 0.7),
    'maxiter': (100, 30, 200),
    'exclusion': (0.065, 0.02, 0.15)
}
N = 21
I, J = np.triu_indices(N, 1)
K = np.arange(N)
M = len(I)

class StopSearch(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0] - a[J, 0]) ** 2 + (a[I, 1] - a[J, 1]) ** 2)
    if np.any(d < a[I, 2] + a[J, 2]):
        return -np.inf
    x = list(map(Fraction, map(float, a[:, 0])))
    y = list(map(Fraction, map(float, a[:, 1])))
    r = list(map(Fraction, map(float, a[:, 2])))
    width = max(map(lambda q: q[0] + q[1], zip(x, r))) - min(map(lambda q: q[0] - q[1], zip(x, r)))
    height = max(map(lambda q: q[0] + q[1], zip(y, r))) - min(map(lambda q: q[0] - q[1], zip(y, r)))
    if width + height > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))

def repair(z):
    a = np.column_stack((z[:N], z[N:2*N], z[2*N:3*N])).copy()
    if not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
        return None
    d = np.sqrt((a[I, 0] - a[J, 0]) ** 2 + (a[I, 1] - a[J, 1]) ** 2)
    ratio = min(1.0, float(np.min(d / (a[I, 2] + a[J, 2]))))
    a[:, 2] *= ratio * (1.0 - 3e-14)
    lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
    hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
    perimeter = float(np.sum(hi - lo))
    a[:, :2] -= lo
    a *= min(1.0, 2.0 / perimeter) * (1.0 - 3e-14)
    return a

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        xx, yy = np.meshgrid(np.arange(3), np.arange(7))
        a = np.column_stack(((xx.ravel() + 0.5) * 0.2, (yy.ravel() + 0.5) * 0.2, np.full(N, 0.099999999999)))
        best = {'circles': a.tolist()}
    else:
        best = init
        a = np.asarray(init['circles'], dtype=float).copy()
    best_score = exact_score(a)
    lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
    side = np.max(a[:, :2] + a[:, 2, None], axis=0) - lo
    a[:, :2] -= lo
    w = float(side[0])
    base = np.r_[a[:, 0], a[:, 1], a[:, 2], w]
    objective_jac = np.r_[np.zeros(2*N), -np.ones(N), 0.0]
    bounds = [(0.0, 2.0)] * (2*N) + [(1e-7, 0.8)] * N + [(0.03, 1.97)]
    grid_n = int(params['grid'])
    gx, gy = np.meshgrid(np.linspace(0, w, grid_n), np.linspace(0, 2-w, grid_n))
    grid = np.column_stack((gx.ravel(), gy.ravel()))
    wall_clearance = np.minimum.reduce((grid[:, 0], w-grid[:, 0], grid[:, 1], 2-w-grid[:, 1]))
    trial = 0

    def consider(z):
        nonlocal best, best_score
        if time.monotonic() >= end:
            return
        if float(np.sum(z[2*N:3*N])) <= best_score:
            return
        candidate = repair(z)
        if candidate is not None:
            score = exact_score(candidate)
            if score > best_score:
                best_score = score
                best = {'circles': candidate.tolist()}

    while time.monotonic() < end - 0.045:
        count = int(params['cluster_size']) if trial % 3 != 2 else 1
        small = np.argsort(a[:, 2])[:9]
        anchor = int(small[trial % len(small)])
        proximity = np.linalg.norm(a[:, :2] - a[anchor, :2], axis=1)
        priority = proximity + 0.45 * a[:, 2]
        priority[anchor] = -1
        removed = np.argsort(priority)[:count]
        keep = np.ones(N, dtype=bool)
        keep[removed] = False
        work = a.copy()
        old = a[removed, :2].copy()
        clearance = wall_clearance.copy()
        dist = np.sqrt(np.sum((grid[:, None, :] - a[None, keep, :2]) ** 2, axis=2)) - a[None, keep, 2]
        clearance = np.minimum(clearance, np.min(dist, axis=1))
        feasible = True
        for idx in removed:
            if time.monotonic() >= end - 0.035:
                feasible = False
                break
            exclusion = np.min(np.linalg.norm(grid[:, None, :] - old[None, :, :], axis=2), axis=1)
            scores = clearance.copy()
            scores[exclusion < float(params['exclusion'])] = -1
            peaks = scores.reshape(grid_n, grid_n)
            is_peak = peaks >= maximum_filter(peaks, size=5, mode='constant', cval=-2)
            choices = np.flatnonzero(is_peak.ravel() & (scores > 0.002))
            if not len(choices):
                feasible = False
                break
            order = choices[np.argsort(scores[choices])[::-1]]
            pick = int(order[0 if trial == 0 else rng.integers(min(4, len(order)))])
            work[idx, :2] = grid[pick]
            work[idx, 2] = scores[pick] * 0.96
            clearance = np.minimum(clearance, np.linalg.norm(grid - work[idx, :2], axis=1) - work[idx, 2])
        trial += 1
        if not feasible:
            continue
        z0 = np.r_[work[:, 0], work[:, 1], work[:, 2], w]
        local_end = min(end - 0.012, time.monotonic() + float(params['trial_seconds']))
        last = [z0]

        def check():
            if time.monotonic() >= local_end:
                raise StopSearch()

        def fun(z):
            check()
            return -float(np.sum(z[2*N:3*N]))

        def jac(z):
            check()
            return objective_jac

        def constraints(z):
            check()
            x, y, r, width = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
            d = np.sqrt((x[I]-x[J])**2 + (y[I]-y[J])**2)
            return np.r_[d-r[I]-r[J], x-r, width-x-r, y-r, 2-width-y-r]

        def constraint_jac(z):
            check()
            x, y = z[:N], z[N:2*N]
            dx, dy = x[I]-x[J], y[I]-y[J]
            d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-20)
            out = np.zeros((M+4*N, 3*N+1))
            rows = np.arange(M)
            out[rows, I] = dx/d
            out[rows, J] = -dx/d
            out[rows, N+I] = dy/d
            out[rows, N+J] = -dy/d
            out[rows, 2*N+I] = -1
            out[rows, 2*N+J] = -1
            out[M+K, K] = 1
            out[M+N+K, K] = -1
            out[M+2*N+K, N+K] = 1
            out[M+3*N+K, N+K] = -1
            out[M+K, 2*N+K] = -1
            out[M+N+K, 2*N+K] = -1
            out[M+2*N+K, 2*N+K] = -1
            out[M+3*N+K, 2*N+K] = -1
            out[M+N+K, -1] = 1
            out[M+3*N+K, -1] = -1
            return out

        def callback(z):
            last[0] = z.copy()
            consider(z)
            check()

        try:
            result = minimize(fun, z0, jac=jac, method='SLSQP', bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraints, 'jac': constraint_jac},
                callback=callback, options={'maxiter': int(params['maxiter']), 'ftol': 2e-12, 'disp': False})
            consider(result.x)
        except StopSearch:
            consider(last[0])
    return best
