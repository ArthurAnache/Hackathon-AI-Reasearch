import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {'project_steps': (55, 25, 100), 'lattice_steps': (48, 20, 90), 'polish_steps': (110, 40, 180), 'first_share': (0.27, 0.15, 0.4), 'second_share': (0.20, 0.1, 0.3)}


def canonical(obj):
    if 'circles' in obj:
        return obj
    c = []
    for x, y, r, m in obj['orbits']:
        c.append([x, y, r])
        if m == 2:
            c.append([x, -y, r])
    return {'circles': c}


def exact_score(obj):
    try:
        a = np.asarray(canonical(obj)['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        ii, jj = np.triu_indices(21, 1)
        d = a[ii, :2] - a[jj, :2]
        if np.any(np.sqrt(d[:, 0]**2 + d[:, 1]**2) < a[ii, 2] + a[jj, 2]):
            return -float('inf')
        x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
        width = max(u+v for u, v in zip(x, r))-min(u-v for u, v in zip(x, r))
        height = max(u+v for u, v in zip(y, r))-min(u-v for u, v in zip(y, r))
        if width+height > 2:
            return -float('inf')
        return float(np.sum(a[:, 2]))
    except (TypeError, ValueError, KeyError, OverflowError):
        return -float('inf')


class StopStage(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    stage_end = deadline
    best = init
    best_score = exact_score(init) if init is not None else -float('inf')
    if best_score == -float('inf'):
        best = {'circles': [[(i % 7-3)*0.17, (i//7-1)*0.17, 0.08] for i in range(21)]}
        best_score = exact_score(best)

    def tick():
        if time.monotonic() >= min(deadline, stage_end):
            raise StopStage()

    oi = np.r_[np.arange(12), np.arange(3, 12)]
    signs = np.r_[np.ones(12), -np.ones(9)]
    X = np.zeros((21, 34))
    Y = np.zeros_like(X)
    R = np.zeros_like(X)
    for i, g in enumerate(oi):
        if time.monotonic() >= deadline:
            return best
        X[i, g] = 1
        R[i, 21+g] = 1
        if g >= 3:
            Y[i, 9+g] = signs[i]
    edges = []
    seen = set()
    for i in range(21):
        if time.monotonic() >= deadline:
            return best
        for j in range(i):
            if time.monotonic() >= deadline:
                return best
            key = (min(oi[i], oi[j]), max(oi[i], oi[j]), int(signs[i]*signs[j]) if oi[i] >= 3 and oi[j] >= 3 else 0)
            if key not in seen:
                seen.add(key)
                edges.append((i, j))
    p, q = np.asarray(edges).T
    DX, DY, RR = X[p]-X[q], Y[p]-Y[q], R[p]+R[q]
    L = np.zeros((36, 34))
    offset = np.zeros(36)
    for g in range(12):
        if time.monotonic() >= deadline:
            return best
        L[g, g], L[g, 21+g], L[g, 33] = 1, -1, 1
        L[12+g, g], L[12+g, 21+g], L[12+g, 33] = -1, -1, 1
        L[24+g, 21+g], L[24+g, 33] = -1, -1
        offset[24+g] = 1
        if g >= 3:
            L[24+g, 9+g] = -1
    gradient = np.zeros(34)
    gradient[21:33] = -np.r_[np.ones(3), np.full(9, 2.)]
    bounds = [(-1., 1.)]*12 + [(0., 1.)]*9 + [(1e-6, .5)]*12 + [(.05, .95)]
    ii, jj = np.triu_indices(21, 1)
    cache_z = cache_c = cache_j = None
    basin_z = None
    basin_score = -float('inf')
    current = None
    counter = 0

    def retain(z):
        nonlocal best, best_score, basin_z, basin_score
        if time.monotonic() >= deadline:
            return
        if not np.isfinite(z).all() or np.any(z[21:33] <= 0):
            return
        c = np.column_stack((X@z, Y@z, R@z))
        d = c[ii, :2]-c[jj, :2]
        ratio = float(np.min(np.sqrt(d[:, 0]**2+d[:, 1]**2)/(c[ii, 2]+c[jj, 2])))
        if ratio <= 0:
            return
        c[:, 2] *= min(1., ratio)*(1.-1.2e-14)
        lo = np.min(c[:, :2]-c[:, 2, None], axis=0)
        hi = np.max(c[:, :2]+c[:, 2, None], axis=0)
        scale = 2./float(np.sum(hi-lo))*(1.-1.2e-14)
        c[:, :2] -= (lo+hi)/2
        c *= scale
        score = float(np.sum(c[:, 2]))
        if score > basin_score:
            basin_score = score
            basin_z = z.copy()
        if score > best_score:
            obj = {'circles': c.tolist()}
            score = exact_score(obj)
            if score > best_score:
                best, best_score = obj, score

    def constraints(z):
        nonlocal cache_z, cache_c, cache_j
        tick()
        if cache_z is None or not np.array_equal(cache_z, z):
            dx, dy = DX@z, DY@z
            dist = np.sqrt(dx*dx+dy*dy)
            safe = np.maximum(dist, 1e-30)
            cache_c = np.r_[dist-RR@z, L@z+offset]
            cache_j = np.vstack(((dx/safe)[:, None]*DX+(dy/safe)[:, None]*DY-RR, L))
            cache_z = z.copy()
        return cache_c

    def jac(z):
        constraints(z)
        return cache_j

    def objective(z):
        tick()
        return float(gradient@z)

    def callback(z):
        nonlocal current, counter
        tick()
        current = z.copy()
        counter += 1
        if counter % 4 == 0:
            retain(z)

    def solve(z, steps, until):
        nonlocal stage_end, current
        stage_end = min(until, deadline)
        current = z.copy()
        try:
            result = minimize(objective, z, jac=lambda z: gradient, method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': constraints, 'jac': jac},
                              callback=callback, options={'maxiter': steps, 'ftol': 3e-14, 'disp': False})
            current = result.x.copy()
        except (StopStage, ValueError, FloatingPointError):
            pass
        retain(current)
        return current.copy()

    def pack(g):
        z = np.zeros(34)
        z[:12] = g[:, 0]
        z[12:21] = np.maximum(g[3:, 1], 0.)
        z[21:33] = np.maximum(g[:, 2], 1e-5)
        z[33] = np.clip((np.max(g[:, 0]+g[:, 2])-np.min(g[:, 0]-g[:, 2]))/2, .05, .95)
        return z

    a = np.asarray(canonical(best)['circles'], float).copy()
    lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
    hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
    a[:, :2] -= (lo+hi)/2
    projections = []
    for swap in (False, True):
        if time.monotonic() >= deadline:
            return best
        b = a.copy()
        if swap:
            b[:, [0, 1]] = b[:, [1, 0]]
        order = np.argsort(b[:, 1])
        low, singles, high = order[:9], order[9:12], order[12:]
        ref = b[low].copy()
        ref[:, 1] *= -1
        cost = np.sum((b[high, None, :]-ref[None, :, :])**2, axis=2)
        rows, cols = linear_sum_assignment(cost)
        g = np.vstack((np.column_stack((b[singles, 0], np.zeros(3), b[singles, 2])), (b[high[rows]]+ref[cols])/2))
        projections.append((float(np.sum(cost[rows, cols])+np.sum(b[singles, 1]**2)), pack(g)))
    projections.sort(key=lambda t: t[0])
    z = solve(projections[0][1], int(params['project_steps']), start+float(params['first_share'])*budget_s)
    g = np.zeros((12, 3))
    g[:3, 0] = [-2, 0, 2]
    g[3:7, 0] = [-3, -1, 1, 3]
    g[7:, 0] = [-4, -2, 0, 2, 4]
    g[3:7, 1] = np.sqrt(3.)
    g[7:, 1] = 2*np.sqrt(3.)
    g[:, 2] = 1
    g *= 2./(12+4*np.sqrt(3.))
    second_end = start+(float(params['first_share'])+float(params['second_share']))*budget_s
    solve(pack(g), int(params['lattice_steps']), second_end)
    if basin_z is not None and time.monotonic() < deadline:
        solve(basin_z.copy(), int(params['polish_steps']), deadline)
    return best
