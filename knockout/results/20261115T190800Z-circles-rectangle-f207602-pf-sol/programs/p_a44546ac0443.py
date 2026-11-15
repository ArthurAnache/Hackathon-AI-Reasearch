import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {'release': (0.014, 0.002, 0.05), 'bias': (0.22, 0.02, 0.8), 'contacts': (4, 3, 7), 'iterations': (75, 30, 150)}

class StopStage(Exception):
    pass


def canonical(obj):
    if obj is None:
        return None
    if 'circles' in obj:
        return obj
    out = []
    for x, y, r, m in obj['orbits']:
        out.append([x, y, r])
        if m == 2:
            out.append([x, -y, r])
    return {'circles': out}


def score(obj):
    try:
        a = np.asarray(canonical(obj)['circles'], dtype=np.float64)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
        w = max(u+v for u, v in zip(x, r))-min(u-v for u, v in zip(x, r))
        h = max(u+v for u, v in zip(y, r))-min(u-v for u, v in zip(y, r))
        if w+h > 2:
            return -float('inf')
        i, j = np.triu_indices(21, 1)
        d = np.sqrt((a[i, 0]-a[j, 0])**2+(a[i, 1]-a[j, 1])**2)
        if np.any(d < a[i, 2]+a[j, 2]):
            return -float('inf')
        return float(np.sum(a[:, 2]))
    except Exception:
        return -float('inf')


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.69*budget_s
    stage_end = deadline
    rng = np.random.default_rng(seed)
    best = init
    best_score = score(init)
    fallback = {'circles': [[(k % 7)*0.2, (k//7)*0.2, 0.09999999999999] for k in range(21)]}
    fs = score(fallback)
    if fs > best_score:
        best, best_score = fallback, fs

    def tick():
        if time.monotonic() >= min(deadline, stage_end):
            raise StopStage()

    groups = np.r_[np.arange(3), np.repeat(np.arange(3, 12), 2)]
    signs = np.r_[np.zeros(3), np.tile([1., -1.], 9)]
    X = np.zeros((21, 34)); Y = X.copy(); R = X.copy()
    X[np.arange(21), groups] = 1.
    R[np.arange(21), 21+groups] = 1.
    mask = groups >= 3
    Y[np.flatnonzero(mask), 9+groups[mask]] = signs[mask]
    ii, jj = np.triu_indices(21, 1)
    keys = np.column_stack((np.minimum(groups[ii], groups[jj]), np.maximum(groups[ii], groups[jj]), signs[ii]*signs[jj]))
    _, keep = np.unique(keys, axis=0, return_index=True)
    pi, pj = ii[keep], jj[keep]
    ga, gb = groups[pi], groups[pj]
    DX = X[pi]-X[pj]; DY = Y[pi]-Y[pj]; RR = R[pi]+R[pj]
    W = np.zeros((36, 34)); off = np.zeros(36)
    q = np.arange(12)
    W[q, q] = 1.; W[q, 21+q] = -1.; W[q, 33] = .5
    W[12+q, q] = -1.; W[12+q, 21+q] = -1.; W[12+q, 33] = .5
    W[24+q, 21+q] = -1.; W[24+q, 33] = -.5
    W[27+np.arange(9), 12+np.arange(9)] = -1.
    off[24:] = 1.
    weights = np.r_[np.ones(3), np.full(9, 2.)]
    objective = np.zeros(34); objective[21:33] = -weights
    base_objective = objective.copy()
    bounds = [(-1., 1.)]*12 + [(0., 1.)]*9 + [(1e-6, .6)]*12 + [(.12, 1.88)]
    release = np.zeros(len(pi))
    cache_v = None
    cache_c = None
    cache_j = None
    best_z = None
    last = None
    calls = 0

    def retain(z):
        nonlocal best, best_score, best_z
        if z is None or not np.isfinite(z).all() or np.any(z[21:33] <= 0):
            return
        c = np.column_stack((X@z, Y@z, R@z))
        dist = np.sqrt((c[ii, 0]-c[jj, 0])**2+(c[ii, 1]-c[jj, 1])**2)
        shrink = min(1., float(np.min(dist/(c[ii, 2]+c[jj, 2]))))*(1.-2e-14)
        if shrink <= 0:
            return
        c[:, 2] *= shrink
        extent = np.max(c[:, :2]+c[:, 2, None], axis=0)-np.min(c[:, :2]-c[:, 2, None], axis=0)
        scale = 2./float(np.sum(extent))*(1.-2e-14)
        c *= scale
        if float(np.sum(c[:, 2])) <= best_score:
            return
        candidate = {'circles': c.tolist()}
        val = score(candidate)
        if val > best_score:
            best, best_score = candidate, val
            best_z = z.copy()
            best_z[:33] *= scale
            best_z[21:33] *= shrink
            low = np.min(c[:, 0]-c[:, 2]); high = np.max(c[:, 0]+c[:, 2])
            best_z[:12] -= (low+high)/2
            best_z[33] = high-low

    def calculate(z):
        nonlocal cache_v, cache_c, cache_j
        tick()
        if cache_v is None or not np.array_equal(z, cache_v):
            dx = DX@z; dy = DY@z
            d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-24)
            cache_c = np.r_[d-RR@z-release, W@z+off]
            cache_j = np.vstack(((dx/d)[:, None]*DX+(dy/d)[:, None]*DY-RR, W))
            cache_v = z.copy()
        return cache_c, cache_j

    def fun(z):
        tick()
        return float(objective@z)

    def jac(z):
        tick()
        return objective

    def callback(z):
        nonlocal last, calls
        last = z.copy()
        calls += 1
        tick()
        if calls % 5 == 0:
            retain(z)

    def solve(z, iterations, seconds):
        nonlocal stage_end, cache_v, last
        stage_end = min(deadline, time.monotonic()+max(0., seconds))
        cache_v = None
        last = z.copy()
        try:
            tick()
            ans = minimize(fun, z, jac=jac, method='SLSQP', bounds=bounds,
                           constraints={'type': 'ineq', 'fun': lambda v: calculate(v)[0], 'jac': lambda v: calculate(v)[1]},
                           callback=callback, options={'maxiter': int(iterations), 'ftol': 1e-13, 'disp': False})
            last = ans.x.copy()
        except (StopStage, ValueError, FloatingPointError):
            pass
        if time.monotonic() < deadline:
            retain(last)
        return last.copy()

    def lattice():
        z = np.zeros(34)
        z[:12] = [-2., 0., 2., -3., -1., 1., 3., -4., -2., 0., 2., 4.]
        z[12:21] = np.r_[np.full(4, np.sqrt(3.)), np.full(5, 2*np.sqrt(3.))]
        z[21:33] = 1.
        width = np.ptp(z[:12])+2
        s = 2./(width+2*(np.max(z[12:21])+1))
        z[:33] *= s; z[33] = width*s
        return z

    projections = []
    try:
        a = np.asarray(canonical(init)['circles'], dtype=float).copy()
        low = np.min(a[:, :2]-a[:, 2, None], axis=0)
        high = np.max(a[:, :2]+a[:, 2, None], axis=0)
        a[:, :2] -= (low+high)/2
        for swap in (False, True):
            if time.monotonic() >= deadline:
                break
            b = a[:, [1, 0, 2]].copy() if swap else a.copy()
            order = np.argsort(b[:, 1])
            lower, axis, upper = order[:9], order[9:12], order[12:]
            ref = b[lower].copy(); ref[:, 1] *= -1
            cost = np.sum((b[upper, None, :]-ref[None, :, :])**2, axis=2)
            u, v = linear_sum_assignment(cost)
            g = np.vstack((b[axis], (b[upper[u]]+ref[v])/2))
            z = np.zeros(34)
            z[:12] = g[:, 0]; z[12:21] = np.abs(g[3:, 1]); z[21:33] = g[:, 2]
            z[33] = np.max(g[:, 0]+g[:, 2])-np.min(g[:, 0]-g[:, 2])
            error = float(np.sum(cost[u, v])+np.sum(b[axis, 1]**2))
            projections.append((error, z))
    except (TypeError, KeyError, ValueError, IndexError):
        pass
    projections.sort(key=lambda item: item[0])
    anchor = projections[0][1].copy() if projections else lattice()
    retain(anchor)
    polished = solve(anchor, params['iterations'], .18 if best_score > 2.36 else .35)
    if best_z is not None:
        anchor = best_z.copy()
    else:
        anchor = polished.copy()
    attempt = 0
    while time.monotonic() < deadline-.045:
        stage_end = deadline
        remaining = deadline-time.monotonic()
        if remaining < .11:
            objective[:] = base_objective; release[:] = 0
            solve(anchor, params['iterations'], remaining)
            break
        gaps = np.sqrt((DX@anchor)**2+(DY@anchor)**2)-RR@anchor
        near = np.flatnonzero(gaps < .0025)
        if len(near) < 3:
            break
        pivot = int(rng.integers(12))
        incident = near[(ga[near] == pivot) | (gb[near] == pivot)]
        if len(incident) >= 3 and attempt % 3 != 2:
            chosen = rng.choice(incident, min(int(params['contacts']), len(incident)), replace=False)
        else:
            chosen = rng.choice(near, min(int(params['contacts']), len(near)), replace=False)
        release[:] = 0
        magnitude = float(params['release'])*[.45, 1., 1.9][attempt % 3]
        release[chosen] = magnitude*rng.uniform(.7, 1.3, len(chosen))
        objective[:] = base_objective
        bias = rng.normal(size=12)
        bias -= np.mean(bias)
        objective[21:33] -= float(params['bias'])*weights*bias
        displaced = solve(anchor, 23, min(.115, remaining*.30))
        release[:] = 0
        objective[:] = base_objective
        restored = solve(displaced, params['iterations'], min(.25, max(.01, deadline-time.monotonic())))
        if best_z is not None:
            anchor = best_z.copy()
        elif np.isfinite(restored).all():
            anchor = restored.copy()
        attempt += 1
    return best
