import time
import numpy as np
from fractions import Fraction
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {"amplitude": (0.055, 0.005, 0.18), "iterations": (75, 20, 150)}


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)

    def exact_score(c):
        if time.monotonic() >= start + 0.94 * budget_s:
            return -np.inf
        a = np.asarray(c, dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -np.inf
        d = a[ii, :2] - a[jj, :2]
        if np.any(np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2) < a[ii, 2] + a[jj, 2]):
            return -np.inf
        xs, ys, rs = [], [], []
        for x, y, r in a:
            if time.monotonic() >= start + 0.94 * budget_s:
                return -np.inf
            xs.append(Fraction(float(x)))
            ys.append(Fraction(float(y)))
            rs.append(Fraction(float(r)))
        p = max(x+r for x, r in zip(xs, rs)) - min(x-r for x, r in zip(xs, rs))
        p += max(y+r for y, r in zip(ys, rs)) - min(y-r for y, r in zip(ys, rs))
        return float(np.sum(a[:, 2])) if p <= 2 else -np.inf

    if init is None:
        a = np.array([[k % 7, k // 7, 0.499999999999] for k in range(21)], float)
        a /= 5.0
        best = {"circles": a.tolist()}
    else:
        best = init
    best_score = exact_score(best['circles'])
    original = np.array(best['circles'], float)

    def repair(c):
        a = np.array(c, float, copy=True)
        if not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return None
        a[:, :2] -= np.min(a[:, :2] - a[:, 2, None], axis=0)
        d = a[ii, :2] - a[jj, :2]
        distances = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
        ratio = np.min(distances / (a[ii, 2] + a[jj, 2]))
        a[:, 2] *= min(1.0, ratio) * (1.0 - 5e-15)
        span = np.max(a[:, :2] + a[:, 2, None], axis=0) - np.min(a[:, :2] - a[:, 2, None], axis=0)
        a *= (2.0 / np.sum(span)) * (1.0 - 5e-15)
        return a

    def consider(c):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        a = repair(c)
        if a is None or float(np.sum(a[:, 2])) <= best_score:
            return
        score = exact_score(a)
        if score > best_score:
            best_score = score
            best = {"circles": a.tolist()}

    # Detect which coordinate is the reflection-normal coordinate.
    projections = []
    for swap in (False, True):
        if time.monotonic() >= deadline:
            return best
        a = original.copy()
        if swap:
            a[:, [0, 1]] = a[:, [1, 0]]
        lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
        hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
        a[:, :2] -= lo
        a *= 2.0 / np.sum(hi - lo)
        w = (hi[0] - lo[0]) * 2.0 / np.sum(hi - lo)
        mid = (2.0-w)/2.0
        singles = np.argsort(np.abs(a[:, 1]-mid))[:3]
        rem = np.setdiff1d(np.arange(21), singles)
        order = rem[np.argsort(a[rem, 1])]
        lower, upper = order[:9], order[9:]
        reflected = a[upper].copy()
        reflected[:, 1] = 2*mid-reflected[:, 1]
        cost = np.sum((a[lower, None, :] - reflected[None, :, :])**2, axis=2)
        rows, cols = linear_sum_assignment(cost)
        pairs = np.column_stack((lower[rows], upper[cols]))
        err = float(np.sum(cost[rows, cols]) + np.sum((a[singles, 1]-mid)**2))
        projections.append((err, a, singles, pairs, w, swap))
    _, a, singles, pairs, w, swap = min(projections, key=lambda p: p[0])

    # Linear expansion: 3 axis circles, 9 mirrored pairs, rectangle width.
    M = np.zeros((64, 34))
    offset = np.zeros(64)
    q = np.zeros(34)
    q[-1] = w
    for k, s in enumerate(singles):
        if time.monotonic() >= deadline:
            return best
        M[3*k, 2*k] = 1
        M[3*k+1, -1] = -0.5
        offset[3*k+1] = 1
        M[3*k+2, 2*k+1] = 1
        q[2*k:2*k+2] = a[s, [0, 2]]
    for k, (l, u) in enumerate(pairs):
        if time.monotonic() >= deadline:
            return best
        col = 6+3*k
        p = 3+2*k
        M[3*p, col] = M[3*(p+1), col] = 1
        M[3*p+1, col+1] = 1
        M[3*(p+1)+1, col+1] = -1
        M[3*(p+1)+1, -1] = -1
        offset[3*(p+1)+1] = 2
        M[3*p+2, col+2] = M[3*(p+1)+2, col+2] = 1
        q[col:col+3] = [(a[l, 0]+a[u, 0])/2, (a[l, 1]+2-w-a[u, 1])/2, (a[l, 2]+a[u, 2])/2]
    M[63, 33] = 1
    X, Y, R = M[0:63:3], M[1:63:3], M[2:63:3]
    ox, oy, ore = offset[0:63:3], offset[1:63:3], offset[2:63:3]
    W = M[63]
    walls = np.vstack((X-R, W-X-R, Y-R, -W-Y-R))
    wall_offset = np.concatenate((ox-ore, -ox-ore, oy-ore, 2-oy-ore))
    DX, DY, RR = X[ii]-X[jj], Y[ii]-Y[jj], R[ii]+R[jj]
    dx0, dy0 = ox[ii]-ox[jj], oy[ii]-oy[jj]
    objgrad = -np.sum(R, axis=0)
    lower = np.full(34, -2.0)
    upper = np.full(34, 2.0)
    radius_cols = [1, 3, 5] + list(range(8, 33, 3))
    lower[radius_cols] = 0.012
    lower[-1], upper[-1] = 0.45, 1.55
    bounds = list(zip(lower, upper))

    class StopSearch(Exception):
        pass

    def constraints(z):
        if time.monotonic() >= deadline:
            raise StopSearch()
        dx, dy = DX@z+dx0, DY@z+dy0
        return np.concatenate((walls@z+wall_offset, np.sqrt(dx*dx+dy*dy)-RR@z))

    def jacobian(z):
        if time.monotonic() >= deadline:
            raise StopSearch()
        dx, dy = DX@z+dx0, DY@z+dy0
        d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-15)
        return np.vstack((walls, (dx/d)[:, None]*DX + (dy/d)[:, None]*DY-RR))

    latest = q.copy()
    def callback(z):
        nonlocal latest
        latest = z.copy()
        if time.monotonic() >= deadline:
            raise StopSearch()

    def decode(z):
        c = (M@z+offset)[:63].reshape(21, 3).copy()
        if swap:
            c[:, [0, 1]] = c[:, [1, 0]]
        return c

    anchor = q.copy()
    local_score = -np.inf
    attempt = 0
    while time.monotonic() < deadline:
        z = anchor.copy()
        if attempt:
            amp = params['amplitude'] * (0.45, 0.85, 1.3, 1.9, 2.5)[(attempt-1) % 5]
            chosen = rng.choice(9, 3 if attempt % 3 else 5, replace=False)
            for k in chosen:
                if time.monotonic() >= deadline:
                    return best
                j = 6+3*k
                z[j:j+2] += rng.normal(0, amp, 2)
                z[j+2] *= np.exp(rng.normal(0, 0.10))
            z[-1] += rng.normal(0, amp*0.4)
        latest = z.copy()
        try:
            result = minimize(lambda v: float(objgrad@v), z, jac=lambda v: objgrad,
                              method='SLSQP', bounds=bounds,
                              constraints={'type':'ineq', 'fun':constraints, 'jac':jacobian},
                              callback=callback,
                              options={'maxiter':int(params['iterations']), 'ftol':2e-13, 'disp':False})
            latest = result.x
        except StopSearch:
            break
        c = decode(latest)
        consider(c)
        if time.monotonic() >= deadline:
            break
        feasible = np.min(constraints(latest)) > -1e-8
        value = -float(objgrad@latest)
        if feasible and value > local_score:
            local_score = value
            anchor = latest.copy()
        attempt += 1
    return best
