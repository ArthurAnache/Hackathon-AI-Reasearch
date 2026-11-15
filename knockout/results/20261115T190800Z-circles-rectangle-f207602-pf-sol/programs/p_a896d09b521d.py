import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {"first_fraction": (0.47, 0.25, 0.6), "ftol": (2e-13, 1e-14, 1e-8, "log")}


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -float('inf')

    def exact(obj):
        try:
            c = np.asarray(obj['circles'], dtype=float)
            if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
                return -float('inf')
            f = [[Fraction(float(t)) for t in row] for row in c]
            p = sum(max(row[k] + row[2] for row in f) - min(row[k] - row[2] for row in f) for k in (0, 1))
            if p > 2:
                return -float('inf')
            for i in range(21):
                if time.monotonic() > deadline + 0.08:
                    return -float('inf')
                for j in range(i):
                    if np.sqrt((c[i, 0]-c[j, 0])**2 + (c[i, 1]-c[j, 1])**2) < c[i, 2]+c[j, 2]:
                        return -float('inf')
            return float(np.sum(c[:, 2]))
        except Exception:
            return -float('inf')

    if init is not None:
        if 'circles' not in init:
            cc = []
            for x, y, r, m in init['orbits']:
                cc.append([x, y, r])
                if m == 2:
                    cc.append([x, -y, r])
            init = {'circles': cc}
            best = init
        best_score = exact(init)

    # Variables: twelve x coordinates, nine positive y coordinates,
    # twelve radii, and rectangle width; rectangle height is 2-width.
    X = np.zeros((21, 34)); Y = X.copy(); R = X.copy()
    orbit = list(range(3))
    signs = [0, 0, 0]
    for k in range(3, 12):
        orbit.extend([k, k]); signs.extend([1, -1])
    for i, (k, s) in enumerate(zip(orbit, signs)):
        X[i, k] = 1
        R[i, 21+k] = 1
        if s:
            Y[i, 12+k-3] = s
    ii, jj = np.triu_indices(21, 1)
    DX, DY, DR = X[ii]-X[jj], Y[ii]-Y[jj], R[ii]+R[jj]
    # Reflected contacts have identical squared distances; retain one copy.
    for a in (DX, DY):
        for row in a:
            if time.monotonic() >= deadline:
                return best
            nz = np.flatnonzero(row)
            if len(nz) and row[nz[0]] < 0:
                row *= -1
    _, inds = np.unique(np.concatenate([DX, DY, DR], axis=1), axis=0, return_index=True)
    DX, DY, DR = DX[inds], DY[inds], DR[inds]
    rows = []; offsets = []
    for k in range(12):
        if time.monotonic() >= deadline:
            return best
        a = np.zeros(34); a[k] = 1; a[21+k] = -1
        rows.append(a); offsets.append(0.)
        a = np.zeros(34); a[33] = 1; a[k] = -1; a[21+k] = -1
        rows.append(a); offsets.append(0.)
        a = np.zeros(34); a[33] = -.5; a[21+k] = -1
        if k >= 3:
            a[12+k-3] = -1
        rows.append(a); offsets.append(1.)
        if k >= 3:
            a = np.zeros(34); a[12+k-3] = 1; a[21+k] = -1
            rows.append(a); offsets.append(0.)
    A = np.asarray(rows); b = np.asarray(offsets)
    weights = np.r_[np.ones(3), np.full(9, 2.)]
    grad = np.zeros(34); grad[21:33] = -weights
    bounds = [(0., 2.)]*12 + [(0., 1.)]*9 + [(1e-7, .5)]*12 + [(.3, 1.7)]

    def unpack(z):
        return np.column_stack((X@z, Y@z, R@z))

    def consider(z):
        nonlocal best, best_score
        if time.monotonic() > deadline:
            return
        c = unpack(z)
        if not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return
        dx = c[ii, 0]-c[jj, 0]; dy = c[ii, 1]-c[jj, 1]
        ratio = np.min(np.sqrt(dx*dx+dy*dy)/(c[ii, 2]+c[jj, 2]))
        c[:, 2] *= min(1., ratio) * (1.-3e-14)
        ext = np.max(c[:, :2]+c[:, 2, None], axis=0)-np.min(c[:, :2]-c[:, 2, None], axis=0)
        c *= min(1., 2./sum(ext)) * (1.-3e-14)
        if np.sum(c[:, 2]) <= best_score:
            return
        obj = {'circles': c.tolist()}
        val = exact(obj)
        if val > best_score:
            best, best_score = obj, val

    def project(c):
        c = c.copy()
        c[:, 1] -= np.median(c[:, 1])
        axis = np.argsort(np.abs(c[:, 1]))[:3]
        rest = np.array([i for i in range(21) if i not in axis])
        rest = rest[np.argsort(c[rest, 1])]
        lo, hi = rest[:9], rest[9:]
        cost = (c[hi, None, 0]-c[lo, 0])**2 + (c[hi, None, 1]+c[lo, 1])**2 + (c[hi, None, 2]-c[lo, 2])**2
        u, v = linear_sum_assignment(cost)
        pairs = (c[hi[u]]+c[lo[v]])*.5
        ys = (c[hi[u], 1]-c[lo[v], 1])*.5
        z = np.zeros(34)
        z[:12] = np.r_[c[axis, 0], pairs[:, 0]]
        z[12:21] = ys
        z[21:33] = np.r_[c[axis, 2], pairs[:, 2]]
        width = np.max(z[:12]+z[21:33])-np.min(z[:12]-z[21:33])
        height = 2*np.max(np.r_[z[21:24], ys+z[24:33]])
        scale = 2/(width+height)
        z[:12] -= np.min(z[:12]-z[21:33])
        z[:33] *= scale
        z[33] = width*scale
        quality = np.sum(cost[u, v])+np.sum(c[axis, 1]**2)
        return quality, z

    class Timeout(Exception):
        pass

    def solve(z, until):
        cache_z = None; cache_v = None; cache_j = None
        last = z.copy()
        def calc(t):
            nonlocal cache_z, cache_v, cache_j
            if time.monotonic() >= until:
                raise Timeout()
            if cache_z is None or not np.array_equal(t, cache_z):
                dx, dy = DX@t, DY@t
                dist = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-15)
                cache_v = np.r_[A@t+b, dist-DR@t]
                cache_j = np.vstack((A, (dx/dist)[:, None]*DX+(dy/dist)[:, None]*DY-DR))
                cache_z = t.copy()
            return cache_v, cache_j
        def callback(t):
            nonlocal last
            if time.monotonic() >= until:
                raise Timeout()
            last = t.copy()
        try:
            res = minimize(lambda t: float(grad@t), z, jac=lambda t: grad,
                           method='SLSQP', bounds=bounds,
                           constraints=[{'type':'ineq', 'fun':lambda t:calc(t)[0], 'jac':lambda t:calc(t)[1]}],
                           callback=callback, options={'maxiter':160, 'ftol':params['ftol'], 'disp':False})
            last = res.x
        except Timeout:
            pass
        consider(last)
        return last

    seeds = []
    if init is not None:
        try:
            c = np.asarray(init['circles'], float)
            seeds.append(project(c))
            seeds.append(project(c[:, [1, 0, 2]]))
        except Exception:
            pass
    seeds.sort(key=lambda t:t[0])
    z = np.zeros(34)
    z[:3] = [1., 4., 7.]
    z[3:8] = np.arange(5)*2
    z[8:12] = np.arange(4)*2+1
    z[12:17] = np.sqrt(3.)
    z[17:21] = 2*np.sqrt(3.)
    z[21:33] = 1.
    z[:12] += 1.
    sc = 2/(10+2+4*np.sqrt(3.))
    z[:33] *= sc; z[33] = 10*sc
    seeds.append((1e9, z))
    winner = None
    for k, (_, z) in enumerate(seeds):
        if time.monotonic() >= deadline-.08:
            break
        consider(z)
        remaining = deadline-time.monotonic()
        if k == 0:
            duration = min(remaining-.04, params['first_fraction']*budget_s)
        else:
            duration = remaining*.65
        if duration <= 0:
            break
        out = solve(z, time.monotonic()+duration)
        if winner is None or grad@out < grad@winner:
            winner = out
        if best_score > 2.3658323:
            break
    if winner is not None and time.monotonic() < deadline-.035:
        solve(winner, deadline-.01)
    return best
