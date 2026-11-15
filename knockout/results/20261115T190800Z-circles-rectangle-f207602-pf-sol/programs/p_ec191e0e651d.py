import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {"first_share": (0.43, 0.25, 0.65), "ftol": (2e-12, 1e-14, 1e-8, "log")}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    n = 12
    oi = np.array([0, 1, 2] + list(range(3, 12)) + list(range(3, 12)))
    signs = np.array([0., 0., 0.] + [1.] * 9 + [-1.] * 9)
    ex = np.zeros((21, 34)); ey = np.zeros((21, 34)); er = np.zeros((21, 34))
    ex[np.arange(21), oi] = 1
    for k in range(21):
        if time.monotonic() >= deadline:
            return init
        if oi[k] >= 3:
            ey[k, 12 + oi[k] - 3] = signs[k]
    er[np.arange(21), 21 + oi] = 1
    ii, jj = np.triu_indices(21, 1)
    # Remove distances duplicated by reflection.
    seen = set(); edges = []
    for a, b in zip(ii, jj):
        if time.monotonic() >= deadline:
            return init
        aa, bb = int(oi[a]), int(oi[b])
        if aa == bb:
            continue
        if aa > bb:
            aa, bb = bb, aa
            sa, sb = signs[b], signs[a]
        else:
            sa, sb = signs[a], signs[b]
        kind = 0 if aa < 3 else int(sa != sb)
        key = (aa, bb, kind)
        if key not in seen:
            seen.add(key); edges.append((a, b))
    ea, eb = np.array(edges).T
    DX = ex[ea] - ex[eb]
    DY = ey[ea] - ey[eb]
    RR = er[ea] + er[eb]
    linear = []; const = []
    for k in range(12):
        if time.monotonic() >= deadline:
            return init
        for s in (-1., 1.):
            row = np.zeros(34); row[k] = s; row[21+k] = -1; row[33] = .5
            linear.append(row); const.append(0.)
        row = np.zeros(34); row[21+k] = -1; row[33] = -.5
        if k >= 3:
            row[12+k-3] = -1
        linear.append(row); const.append(1.)
        if k >= 3:
            row = np.zeros(34); row[12+k-3] = 1; row[21+k] = -1
            linear.append(row); const.append(0.)
    A = np.array(linear); B = np.array(const)
    weights = np.array([1.] * 3 + [2.] * 9)
    grad = np.zeros(34); grad[21:33] = -weights
    bounds = [(-1., 1.)]*12 + [(0., 1.)]*9 + [(1e-7, .5)]*12 + [(.2, 1.8)]
    best = init
    best_score = -float('inf')

    def exact(c):
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return -float('inf')
        d = np.sqrt((c[ii, 0]-c[jj, 0])**2 + (c[ii, 1]-c[jj, 1])**2)
        if np.any(d < c[ii, 2]+c[jj, 2]):
            return -float('inf')
        x = list(map(lambda v: Fraction(float(v)), c[:, 0]))
        y = list(map(lambda v: Fraction(float(v)), c[:, 1]))
        r = list(map(lambda v: Fraction(float(v)), c[:, 2]))
        per = max(a+b for a,b in zip(x,r))-min(a-b for a,b in zip(x,r))
        per += max(a+b for a,b in zip(y,r))-min(a-b for a,b in zip(y,r))
        return float(np.sum(c[:, 2])) if per <= 2 else -float('inf')

    raw = None
    if init is not None:
        try:
            if 'circles' in init:
                raw = np.asarray(init['circles'], dtype=float)
            else:
                rows = []
                for x,y,r,m in init['orbits']:
                    if time.monotonic() >= deadline:
                        return init
                    rows.append([x,y,r])
                    if m == 2:
                        rows.append([x,-y,r])
                raw = np.array(rows)
            best_score = exact(raw)
        except Exception:
            raw = None

    best_z = None
    def repaired(c):
        c = c.copy()
        if not np.isfinite(c).all():
            return None
        c[:, 2] = np.maximum(c[:, 2], 1e-9)
        dist = np.sqrt(np.sum((c[ii,:2]-c[jj,:2])**2, axis=1))
        factor = min(1., float(np.min(dist/(c[ii,2]+c[jj,2]))))
        if factor <= 0:
            return None
        c[:,2] *= factor * (1.-2e-14)
        lo = np.min(c[:,:2]-c[:,2,None], axis=0)
        hi = np.max(c[:,:2]+c[:,2,None], axis=0)
        scale = (2.-5e-14)/float(np.sum(hi-lo))
        c[:,:2] -= (hi+lo)/2
        c *= scale
        return c

    def keep(z):
        nonlocal best, best_score, best_z
        c = np.column_stack((ex@z, ey@z, er@z))
        c = repaired(c)
        if c is not None and float(np.sum(c[:,2])) > best_score:
            score = exact(c)
            if score > best_score:
                best_score = score; best = {'circles': c.tolist()}; best_z = z.copy()

    def project(c, transpose):
        c = c.copy()
        if transpose:
            c[:,[0,1]] = c[:,[1,0]]
        lo = np.min(c[:,:2]-c[:,2,None], axis=0)
        hi = np.max(c[:,:2]+c[:,2,None], axis=0)
        c[:,:2] -= (lo+hi)/2
        axis = np.argsort(np.abs(c[:,1]))[:3]
        rest = np.array([k for k in range(21) if k not in axis])
        rest = rest[np.argsort(c[rest,1])]
        neg, pos = rest[:9], rest[9:]
        cost = (c[pos,None,0]-c[neg,0])**2 + (c[pos,None,1]+c[neg,1])**2
        rr, cc = linear_sum_assignment(cost)
        p, q = pos[rr], neg[cc]
        z = np.zeros(34)
        z[:3] = c[axis,0]; z[21:24] = c[axis,2]
        z[3:12] = (c[p,0]+c[q,0])/2
        z[12:21] = np.abs(c[p,1]-c[q,1])/2
        z[24:33] = (c[p,2]+c[q,2])/2
        z[33] = np.clip(hi[0]-lo[0], .25, 1.75)
        error = np.sum(cost[rr,cc]) + 4*np.sum(c[axis,1]**2)
        return error, z

    seeds = []
    if raw is not None and raw.shape == (21,3) and np.isfinite(raw).all():
        seeds = sorted([project(raw, False), project(raw, True)], key=lambda p:p[0])
        seeds = [p[1] for p in seeds]
    z = np.zeros(34)
    z[:3] = [-2.,0.,2.]
    z[3:8] = np.arange(-4.,5.,2.)
    z[8:12] = np.arange(-3.,4.,2.)
    z[12:17] = np.sqrt(3.)
    z[17:21] = 2*np.sqrt(3.)
    z[21:33] = 1.
    z[:33] *= 2/(10+2+4*np.sqrt(3.))
    z[33] = 20/(12+4*np.sqrt(3.))
    seeds.append(z)
    local_end = deadline
    cache_z = None; cache_v = None; cache_j = None
    def calc(z):
        nonlocal cache_z, cache_v, cache_j
        if time.monotonic() >= local_end:
            raise StopSearch()
        if cache_z is None or not np.array_equal(cache_z,z):
            dx = DX@z; dy = DY@z
            d = np.maximum(np.hypot(dx,dy), 1e-14)
            cache_v = np.concatenate((d-RR@z, A@z+B))
            cache_j = np.vstack(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RR, A))
            cache_z = z.copy()
        return cache_v
    def jac(z):
        calc(z)
        return cache_j
    def callback(z):
        if time.monotonic() >= local_end:
            raise StopSearch()
        keep(z)
    def run(z, seconds):
        nonlocal local_end, cache_z
        local_end = min(deadline, time.monotonic()+seconds)
        cache_z = None
        try:
            keep(z)
            out = minimize(lambda v: float(grad@v), z, jac=lambda v:grad,
                method='SLSQP', bounds=bounds,
                constraints={'type':'ineq', 'fun':calc, 'jac':jac},
                callback=callback,
                options={'maxiter':180, 'ftol':params['ftol'], 'disp':False})
            keep(out.x)
        except StopSearch:
            pass

    remaining = deadline-time.monotonic()
    if seeds and remaining > .02:
        run(seeds[0], remaining*params['first_share'])
    if len(seeds)>1 and deadline-time.monotonic()>.04:
        run(seeds[1], (deadline-time.monotonic())*.48)
    if best_z is not None and deadline-time.monotonic()>.03:
        run(best_z.copy(), (deadline-time.monotonic())*.64)
    if deadline-time.monotonic()>.04:
        run(seeds[-1], deadline-time.monotonic()-.008)
    return best
