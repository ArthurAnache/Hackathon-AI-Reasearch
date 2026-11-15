import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'release': (0.035, 0.002, 0.12),
    'contacts': (3, 3, 7),
    'iterations': (48, 15, 100),
    'pool': (16, 6, 35),
}
N = 21
II, JJ = np.triu_indices(N, 1)

class Finished(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    bestscore = -np.inf

    def check():
        if time.monotonic() >= end:
            raise Finished()

    def exact(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -np.inf
        d = np.sqrt((a[II, 0]-a[JJ, 0])**2 + (a[II, 1]-a[JJ, 1])**2)
        if np.any(d < a[II, 2]+a[JJ, 2]):
            return -np.inf
        f = [[Fraction(float(v)) for v in row] for row in a]
        perimeter = sum(max(row[k]+row[2] for row in f)-min(row[k]-row[2] for row in f) for k in (0, 1))
        if perimeter > 2:
            return -np.inf
        return float(np.sum(a[:, 2]))

    if init is None:
        a = np.array([[(i % 7 + 0.5)*0.2, (i//7+0.5)*0.2, 0.1*(1-1e-13)] for i in range(21)])
        best = {'circles': a.tolist()}
    else:
        a = np.array(init['circles'], dtype=float)
    bestscore = exact(a)
    original = a.copy()
    lo = (a[:, :2]-a[:, 2, None]).min(axis=0)
    hi = (a[:, :2]+a[:, 2, None]).max(axis=0)
    midpoint = (lo+hi)/2

    # Find the better of the two possible reflection directions.
    qualities = []
    for axis in (0, 1):
        check()
        reflected = a[:, :2].copy()
        reflected[:, axis] = 2*midpoint[axis]-reflected[:, axis]
        dd = ((reflected[:, None, :]-a[None, :, :2])**2).sum(axis=2)
        qualities.append(float(np.min(dd, axis=1).sum()))
    transpose = qualities[0] < qualities[1]
    if transpose:
        a[:, [0, 1]] = a[:, [1, 0]]
    lo = (a[:, :2]-a[:, 2, None]).min(axis=0)
    hi = (a[:, :2]+a[:, 2, None]).max(axis=0)
    scale = 2/(hi-lo).sum()
    a[:, :2] = (a[:, :2]-lo)*scale
    a[:, 2] *= scale
    w = (hi[0]-lo[0])*scale
    mid = (2-w)/2
    singles = list(np.argsort(np.abs(a[:, 1]-mid))[:3])
    remaining = set(range(21))-set(singles)
    pairs = []
    while remaining:
        check()
        i = min(remaining)
        remaining.remove(i)
        target = np.array([a[i, 0], 2*mid-a[i, 1]])
        j = min(remaining, key=lambda j: float(np.sum((a[j, :2]-target)**2)))
        remaining.remove(j)
        if a[i, 1] < a[j, 1]:
            i, j = j, i
        pairs.append((i, j))

    # z = T q + offset, with 12 x coordinates, 12 radii,
    # nine signed offsets from the reflection axis, and box width.
    T = np.zeros((64, 34))
    offset = np.zeros(64)
    q = np.zeros(34)
    mirror = np.arange(21)
    for g, group in enumerate([(i,) for i in singles]+pairs):
        check()
        ids = list(group)
        q[g] = np.mean(a[ids, 0])
        q[12+g] = np.mean(a[ids, 2])
        for i in ids:
            T[i, g] = 1
            T[42+i, 12+g] = 1
            T[21+i, 33] = -0.5
            offset[21+i] = 1
        if len(ids) == 2:
            i, j = ids
            k = g-3
            T[21+i, 24+k] = 1
            T[21+j, 24+k] = -1
            q[24+k] = (a[i, 1]-a[j, 1])/2
            mirror[i], mirror[j] = j, i
    T[63, 33] = 1
    q[33] = w

    # Only one representative per reflected circle-pair constraint.
    pi, pj = [], []
    seen = set()
    for i, j in zip(II, JJ):
        check()
        key = tuple(sorted((int(mirror[i]), int(mirror[j]))))
        own = (int(i), int(j))
        canonical = min(own, key)
        if canonical not in seen:
            seen.add(canonical)
            pi.append(i)
            pj.append(j)
    pi, pj = np.array(pi), np.array(pj)
    dxT = T[pi]-T[pj]
    dyT = T[21+pi]-T[21+pj]
    rsT = T[42+pi]+T[42+pj]
    B = []
    bc = []
    for i in range(21):
        check()
        for row, c in ((T[i]-T[42+i], offset[i]-offset[42+i]),
                       (T[63]-T[i]-T[42+i], -offset[i]-offset[42+i]),
                       (T[21+i]-T[42+i], offset[21+i]-offset[42+i]),
                       (-T[63]-T[21+i]-T[42+i], 2-offset[21+i]-offset[42+i])):
            if not any(np.array_equal(row, old) and c == oldc for old, oldc in zip(B, bc)):
                B.append(row.copy())
                bc.append(c)
    B, bc = np.array(B), np.array(bc)
    cost = -T[42:63].sum(axis=0)
    bounds = [(None, None)]*12 + [(1e-6, 0.4)]*12 + [(0, 1)]*9 + [(0.3, 1.7)]

    def constraints(v):
        check()
        z = T@v+offset
        dx = z[pi]-z[pj]
        dy = z[21+pi]-z[21+pj]
        d = np.sqrt(dx*dx+dy*dy)
        d = np.maximum(d, 1e-30)
        values = np.r_[d-z[42+pi]-z[42+pj], B@v+bc]
        jac = np.vstack(((dx[:, None]*dxT+dy[:, None]*dyT)/d[:, None]-rsT, B))
        return values, jac

    def consider(v):
        nonlocal best, bestscore
        check()
        z = T@v+offset
        c = np.column_stack((z[:21], z[21:42], z[42:63]))
        if np.any(c[:, 2] <= 0):
            return
        d = np.sqrt((c[II, 0]-c[JJ, 0])**2+(c[II, 1]-c[JJ, 1])**2)
        ratio = min(1.0, float(np.min(d/(c[II, 2]+c[JJ, 2]))))
        c[:, 2] *= ratio*(1-2e-14)
        low = (c[:, :2]-c[:, 2, None]).min(axis=0)
        high = (c[:, :2]+c[:, 2, None]).max(axis=0)
        s = min(1.0, 2/float(np.sum(high-low)))*(1-2e-14)
        c[:, :2] = (c[:, :2]-low)*s
        c[:, 2] *= s
        if transpose:
            c[:, [0, 1]] = c[:, [1, 0]]
        score = exact(c)
        if score > bestscore:
            bestscore = score
            best = {'circles': c.tolist()}

    def solve(v, relaxation, iterations):
        cache = [None, None]
        def both(x):
            check()
            if cache[0] is None or not np.array_equal(cache[0], x):
                cache[0] = x.copy()
                cache[1] = constraints(x)
            return cache[1]
        return minimize(lambda x: float(cost@x), v, jac=lambda x: cost,
                        bounds=bounds, method='SLSQP',
                        constraints={'type': 'ineq', 'fun': lambda x: both(x)[0]+relaxation,
                                     'jac': lambda x: both(x)[1]},
                        options={'maxiter': iterations, 'ftol': 2e-13, 'disp': False}).x

    try:
        q = solve(q, 0.0, 35)
        consider(q)
        anchor = q.copy()
        vals, jac = constraints(anchor)
        active = np.flatnonzero(vals < 1e-6)
        multipliers = np.linalg.lstsq(jac[active].T, cost, rcond=1e-10)[0]
        order = active[np.argsort(np.abs(multipliers))]
        pool = order[:min(len(order), int(params['pool']))]
        attempt = 0
        while time.monotonic() < end:
            check()
            count = min(int(params['contacts']), len(pool))
            chosen = rng.choice(pool, count, replace=False)
            slack = np.zeros(len(vals))
            magnitude = float(params['release'])*(0.6+0.5*(attempt % 4))
            slack[chosen] = magnitude
            trial = solve(anchor.copy(), slack, int(params['iterations']))
            trial = solve(trial, 0.0, int(params['iterations'])+15)
            consider(trial)
            attempt += 1
    except Finished:
        pass
    return best
