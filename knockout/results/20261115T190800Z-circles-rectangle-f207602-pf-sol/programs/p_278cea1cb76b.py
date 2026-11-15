import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {
    'release': (0.009, 0.001, 0.06, 'log'),
    'exploration': (0.018, 0.001, 0.1, 'log'),
    'polish_iterations': (35, 10, 100),
}


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    class Stop(Exception):
        pass

    def check():
        if time.monotonic() >= deadline:
            raise Stop()

    def canonical(obj):
        if obj is None or 'circles' in obj:
            return obj
        out = []
        for x, y, r, m in obj['orbits']:
            check()
            out.append([x, y, r])
            if m == 2:
                out.append([x, -y, r])
        return {'circles': out}

    ii, jj = np.triu_indices(21, 1)
    def exact_score(obj):
        if obj is None:
            return -float('inf')
        a = np.asarray(obj['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return -float('inf')
        d = a[ii, :2] - a[jj, :2]
        dist = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
        if np.any(dist < a[ii, 2] + a[jj, 2]):
            return -float('inf')
        xx, yy, rr = [], [], []
        for x, y, r in a:
            check()
            xx.append(Fraction(float(x)))
            yy.append(Fraction(float(y)))
            rr.append(Fraction(float(r)))
        w = max(x+r for x, r in zip(xx, rr)) - min(x-r for x, r in zip(xx, rr))
        h = max(y+r for y, r in zip(yy, rr)) - min(y-r for y, r in zip(yy, rr))
        if w+h > 2:
            return -float('inf')
        return float(np.sum(a[:, 2]))

    original = init
    base = canonical(init)
    best = original
    bestscore = exact_score(base)

    # Twelve orbit representatives: three singles and nine pairs.
    orbit = np.r_[np.arange(12), np.arange(3, 12)]
    sign = np.r_[np.zeros(3), np.ones(9), -np.ones(9)]
    counts = np.r_[np.ones(3), np.full(9, 2.0)]
    pairs = []
    for i in range(12):
        check()
        for j in range(i+1, 12):
            check()
            pairs.append((i, j, 1.0))
            if i >= 3:
                pairs.append((i, j, -1.0))
    for i in range(3, 12):
        check()
        pairs.append((i, i, -1.0))
    pi = np.array([p[0] for p in pairs])
    pj = np.array([p[1] for p in pairs])
    ps = np.array([p[2] for p in pairs])
    npair = len(pi)
    yi = np.maximum(pi-3, 0)
    yj = np.maximum(pj-3, 0)
    iy = pi >= 3
    jy = pj >= 3
    rows = np.arange(npair)
    objective = np.zeros(34)
    objective[21:33] = -counts

    def expand(z):
        y = np.r_[np.zeros(3), z[12:21]]
        return np.column_stack((z[:12][orbit], y[orbit]*sign, z[21:33][orbit]))

    def repaired(z):
        a = expand(z).copy()
        if not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return None
        d = a[ii, :2]-a[jj, :2]
        dist = np.sqrt(d[:, 0]**2+d[:, 1]**2)
        shrink = min(1.0, float(np.min(dist/(a[ii, 2]+a[jj, 2]))))
        a[:, 2] *= max(0.0, shrink*(1.0-3e-15))
        extent = np.max(a[:, :2]+a[:, 2, None], axis=0)-np.min(a[:, :2]-a[:, 2, None], axis=0)
        a *= min(1.0, 2.0/float(extent.sum()))*(1.0-3e-15)
        return {'circles': a.tolist()}

    bestz = None
    def consider(z):
        nonlocal best, bestscore, bestz
        check()
        if -float(objective @ z) < bestscore-1e-6:
            return
        obj = repaired(z)
        val = exact_score(obj)
        if val > bestscore:
            bestscore, best, bestz = val, obj, z.copy()

    cache_z = None
    cache_c = None
    cache_j = None
    def geometry(z):
        nonlocal cache_z, cache_c, cache_j
        check()
        if cache_z is not None and np.array_equal(z, cache_z):
            return cache_c, cache_j
        x = z[:12]
        y = np.r_[np.zeros(3), z[12:21]]
        r = z[21:33]
        dx = x[pi]-x[pj]
        dy = y[pi]-ps*y[pj]
        dist = np.sqrt(dx*dx+dy*dy)
        inv = 1.0/np.maximum(dist, 1e-14)
        c = np.r_[dist-r[pi]-r[pj], x-r, z[33]-x-r, (2.0-z[33])/2.0-y-r]
        J = np.zeros((npair+36, 34))
        np.add.at(J, (rows, pi), dx*inv)
        np.add.at(J, (rows, pj), -dx*inv)
        np.add.at(J, (rows[iy], 12+yi[iy]), (dy*inv)[iy])
        np.add.at(J, (rows[jy], 12+yj[jy]), (-ps*dy*inv)[jy])
        np.add.at(J, (rows, 21+pi), -1.0)
        np.add.at(J, (rows, 21+pj), -1.0)
        k = np.arange(12)
        J[npair+k, k] = 1
        J[npair+k, 21+k] = -1
        J[npair+12+k, k] = -1
        J[npair+12+k, 21+k] = -1
        J[npair+12+k, 33] = 1
        J[npair+24+k, 21+k] = -1
        J[npair+24+k, 33] = -0.5
        J[npair+24+np.arange(3, 12), 12+np.arange(9)] = -1
        cache_z, cache_c, cache_j = z.copy(), c, J
        return c, J

    bounds = [(0.0, 2.0)]*12 + [(0.0, 1.0)]*9 + [(1e-6, 0.45)]*12 + [(0.35, 1.65)]
    def solve(z, iterations, releases=None):
        check()
        margin = np.zeros(npair+36)
        if releases is not None:
            ids, amount = releases
            margin[ids] = amount
        def con(v):
            return geometry(v)[0]-margin
        def jac(v):
            return geometry(v)[1]
        def callback(v):
            check()
            if -objective @ v > bestscore+1e-11 and np.min(geometry(v)[0]) > -1e-7:
                consider(v)
        result = minimize(lambda v: float(objective @ v), z, jac=lambda v: objective,
                          method='SLSQP', bounds=bounds,
                          constraints={'type': 'ineq', 'fun': con, 'jac': jac},
                          callback=callback,
                          options={'maxiter': int(iterations), 'ftol': 7e-14, 'disp': False})
        consider(result.x)
        return result.x

    def project(a, swap):
        a = a.copy()
        if swap:
            a[:, [0, 1]] = a[:, [1, 0]]
        center = float(np.mean(a[:, 1]))
        a[:, 1] -= center
        axis = np.argsort(np.abs(a[:, 1]))[:3]
        rest = np.setdiff1d(np.arange(21), axis)
        rest = rest[np.argsort(a[rest, 1])]
        neg, pos = rest[:9], rest[9:]
        cost = ((a[neg, None, 0]-a[pos, 0])**2 +
                (a[neg, None, 1]+a[pos, 1])**2 +
                (a[neg, None, 2]-a[pos, 2])**2)
        u, v = linear_sum_assignment(cost)
        rep = (a[neg[u]]+a[pos[v]])/2
        rep[:, 1] = (a[pos[v], 1]-a[neg[u], 1])/2
        singles = a[axis].copy()
        singles[:, 1] = 0
        q = np.vstack((singles, rep))
        left = float(np.min(q[:, 0]-q[:, 2]))
        q[:, 0] -= left
        w = float(np.max(q[:, 0]+q[:, 2]))
        total = w+2*float(np.max(q[:, 1]+q[:, 2]))
        q *= 2/total
        z = np.r_[q[:, 0], q[3:, 1], q[:, 2], w*2/total]
        error = float(cost[u, v].sum()+np.sum(a[axis, 1]**2))
        return error, z

    try:
        if base is not None:
            a = np.asarray(base['circles'], dtype=float)
            candidates = [project(a, False), project(a, True)]
            z = min(candidates, key=lambda p: p[0])[1]
        else:
            q = []
            for col in range(3):
                check()
                q.append([0.15+col*0.3, 0.0, 0.14])
            for row in range(1, 4):
                check()
                for col in range(3):
                    check()
                    q.append([0.15+col*0.3, row*0.3, 0.14])
            q = np.array(q)
            z = np.r_[q[:, 0], q[3:, 1], q[:, 2], 0.9]
        anchor = solve(z, 45)
        if bestz is not None:
            anchor = bestz.copy()
        attempt = 0
        while time.monotonic() < deadline:
            check()
            attempt += 1
            z = anchor.copy()
            if attempt % 3 != 0:
                slack = geometry(z)[0][:npair]
                active = np.flatnonzero(slack < 0.002)
                if len(active) >= 3:
                    first = int(rng.choice(active))
                    mids = (z[:12][pi]+z[:12][pj])/2
                    weights = np.exp(-((mids[active]-mids[first])/0.25)**2)
                    weights /= weights.sum()
                    ids = rng.choice(active, 3, replace=False, p=weights)
                    amount = params['release']*float(rng.uniform(0.5, 2.5))
                    z = solve(z, 10, (ids, amount))
            else:
                amplitude = params['exploration']*float(rng.uniform(0.4, 2.5))
                z[:21] += rng.normal(0, amplitude, 21)
                z[21:33] *= np.exp(rng.normal(0, amplitude*2, 12))
                z[:21] = np.maximum(z[:21], 1e-6)
            z = solve(z, params['polish_iterations'])
            if bestz is not None:
                anchor = bestz.copy()
    except Stop:
        pass
    return best if best is not None else {'circles': [[0.02+0.04*(i%3), 0.02+0.04*(i//3), 0.019] for i in range(21)]}
