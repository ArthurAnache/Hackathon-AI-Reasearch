import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {
    'release_gap': (0.004, 0.0001, 0.03, 'log'),
    'jitter': (0.018, 0.002, 0.08, 'log'),
    'initial_fraction': (0.31, 0.15, 0.45),
}

class Stop(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + 0.70 * budget_s
    local_deadline = deadline
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)

    def check():
        if time.monotonic() >= min(deadline, local_deadline):
            raise Stop()

    def canonical(obj):
        if obj is None or 'circles' in obj:
            return obj
        c = []
        for x, y, r, m in obj['orbits']:
            check()
            c.append([x, y, r])
            if m == 2:
                c.append([x, -y, r])
        return {'circles': c}

    def score(obj):
        if obj is None:
            return -float('inf')
        a = np.asarray(obj['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return -float('inf')
        d = a[ii, :2] - a[jj, :2]
        if np.any(np.sqrt(d[:, 0]**2 + d[:, 1]**2) < a[ii, 2] + a[jj, 2]):
            return -float('inf')
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        w = max(u+v for u, v in zip(x, r))-min(u-v for u, v in zip(x, r))
        h = max(u+v for u, v in zip(y, r))-min(u-v for u, v in zip(y, r))
        if w+h > 2:
            return -float('inf')
        return float(np.sum(a[:, 2]))

    fallback = {'circles': [[0.05 + 0.1*(i % 3), 0.05 + 0.1*(i // 3), 0.049] for i in range(21)]}
    base = canonical(init)
    best = init if init is not None else fallback
    bestscore = score(base if base is not None else fallback)
    if not np.isfinite(bestscore):
        best, bestscore = fallback, score(fallback)

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
    rows = np.arange(npair)
    iy, jy = pi >= 3, pj >= 3
    orbit = np.r_[np.arange(12), np.arange(3, 12)]
    signs = np.r_[np.zeros(3), np.ones(9), -np.ones(9)]
    objective = np.zeros(34)
    objective[21:33] = -np.r_[np.ones(3), np.full(9, 2.0)]
    k = np.arange(12)
    fixed = np.zeros((npair+36, 34))
    np.add.at(fixed, (rows, 21+pi), -1.0)
    np.add.at(fixed, (rows, 21+pj), -1.0)
    fixed[npair+k, k] = 1
    fixed[npair+k, 21+k] = -1
    fixed[npair+12+k, k] = -1
    fixed[npair+12+k, 21+k] = -1
    fixed[npair+12+k, 33] = 1
    fixed[npair+24+k, 21+k] = -1
    fixed[npair+24+k, 33] = -0.5
    fixed[npair+24+np.arange(3, 12), 12+np.arange(9)] = -1
    cache = [None, None, None]

    def geometry(z):
        check()
        if cache[0] is not None and np.array_equal(z, cache[0]):
            return cache[1], cache[2]
        x, r = z[:12], z[21:33]
        y = np.r_[np.zeros(3), z[12:21]]
        dx, dy = x[pi]-x[pj], y[pi]-ps*y[pj]
        d = np.sqrt(dx*dx+dy*dy)
        inv = 1/np.maximum(d, 1e-30)
        g = np.r_[d-r[pi]-r[pj], x-r, z[33]-x-r, (2-z[33])/2-y-r]
        J = fixed.copy()
        np.add.at(J, (rows, pi), dx*inv)
        np.add.at(J, (rows, pj), -dx*inv)
        np.add.at(J, (rows[iy], 12+pi[iy]-3), (dy*inv)[iy])
        np.add.at(J, (rows[jy], 12+pj[jy]-3), (-ps*dy*inv)[jy])
        cache[:] = [z.copy(), g, J]
        return g, J

    def expand(z):
        y = np.r_[np.zeros(3), z[12:21]]
        return np.column_stack((z[:12][orbit], y[orbit]*signs, z[21:33][orbit]))

    bestz = None
    def save(z):
        nonlocal best, bestscore, bestz
        check()
        if -objective @ z < bestscore-1e-6:
            return
        a = expand(z)
        if not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return
        a[:, :2] -= np.mean(a[:, :2], axis=0)
        d = a[ii, :2]-a[jj, :2]
        distances = np.sqrt(d[:, 0]**2+d[:, 1]**2)
        ratio = min(1.0, float(np.min(distances/(a[ii, 2]+a[jj, 2]))))
        a[:, 2] *= max(0.0, ratio)*(1-5e-15)
        extent = np.max(a[:, :2]+a[:, 2, None], axis=0)-np.min(a[:, :2]-a[:, 2, None], axis=0)
        a *= min(1.0, 2.0/float(extent.sum()))*(1-5e-15)
        obj = {'circles': a.tolist()}
        val = score(obj)
        if val > bestscore:
            best, bestscore, bestz = obj, val, z.copy()

    bounds = [(0, 2)]*12+[(0, 1)]*9+[(1e-6, 0.49)]*12+[(0.15, 1.85)]
    def run(z, seconds, iterations=90, release=None):
        nonlocal local_deadline
        local_deadline = min(deadline, time.monotonic()+max(0.001, seconds))
        last = z.copy()
        margin = np.zeros(npair+36)
        if release is not None:
            margin[release[0]] = release[1]
        def callback(v):
            nonlocal last
            check()
            last = v.copy()
            if -objective @ v > bestscore and np.min(geometry(v)[0]) > -1e-7:
                save(v)
        try:
            res = minimize(lambda v: float(objective @ v), z, jac=lambda v: objective,
                method='SLSQP', bounds=bounds,
                constraints={'type': 'ineq', 'fun': lambda v: geometry(v)[0]-margin,
                             'jac': lambda v: geometry(v)[1]},
                callback=callback, options={'maxiter': iterations, 'ftol': 2e-14, 'disp': False})
            last = res.x.copy()
            save(last)
        except (Stop, ValueError, np.linalg.LinAlgError):
            pass
        local_deadline = deadline
        return last

    def project(raw, swap):
        a = raw.copy()
        if swap:
            a[:, [0, 1]] = a[:, [1, 0]]
        a[:, 1] -= (np.max(a[:, 1]+a[:, 2])+np.min(a[:, 1]-a[:, 2]))/2
        singles = np.argsort(np.abs(a[:, 1]))[:3]
        rest = np.setdiff1d(np.arange(21), singles)
        rest = rest[np.argsort(a[rest, 1])]
        neg, pos = rest[:9], rest[9:]
        cost = (a[neg, None, 0]-a[pos, 0])**2+(a[neg, None, 1]+a[pos, 1])**2+(a[neg, None, 2]-a[pos, 2])**2
        u, v = linear_sum_assignment(cost)
        q = np.vstack((a[singles], (a[neg[u]]+a[pos[v]])/2))
        q[:3, 1] = 0
        q[3:, 1] = (a[pos[v], 1]-a[neg[u], 1])/2
        q[:, 0] -= np.min(q[:, 0]-q[:, 2])
        w = np.max(q[:, 0]+q[:, 2])
        scale = 2/(w+2*np.max(q[:, 1]+q[:, 2]))
        q *= scale
        return float(cost[u, v].sum()+np.sum(a[singles, 1]**2)), np.r_[q[:, 0], q[3:, 1], q[:, 2], w*scale]

    try:
        projections = []
        if base is not None:
            raw = np.asarray(base['circles'], dtype=float)
            if raw.shape == (21, 3) and np.isfinite(raw).all():
                for swap in (False, True):
                    check()
                    projections.append(project(raw, swap))
        if projections:
            z = min(projections, key=lambda p: p[0])[1]
            z = run(z, params['initial_fraction']*budget_s)
            g, J = geometry(z)
            active = np.flatnonzero(g < 2e-7)
            if len(active) >= 34:
                for iteration in range(4):
                    check()
                    g, J = geometry(z)
                    step = np.linalg.lstsq(J[active], -g[active], rcond=1e-12)[0]
                    if np.linalg.norm(step) > 0.015:
                        break
                    z += step
                    save(z)
        else:
            z = None
        anchor = bestz.copy() if bestz is not None else z
        # A compact five-row family with three axis circles and nine pairs.
        q = np.array([[-0.2, 0, 0.095], [0, 0, 0.095], [0.2, 0, 0.095]]+
                     [[(i-2)*0.2, 0.18, 0.095] for i in range(5)]+
                     [[(i-1.5)*0.2, 0.36, 0.095] for i in range(4)])
        q[:, 0] -= np.min(q[:, 0]-q[:, 2])
        w = np.max(q[:, 0]+q[:, 2])
        scale = 2/(w+2*np.max(q[:, 1]+q[:, 2]))
        q *= scale
        lattice = np.r_[q[:, 0], q[3:, 1], q[:, 2], w*scale]
        if time.monotonic()+0.08 < deadline:
            run(lattice, min(0.18*budget_s, deadline-time.monotonic()), 85)
        attempt = 0
        while time.monotonic()+0.055 < deadline:
            check()
            attempt += 1
            if bestz is not None:
                anchor = bestz.copy()
            if anchor is None:
                anchor = lattice.copy()
            z = anchor.copy()
            if attempt % 3 != 0:
                g, J = geometry(z)
                active = np.flatnonzero(g < 3e-6)
                multipliers = np.linalg.lstsq(J[active].T, objective, rcond=1e-10)[0]
                eligible = active < npair
                contacts = active[eligible]
                strength = np.abs(multipliers[eligible])
                if len(contacts) >= 3:
                    pool = contacts[np.argsort(strength)[:max(6, len(contacts)//2)]]
                    selected = rng.choice(pool, 3, replace=False)
                    z = run(z, min(0.07*budget_s, 0.11), 18,
                            (selected, params['release_gap']*rng.uniform(0.6, 2.0)))
            else:
                z[:21] += rng.normal(0, params['jitter'], 21)
                z[:21] = np.maximum(z[:21], 1e-6)
            if time.monotonic()+0.025 < deadline:
                run(z, min(0.13*budget_s, deadline-time.monotonic()), 65)
    except (Stop, ValueError, np.linalg.LinAlgError):
        pass
    return best
