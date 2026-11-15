import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    "ftol": (2e-11, 1e-13, 1e-7, "log"),
    "maxiter": (180, 40, 400),
    "jitter": (0.025, 0.0, 0.15)
}

class StopSearch(Exception):
    pass


def construct(params: dict, seed: int, budget_s: float, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    tri = np.triu_indices(21, 1)
    best = init
    best_score = -float('inf')

    def exact_score(data):
        try:
            c = np.asarray(data['circles'], dtype=np.float64)
            if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
                return -float('inf')
            dx = c[tri[0], 0] - c[tri[1], 0]
            dy = c[tri[0], 1] - c[tri[1], 1]
            if np.any(np.sqrt(dx**2 + dy**2) < c[tri[0], 2] + c[tri[1], 2]):
                return -float('inf')
            xs, ys, rs = [], [], []
            for x, y, r in c:
                if time.monotonic() >= deadline:
                    return -float('inf')
                xs.append(Fraction(float(x)))
                ys.append(Fraction(float(y)))
                rs.append(Fraction(float(r)))
            w = max(x+r for x, r in zip(xs, rs)) - min(x-r for x, r in zip(xs, rs))
            h = max(y+r for y, r in zip(ys, rs)) - min(y-r for y, r in zip(ys, rs))
            if w+h > 2:
                return -float('inf')
            return float(np.sum(c[:, 2]))
        except (ValueError, TypeError, KeyError, OverflowError):
            return -float('inf')

    if init is not None:
        best_score = exact_score(init)

    def retain(c):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        c = np.array(c, dtype=float, copy=True)
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return
        dx = c[tri[0], 0] - c[tri[1], 0]
        dy = c[tri[0], 1] - c[tri[1], 1]
        ratios = np.sqrt(dx**2 + dy**2) / (c[tri[0], 2] + c[tri[1], 2])
        shrink = min(1.0, float(np.min(ratios))) * (1.0 - 3e-14)
        if shrink <= 0:
            return
        c[:, 2] *= shrink
        lo = np.min(c[:, :2] - c[:, 2, None], axis=0)
        hi = np.max(c[:, :2] + c[:, 2, None], axis=0)
        c[:, :2] -= (lo+hi) * 0.5
        scale = 2.0 / float(np.sum(hi-lo)) * (1.0 - 3e-14)
        c *= scale
        if float(np.sum(c[:, 2])) <= best_score:
            return
        data = {'circles': c.tolist()}
        score = exact_score(data)
        if score > best_score:
            best, best_score = data, score

    # This fallback is feasible independently of all nonlinear optimization.
    fallback = np.zeros((21, 3))
    fallback[:, 0] = np.arange(21) / 22.0
    fallback[:, 2] = (1.0 / 22.0) * (1.0 - 1e-12)
    retain(fallback)

    families = [(4, 3, 4, 3, 4, 3), (5, 4, 3, 4, 5), (5, 4, 5, 4, 3)]
    round_index = 0
    while time.monotonic() < deadline:
        rows = families[round_index % len(families)]
        radius = 1.0 / (max(rows) + 1.0 + (len(rows)-1)*np.sqrt(3.0)/2.0)
        # radius above is twice the required radius for width+height=2.
        radius *= 0.5
        xs, ys, flags = [], [], []
        for row, count in enumerate(rows):
            if time.monotonic() >= deadline:
                break
            for k in range((count+1)//2):
                if time.monotonic() >= deadline:
                    break
                u = (2*k + (1 if count % 2 == 0 else 0))*radius
                xs.append(u)
                ys.append((row-(len(rows)-1)/2)*np.sqrt(3.0)*radius)
                flags.append(u > 0)
        if len(xs) != 12 or time.monotonic() >= deadline:
            break
        paired = np.asarray(flags, dtype=bool)
        pids = np.flatnonzero(paired)
        m = 12
        z = np.concatenate((xs, ys, np.full(m, radius), [max(rows)*radius]))
        if round_index >= len(families):
            z[:m] += rng.normal(0, params['jitter']*radius, m)*paired
            z[m:2*m] += rng.normal(0, params['jitter']*radius, m)
            z[:m] = np.maximum(z[:m], 0)
        
        def expand(v):
            base = np.column_stack((v[:m], v[m:2*m], v[2*m:3*m]))
            mirrors = base[pids].copy()
            mirrors[:, 0] *= -1
            return np.concatenate((base, mirrors), axis=0)

        retain(expand(z))
        si, sj = np.triu_indices(m, 1)
        oi0, oj0 = np.triu_indices(len(pids), 1)
        oi, oj = pids[oi0], pids[oj0]
        ii = np.concatenate((si, oi))
        jj = np.concatenate((sj, oj))
        signs = np.concatenate((-np.ones(len(si)), np.ones(len(oi))))
        nc = len(ii)
        total = nc + len(pids) + 3*m
        rr = np.arange(nc)
        grad = np.zeros(3*m+1)
        grad[2*m:3*m] = -np.where(paired, 2.0, 1.0)
        bounds = [(0.0, 1.0) if paired[k] else (0.0, 0.0) for k in range(m)]
        bounds += [(-1.0, 1.0)]*m + [(1e-8, 0.7)]*m + [(0.03, 0.97)]
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            break
        # Reserve time for at least two different contact families.
        allowance = min(remaining, budget_s*(0.29 if round_index == 0 else 0.23))
        local_end = time.monotonic()+allowance
        cache_z = None
        cache_c = None
        cache_j = None
        last = z.copy()
        ticks = 0

        def check():
            if time.monotonic() >= min(deadline, local_end):
                raise StopSearch()

        def calculate(v):
            nonlocal cache_z, cache_c, cache_j
            check()
            if cache_z is not None and np.array_equal(v, cache_z):
                return cache_c, cache_j
            x, y, r, a = v[:m], v[m:2*m], v[2*m:3*m], v[-1]
            dx = x[ii] + signs*x[jj]
            dy = y[ii] - y[jj]
            dist = np.sqrt(dx*dx + dy*dy)
            safe = np.maximum(dist, 1e-20)
            c = np.empty(total)
            jac = np.zeros((total, 3*m+1))
            c[:nc] = dist-r[ii]-r[jj]
            jac[rr, ii] = dx/safe
            jac[rr, jj] = signs*dx/safe
            jac[rr, m+ii] = dy/safe
            jac[rr, m+jj] = -dy/safe
            jac[rr, 2*m+ii] = -1
            jac[rr, 2*m+jj] = -1
            off = nc
            ids = np.arange(off, off+len(pids))
            c[ids] = x[pids]-r[pids]
            jac[ids, pids] = 1
            jac[ids, 2*m+pids] = -1
            off += len(pids)
            ids = np.arange(off, off+m)
            k = np.arange(m)
            c[ids] = a-x-r
            jac[ids, k] = -1
            jac[ids, 2*m+k] = -1
            jac[ids, -1] = 1
            ids = ids+m
            c[ids] = 1-a-y-r
            jac[ids, m+k] = -1
            jac[ids, 2*m+k] = -1
            jac[ids, -1] = -1
            ids = ids+m
            c[ids] = 1-a+y-r
            jac[ids, m+k] = 1
            jac[ids, 2*m+k] = -1
            jac[ids, -1] = -1
            cache_z, cache_c, cache_j = v.copy(), c, jac
            return c, jac

        def objective(v):
            check()
            return float(grad @ v)

        def callback(v):
            nonlocal last, ticks
            check()
            last = v.copy()
            ticks += 1
            if ticks % 5 == 0:
                retain(expand(v))

        try:
            result = minimize(objective, z, jac=lambda v: grad, bounds=bounds,
                              constraints={'type': 'ineq', 'fun': lambda v: calculate(v)[0],
                                           'jac': lambda v: calculate(v)[1]},
                              method='SLSQP', callback=callback,
                              options={'maxiter': int(params['maxiter']), 'ftol': params['ftol'], 'disp': False})
            last = result.x
        except StopSearch:
            pass
        if time.monotonic() < deadline:
            retain(expand(last))
        round_index += 1
    return best
