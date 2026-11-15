import time
import numpy as np
from scipy.optimize import linprog, minimize
from fractions import Fraction

PARAMS = {
    'screen_fraction': (0.34, 0.15, 0.45),
    'lp_steps': (13, 5, 24),
    'radius_floor': (0.000001, 0.00000001, 0.001, 'log'),
    'polish_iterations': (100, 30, 180)
}

class StopSearch(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + 0.70 * budget_s
    screen_end = started + float(params['screen_fraction']) * budget_s
    n = 21
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    ids = np.arange(n)
    rows = np.arange(m)
    best = None
    best_score = -float('inf')

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def unpack(obj):
        if 'circles' in obj:
            return np.asarray(obj['circles'], dtype=float)
        return np.asarray(obj['coordinate_rows'], dtype=float).T

    def exact_score(c):
        check()
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return -float('inf')
        x, y, r = c.T
        d = np.sqrt((x[ii] - x[jj]) ** 2 + (y[ii] - y[jj]) ** 2)
        if np.any(d < r[ii] + r[jj]):
            return -float('inf')
        left, right, bottom, top = [], [], [], []
        for a, b, rad in c:
            check()
            a, b, rad = Fraction(float(a)), Fraction(float(b)), Fraction(float(rad))
            left.append(a - rad)
            right.append(a + rad)
            bottom.append(b - rad)
            top.append(b + rad)
        if max(right) - min(left) + max(top) - min(bottom) > 2:
            return -float('inf')
        return float(np.sum(r))

    def offer(z):
        nonlocal best, best_score
        check()
        c = np.column_stack((z[:n], z[n:2*n], z[2*n:3*n])).copy()
        if not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return
        if float(np.sum(c[:, 2])) < best_score - 1e-8:
            return
        x, y, r = c.T
        d = np.sqrt((x[ii]-x[jj])**2 + (y[ii]-y[jj])**2)
        factor = min(1.0, float(np.min(d / (r[ii]+r[jj]))))
        r *= factor * (1.0 - 4e-14)
        lo = np.min(c[:, :2] - r[:, None], axis=0)
        hi = np.max(c[:, :2] + r[:, None], axis=0)
        c[:, :2] -= (lo + hi) * 0.5
        c *= 2.0 * (1.0 - 4e-14) / float(np.sum(hi-lo))
        score = exact_score(c)
        if score > best_score:
            best_score = score
            best = {'coordinate_rows': c.T.tolist()}

    # Four walls per circle; the rectangle has height 2-W.
    wall = np.zeros((4*n, 3*n+1))
    off = np.zeros(4*n)
    wall[ids, ids] = 1
    wall[ids, 2*n+ids] = -1
    wall[n+ids, ids] = -1
    wall[n+ids, 2*n+ids] = -1
    wall[n+ids, -1] = 1
    wall[2*n+ids, n+ids] = 1
    wall[2*n+ids, 2*n+ids] = -1
    wall[3*n+ids, n+ids] = -1
    wall[3*n+ids, 2*n+ids] = -1
    wall[3*n+ids, -1] = -1
    off[3*n:] = 2
    objective = np.zeros(3*n+1)
    objective[2*n:3*n] = -1
    bounds = [(0.0, 2.0)]*(2*n) + [(float(params['radius_floor']), 0.6)]*n + [(0.12, 1.88)]
    A = np.zeros((m+4*n, 3*n+1))
    A[rows, 2*n+ii] = 1
    A[rows, 2*n+jj] = 1
    A[m:] = -wall
    rhs = np.concatenate((np.zeros(m), off))

    def encode(c):
        c = c.copy()
        r = c[:, 2]
        lo = np.min(c[:, :2] - r[:, None], axis=0)
        hi = np.max(c[:, :2] + r[:, None], axis=0)
        scale = 2.0 / max(float(np.sum(hi-lo)), 1e-12)
        c[:, :2] -= lo
        c *= scale
        return np.concatenate((c[:, 0], c[:, 1], c[:, 2], [(hi[0]-lo[0])*scale]))

    def row_seed(counts):
        parts = []
        for k, count in enumerate(counts):
            check()
            xx = 2.0*np.arange(count) - (count-1)
            yy = np.full(count, k*np.sqrt(3.0))
            parts.append(np.column_stack((xx, yy, np.ones(count))))
        return encode(np.concatenate(parts))

    def lp_step(z):
        check()
        dx, dy = z[ii]-z[jj], z[n+ii]-z[n+jj]
        d = np.maximum(np.hypot(dx, dy), 1e-15)
        ux, uy = dx/d, dy/d
        A[rows, ii] = -ux
        A[rows, jj] = ux
        A[rows, n+ii] = -uy
        A[rows, n+jj] = uy
        remaining = deadline-time.monotonic()
        if remaining < 0.025:
            raise StopSearch()
        sol = linprog(objective, A_ub=A, b_ub=rhs, bounds=bounds,
                      method='highs', options={'time_limit': max(0.01, min(0.12, remaining)),
                      'primal_feasibility_tolerance': 1e-9,
                      'dual_feasibility_tolerance': 1e-9})
        if sol.x is None:
            return z
        return sol.x

    def constraints(z):
        check()
        dx, dy = z[ii]-z[jj], z[n+ii]-z[n+jj]
        rr = z[2*n+ii]+z[2*n+jj]
        return np.concatenate((dx*dx+dy*dy-rr*rr, wall@z+off))

    def jacobian(z):
        check()
        J = np.zeros((m+4*n, 3*n+1))
        dx = 2*(z[ii]-z[jj])
        dy = 2*(z[n+ii]-z[n+jj])
        rr = -2*(z[2*n+ii]+z[2*n+jj])
        J[rows, ii], J[rows, jj] = dx, -dx
        J[rows, n+ii], J[rows, n+jj] = dy, -dy
        J[rows, 2*n+ii], J[rows, 2*n+jj] = rr, rr
        J[m:] = wall
        return J

    try:
        if init is not None:
            original = unpack(init)
            score = exact_score(original)
            if np.isfinite(score):
                best, best_score = init, score
        fallback = row_seed([7, 7, 7])
        offer(fallback)
        starts = []
        if init is not None:
            c = unpack(init)
            if c.shape == (21, 3) and np.isfinite(c).all() and np.min(c[:, 2]) > 0:
                starts.append(encode(c))
        starts.extend([row_seed([4, 3, 4, 3, 4, 3]),
                       row_seed([4, 5, 4, 5, 3]),
                       row_seed([4, 5, 3, 5, 4])])
        pool = []
        for index, z in enumerate(starts):
            check()
            local_end = min(screen_end, time.monotonic() + max(0.035, (screen_end-time.monotonic())/(len(starts)-index)))
            for iteration in range(int(params['lp_steps'])):
                check()
                if time.monotonic() >= local_end and iteration >= 2:
                    break
                previous = z
                z = lp_step(z)
                if iteration % 4 == 3:
                    offer(z)
                if np.max(np.abs(z-previous)) < 2e-7:
                    break
            offer(z)
            pool.append(z)
            if time.monotonic() >= screen_end:
                break
        pool.sort(key=lambda v: -float(np.sum(v[2*n:3*n])))
        for rank, z in enumerate(pool[:2]):
            check()
            local_end = deadline if rank else time.monotonic() + 0.64*(deadline-time.monotonic())
            last = [z]
            def callback(v):
                check()
                last[0] = v.copy()
                offer(v)
                if time.monotonic() >= local_end:
                    raise StopSearch()
            try:
                res = minimize(lambda v: float(objective@v), z, jac=lambda v: objective,
                    method='SLSQP', bounds=bounds,
                    constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                    callback=callback,
                    options={'maxiter': int(params['polish_iterations']), 'ftol': 2e-12, 'disp': False})
                offer(res.x)
            except StopSearch:
                if time.monotonic() >= deadline:
                    break
        while time.monotonic() < deadline-0.04:
            check()
            z = encode(unpack(best))
            updated = lp_step(z)
            offer(updated)
            if np.max(np.abs(updated-z)) < 1e-8:
                break
    except StopSearch:
        pass
    if best is not None:
        return best
    # A uniformly spaced rectangular grid is an unconditional valid fallback.
    x = np.tile(np.arange(7)*0.2, 3)
    y = np.repeat(np.arange(3)*0.2, 7)
    return {'coordinate_rows': [x.tolist(), y.tolist(), [0.099999999999]*21]}
