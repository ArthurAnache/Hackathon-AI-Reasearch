import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'release_size': (0.055, 0.015, 0.12),
    'weak_pool': (16, 6, 32),
    'attempts': (3, 1, 6),
    'ftol': (2e-13, 1e-14, 1e-9, 'log')
}

class StopSearch(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    stage_end = deadline
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    ar = np.arange(21)
    rows = np.arange(210)
    best = init
    best_score = -float('inf')

    def guard():
        if time.monotonic() >= stage_end:
            raise StopSearch()

    def exact(c):
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return -float('inf')
        d = c[ii, :2] - c[jj, :2]
        if np.any(np.sqrt(d[:, 0]**2 + d[:, 1]**2) < c[ii, 2] + c[jj, 2]):
            return -float('inf')
        lo = [[], []]
        hi = [[], []]
        for x, y, r in c:
            if time.monotonic() >= deadline:
                return -float('inf')
            x, y, r = Fraction(float(x)), Fraction(float(y)), Fraction(float(r))
            lo[0].append(x-r)
            hi[0].append(x+r)
            lo[1].append(y-r)
            hi[1].append(y+r)
        perimeter = max(hi[0])-min(lo[0])+max(hi[1])-min(lo[1])
        if perimeter > 2:
            return -float('inf')
        return float(np.sum(c[:, 2]))

    def decode(c):
        c = np.asarray(c, dtype=np.float64).copy()
        d = c[ii, :2]-c[jj, :2]
        ds = np.sqrt(d[:, 0]**2+d[:, 1]**2)
        ratio = min(1.0, float(np.min(ds/(c[ii, 2]+c[jj, 2]))))
        c[:, 2] *= ratio*(1.0-4e-15)
        c[:, :2] -= np.min(c[:, :2]-c[:, 2, None], axis=0)
        left = []; right = []; bottom = []; top = []
        for x, y, r in c:
            if time.monotonic() >= deadline:
                raise StopSearch()
            x, y, r = Fraction(float(x)), Fraction(float(y)), Fraction(float(r))
            left.append(x-r); right.append(x+r)
            bottom.append(y-r); top.append(y+r)
        span = max(right)-min(left)+max(top)-min(bottom)
        c *= float(Fraction(2)/span)*(1.0-4e-15)
        return c

    if init is not None:
        c0 = np.asarray(init['circles'], dtype=float)
        best_score = exact(c0)
    else:
        grid = []
        for k in range(21):
            if time.monotonic() >= deadline:
                return None
            row, col = divmod(k, 5)
            grid.append([2.0*col+(row % 2), np.sqrt(3.0)*row, 1.0])
        c0 = np.asarray(grid, dtype=float)

    c0 = c0.copy()
    c0[:, :2] -= np.min(c0[:, :2]-c0[:, 2, None], axis=0)
    c0 /= np.sum(c0[:, 2])
    wh = np.max(c0[:, :2]+c0[:, 2, None], axis=0)
    base = np.r_[c0.ravel(), wh]
    g = np.zeros(65)
    g[63:] = 1.0
    eqjac = np.zeros(65)
    eqjac[2:63:3] = 1.0
    bounds = [(None, None), (None, None), (1e-7, 0.3)]*21 + [(0.05, 2.0)]*2

    def consider(z):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        c = z[:63].reshape(21, 3)
        if not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return
        # Cheap upper bound: repair can only reduce this normalized score.
        spans = np.max(c[:, :2]+c[:, 2, None], axis=0)-np.min(c[:, :2]-c[:, 2, None], axis=0)
        if 2*np.sum(c[:, 2])/np.sum(spans) <= best_score:
            return
        canonical = decode(c)
        score = exact(canonical)
        if score > best_score:
            best_score = score
            best = {'normalized_disks': c.tolist()}

    def objective(z):
        guard()
        return float(z[63]+z[64])

    def equality(z):
        guard()
        return float(np.sum(z[2:63:3])-1.0)

    def constraints(z):
        guard()
        x, y, r = z[:63].reshape(21, 3).T
        dx = x[ii]-x[jj]
        dy = y[ii]-y[jj]
        dist = np.sqrt(dx*dx+dy*dy)
        return np.r_[dist-r[ii]-r[jj], x-r, z[63]-x-r, y-r, z[64]-y-r]

    def jacobian(z):
        guard()
        x, y, r = z[:63].reshape(21, 3).T
        dx = x[ii]-x[jj]
        dy = y[ii]-y[jj]
        dist = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-100)
        dx /= dist
        dy /= dist
        j = np.zeros((294, 65))
        j[rows, 3*ii] = dx
        j[rows, 3*jj] = -dx
        j[rows, 3*ii+1] = dy
        j[rows, 3*jj+1] = -dy
        j[rows, 3*ii+2] = -1
        j[rows, 3*jj+2] = -1
        j[210+ar, 3*ar] = 1
        j[210+ar, 3*ar+2] = -1
        j[231+ar, 3*ar] = -1
        j[231+ar, 3*ar+2] = -1
        j[231+ar, 63] = 1
        j[252+ar, 3*ar+1] = 1
        j[252+ar, 3*ar+2] = -1
        j[273+ar, 3*ar+1] = -1
        j[273+ar, 3*ar+2] = -1
        j[273+ar, 64] = 1
        return j

    def callback(z):
        guard()
        consider(z)

    def optimize(z):
        try:
            consider(z)
            result = minimize(objective, z, jac=lambda z: g,
                method='SLSQP', bounds=bounds,
                constraints=[{'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                             {'type': 'eq', 'fun': equality, 'jac': lambda z: eqjac}],
                callback=callback,
                options={'maxiter': 100, 'ftol': params['ftol'], 'disp': False})
            consider(result.x)
            return result.x
        except StopSearch:
            return z

    stage_end = min(deadline, start+0.16*budget_s)
    polished = optimize(base)
    stage_end = deadline
    if time.monotonic() >= deadline-0.06:
        return best
    try:
        if np.min(constraints(polished)) >= -1e-8:
            base = polished
        values = constraints(base)
        active = np.flatnonzero(values < 2e-7)
        a = np.vstack([jacobian(base)[active], eqjac])
        inverse = np.linalg.pinv(a, rcond=1e-10)
        guard()
        multipliers = np.linalg.lstsq(a.T, g, rcond=1e-10)[0][:-1]
        contact_rows = np.flatnonzero(active < 210)
        order = contact_rows[np.argsort(np.abs(multipliers[contact_rows]))]
        pool = order[:min(int(params['weak_pool']), len(order))]
        attempts = int(params['attempts'])
        for attempt in range(attempts):
            if time.monotonic() >= deadline-0.045 or len(pool) < 3:
                break
            chosen = rng.choice(pool, size=3, replace=False)
            opening = np.zeros(len(active)+1)
            opening[chosen] = rng.uniform(0.7, 1.3, 3)
            direction = inverse @ opening
            movement = np.max(np.linalg.norm(direction[:63].reshape(21, 3)[:, :2], axis=1))
            if movement < 1e-12:
                continue
            amplitude = params['release_size']*(0.75+0.5*attempt)
            step = amplitude/movement
            decreasing = direction[2:63:3] < 0
            if np.any(decreasing):
                step = min(step, float(np.min(0.72*base[2:63:3][decreasing]/(-direction[2:63:3][decreasing]))))
            trial = base+step*direction
            remaining = deadline-time.monotonic()
            stage_end = min(deadline, time.monotonic()+remaining/(attempts-attempt))
            optimize(trial)
            stage_end = deadline
    except (StopSearch, np.linalg.LinAlgError):
        pass
    return best
