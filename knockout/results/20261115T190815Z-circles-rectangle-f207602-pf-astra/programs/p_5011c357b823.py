import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'release_size': (0.095, 0.025, 0.22),
    'active_threshold': (3e-7, 1e-9, 1e-5, 'log'),
    'restarts': (5, 1, 12),
}

class Deadline(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    rows = np.arange(m)
    ids = np.arange(n)

    def check():
        if time.monotonic() >= deadline:
            raise Deadline()

    def exact_score(c):
        check()
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return -float('inf')
        d = np.sqrt((c[ii, 0] - c[jj, 0]) ** 2 + (c[ii, 1] - c[jj, 1]) ** 2)
        if np.any(d < c[ii, 2] + c[jj, 2]):
            return -float('inf')
        left, right, bottom, top = [], [], [], []
        for x, y, r in c:
            check()
            x, y, r = Fraction(float(x)), Fraction(float(y)), Fraction(float(r))
            left.append(x-r)
            right.append(x+r)
            bottom.append(y-r)
            top.append(y+r)
        if max(right)-min(left)+max(top)-min(bottom) > 2:
            return -float('inf')
        return float(np.sum(c[:, 2]))

    if init is None:
        c0 = np.empty((21, 3))
        for k in range(21):
            if time.monotonic() >= deadline:
                break
            c0[k] = (0.1 + 0.2*(k % 7), 0.1 + 0.2*(k // 7), 0.09999999)
        # Vectorized fallback remains valid even with an extremely short budget.
        k = np.arange(21)
        c0[:, 0] = 0.1 + 0.2*(k % 7)
        c0[:, 1] = 0.1 + 0.2*(k // 7)
        c0[:, 2] = 0.09999999
        best = {'circles': c0.tolist()}
    else:
        best = init
        c0 = np.asarray(init['circles'], dtype=float).copy()
    best_score = -float('inf')

    def certify(c):
        nonlocal best, best_score
        check()
        c = np.asarray(c, dtype=float).copy()
        if not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return
        dist = np.sqrt((c[ii, 0]-c[jj, 0])**2 + (c[ii, 1]-c[jj, 1])**2)
        ratio = float(np.min(dist/(c[ii, 2]+c[jj, 2])))
        c[:, 2] *= min(1.0, ratio)*(1.0-1.8e-15)
        lo = np.min(c[:, :2]-c[:, 2, None], axis=0)
        hi = np.max(c[:, :2]+c[:, 2, None], axis=0)
        c[:, :2] -= (lo+hi)*0.5
        perimeter = float(np.sum(hi-lo))
        if perimeter <= 0:
            return
        c *= (2.0-1.0e-14)/perimeter
        if float(np.sum(c[:, 2])) <= best_score:
            return
        value = exact_score(c)
        if value > best_score:
            best_score = value
            best = {'circles': c.tolist()}

    def unpack(z):
        return np.column_stack((z[:n], z[n:2*n], z[2*n:3*n]))

    fixed = np.zeros((4*n, 64))
    fixed[ids, ids] = 1
    fixed[ids, 2*n+ids] = -1
    fixed[n+ids, ids] = -1
    fixed[n+ids, 2*n+ids] = -1
    fixed[n+ids, 63] = 1
    fixed[2*n+ids, n+ids] = 1
    fixed[2*n+ids, 2*n+ids] = -1
    fixed[3*n+ids, n+ids] = -1
    fixed[3*n+ids, 2*n+ids] = -1
    fixed[3*n+ids, 63] = -1
    objective_grad = np.zeros(64)
    objective_grad[2*n:3*n] = -1

    def constraints(z):
        check()
        x, y, r, w = z[:n], z[n:2*n], z[2*n:3*n], z[63]
        dx, dy = x[ii]-x[jj], y[ii]-y[jj]
        return np.concatenate((np.sqrt(dx*dx+dy*dy)-r[ii]-r[jj],
                               x-r, w-x-r, y-r, 2-w-y-r))

    def jacobian(z):
        check()
        dx, dy = z[ii]-z[jj], z[n+ii]-z[n+jj]
        d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-30)
        a = np.zeros((m+4*n, 64))
        a[rows, ii] = dx/d
        a[rows, jj] = -dx/d
        a[rows, n+ii] = dy/d
        a[rows, n+jj] = -dy/d
        a[rows, 2*n+ii] = -1
        a[rows, 2*n+jj] = -1
        a[m:] = fixed
        return a

    def objective(z):
        check()
        return -float(np.sum(z[2*n:3*n]))

    def callback(z):
        check()
        certify(unpack(z))

    bounds = [(0.0, 2.0)]*(2*n) + [(1e-7, 0.5)]*n + [(0.05, 1.95)]

    def polish(z, iterations):
        check()
        result = minimize(objective, z, jac=lambda v: objective_grad,
                          bounds=bounds,
                          constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                          method='SLSQP', callback=callback,
                          options={'maxiter': iterations, 'ftol': 2e-14, 'disp': False})
        check()
        certify(unpack(result.x))
        return result.x

    try:
        best_score = exact_score(c0)
        certify(c0)
        lo = np.min(c0[:, :2]-c0[:, 2, None], axis=0)
        hi = np.max(c0[:, :2]+c0[:, 2, None], axis=0)
        c0[:, :2] -= lo
        c0 *= 2.0/float(np.sum(hi-lo))
        z = np.concatenate((c0[:, 0], c0[:, 1], c0[:, 2],
                            [2.0*(hi[0]-lo[0])/float(np.sum(hi-lo))]))

        # Remove residual error in the incumbent's contact equations first.
        active = np.flatnonzero(constraints(z) < params['active_threshold'])
        if len(active) >= 60:
            for iteration in range(5):
                check()
                g = constraints(z)[active]
                a = jacobian(z)[active]
                step = np.linalg.lstsq(a, -g, rcond=1e-12)[0]
                if np.max(np.abs(step)) > 0.015:
                    break
                z = z + step
                certify(unpack(z))
                if np.max(np.abs(step)) < 2e-15:
                    break
        z = polish(z, 65)
        base = z.copy()
        gaps = constraints(base)
        active = np.flatnonzero(gaps < params['active_threshold'])
        a = jacobian(base)[active]
        if len(active) < 3:
            return best
        multipliers = np.linalg.lstsq(a.T, objective_grad, rcond=1e-11)[0]
        contact_positions = np.flatnonzero(active < m)
        ranked = contact_positions[np.argsort(multipliers[contact_positions])]
        if len(ranked) < 3:
            return best
        inverse = np.linalg.pinv(a, rcond=1e-11)
        centers = unpack(base)[:, :2]
        mids = 0.5*(centers[ii]+centers[jj])

        for attempt in range(int(params['restarts'])):
            check()
            if deadline-time.monotonic() < 0.14:
                break
            pivot = int(ranked[attempt % min(8, len(ranked))])
            pool = ranked[:min(24, len(ranked))]
            distances = np.linalg.norm(mids[active[pool]]-mids[active[pivot]], axis=1)
            order = np.argsort(distances + 0.08*rng.random(len(pool)))
            chosen = pool[order[:3]]
            releases = np.zeros(len(active))
            releases[chosen] = rng.uniform(0.65, 1.35, 3)
            direction = inverse @ releases
            motion = np.max(np.sqrt(direction[:n]**2+direction[n:2*n]**2))
            if motion < 1e-12:
                continue
            amplitude = params['release_size']*(1.0+0.35*(attempt % 3))
            trial = base + (amplitude/motion)*direction
            trial[:2*n] = np.clip(trial[:2*n], 1e-5, 1.99999)
            trial[2*n:3*n] = np.clip(trial[2*n:3*n], 0.015, 0.3)
            trial[63] = np.clip(trial[63], 0.2, 1.8)
            polish(trial, 80)
    except Deadline:
        pass
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        pass
    return best
