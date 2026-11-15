import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import lstsq

PARAMS = {
    'opening': (0.045, 0.005, 0.15),
    'contacts': (4, 3, 7),
    'iterations': (65, 20, 120),
    'active_tol': (2e-7, 1e-9, 1e-5, 'log')
}


def construct(params: dict, seed: int, budget_s: float, init=None):
    started = time.monotonic()
    deadline = started + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    ar = np.arange(n)

    class Stop(Exception):
        pass

    def check():
        if time.monotonic() >= deadline:
            raise Stop()

    def exact_score(data):
        check()
        try:
            c = np.asarray(data['circles'], dtype=np.float64)
            if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
                return -float('inf')
            xs, ys, rs = [], [], []
            for x, y, r in c:
                check()
                xs.append(Fraction(float(x)))
                ys.append(Fraction(float(y)))
                rs.append(Fraction(float(r)))
            right, left, top, bottom = [], [], [], []
            for x, y, r in zip(xs, ys, rs):
                check()
                right.append(x + r)
                left.append(x - r)
                top.append(y + r)
                bottom.append(y - r)
            if max(right) - min(left) + max(top) - min(bottom) > Fraction(2):
                return -float('inf')
            for a, b in itertools.combinations(c, 2):
                check()
                dist = np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
                if dist < a[2] + b[2]:
                    return -float('inf')
            return float(np.sum(c[:, 2]))
        except (ValueError, KeyError, TypeError, OverflowError):
            return -float('inf')

    fallback = np.column_stack((0.2 * (ar % 7), 0.2 * (ar // 7), np.full(n, 0.09999999999999)))
    best = init if init is not None else {'circles': fallback.tolist()}
    best_score = -float('inf')
    try:
        best_score = exact_score(best)
    except Stop:
        return best

    def encode(c):
        c = np.array(c, dtype=float, copy=True)
        left = np.min(c[:, 0] - c[:, 2])
        bottom = np.min(c[:, 1] - c[:, 2])
        width = np.max(c[:, 0] + c[:, 2]) - left
        height = np.max(c[:, 1] + c[:, 2]) - bottom
        c[:, 0] -= left
        c[:, 1] -= bottom
        scale = 2.0 / (width + height)
        c *= scale
        return np.r_[c[:, 0], c[:, 1], c[:, 2], width * scale]

    def decode(z):
        return np.column_stack((z[:n], z[n:2*n], z[2*n:3*n]))

    def certify(z):
        nonlocal best, best_score
        check()
        c = decode(z)
        if not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return
        if np.sum(c[:, 2]) <= best_score - 1e-11:
            return
        c[:, :2] -= np.mean(c[:, :2], axis=0)
        d = np.sqrt(np.sum((c[ii, :2] - c[jj, :2])**2, axis=1))
        ratio = float(np.min(d / (c[ii, 2] + c[jj, 2])))
        c[:, 2] *= min(1.0, ratio) * (1.0 - 2e-15)
        width = np.max(c[:, 0] + c[:, 2]) - np.min(c[:, 0] - c[:, 2])
        height = np.max(c[:, 1] + c[:, 2]) - np.min(c[:, 1] - c[:, 2])
        c *= min(1.0, 2.0 / (width + height)) * (1.0 - 2e-15)
        for attempt in range(3):
            check()
            if float(np.sum(c[:, 2])) <= best_score:
                return
            data = {'circles': c.tolist()}
            score = exact_score(data)
            if score > best_score:
                best, best_score = data, score
                return
            c[:, 2] *= 1.0 - 1e-14

    wall_jac = np.zeros((4*n, 3*n+1))
    wall_jac[ar, ar] = 1.0
    wall_jac[ar, 2*n+ar] = -1.0
    wall_jac[n+ar, ar] = -1.0
    wall_jac[n+ar, 2*n+ar] = -1.0
    wall_jac[n+ar, -1] = 1.0
    wall_jac[2*n+ar, n+ar] = 1.0
    wall_jac[2*n+ar, 2*n+ar] = -1.0
    wall_jac[3*n+ar, n+ar] = -1.0
    wall_jac[3*n+ar, 2*n+ar] = -1.0
    wall_jac[3*n+ar, -1] = -1.0
    pair_rows = np.arange(m)
    cache_z = None
    cache_c = None
    cache_j = None

    def geometry(z):
        nonlocal cache_z, cache_c, cache_j
        check()
        if cache_z is not None and np.array_equal(z, cache_z):
            return cache_c, cache_j
        x, y, r, w = z[:n], z[n:2*n], z[2*n:3*n], z[-1]
        dx, dy = x[ii]-x[jj], y[ii]-y[jj]
        d = np.sqrt(dx*dx + dy*dy)
        safe = np.maximum(d, 1e-100)
        g = np.concatenate((d-r[ii]-r[jj], x-r, w-x-r, y-r, 2.0-w-y-r))
        j = np.zeros((m+4*n, 3*n+1))
        j[pair_rows, ii] = dx/safe
        j[pair_rows, jj] = -dx/safe
        j[pair_rows, n+ii] = dy/safe
        j[pair_rows, n+jj] = -dy/safe
        j[pair_rows, 2*n+ii] = -1.0
        j[pair_rows, 2*n+jj] = -1.0
        j[m:] = wall_jac
        cache_z, cache_c, cache_j = z.copy(), g, j
        return g, j

    grad = np.zeros(3*n+1)
    grad[2*n:3*n] = -1.0

    def objective(z):
        check()
        return -float(np.sum(z[2*n:3*n]))

    def objective_jac(z):
        check()
        return grad

    def contact_polish(z):
        check()
        z = z.copy()
        g, j = geometry(z)
        active = np.flatnonzero(np.abs(g) < params['active_tol'])
        if len(active) < 3*n+1 or np.min(g) < -1e-6:
            return z
        for iteration in range(7):
            check()
            g, j = geometry(z)
            delta = lstsq(j[active], -g[active], cond=1e-12, check_finite=False, lapack_driver='gelsy')[0]
            if not np.isfinite(delta).all() or np.max(np.abs(delta)) > 0.01:
                break
            z += delta
            certify(z)
            if np.max(np.abs(delta)) < 2e-16:
                break
        return z

    bounds = [(None, None)] * (2*n) + [(1e-8, 0.5)] * n + [(0.15, 1.85)]
    callback_count = 0
    latest = None

    def callback(z):
        nonlocal callback_count, latest
        check()
        latest = z.copy()
        callback_count += 1
        if callback_count % 4 == 0:
            certify(z)

    def optimize(z, maxiter):
        check()
        result = minimize(objective, z, jac=objective_jac, method='SLSQP',
                          bounds=bounds,
                          constraints={'type': 'ineq', 'fun': lambda q: geometry(q)[0],
                                       'jac': lambda q: geometry(q)[1]},
                          callback=callback,
                          options={'maxiter': maxiter, 'ftol': 2e-13, 'disp': False})
        check()
        certify(result.x)
        return contact_polish(result.x)

    try:
        start = encode(best['circles'])
        certify(start)
        start = contact_polish(start)
        start = optimize(start, 35)
        anchor = encode(best['circles'])
        anchor = contact_polish(anchor)
        g, j = geometry(anchor)
        active = np.flatnonzero(g < 2e-6)
        a = j[active].copy()
        weights = lstsq(a.T, grad, cond=1e-11, check_finite=False, lapack_driver='gelsy')[0]
        weights = np.maximum(np.abs(weights), 1e-5)
        probabilities = 1.0 / np.sqrt(weights)
        probabilities /= probabilities.sum()
        for trial in range(12):
            check()
            if time.monotonic() - started > 0.63 * budget_s:
                break
            pivot = int(rng.choice(len(active), p=probabilities))
            row = int(active[pivot])
            if row < m:
                touched = {int(ii[row]), int(jj[row])}
            else:
                touched = {(row-m) % n}
            local = []
            for k, constraint in enumerate(active):
                check()
                if constraint < m:
                    if int(ii[constraint]) in touched or int(jj[constraint]) in touched:
                        local.append(k)
                elif (int(constraint)-m) % n in touched:
                    local.append(k)
            count = min(int(params['contacts']), len(active))
            if len(local) < count or trial % 3 == 2:
                chosen = rng.choice(len(active), size=count, replace=False, p=probabilities)
            else:
                chosen = rng.choice(np.asarray(local), size=count, replace=False)
            opening = np.zeros(len(active))
            opening[chosen] = rng.uniform(0.5, 1.5, count)
            direction = lstsq(a, opening, cond=1e-11, check_finite=False, lapack_driver='gelsy')[0]
            norm = np.max(np.abs(direction[:2*n]))
            if not np.isfinite(norm) or norm < 1e-12:
                continue
            amplitude = params['opening'] * (0.7 + 0.5*(trial % 3))
            trial_z = anchor + direction * (amplitude/norm)
            trial_z[2*n:3*n] = np.clip(trial_z[2*n:3*n], 0.025, 0.25)
            trial_z[-1] = np.clip(trial_z[-1], 0.5, 1.5)
            optimize(trial_z, int(params['iterations']))
    except Stop:
        pass
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        pass
    return best
