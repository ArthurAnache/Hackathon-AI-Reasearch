import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'trajectory_fraction': (0.77, 0.5, 0.9),
    'stage_iterations': (28, 12, 65),
    'contact_margin': (0.075, 0.025, 0.2),
    'restart_noise': (0.016, 0.002, 0.06, 'log'),
}


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.69 * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    best = init
    best_score = -float('inf')
    best_a = None

    def score(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        delta = a[ii, :2] - a[jj, :2]
        if np.any(np.sqrt(delta[:, 0]**2 + delta[:, 1]**2) < a[ii, 2] + a[jj, 2]):
            return -float('inf')
        x = list(map(Fraction, map(float, a[:, 0])))
        y = list(map(Fraction, map(float, a[:, 1])))
        r = list(map(Fraction, map(float, a[:, 2])))
        w = max(u+v for u, v in zip(x, r)) - min(u-v for u, v in zip(x, r))
        h = max(u+v for u, v in zip(y, r)) - min(u-v for u, v in zip(y, r))
        if w+h > 2:
            return -float('inf')
        return float(np.sum(a[:, 2]))

    def offer(a):
        nonlocal best, best_score, best_a
        if time.monotonic() >= deadline:
            return
        a = np.asarray(a, dtype=float).copy()
        if a.shape != (21, 3) or not np.isfinite(a).all():
            return
        if np.any(a[:, 2] <= 0):
            return
        d = a[ii, :2] - a[jj, :2]
        distances = np.sqrt(d[:, 0]**2+d[:, 1]**2)
        shrink = min(1.0, float(np.min(distances/(a[ii, 2]+a[jj, 2]))))
        if shrink <= 0:
            return
        a[:, 2] *= shrink * (1-2e-12)
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        perimeter = float(np.sum(hi-lo))
        if perimeter <= 0:
            return
        a[:, :2] -= lo
        a *= (2/perimeter)*(1-2e-12)
        if float(np.sum(a[:, 2])) <= best_score:
            return
        s = score(a)
        if s > best_score:
            best_score = s
            best_a = a.copy()
            best = {'circles': a.tolist()}

    a0 = None
    if init is not None:
        try:
            a0 = np.asarray(init['circles'], dtype=float).copy()
            s = score(a0)
            if np.isfinite(s):
                best_score = s
                best_a = a0.copy()
            if a0.shape != (21, 3) or not np.isfinite(a0).all():
                a0 = None
        except (ValueError, TypeError, KeyError, OverflowError):
            a0 = None
    grid = np.array([[0.1+0.2*x, 0.1+0.2*y, 0.1] for y in range(5) for x in range(5)
                     if (x, y) not in ((0, 0), (4, 0), (0, 4), (4, 4))])
    offer(grid)
    if a0 is None:
        a0 = grid.copy()
        a0[:, :2] += rng.normal(0, 0.012, (21, 2))
    else:
        offer(a0)

    k = np.arange(n)
    B = np.zeros((84, 64))
    B[k, k] = 1
    B[k, 42+k] = -1
    B[21+k, k] = -1
    B[21+k, 42+k] = -1
    B[21+k, 63] = 1
    B[42+k, 21+k] = 1
    B[42+k, 42+k] = -1
    B[63+k, 21+k] = -1
    B[63+k, 42+k] = -1
    B[63+k, 63] = -1
    bc = np.zeros(84)
    bc[63:] = 2
    grad = np.zeros(64)
    grad[42:63] = -1
    bounds = [(None, None)]*42 + [(1e-8, 0.7)]*21 + [(0.12, 1.88)]

    class Stop(Exception):
        pass

    def unpack(z):
        return np.column_stack((z[:21], z[21:42], z[42:63]))

    def pack(a):
        a = a.copy()
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        dim = hi-lo
        scale = 2/max(1e-12, float(np.sum(dim)))
        a[:, :2] = (a[:, :2]-lo)*scale
        a[:, 2] = np.clip(a[:, 2]*scale, 1e-7, 0.65)
        return np.r_[a[:, 0], a[:, 1], a[:, 2], np.clip(dim[0]*scale, 0.12, 1.88)]

    def trajectory(a, until):
        z = pack(a)
        active = np.zeros(len(ii), dtype=bool)
        stage = 0
        while time.monotonic() < until-0.012:
            dx = z[ii]-z[jj]
            dy = z[21+ii]-z[21+jj]
            gaps = np.sqrt(dx*dx+dy*dy)-z[42+ii]-z[42+jj]
            active |= gaps < float(params['contact_margin'])
            # Each circle also retains its six closest neighbours.
            distances = np.full((n, n), np.inf)
            distances[ii, jj] = gaps
            distances[jj, ii] = gaps
            nearest = np.argpartition(distances, 5, axis=1)[:, :6]
            mask = np.zeros((n, n), dtype=bool)
            mask[np.arange(n)[:, None], nearest] = True
            active |= mask[ii, jj] | mask[jj, ii]
            if stage >= 3:
                active[:] = True
            ai, aj = ii[active], jj[active]
            rr = np.arange(len(ai))
            last = z.copy()
            counter = 0

            def check():
                if time.monotonic() >= until:
                    raise Stop()

            def fun(v):
                check()
                return -float(np.sum(v[42:63]))

            def con(v):
                check()
                dx = v[ai]-v[aj]
                dy = v[21+ai]-v[21+aj]
                return np.r_[B@v+bc, np.sqrt(dx*dx+dy*dy)-v[42+ai]-v[42+aj]]

            def jac(v):
                check()
                dx = v[ai]-v[aj]
                dy = v[21+ai]-v[21+aj]
                inv = 1/np.maximum(np.sqrt(dx*dx+dy*dy), 1e-15)
                J = np.zeros((len(ai), 64))
                J[rr, ai] = dx*inv
                J[rr, aj] = -dx*inv
                J[rr, 21+ai] = dy*inv
                J[rr, 21+aj] = -dy*inv
                J[rr, 42+ai] = -1
                J[rr, 42+aj] = -1
                return np.vstack((B, J))

            def callback(v):
                nonlocal last, counter
                check()
                last = v.copy()
                counter += 1
                if counter % 3 == 0:
                    offer(unpack(v))

            try:
                result = minimize(fun, z, jac=lambda v: grad, method='SLSQP', bounds=bounds,
                                  constraints={'type': 'ineq', 'fun': con, 'jac': jac},
                                  callback=callback,
                                  options={'maxiter': int(params['stage_iterations']), 'ftol': 1e-11, 'disp': False})
                z = result.x
                offer(unpack(z))
                if result.success and stage >= 3:
                    break
            except Stop:
                offer(unpack(last))
                break
            except (ValueError, FloatingPointError):
                break
            stage += 1
        return z

    first_end = min(deadline, start + 0.69*budget_s*float(params['trajectory_fraction']))
    trajectory(a0, first_end)
    attempt = 0
    while time.monotonic() < deadline-0.045:
        if best_a is None:
            trial = grid.copy()
        elif best_score < 2.29 and attempt == 0:
            trial = grid.copy()
            trial[:, :2] += rng.normal(0, 0.009, (21, 2))
        else:
            trial = best_a.copy()
            trial[:, :2] += rng.normal(0, float(params['restart_noise'])*(1+attempt%3), (21, 2))
            trial[:, 2] *= rng.uniform(0.97, 1.015, 21)
        trajectory(trial, min(deadline, time.monotonic()+0.31*budget_s))
        attempt += 1
    return best
