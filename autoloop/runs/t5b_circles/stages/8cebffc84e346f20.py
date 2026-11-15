import time
import numpy as np
from scipy.optimize import minimize


def refine(v, score, budget_s, rng):
    start = time.monotonic()
    deadline = start + max(0.0, float(budget_s) - max(0.025, 0.035 * float(budget_s)))
    best = np.asarray(v, dtype=float).reshape(-1).copy()
    if deadline - start < 0.025:
        return best
    n = 21
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    rows = np.arange(m)
    best_sum = float(best.reshape(n, 3)[:, 2].sum())
    best_score = -best_sum

    class Finished(Exception):
        pass

    def check_time():
        if time.monotonic() >= deadline:
            raise Finished()

    def encode(c):
        c = np.asarray(c).reshape(n, 3).copy()
        lo = np.min(c[:, :2] - c[:, 2, None], axis=0)
        hi = np.max(c[:, :2] + c[:, 2, None], axis=0)
        size = hi - lo
        spare = max(0.0, 2.0 - size.sum())
        c[:, :2] -= lo
        c[:, :2] += spare / 4.0
        return np.r_[c.ravel(), size[0] + spare / 2.0]

    def consider(z):
        nonlocal best, best_sum, best_score
        if time.monotonic() >= deadline:
            return
        c = np.asarray(z[:63]).reshape(n, 3).copy()
        if not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return
        # Repair small contact errors, then use the entire perimeter allowance.
        d = c[ii, :2] - c[jj, :2]
        dist = np.sqrt(np.einsum('ij,ij->i', d, d))
        ratio = np.min(dist / (c[ii, 2] + c[jj, 2]))
        if ratio <= 0:
            return
        c[:, 2] *= min(1.0, ratio) * (1.0 - 3e-14)
        lo = np.min(c[:, :2] - c[:, 2, None], axis=0)
        hi = np.max(c[:, :2] + c[:, 2, None], axis=0)
        perimeter = float(np.sum(hi - lo))
        if not perimeter > 0:
            return
        c[:, :2] -= (lo + hi) / 2.0
        c *= (2.0 / perimeter) * (1.0 - 4e-14)
        value = float(c[:, 2].sum())
        if value <= best_sum + 2e-14:
            return
        candidate = c.ravel().copy()
        actual = score(candidate)
        if np.isfinite(actual) and actual < best_score:
            best = candidate
            best_sum = value
            best_score = actual

    # Coordinates and radii are interleaved; the last variable is width.
    # Height is 2-width, making the perimeter constraint exact in the model.
    border_jac = np.zeros((4 * n, 64))
    k = np.arange(n)
    border_jac[k, 3*k] = 1
    border_jac[k, 3*k+2] = -1
    border_jac[n+k, 3*k] = -1
    border_jac[n+k, 3*k+2] = -1
    border_jac[n+k, 63] = 1
    border_jac[2*n+k, 3*k+1] = 1
    border_jac[2*n+k, 3*k+2] = -1
    border_jac[3*n+k, 3*k+1] = -1
    border_jac[3*n+k, 3*k+2] = -1
    border_jac[3*n+k, 63] = -1
    base_jac = np.zeros((m + 4*n, 64))
    base_jac[m:] = border_jac
    base_jac[rows, 3*ii+2] = -1
    base_jac[rows, 3*jj+2] = -1
    objective_jac = np.zeros(64)
    objective_jac[2:63:3] = -1
    cache_z = None
    cache_fun = None
    cache_jac = None

    def constraints(z):
        nonlocal cache_z, cache_fun, cache_jac
        check_time()
        if cache_z is not None and np.array_equal(z, cache_z):
            return cache_fun
        c = z[:63].reshape(n, 3)
        x, y, r = c.T
        delta = c[ii, :2] - c[jj, :2]
        distance = np.sqrt(np.einsum('ij,ij->i', delta, delta))
        directions = delta / np.maximum(distance[:, None], 1e-100)
        values = np.concatenate((distance-r[ii]-r[jj], x-r,
                                 z[63]-x-r, y-r, 2-z[63]-y-r))
        jac = base_jac.copy()
        jac[rows, 3*ii] = directions[:, 0]
        jac[rows, 3*ii+1] = directions[:, 1]
        jac[rows, 3*jj] = -directions[:, 0]
        jac[rows, 3*jj+1] = -directions[:, 1]
        cache_z, cache_fun, cache_jac = z.copy(), values, jac
        return values

    def jacobian(z):
        constraints(z)
        return cache_jac

    def objective(z):
        check_time()
        return -np.sum(z[2:63:3])

    def callback(z):
        check_time()
        consider(z)

    bounds = [(None, None)] * 64
    for t in range(n):
        bounds[3*t+2] = (1e-12, 0.5)
    bounds[63] = (1e-9, 2.0-1e-9)
    attempt = 0
    try:
        while time.monotonic() < deadline:
            z = encode(best)
            if attempt:
                c = z[:63].reshape(n, 3)
                amplitude = (0.00025, 0.001, 0.004, 0.012, 0.025)[(attempt-1) % 5]
                c[:, :2] += rng.normal(0.0, amplitude, (n, 2))
                c[:, 2] *= np.exp(rng.normal(0.0, amplitude * 2.0, n))
                c[:, 2] = np.clip(c[:, 2], 1e-12, 0.499999)
                if attempt % 7 == 0:
                    small = np.argsort(c[:, 2])[:5]
                    t = int(rng.choice(small))
                    c[t, 0] = rng.uniform(c[t, 2], max(c[t, 2], z[63]-c[t, 2]))
                    c[t, 1] = rng.uniform(c[t, 2], max(c[t, 2], 2-z[63]-c[t, 2]))
            result = minimize(objective, z, jac=lambda z: objective_jac,
                              method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                              callback=callback,
                              options={'ftol': 2e-12, 'maxiter': 160, 'disp': False})
            consider(result.x)
            attempt += 1
    except Finished:
        pass
    return best
