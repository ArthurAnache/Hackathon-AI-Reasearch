import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'removed': (3, 2, 5),
    'attempt_seconds': (0.32, 0.12, 0.7),
    'grid': (25, 12, 40),
    'iterations': (95, 30, 200),
    'novelty': (0.045, 0.0, 0.15)
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class StopSearch(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    dx = a[I, 0] - a[J, 0]
    dy = a[I, 1] - a[J, 1]
    if np.any(np.sqrt(dx * dx + dy * dy) < a[I, 2] + a[J, 2]):
        return -np.inf
    x = list(map(Fraction, map(float, a[:, 0])))
    y = list(map(Fraction, map(float, a[:, 1])))
    r = list(map(Fraction, map(float, a[:, 2])))
    perimeter = (max(t + q for t, q in zip(x, r)) - min(t - q for t, q in zip(x, r))
                 + max(t + q for t, q in zip(y, r)) - min(t - q for t, q in zip(y, r)))
    if perimeter > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))

def repair(z, deadline):
    if time.monotonic() >= deadline:
        return None
    a = np.column_stack((z[:N], z[N:2*N], z[2*N:3*N])).copy()
    if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return None
    d = np.sqrt((a[I, 0] - a[J, 0])**2 + (a[I, 1] - a[J, 1])**2)
    factor = min(1.0, float(np.min(d / (a[I, 2] + a[J, 2]))))
    if factor <= 0:
        return None
    a[:, 2] *= factor * (1 - 8e-15)
    # Translation keeps coordinates small; uniform scaling preserves all contacts.
    lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
    a[:, :2] -= lo
    sides = np.max(a[:, :2] + a[:, 2, None], axis=0) - np.min(a[:, :2] - a[:, 2, None], axis=0)
    a *= min(1.0, 2.0 / float(sides.sum())) * (1 - 8e-15)
    for k in range(3):
        if time.monotonic() >= deadline:
            return None
        if np.isfinite(exact_score(a)):
            return a
        a[:, 2] *= 1 - 1e-13
        a *= 1 - 1e-13
    return None

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a = np.zeros((N, 3))
        a[:, 0] = (np.arange(N) % 5 + 0.5) / 5
        a[:, 1] = (np.arange(N) // 5 + 0.5) / 5
        a[:, 2] = 0.099
        best = {'circles': a.tolist()}
    else:
        a = np.asarray(init['circles'], dtype=float).copy()
        best = init
    best_score = exact_score(a)
    lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
    hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
    scale = 2.0 / float(np.sum(hi - lo))
    base = a.copy()
    base[:, :2] = (base[:, :2] - lo) * scale
    base[:, 2] *= scale
    w0 = float((hi[0] - lo[0]) * scale)
    objective_gradient = np.zeros(64)
    objective_gradient[42:63] = -1
    rows = np.arange(M)
    boundary_jac = np.zeros((4*N, 64))
    ids = np.arange(N)
    boundary_jac[ids, ids] = 1
    boundary_jac[ids, 42 + ids] = -1
    boundary_jac[N + ids, ids] = -1
    boundary_jac[N + ids, 42 + ids] = -1
    boundary_jac[N + ids, 63] = 1
    boundary_jac[2*N + ids, N + ids] = 1
    boundary_jac[2*N + ids, 42 + ids] = -1
    boundary_jac[3*N + ids, N + ids] = -1
    boundary_jac[3*N + ids, 42 + ids] = -1
    boundary_jac[3*N + ids, 63] = -1
    attempt = 0
    while time.monotonic() < deadline - 0.045:
        attempt_end = min(deadline - 0.012, time.monotonic() + float(params['attempt_seconds']))
        trial = base.copy()
        w = w0
        if attempt > 0 or init is not None:
            # Remove a cluster around one of the smaller circles, not independent contacts.
            order = np.argsort(base[:, 2])
            anchor = int(order[(attempt * 3 + seed % 7) % 12])
            dist = np.linalg.norm(base[:, :2] - base[anchor, :2], axis=1)
            ranking = dist + 0.7 * base[:, 2]
            ranking[anchor] = -1
            removed = np.argsort(ranking)[:int(params['removed'])]
            kept = [int(i) for i in np.setdiff1d(np.arange(N), removed)]
            g = int(params['grid'])
            gx, gy = np.meshgrid(np.linspace(0.015, w - 0.015, g),
                                 np.linspace(0.015, 2 - w - 0.015, g))
            pool = np.column_stack((gx.ravel(), gy.ravel()))
            pool += rng.uniform(-0.008, 0.008, pool.shape)
            pool = np.vstack((pool, base[removed, :2]))
            wall = np.minimum.reduce((pool[:, 0], w - pool[:, 0], pool[:, 1], 2 - w - pool[:, 1]))
            for step, idx in enumerate(removed):
                if time.monotonic() >= attempt_end:
                    break
                pp = trial[kept, :2]
                rr = trial[kept, 2]
                clearance = np.minimum(wall, np.min(np.linalg.norm(pool[:, None, :] - pp[None, :, :], axis=2) - rr, axis=1))
                # The first insertion explicitly competes with the deleted cluster's old sites.
                quality = clearance.copy()
                if step == 0:
                    olddist = np.min(np.linalg.norm(pool[:, None, :] - base[removed, :2][None, :, :], axis=2), axis=1)
                    quality -= float(params['novelty']) * np.exp(-(olddist / 0.085)**2)
                quality += rng.uniform(0, 0.003, len(quality))
                chosen = int(np.argmax(quality))
                trial[idx, :2] = pool[chosen]
                trial[idx, 2] = max(1e-5, 0.99 * clearance[chosen])
                kept.append(int(idx))
        z = np.r_[trial[:, 0], trial[:, 1], trial[:, 2], w]
        last = z.copy()
        cached_z = None
        cached_c = None
        cached_j = None
        def check():
            if time.monotonic() >= attempt_end:
                raise StopSearch()
        def objective(v):
            check()
            return -float(np.sum(v[42:63])), objective_gradient
        def calculate(v):
            nonlocal cached_z, cached_c, cached_j
            check()
            if cached_z is not None and np.array_equal(v, cached_z):
                return
            x, y, r, width = v[:21], v[21:42], v[42:63], v[63]
            dx, dy = x[I] - x[J], y[I] - y[J]
            d = np.maximum(np.sqrt(dx*dx + dy*dy), 1e-18)
            cached_c = np.r_[d - r[I] - r[J], x-r, width-x-r, y-r, 2-width-y-r]
            jac = np.zeros((M + 84, 64))
            jac[rows, I] = dx/d
            jac[rows, J] = -dx/d
            jac[rows, 21+I] = dy/d
            jac[rows, 21+J] = -dy/d
            jac[rows, 42+I] = -1
            jac[rows, 42+J] = -1
            jac[M:] = boundary_jac
            cached_j = jac
            cached_z = v.copy()
        def constraint(v):
            calculate(v)
            return cached_c
        def jacobian(v):
            calculate(v)
            return cached_j
        def callback(v):
            nonlocal last
            check()
            last = v.copy()
        try:
            result = minimize(objective, z, jac=True, method='SLSQP',
                              bounds=[(0, 2)]*42 + [(1e-7, 0.5)]*21 + [(0.35, 1.65)],
                              constraints={'type': 'ineq', 'fun': constraint, 'jac': jacobian},
                              callback=callback,
                              options={'maxiter': int(params['iterations']), 'ftol': 2e-12, 'disp': False})
            last = result.x
        except StopSearch:
            pass
        candidate = repair(last, deadline)
        if candidate is not None:
            score = exact_score(candidate)
            if score > best_score:
                best_score = score
                best = {'circles': candidate.tolist()}
                base = candidate.copy()
                lower = np.min(base[:, :2] - base[:, 2, None], axis=0)
                upper = np.max(base[:, :2] + base[:, 2, None], axis=0)
                factor = 2 / float(np.sum(upper-lower))
                base[:, :2] = (base[:, :2]-lower)*factor
                base[:, 2] *= factor
                w0 = float((upper[0]-lower[0])*factor)
        attempt += 1
    return best
