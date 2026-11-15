import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'ftol': (2e-13, 1e-15, 1e-9, 'log'),
    'perturbation': (0.012, 0.0001, 0.08, 'log'),
    'maxiter': (100, 20, 250),
    'time_fraction': (0.69, 0.4, 0.72),
}
N = 21
II, JJ = np.triu_indices(N, 1)
M = len(II)
AR = np.arange(N)

class StopSearch(Exception):
    pass


def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[II, 0] - a[JJ, 0]) ** 2 + (a[II, 1] - a[JJ, 1]) ** 2)
    if np.any(d < a[II, 2] + a[JJ, 2]):
        return -np.inf
    xs = [Fraction(float(v)) for v in a[:, 0]]
    ys = [Fraction(float(v)) for v in a[:, 1]]
    rs = [Fraction(float(v)) for v in a[:, 2]]
    perimeter = (max(x+r for x, r in zip(xs, rs)) - min(x-r for x, r in zip(xs, rs))
                 + max(y+r for y, r in zip(ys, rs)) - min(y-r for y, r in zip(ys, rs)))
    if perimeter > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))


def repair(z, deadline):
    a = np.column_stack((z[:N], z[N:2*N], z[2*N:3*N])).copy()
    if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return None
    d = np.sqrt((a[II, 0] - a[JJ, 0])**2 + (a[II, 1] - a[JJ, 1])**2)
    factor = min(1.0, float(np.min(d / (a[II, 2] + a[JJ, 2]))))
    a[:, 2] *= factor * (1.0 - 4e-15)
    lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
    hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
    scale = (2.0 / np.sum(hi-lo)) * (1.0 - 4e-15)
    a[:, :2] = (a[:, :2] - lo) * scale
    a[:, 2] *= scale
    for k in range(4):
        if time.monotonic() >= deadline:
            break
        if np.isfinite(exact_score(a)):
            return a
        a *= (1.0 - 2e-14)
        a[:, 2] *= (1.0 - 2e-14)
    return a if np.isfinite(exact_score(a)) else None


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + float(params['time_fraction']) * budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a = np.array([[(i % 7 + 0.5)*0.2, (i//7 + 0.5)*0.2, 0.1-1e-14] for i in range(N)])
        best = {'circles': a.tolist()}
    else:
        best = init
        a = np.asarray(init['circles'], dtype=float).copy()
    best_score = exact_score(a)
    lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
    hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
    scale = 2.0 / np.sum(hi-lo)
    a[:, :2] = (a[:, :2]-lo)*scale
    a[:, 2] *= scale
    zbase = np.r_[a[:, 0], a[:, 1], a[:, 2], (hi[0]-lo[0])*scale]
    objective_gradient = np.r_[np.zeros(2*N), -np.ones(N), 0.0]
    bounds = [(None, None)]*(2*N) + [(1e-7, None)]*N + [(0.05, 1.95)]
    latest = zbase.copy()
    local_end = deadline

    def check():
        if time.monotonic() >= local_end:
            raise StopSearch()

    def objective(z):
        nonlocal latest
        check()
        latest = z.copy()
        return -float(np.sum(z[2*N:3*N]))

    def jac_objective(z):
        check()
        return objective_gradient

    def constraints(z):
        check()
        x, y, r, w = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
        d = np.sqrt((x[II]-x[JJ])**2 + (y[II]-y[JJ])**2 + 1e-300)
        return np.r_[d-r[II]-r[JJ], x-r, w-x-r, y-r, 2-w-y-r]

    def jac_constraints(z):
        check()
        x, y = z[:N], z[N:2*N]
        dx, dy = x[II]-x[JJ], y[II]-y[JJ]
        d = np.sqrt(dx*dx + dy*dy + 1e-300)
        j = np.zeros((M+4*N, 3*N+1))
        rows = np.arange(M)
        j[rows, II] = dx/d
        j[rows, JJ] = -dx/d
        j[rows, N+II] = dy/d
        j[rows, N+JJ] = -dy/d
        j[rows, 2*N+II] = -1
        j[rows, 2*N+JJ] = -1
        rows = M+AR
        j[rows, AR] = 1
        j[rows, 2*N+AR] = -1
        rows = M+N+AR
        j[rows, AR] = -1
        j[rows, 2*N+AR] = -1
        j[rows, -1] = 1
        rows = M+2*N+AR
        j[rows, N+AR] = 1
        j[rows, 2*N+AR] = -1
        rows = M+3*N+AR
        j[rows, N+AR] = -1
        j[rows, 2*N+AR] = -1
        j[rows, -1] = -1
        return j

    # Find the more accurately reflected axis, without assuming circle ordering.
    p = a[:, :2]
    symmetry = []
    for axis in range(2):
        if time.monotonic() >= deadline:
            return best
        reflected = p.copy()
        reflected[:, axis] = np.min(p[:, axis]-a[:, 2]) + np.max(p[:, axis]+a[:, 2]) - reflected[:, axis]
        costs = np.sum((reflected[:, None, :]-p[None, :, :])**2, axis=2)
        partner = np.argmin(costs, axis=1)
        symmetry.append((float(np.sum(np.min(costs, axis=1))), axis, partner))
    _, axis, partner = min(symmetry, key=lambda v: v[0])

    attempt = 0
    while time.monotonic() < deadline - 0.025:
        z = zbase.copy()
        if attempt:
            amp = float(params['perturbation']) * (0.35, 1.0, 2.5, 0.1)[(attempt-1) % 4]
            noise = rng.normal(size=(N, 3))
            reflected_noise = noise[partner].copy()
            reflected_noise[:, axis] *= -1
            noise = 0.5*(noise+reflected_noise)
            z[:N] += amp*noise[:, 0]
            z[N:2*N] += amp*noise[:, 1]
            z[2*N:3*N] = np.maximum(1e-5, z[2*N:3*N] + 0.25*amp*noise[:, 2])
        latest = z.copy()
        remaining = deadline-time.monotonic()
        local_end = min(deadline-0.012, time.monotonic() + (min(0.55*budget_s, remaining-0.015) if attempt == 0 else min(0.22*budget_s, remaining-0.015)))
        try:
            result = minimize(objective, z, jac=jac_objective, bounds=bounds,
                              constraints={'type': 'ineq', 'fun': constraints, 'jac': jac_constraints},
                              method='SLSQP', options={'ftol': float(params['ftol']), 'maxiter': int(params['maxiter']), 'disp': False})
            latest = result.x
        except StopSearch:
            pass
        if time.monotonic() >= deadline-0.006:
            break
        candidate = repair(latest, deadline)
        if candidate is not None:
            score = exact_score(candidate)
            if score > best_score:
                best_score = score
                best = {'circles': candidate.tolist()}
                lo2 = np.min(candidate[:, :2]-candidate[:, 2, None], axis=0)
                hi2 = np.max(candidate[:, :2]+candidate[:, 2, None], axis=0)
                zbase = np.r_[candidate[:, 0]-lo2[0], candidate[:, 1]-lo2[1], candidate[:, 2], hi2[0]-lo2[0]]
        attempt += 1
    return best
