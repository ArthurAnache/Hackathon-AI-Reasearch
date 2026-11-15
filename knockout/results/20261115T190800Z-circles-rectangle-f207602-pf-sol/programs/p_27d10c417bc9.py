import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'iterations': (125, 30, 250),
    'time_fraction': (0.69, 0.4, 0.7),
    'perturbation': (0.007, 0.001, 0.05, 'log')
}


def exact_score(data):
    try:
        a = np.asarray(data['circles'], dtype=np.float64)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
        perimeter = (max(xi+ri for xi, ri in zip(x, r)) - min(xi-ri for xi, ri in zip(x, r))
                     + max(yi+ri for yi, ri in zip(y, r)) - min(yi-ri for yi, ri in zip(y, r)))
        if perimeter > 2:
            return -float('inf')
        for i in range(21):
            for j in range(i):
                if np.sqrt((a[i, 0]-a[j, 0])**2 + (a[i, 1]-a[j, 1])**2) < a[i, 2]+a[j, 2]:
                    return -float('inf')
        return float(np.sum(a[:, 2]))
    except Exception:
        return -float('inf')


class StopSearch(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + float(params['time_fraction'])*budget_s
    local_deadline = deadline
    rng = np.random.default_rng(seed)
    best = init
    best_score = exact_score(init)
    ii, jj = np.triu_indices(21, 1)
    rows = np.arange(21)
    pairrows = np.arange(210)+84
    gradient = np.zeros(64)
    gradient[2:63:3] = -1
    candidates = []
    leading = None
    leading_score = best_score

    def check():
        if time.monotonic() >= min(deadline, local_deadline):
            raise StopSearch()

    def repair(a):
        a = np.asarray(a, dtype=float).reshape(21, 3).copy()
        if not np.isfinite(a).all():
            return None
        a[:, 2] = np.maximum(a[:, 2], 1e-9)
        delta = a[ii, :2]-a[jj, :2]
        d = np.sqrt(np.sum(delta*delta, axis=1))
        factor = min(1.0, float(np.min(d/(a[ii, 2]+a[jj, 2]))))
        if factor <= 0:
            return None
        a[:, 2] *= factor*(1-3e-13)
        a[:, :2] -= np.min(a[:, :2]-a[:, 2, None], axis=0)
        extent = np.max(a[:, :2]+a[:, 2, None], axis=0)
        a *= (2-2e-12)/float(np.sum(extent))
        return a

    def consider(z):
        nonlocal leading, leading_score
        a = repair(np.asarray(z)[:63])
        if a is None:
            return
        score = float(np.sum(a[:, 2]))
        if score > leading_score:
            leading_score = score
            leading = a.copy()
            candidates.append((score, a))
            if len(candidates) > 8:
                candidates.pop(0)

    base_jac = np.zeros((294, 64))
    base_jac[rows, 3*rows] = 1
    base_jac[rows, 3*rows+2] = -1
    base_jac[21+rows, 3*rows] = -1
    base_jac[21+rows, 3*rows+2] = -1
    base_jac[21+rows, 63] = 1
    base_jac[42+rows, 3*rows+1] = 1
    base_jac[42+rows, 3*rows+2] = -1
    base_jac[63+rows, 3*rows+1] = -1
    base_jac[63+rows, 3*rows+2] = -1
    base_jac[63+rows, 63] = -1
    base_jac[pairrows, 3*ii+2] = -1
    base_jac[pairrows, 3*jj+2] = -1

    def constraints(z):
        check()
        x, y, r = z[:63].reshape(21, 3).T
        d = np.hypot(x[ii]-x[jj], y[ii]-y[jj])
        return np.concatenate((x-r, z[63]-x-r, y-r, 2-z[63]-y-r, d-r[ii]-r[jj]))

    def jacobian(z):
        check()
        a = z[:63].reshape(21, 3)
        dx = a[ii, 0]-a[jj, 0]
        dy = a[ii, 1]-a[jj, 1]
        d = np.maximum(np.hypot(dx, dy), 1e-25)
        out = base_jac.copy()
        out[pairrows, 3*ii] = dx/d
        out[pairrows, 3*jj] = -dx/d
        out[pairrows, 3*ii+1] = dy/d
        out[pairrows, 3*jj+1] = -dy/d
        return out

    def callback(z):
        consider(z)
        check()

    def optimize(a):
        a = repair(a)
        if a is None:
            return
        z = np.r_[a.ravel(), np.max(a[:, 0]+a[:, 2])]
        consider(z)
        try:
            result = minimize(lambda q: float(-np.sum(q[2:63:3])), z,
                              jac=lambda q: gradient, method='SLSQP',
                              bounds=[(0, 2), (0, 2), (1e-8, 0.7)]*21+[(0.1, 1.9)],
                              constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                              callback=callback,
                              options={'maxiter': int(params['iterations']), 'ftol': 8e-13, 'disp': False})
            consider(result.x)
        except (StopSearch, ValueError, FloatingPointError):
            pass

    original = None
    try:
        trial = np.asarray(init['circles'], dtype=float)
        if trial.shape == (21, 3) and np.isfinite(trial).all():
            original = repair(trial)
    except Exception:
        pass

    if original is not None:
        local_deadline = min(deadline, started+0.43*budget_s)
        optimize(original)

    attempt = 0
    while time.monotonic() < deadline-0.035:
        local_deadline = deadline
        if attempt == 1 or (original is None and attempt == 0):
            # Three circles on the horizontal axis, nine reflected pairs.
            counts = [4, 5, 3, 5, 4]
            coords = []
            for row, count in enumerate(counts):
                if time.monotonic() >= deadline:
                    break
                for k in range(count):
                    coords.append([2*k-(count-1), (row-2)*1.76, 1.0])
            if len(coords) != 21:
                break
            a = np.asarray(coords, dtype=float)
            a[:, 2] *= rng.uniform(0.94, 1.04, 21)
        else:
            a = (leading if leading is not None else original)
            if a is None:
                break
            a = a.copy()
            amount = float(params['perturbation'])*(1.0+0.6*(attempt % 3))
            # A smooth shear moves several contacts together rather than
            # independently displacing a single jammed circle.
            centered = a[:, :2]-np.mean(a[:, :2], axis=0)
            shear = rng.normal(0, amount*1.5, (2, 2))
            a[:, :2] += centered @ shear
            a[:, :2] += rng.normal(0, amount*0.35, (21, 2))
            a[:, 2] *= 1+rng.normal(0, amount*0.4, 21)
        optimize(a)
        attempt += 1

    for score, a in reversed(candidates):
        if time.monotonic() >= started+0.91*budget_s:
            break
        candidate = {'circles': a.tolist()}
        verified = exact_score(candidate)
        if verified > best_score:
            best, best_score = candidate, verified
    if best_score == -float('inf'):
        r = 0.099999999999
        fallback = {'circles': [[(i % 7)*0.2+0.1, (i//7)*0.2+0.1, r] for i in range(21)]}
        if exact_score(fallback) > best_score:
            best = fallback
    return best
