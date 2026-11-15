import time
import numpy as np
from fractions import Fraction
from scipy.optimize import minimize
from scipy.linalg import qr

PARAMS = {"precision": (1e-13, 1e-15, 1e-10, "log"), "perturbation": (0.002, 0.00001, 0.05, "log")}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    rows = np.arange(len(ii))
    best = init
    best_score = -float('inf')

    def exact_score(obj):
        if time.monotonic() >= deadline:
            return -float('inf')
        try:
            c = np.asarray(obj['circles'], dtype=float)
            if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
                return -float('inf')
            dx = c[ii, 0] - c[jj, 0]
            dy = c[ii, 1] - c[jj, 1]
            if np.any(np.sqrt(dx ** 2 + dy ** 2) < c[ii, 2] + c[jj, 2]):
                return -float('inf')
            xx, yy, rr = [], [], []
            for x, y, r in c:
                if time.monotonic() >= deadline:
                    return -float('inf')
                xx.append(Fraction(float(x)))
                yy.append(Fraction(float(y)))
                rr.append(Fraction(float(r)))
            width = max(x+r for x, r in zip(xx, rr)) - min(x-r for x, r in zip(xx, rr))
            height = max(y+r for y, r in zip(yy, rr)) - min(y-r for y, r in zip(yy, rr))
            if width + height > 2:
                return -float('inf')
            return float(np.sum(c[:, 2]))
        except (ValueError, KeyError, TypeError):
            return -float('inf')

    if init is not None:
        if 'circles' in init:
            original = init
        else:
            cc = []
            for x, y, r, m in init['orbits']:
                if time.monotonic() >= deadline:
                    return init
                cc.append([x, y, r])
                if m == 2:
                    cc.append([x, -y, r])
            original = {'circles': cc}
        best_score = exact_score(original)
        c0 = np.array(original['circles'], dtype=float)
    else:
        c0 = np.array([[0.1 + .2*x, 0.1 + .2*y, .1*(1-1e-12)] for x in range(7) for y in range(3)])

    def consider(c):
        nonlocal best, best_score
        if time.monotonic() >= deadline or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return
        c = c.copy()
        d = np.sqrt((c[ii, 0]-c[jj, 0])**2 + (c[ii, 1]-c[jj, 1])**2)
        factor = min(1.0, float(np.min(d/(c[ii, 2]+c[jj, 2]))))
        c[:, 2] *= factor * (1.0 - 3e-15)
        lo = np.min(c[:, :2]-c[:, 2, None], axis=0)
        c[:, :2] -= lo
        wh = np.max(c[:, :2]+c[:, 2, None], axis=0) - np.min(c[:, :2]-c[:, 2, None], axis=0)
        scale = min(1.0, 2.0/float(np.sum(wh))) * (1.0-3e-15)
        c *= scale
        c[:, 2] *= 1.0-3e-15
        if float(np.sum(c[:, 2])) <= best_score:
            return
        obj = {'circles': c.tolist()}
        score = exact_score(obj)
        if score > best_score:
            best, best_score = obj, score

    consider(c0)
    lo = np.min(c0[:, :2]-c0[:, 2, None], axis=0)
    c0[:, :2] -= lo
    widths = np.max(c0[:, :2]+c0[:, 2, None], axis=0)
    z0 = np.r_[c0.ravel(), widths[0]]
    objective_gradient = np.zeros(64)
    objective_gradient[2:63:3] = -1
    cache_z = None
    cache_g = None
    cache_j = None

    def calculate(z):
        nonlocal cache_z, cache_g, cache_j
        if time.monotonic() >= deadline:
            raise StopSearch()
        if cache_z is not None and np.array_equal(z, cache_z):
            return cache_g, cache_j
        c = z[:63].reshape(21, 3)
        x, y, r = c.T
        dx, dy = x[ii]-x[jj], y[ii]-y[jj]
        rr = r[ii]+r[jj]
        g = np.r_[x-r, z[63]-x-r, y-r, 2-z[63]-y-r, dx*dx+dy*dy-rr*rr]
        j = np.zeros((294, 64))
        a = np.arange(21)
        j[a, 3*a] = 1
        j[a, 3*a+2] = -1
        j[21+a, 3*a] = -1
        j[21+a, 3*a+2] = -1
        j[21+a, 63] = 1
        j[42+a, 3*a+1] = 1
        j[42+a, 3*a+2] = -1
        j[63+a, 3*a+1] = -1
        j[63+a, 3*a+2] = -1
        j[63+a, 63] = -1
        k = 84+rows
        j[k, 3*ii] = 2*dx
        j[k, 3*jj] = -2*dx
        j[k, 3*ii+1] = 2*dy
        j[k, 3*jj+1] = -2*dy
        j[k, 3*ii+2] = -2*rr
        j[k, 3*jj+2] = -2*rr
        cache_z, cache_g, cache_j = z.copy(), g, j
        return g, j

    history = [z0.copy()]
    local_deadline = deadline
    def callback(z):
        if time.monotonic() >= local_deadline:
            raise StopSearch()
        history.append(z.copy())

    bounds = [(None, None)]*64
    for k in range(21):
        if time.monotonic() >= deadline:
            return best
        bounds[3*k+2] = (1e-7, 0.5)
    bounds[63] = (0.1, 1.9)

    def solve(z, seconds, iterations):
        nonlocal local_deadline
        local_deadline = min(deadline-0.06, time.monotonic()+seconds)
        try:
            result = minimize(lambda v: float(objective_gradient @ v), z,
                jac=lambda v: objective_gradient, method='SLSQP', bounds=bounds,
                constraints={'type': 'ineq', 'fun': lambda v: calculate(v)[0], 'jac': lambda v: calculate(v)[1]},
                callback=callback, options={'maxiter': iterations, 'ftol': params['precision'], 'disp': False})
            history.append(result.x.copy())
            return result.x
        except StopSearch:
            return history[-1].copy()

    z = solve(z0, max(0.05, budget_s*0.30), 70)
    for v in history[-8:]:
        if time.monotonic() >= deadline-0.03:
            break
        consider(v[:63].reshape(21, 3))

    # At a rigid optimum the independent active equations determine the packing.
    try:
        for iteration in range(5):
            if time.monotonic() >= deadline-0.10:
                break
            g, jac = calculate(z)
            active = np.flatnonzero(g < 2e-7)
            if len(active) < 64:
                break
            _, rr, piv = qr(jac[active].T, pivoting=True, mode='economic')
            chosen = active[piv[:64]]
            if np.min(np.abs(np.diag(rr)[:64])) < 1e-10:
                break
            step = np.linalg.solve(jac[chosen], -g[chosen])
            if np.max(np.abs(step)) > 0.02:
                break
            z = z + step
            consider(z[:63].reshape(21, 3))
    except (StopSearch, np.linalg.LinAlgError, ValueError):
        pass

    # Use remaining time for a small topology probe, never sacrificing the incumbent.
    if time.monotonic() < deadline-0.25:
        trial = z.copy()
        trial[:63].reshape(21, 3)[:, :2] += rng.normal(0, params['perturbation'], (21, 2))
        old_len = len(history)
        solve(trial, max(0.02, deadline-time.monotonic()-0.07), 45)
        for v in history[old_len:][-6:]:
            if time.monotonic() >= deadline-0.01:
                break
            consider(v[:63].reshape(21, 3))
    return best
