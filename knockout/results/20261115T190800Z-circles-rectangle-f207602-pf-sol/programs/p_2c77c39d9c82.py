import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import time
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize

PARAMS = {
    'trust': (0.13, 0.02, 0.3),
    'perturbation': (0.036, 0.001, 0.12),
    'weight_noise': (0.30, 0.0, 0.8),
    'polish_iterations': (45, 10, 100),
}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    ar = np.arange(n)
    A = np.zeros((84, 64))
    b = np.zeros(84)
    A[ar, ar] = 1
    A[ar, 42 + ar] = -1
    A[21 + ar, ar] = -1
    A[21 + ar, 42 + ar] = -1
    A[21 + ar, 63] = 1
    A[42 + ar, 21 + ar] = 1
    A[42 + ar, 42 + ar] = -1
    A[63 + ar, 21 + ar] = -1
    A[63 + ar, 42 + ar] = -1
    A[63 + ar, 63] = -1
    b[63:] = 2
    objective = np.zeros(64)
    objective[42:63] = -1
    rows = np.arange(m)
    pairbase = np.zeros((m, 64))
    pairbase[rows, 42 + ii] = -1
    pairbase[rows, 42 + jj] = -1
    best = init
    best_score = -float('inf')

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def exact_score(obj):
        c = np.asarray(obj['circles'], dtype=float)
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return -float('inf')
        dx = c[ii, 0] - c[jj, 0]
        dy = c[ii, 1] - c[jj, 1]
        if np.any(np.sqrt(dx**2 + dy**2) < c[ii, 2] + c[jj, 2]):
            return -float('inf')
        x = list(map(Fraction, map(float, c[:, 0])))
        y = list(map(Fraction, map(float, c[:, 1])))
        r = list(map(Fraction, map(float, c[:, 2])))
        width = max(map(lambda t: t[0] + t[1], zip(x, r))) - min(map(lambda t: t[0] - t[1], zip(x, r)))
        height = max(map(lambda t: t[0] + t[1], zip(y, r))) - min(map(lambda t: t[0] - t[1], zip(y, r)))
        if width + height > 2:
            return -float('inf')
        return float(np.sum(c[:, 2]))

    if init is not None:
        best_score = exact_score(init)
        initial = np.array(init['circles'], dtype=float)
    else:
        xx, yy = np.meshgrid(np.arange(7), np.arange(3))
        initial = np.column_stack(((xx.ravel() + .5) * .2, (yy.ravel() + .5) * .2, np.full(21, .1 - 1e-15)))

    def pack(c):
        c = c.copy()
        lo = np.min(c[:, :2] - c[:, 2, None], axis=0)
        hi = np.max(c[:, :2] + c[:, 2, None], axis=0)
        scale = 2.0 / np.sum(hi - lo)
        c[:, :2] = (c[:, :2] - lo) * scale
        c[:, 2] *= scale
        return np.r_[c[:, 0], c[:, 1], c[:, 2], (hi[0] - lo[0]) * scale]

    def consider(z):
        nonlocal best, best_score
        check()
        c = np.column_stack((z[:21], z[21:42], z[42:63]))
        if not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return
        c[:, :2] -= np.mean(c[:, :2], axis=0)
        distances = np.sqrt((c[ii, 0] - c[jj, 0])**2 + (c[ii, 1] - c[jj, 1])**2)
        ratio = float(np.min(distances / (c[ii, 2] + c[jj, 2])))
        if ratio <= 0:
            return
        c[:, 2] *= min(1.0, ratio) * (1 - 2e-15)
        extent = np.max(c[:, :2] + c[:, 2, None], axis=0) - np.min(c[:, :2] - c[:, 2, None], axis=0)
        c *= (2.0 / np.sum(extent)) * (1 - 2e-15)
        for attempt in range(4):
            check()
            obj = {'circles': c.tolist()}
            score = exact_score(obj)
            if score > best_score:
                best_score, best = score, obj
            if np.isfinite(score):
                break
            c[:, 2] *= 1 - 4e-15 * (attempt + 1)
            c *= 1 - 2e-15

    def constraints(z):
        check()
        dx = z[ii] - z[jj]
        dy = z[21 + ii] - z[21 + jj]
        d = np.sqrt(dx * dx + dy * dy)
        return np.r_[A @ z + b, d - z[42 + ii] - z[42 + jj]]

    def jacobian(z):
        check()
        dx = z[ii] - z[jj]
        dy = z[21 + ii] - z[21 + jj]
        d = np.maximum(np.sqrt(dx * dx + dy * dy), 1e-30)
        P = pairbase.copy()
        P[rows, ii] = dx / d
        P[rows, jj] = -dx / d
        P[rows, 21 + ii] = dy / d
        P[rows, 21 + jj] = -dy / d
        return np.vstack((A, P))

    def contact_polish(z):
        z = z.copy()
        vals = constraints(z)
        active = np.argsort(np.abs(vals))[:64]
        if np.max(np.abs(vals[active])) > 2e-5:
            return z
        for iteration in range(7):
            check()
            residual = constraints(z)[active]
            J = jacobian(z)[active]
            try:
                delta = np.linalg.solve(J, -residual)
            except np.linalg.LinAlgError:
                break
            if not np.isfinite(delta).all() or np.max(np.abs(delta)) > .015:
                break
            z += delta
            consider(z)
            if np.max(np.abs(delta)) < 2e-15:
                break
        return z

    bounds = [(None, None)] * 42 + [(1e-7, .8)] * 21 + [(.3, 1.7)]

    def polish(z, iterations):
        check()
        def fun(v):
            check()
            return float(objective @ v)
        result = minimize(fun, z, jac=lambda v: objective,
                          method='SLSQP', bounds=bounds,
                          constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                          callback=lambda v: check(),
                          options={'maxiter': iterations, 'ftol': 7e-14, 'disp': False})
        consider(result.x)
        return contact_polish(result.x)

    def planes(z, weights=None):
        check()
        J = jacobian(z)
        rhs = np.r_[b, np.zeros(m)]
        cost = objective.copy()
        if weights is not None:
            cost[42:63] = -weights
        trust = params['trust']
        local_bounds = [(float(z[k] - trust), float(z[k] + trust)) for k in range(42)] + bounds[42:]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = linprog(cost, A_ub=-J, b_ub=rhs, bounds=local_bounds,
                             method='highs', options={'threads': 1, 'primal_feasibility_tolerance': 1e-10,
                             'dual_feasibility_tolerance': 1e-10, 'time_limit': max(.001, deadline - time.monotonic())})
        check()
        if result.success:
            consider(result.x)
            return result.x
        return z

    try:
        z0 = pack(initial)
        consider(z0)
        z0 = contact_polish(z0)
        z0 = polish(z0, params['polish_iterations'])
        if best is not None:
            z0 = pack(np.asarray(best['circles'], dtype=float))
        # Identify reflection partners directly, without assuming an orientation.
        xy = np.column_stack((z0[:21], z0[21:42]))
        best_error = float('inf')
        partner = np.arange(21)
        reflection_axis = 1
        for axis in range(2):
            check()
            reflected = xy.copy()
            reflected[:, axis] = np.min(xy[:, axis] - z0[42:63]) + np.max(xy[:, axis] + z0[42:63]) - xy[:, axis]
            distances = np.sum((reflected[:, None, :] - xy[None, :, :])**2, axis=2)
            partners = np.argmin(distances, axis=1)
            error = np.sum(np.min(distances, axis=1))
            if error < best_error:
                best_error, partner, reflection_axis = error, partners, axis
        symmetric = best_error < 1e-7 and np.all(partner[partner] == ar)
        restart = 0
        while time.monotonic() < deadline:
            check()
            restart += 1
            z = z0.copy()
            displacement = rng.normal(size=(21, 2))
            weights = rng.normal(size=21)
            if symmetric:
                other = displacement[partner].copy()
                other[:, reflection_axis] *= -1
                displacement = (displacement + other) * .5
                weights = (weights + weights[partner]) * .5
            amplitude = params['perturbation'] * (1.0 if restart % 3 else 1.8)
            z[:21] += amplitude * displacement[:, 0]
            z[21:42] += amplitude * displacement[:, 1]
            previous = -float('inf')
            for step in range(9):
                check()
                w = np.maximum(.1, 1 + params['weight_noise'] * weights) if step < 2 else None
                z = planes(z, w)
                value = float(np.sum(z[42:63]))
                if step > 3 and abs(value - previous) < 1e-10:
                    break
                previous = value
            if time.monotonic() + .06 < deadline:
                z = polish(z, 25)
            if best is not None:
                z0 = pack(np.asarray(best['circles'], dtype=float))
    except StopSearch:
        pass
    if best is None:
        best = {'circles': initial.tolist()}
    return best
