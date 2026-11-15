import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'first_angle': (32.0, 12.0, 43.0),
    'trial_fraction': (0.23, 0.12, 0.35),
    'maxiter': (100, 30, 200),
    'ftol': (2e-12, 1e-14, 1e-8, 'log'),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
K = np.arange(N)

class StopSearch(Exception):
    pass


def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    r = a[:, 2]
    d = np.sqrt((a[I, 0] - a[J, 0]) ** 2 + (a[I, 1] - a[J, 1]) ** 2)
    if np.any(d < r[I] + r[J]):
        return -np.inf
    x = list(map(lambda v: Fraction(float(v)), a[:, 0]))
    y = list(map(lambda v: Fraction(float(v)), a[:, 1]))
    q = list(map(lambda v: Fraction(float(v)), r))
    hp = (max(map(sum, zip(x, q))) - min(map(lambda t: t[0]-t[1], zip(x, q)))
          + max(map(sum, zip(y, q))) - min(map(lambda t: t[0]-t[1], zip(y, q))))
    if hp > 2:
        return -np.inf
    return float(np.sum(r))


def repair(a):
    a = np.asarray(a, dtype=float).copy()
    if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return None
    a[:, :2] -= a[:, :2].mean(axis=0)
    r = a[:, 2]
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    factor = min(1.0, float(np.min(d / (r[I]+r[J]))))
    r *= factor * (1.0 - 3e-14)
    sides = np.max(a[:, :2] + r[:, None], axis=0) - np.min(a[:, :2] - r[:, None], axis=0)
    a *= 2.0 / float(sides.sum()) * (1.0 - 3e-14)
    a[:, 2] *= 1.0 - 3e-14
    return a


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.69 * budget_s
    rng = np.random.default_rng(seed)
    if init is not None:
        original = np.asarray(init['circles'], dtype=float).copy()
        best = init
        best_score = exact_score(original)
    else:
        original = np.array([[(k % 5 + 0.5) / 5, (k // 5 + 0.5) / 5, 0.095] for k in range(N)])
        original = repair(original)
        best = {'circles': original.tolist()}
        best_score = exact_score(original)
    base = original.copy()
    base[:, :2] -= base[:, :2].mean(axis=0)

    boundary_jac = np.zeros((4*N, 3*N+1))
    boundary_jac[K, K] = 1
    boundary_jac[K, 2*N+K] = -1
    boundary_jac[N+K, K] = -1
    boundary_jac[N+K, 2*N+K] = -1
    boundary_jac[N+K, -1] = 1
    boundary_jac[2*N+K, N+K] = 1
    boundary_jac[2*N+K, 2*N+K] = -1
    boundary_jac[3*N+K, N+K] = -1
    boundary_jac[3*N+K, 2*N+K] = -1
    boundary_jac[3*N+K, -1] = -1
    objective_grad = np.r_[np.zeros(2*N), -np.ones(N), 0.0]
    bounds = [(None, None)] * (2*N) + [(1e-7, 0.7)] * N + [(0.12, 1.88)]

    def consider(z):
        nonlocal best, best_score
        if time.monotonic() >= end:
            return
        a = repair(np.column_stack((z[:N], z[N:2*N], z[2*N:3*N])))
        if a is not None and float(a[:, 2].sum()) > best_score:
            score = exact_score(a)
            if score > best_score:
                best_score = score
                best = {'circles': a.tolist()}

    sign = -1.0 if rng.integers(2) else 1.0
    angles = [sign * float(params['first_angle']), -sign * 22.5, sign * 45.0, -sign * 12.0]
    for angle in angles:
        if time.monotonic() >= end - 0.04:
            break
        trial_end = min(end - 0.012, time.monotonic() + float(params['trial_fraction']) * budget_s)
        theta = np.deg2rad(angle)
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        a = base.copy()
        a[:, :2] = a[:, :2] @ rotation.T
        lower = np.min(a[:, :2] - a[:, 2, None], axis=0)
        side = np.max(a[:, :2] + a[:, 2, None], axis=0) - lower
        scale = 2 / side.sum()
        a[:, :2] = (a[:, :2] - lower) * scale
        a[:, 2] *= scale
        z0 = np.r_[a[:, 0], a[:, 1], a[:, 2], side[0] * scale]
        last = z0.copy()
        iteration = 0

        def check():
            if time.monotonic() >= trial_end:
                raise StopSearch()

        def objective(z):
            check()
            return -float(z[2*N:3*N].sum()), objective_grad

        def constraints(z):
            check()
            x, y, r, w = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
            dx, dy = x[I]-x[J], y[I]-y[J]
            d = np.sqrt(dx*dx + dy*dy)
            return np.r_[d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r]

        def jacobian(z):
            check()
            x, y = z[:N], z[N:2*N]
            dx, dy = x[I]-x[J], y[I]-y[J]
            d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-18)
            jac = np.zeros((M+4*N, 3*N+1))
            rows = np.arange(M)
            jac[rows, I] = dx/d
            jac[rows, J] = -dx/d
            jac[rows, N+I] = dy/d
            jac[rows, N+J] = -dy/d
            jac[rows, 2*N+I] = -1
            jac[rows, 2*N+J] = -1
            jac[M:] = boundary_jac
            return jac

        def callback(z):
            nonlocal last, iteration
            last = z.copy()
            iteration += 1
            check()
            if iteration % 8 == 0:
                consider(last)

        try:
            result = minimize(objective, z0, jac=True, method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                              callback=callback,
                              options={'maxiter': int(params['maxiter']), 'ftol': float(params['ftol']), 'disp': False})
            last = result.x
        except StopSearch:
            pass
        consider(last)
    return best
