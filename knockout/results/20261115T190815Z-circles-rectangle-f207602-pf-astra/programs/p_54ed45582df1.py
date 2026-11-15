import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import lstsq

PARAMS = {
    'contact_tol': (2e-6, 1e-9, 1e-3, 'log'),
    'perturbation': (0.018, 0.0001, 0.1, 'log'),
    'attempts': (10, 1, 40),
    'iterations': (90, 10, 300)
}
N = 21
I, J = np.triu_indices(N, 1)
K = np.arange(N)

class Deadline(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70 * budget_s
    work_end = end - min(0.035, 0.05 * budget_s)
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf

    def check():
        if time.monotonic() >= work_end:
            raise Deadline()

    def exact_score(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -np.inf
        d = np.sqrt((a[I, 0] - a[J, 0]) ** 2 + (a[I, 1] - a[J, 1]) ** 2)
        if np.any(d < a[I, 2] + a[J, 2]):
            return -np.inf
        x, y, r = [], [], []
        for row in a:
            if time.monotonic() >= end:
                return -np.inf
            x.append(Fraction(float(row[0])))
            y.append(Fraction(float(row[1])))
            r.append(Fraction(float(row[2])))
        perimeter = (max(u + v for u, v in zip(x, r)) - min(u - v for u, v in zip(x, r))
                     + max(u + v for u, v in zip(y, r)) - min(u - v for u, v in zip(y, r)))
        return float(np.sum(a[:, 2])) if perimeter <= 2 else -np.inf

    def pack(a):
        lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
        hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
        scale = 2.0 / np.sum(hi - lo)
        p = (a[:, :2] - lo) * scale
        return np.r_[p[:, 0], p[:, 1], a[:, 2] * scale, (hi[0] - lo[0]) * scale]

    def unpack(z):
        return np.column_stack((z[:N], z[N:2*N], z[2*N:3*N]))

    def consider(z):
        nonlocal best, best_score
        check()
        a = unpack(z).copy()
        if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return
        d = np.sqrt((a[I, 0] - a[J, 0]) ** 2 + (a[I, 1] - a[J, 1]) ** 2)
        factor = min(1.0, float(np.min(d / (a[I, 2] + a[J, 2]))))
        a[:, 2] *= factor * (1.0 - 3e-15)
        lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
        hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
        a[:, :2] -= (hi + lo) / 2
        scale = min(1.0, 2.0 / np.sum(hi - lo)) * (1.0 - 3e-15)
        a *= scale
        for _ in range(4):
            check()
            score = exact_score(a)
            if score > best_score:
                best_score = score
                best = {'circles': a.tolist()}
                return
            if np.isfinite(score):
                return
            a[:, 2] *= 1.0 - 2e-14
            a[:, :2] *= 1.0 - 2e-14

    def constraints(z, jac=False):
        check()
        x, y, r, w = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
        dx, dy = x[I] - x[J], y[I] - y[J]
        d = np.sqrt(dx*dx + dy*dy)
        g = np.r_[d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r]
        if not jac:
            return g
        A = np.zeros((294, 64))
        rows = np.arange(210)
        dx = dx / np.maximum(d, 1e-100)
        dy = dy / np.maximum(d, 1e-100)
        A[rows, I], A[rows, J] = dx, -dx
        A[rows, N+I], A[rows, N+J] = dy, -dy
        A[rows, 2*N+I] = -1
        A[rows, 2*N+J] = -1
        A[210+K, K] = 1
        A[231+K, K] = -1
        A[231+K, -1] = 1
        A[252+K, N+K] = 1
        A[273+K, N+K] = -1
        A[273+K, -1] = -1
        A[210+K, 2*N+K] = -1
        A[231+K, 2*N+K] = -1
        A[252+K, 2*N+K] = -1
        A[273+K, 2*N+K] = -1
        return A

    def polish(z):
        check()
        z = z.copy()
        active = np.flatnonzero(np.abs(constraints(z)) < float(params['contact_tol']))
        if len(active) < 60:
            return z
        for _ in range(9):
            check()
            g = constraints(z)[active]
            A = constraints(z, True)[active]
            delta, _, rank, _ = lstsq(A, -g, lapack_driver='gelsy', check_finite=False)
            if rank < 64 or np.max(np.abs(delta)) > 0.025:
                break
            z += delta
            if np.max(np.abs(delta)) < 2e-15:
                break
        return z

    grad = np.r_[np.zeros(42), -np.ones(21), 0.0]
    def objective(z):
        check()
        return -float(np.sum(z[42:63]))

    bounds = [(None, None)] * 42 + [(1e-8, 0.7)] * 21 + [(0.1, 1.9)]
    con = {'type': 'ineq', 'fun': constraints, 'jac': lambda z: constraints(z, True)}
    try:
        if init is not None:
            a0 = np.asarray(init['circles'], dtype=float)
            best_score = exact_score(a0)
            z0 = pack(a0)
        else:
            a0 = np.zeros((21, 3))
            a0[:, 0] = (np.arange(21) % 5 + 0.5) / 5
            a0[:, 1] = (np.arange(21) // 5 + 0.5) / 5
            a0[:, 2] = 0.095
            best = {'circles': a0.tolist()}
            best_score = exact_score(a0)
            z0 = pack(a0)
        z = polish(z0)
        consider(z)
        result = minimize(objective, z, jac=lambda z: grad, constraints=con, bounds=bounds,
                          method='SLSQP', options={'maxiter': int(params['iterations']), 'ftol': 2e-14})
        consider(result.x)
        consider(polish(result.x))
        base = pack(np.asarray(best['circles'], dtype=float))
        a = unpack(base)
        # Infer the closest reflection permutation; perturb entire mirrored groups together.
        permutations = []
        for axis in range(2):
            check()
            reflected = a.copy()
            middle = base[-1]/2 if axis == 0 else (2-base[-1])/2
            reflected[:, axis] = 2*middle-reflected[:, axis]
            costs = np.sum((reflected[:, None, :]-a[None, :, :])**2, axis=2)
            perm = np.argmin(costs, axis=1)
            permutations.append((float(costs[K, perm].sum()), axis, perm))
        error, axis, perm = min(permutations, key=lambda item: item[0])
        for attempt in range(int(params['attempts'])):
            check()
            noise = rng.normal(size=(21, 3))
            if error < 1e-6 and np.all(perm[perm] == K):
                transformed = noise[perm].copy()
                transformed[:, axis] *= -1
                noise = (noise + transformed) / 2
            amplitude = float(params['perturbation']) * (0.5 + 0.3*(attempt % 5))
            trial = base.copy()
            trial[:21] += amplitude * noise[:, 0]
            trial[21:42] += amplitude * noise[:, 1]
            trial[42:63] = np.maximum(1e-5, trial[42:63] + amplitude*0.25*noise[:, 2])
            result = minimize(objective, trial, jac=lambda z: grad, constraints=con, bounds=bounds,
                              method='SLSQP', options={'maxiter': int(params['iterations']), 'ftol': 5e-13})
            consider(result.x)
            consider(polish(result.x))
    except Deadline:
        pass
    return best
