import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    "maxiter": (100, 20, 250),
    "contact_tol": (2e-7, 1e-10, 1e-4, "log"),
    "perturb": (0.014, 0.0001, 0.08, "log"),
    "restarts": (5, 1, 12),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
IDS = np.arange(N)

class TimeUp(Exception):
    pass


def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    p, r = a[:, :2], a[:, 2]
    d = np.sqrt((p[I, 0] - p[J, 0]) ** 2 + (p[I, 1] - p[J, 1]) ** 2)
    if np.any(d < r[I] + r[J]):
        return -np.inf
    x = list(map(lambda v: Fraction(float(v)), p[:, 0]))
    y = list(map(lambda v: Fraction(float(v)), p[:, 1]))
    q = list(map(lambda v: Fraction(float(v)), r))
    hp = (max(a + b for a, b in zip(x, q)) - min(a - b for a, b in zip(x, q))
          + max(a + b for a, b in zip(y, q)) - min(a - b for a, b in zip(y, q)))
    if hp > 2:
        return -np.inf
    return float(np.sum(r))


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf
    if init is not None:
        original = np.asarray(init['circles'], dtype=float)
        best_score = exact_score(original)
    else:
        k = np.arange(N)
        original = np.column_stack(((k % 5 + 0.5) / 5, (k // 5 + 0.5) / 5, np.full(N, 0.09)))
        best = {'circles': original.tolist()}
        best_score = exact_score(original)

    def guard():
        if time.monotonic() >= deadline:
            raise TimeUp()

    def pack(a):
        p = a[:, :2].copy()
        r = a[:, 2].copy()
        lo = np.min(p - r[:, None], axis=0)
        side = np.max(p + r[:, None], axis=0) - lo
        scale = 2.0 / np.sum(side)
        p = (p - lo) * scale
        return np.r_[p[:, 0], p[:, 1], r * scale, side[0] * scale]

    def constraints(z):
        guard()
        x, y, r, w = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
        dx, dy = x[I] - x[J], y[I] - y[J]
        d = np.sqrt(dx*dx + dy*dy)
        return np.r_[d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r]

    def jacobian(z):
        guard()
        x, y = z[:N], z[N:2*N]
        dx, dy = x[I]-x[J], y[I]-y[J]
        d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-30)
        a = np.zeros((M+4*N, 3*N+1))
        rows = np.arange(M)
        a[rows, I] = dx/d
        a[rows, J] = -dx/d
        a[rows, N+I] = dy/d
        a[rows, N+J] = -dy/d
        a[rows, 2*N+I] = -1
        a[rows, 2*N+J] = -1
        a[M+IDS, IDS] = 1
        a[M+N+IDS, IDS] = -1
        a[M+2*N+IDS, N+IDS] = 1
        a[M+3*N+IDS, N+IDS] = -1
        a[M+IDS, 2*N+IDS] = -1
        a[M+N+IDS, 2*N+IDS] = -1
        a[M+2*N+IDS, 2*N+IDS] = -1
        a[M+3*N+IDS, 2*N+IDS] = -1
        a[M+N+IDS, -1] = 1
        a[M+3*N+IDS, -1] = -1
        return a

    gradient = np.r_[np.zeros(2*N), -np.ones(N), 0.0]

    def objective(z):
        guard()
        return -float(np.sum(z[2*N:3*N])), gradient

    def consider(z):
        nonlocal best, best_score
        guard()
        if not np.isfinite(z).all():
            return
        p = np.column_stack((z[:N], z[N:2*N]))
        r = z[2*N:3*N].copy()
        if np.any(r <= 0):
            return
        # Repair pairwise violations with a uniform radius contraction.
        d = np.sqrt((p[I, 0]-p[J, 0])**2 + (p[I, 1]-p[J, 1])**2)
        factor = min(1.0, float(np.min(d/(r[I]+r[J]))))
        r *= factor * (1.0-2e-15)
        if np.any(r <= 0):
            return
        lo = np.min(p-r[:, None], axis=0)
        p -= lo
        side = np.max(p+r[:, None], axis=0) - np.min(p-r[:, None], axis=0)
        factor = 2.0 / float(np.sum(side)) * (1.0-4e-15)
        p *= factor
        r *= factor
        if float(np.sum(r)) <= best_score:
            return
        a = np.column_stack((p, r))
        # A few ulps of extra slack handle scale-rounding at exact tangencies.
        for attempt in range(4):
            guard()
            score = exact_score(a)
            if score > best_score:
                best_score = score
                best = {'circles': a.tolist()}
                return
            if np.isfinite(score):
                return
            a[:, 2] *= 1.0-4e-15
            if float(np.sum(a[:, 2])) <= best_score:
                return

    def polish(z):
        guard()
        z = z.copy()
        active = np.flatnonzero(constraints(z) < float(params['contact_tol']))
        if len(active) == 0:
            return z
        for iteration in range(7):
            guard()
            c = constraints(z)[active]
            if np.max(np.abs(c)) < 2e-16:
                break
            a = jacobian(z)[active]
            delta = np.linalg.lstsq(a, -c, rcond=1e-11)[0]
            if not np.isfinite(delta).all() or np.max(np.abs(delta)) > 0.025:
                break
            trial = z + delta
            if np.min(trial[2*N:3*N]) <= 0:
                break
            if np.max(np.abs(constraints(trial)[active])) > np.max(np.abs(c))*1.05:
                break
            z = trial
            consider(z)
        return z

    try:
        z0 = pack(original)
        consider(z0)
        z0 = polish(z0)
        bounds = [(None, None)]*(2*N) + [(1e-8, 0.6)]*N + [(0.1, 1.9)]
        latest = z0.copy()

        def callback(z):
            nonlocal latest
            guard()
            latest = z.copy()
            consider(z)

        for restart in range(int(params['restarts'])):
            guard()
            if restart == 0:
                z = z0.copy()
            else:
                z = pack(np.asarray(best['circles'], dtype=float))
                amplitude = float(params['perturb']) * (0.15 if restart == 1 else 1.0)
                z[:2*N] += rng.normal(0.0, amplitude, 2*N)
                z[2*N:3*N] *= rng.uniform(0.97, 1.01, N)
            result = minimize(objective, z, jac=True, method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                              callback=callback,
                              options={'maxiter': int(params['maxiter']), 'ftol': 2e-14, 'disp': False})
            guard()
            consider(result.x)
            polish(result.x)
    except TimeUp:
        pass
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        pass
    return best
