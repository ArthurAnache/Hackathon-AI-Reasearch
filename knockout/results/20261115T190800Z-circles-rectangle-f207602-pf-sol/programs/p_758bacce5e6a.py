import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'power': (0.05, -1.0, 0.8),
    'excursion_iterations': (48, 10, 100),
    'return_iterations': (110, 20, 200),
    'floor': (0.0001, 0.000001, 0.01, 'log'),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class Timeout(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -float('inf')

    def check():
        if time.monotonic() >= deadline:
            raise Timeout()

    def score(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        d = np.sqrt((a[I, 0] - a[J, 0]) ** 2 + (a[I, 1] - a[J, 1]) ** 2)
        if np.any(d < a[I, 2] + a[J, 2]):
            return -float('inf')
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        perimeter = (max(u + v for u, v in zip(x, r)) - min(u - v for u, v in zip(x, r))
                     + max(u + v for u, v in zip(y, r)) - min(u - v for u, v in zip(y, r)))
        if perimeter > 2:
            return -float('inf')
        return float(np.sum(a[:, 2]))

    if init is not None:
        a = np.array(init['circles'], dtype=float)
        best_score = score(a)
    else:
        a = np.array([[(k % 5 + 0.5) / 5, (k // 5 + 0.5) / 5, 0.095] for k in range(21)])
        best = {'circles': a.tolist()}
        best_score = score(a)
    lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
    sides = np.max(a[:, :2] + a[:, 2, None], axis=0) - lo
    scale = 2.0 / np.sum(sides)
    p = (a[:, :2] - lo) * scale
    z = np.r_[p[:, 0], p[:, 1], a[:, 2] * scale, sides[0] * scale]

    def save(v):
        nonlocal best, best_score
        check()
        p = np.column_stack((v[:N], v[N:2*N]))
        r = np.maximum(v[2*N:3*N], 1e-12).copy()
        d = np.sqrt(np.sum((p[I] - p[J]) ** 2, axis=1))
        shrink = min(1.0, float(np.min(d / (r[I] + r[J]))))
        r *= shrink * (1.0 - 4e-14)
        for _ in range(4):
            check()
            out = np.column_stack((p, r))
            s = score(out)
            if s > best_score:
                best_score = s
                best = {'circles': out.tolist()}
            if np.isfinite(s):
                return
            lo1 = np.min(p - r[:, None], axis=0)
            hi1 = np.max(p + r[:, None], axis=0)
            factor = min(1.0, 2.0 / float(np.sum(hi1 - lo1))) * (1.0 - 2e-13)
            p = (p - lo1) * factor
            r *= factor

    def constraints(v):
        check()
        x, y, r, w = v[:N], v[N:2*N], v[2*N:3*N], v[-1]
        dx, dy = x[I] - x[J], y[I] - y[J]
        d = np.sqrt(dx * dx + dy * dy + 1e-30)
        return np.r_[d - r[I] - r[J], x-r, w-x-r, y-r, 2-w-y-r]

    def jacobian(v):
        check()
        x, y = v[:N], v[N:2*N]
        dx, dy = x[I]-x[J], y[I]-y[J]
        d = np.sqrt(dx*dx + dy*dy + 1e-30)
        jac = np.zeros((M+4*N, 3*N+1))
        k = np.arange(M)
        jac[k, I] = dx/d
        jac[k, J] = -dx/d
        jac[k, N+I] = dy/d
        jac[k, N+J] = -dy/d
        jac[k, 2*N+I] = -1
        jac[k, 2*N+J] = -1
        k = np.arange(N)
        jac[M+k, k] = 1
        jac[M+N+k, k] = -1
        jac[M+N+k, -1] = 1
        jac[M+2*N+k, N+k] = 1
        jac[M+3*N+k, N+k] = -1
        jac[M+3*N+k, -1] = -1
        for b in range(4):
            check()
            jac[M+b*N+k, 2*N+k] = -1
        return jac

    bounds = [(None, None)]*(2*N) + [(float(params['floor']), 0.5)]*N + [(0.05, 1.95)]
    con = {'type': 'ineq', 'fun': constraints, 'jac': jacobian}

    def run(v, power, iterations):
        def objective(t):
            check()
            r = np.maximum(t[2*N:3*N], 1e-12)
            u = r / 0.11
            g = np.zeros(3*N+1)
            if abs(power) < 1e-9:
                value = -0.11 * np.log(u).sum()
                g[2*N:3*N] = -1/u
            else:
                value = -0.11 * np.sum(np.expm1(power*np.log(u))) / power
                g[2*N:3*N] = -u**(power-1)
            return value, g
        result = minimize(objective, v, jac=True, method='SLSQP', bounds=bounds,
                          constraints=con, callback=lambda _: check(),
                          options={'maxiter': iterations, 'ftol': 2e-12, 'disp': False})
        return result.x

    try:
        q = float(params['power'])
        z = run(z, q, int(params['excursion_iterations']))
        save(z)
        z = run(z, 1.0, int(params['return_iterations']))
        save(z)
        while time.monotonic() < deadline:
            check()
            # A second, weaker excursion probes the new topology rather than restarting the incumbent.
            trial = z.copy()
            trial[2*N:3*N] *= np.exp(rng.normal(0.0, 0.09, N))
            trial = run(trial, 0.45, 25)
            trial = run(trial, 1.0, int(params['return_iterations']))
            save(trial)
            z = trial
    except Timeout:
        pass
    return best
