import time
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog

PARAMS = {
    'trust': (0.19, 0.03, 0.5),
    'steps': (17, 5, 40),
    'recuts': (8, 2, 30),
    'radius_floor': (1e-7, 1e-10, 1e-4, 'log'),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)


def score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    r = a[:, 2]
    d = np.sqrt((a[I, 0] - a[J, 0]) ** 2 + (a[I, 1] - a[J, 1]) ** 2)
    if np.any(d < r[I] + r[J]):
        return -np.inf
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    q = [Fraction(float(v)) for v in r]
    perimeter = (max(u + v for u, v in zip(x, q)) - min(u - v for u, v in zip(x, q))
                 + max(u + v for u, v in zip(y, q)) - min(u - v for u, v in zip(y, q)))
    if perimeter > 2:
        return -np.inf
    return float(np.sum(r))


def repaired(z):
    p = np.column_stack((z[:N], z[N:2*N]))
    r = np.maximum(z[2*N:3*N], 1e-12).copy()
    d = np.sqrt(np.sum((p[I] - p[J]) ** 2, axis=1))
    ratio = float(np.min(d / (r[I] + r[J])))
    r *= min(1.0, ratio) * (1.0 - 3e-15)
    lo = np.min(p - r[:, None], axis=0)
    hi = np.max(p + r[:, None], axis=0)
    s = min(1.0, 2.0 / float(np.sum(hi-lo))) * (1.0 - 3e-15)
    a = np.column_stack(((p-lo)*s, r*s))
    return a


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf
    if init is not None:
        a0 = np.asarray(init['circles'], dtype=float)
        best_score = score(a0)
    else:
        a0 = np.array([[(k % 5 + .5)/5, (k // 5 + .5)/5, .04] for k in range(N)])
        best = {'circles': a0.tolist()}
        best_score = score(a0)
    lo = np.min(a0[:, :2]-a0[:, 2, None], axis=0)
    hi = np.max(a0[:, :2]+a0[:, 2, None], axis=0)
    scale = 2.0 / np.sum(hi-lo)
    p0 = (a0[:, :2]-lo)*scale
    r0 = a0[:, 2]*scale
    w0 = float((hi[0]-lo[0])*scale)
    base = np.r_[p0[:, 0], p0[:, 1], r0, w0]

    # Variables: x[21], y[21], r[21], rectangle width.
    A = np.zeros((M+4*N, 3*N+1))
    b = np.zeros(M+4*N)
    t = np.arange(M)
    A[t, 2*N+I] = 1
    A[t, 2*N+J] = 1
    k = np.arange(N)
    rows = M+k
    A[rows, k] = -1
    A[rows, 2*N+k] = 1
    rows = M+N+k
    A[rows, k] = 1
    A[rows, 2*N+k] = 1
    A[rows, -1] = -1
    rows = M+2*N+k
    A[rows, N+k] = -1
    A[rows, 2*N+k] = 1
    rows = M+3*N+k
    A[rows, N+k] = 1
    A[rows, 2*N+k] = 1
    A[rows, -1] = 1
    b[rows] = 2
    objective = np.r_[np.zeros(2*N), -np.ones(N), 0.0]
    floor = float(params['radius_floor'])
    patterns = [(0., 0.), (.5, 0.), (0., .5), (.5, .5), (.25, 0.), (0., .25), (.25, .5), (.5, .25)]

    for trial in range(int(params['recuts'])):
        if time.monotonic() >= deadline:
            break
        z = base.copy()
        if trial:
            if trial < len(patterns):
                sx, sy = patterns[trial]
            else:
                sx, sy = rng.uniform(.1, .9, 2)
            z[:N] = np.mod(z[:N] + sx*w0, w0)
            z[N:2*N] = np.mod(z[N:2*N] + sy*(2-w0), 2-w0)
            # Start a newly cut outer face without privileging old radii.
            z[2*N:3*N] *= .55
        previous = -np.inf
        stagnant = 0
        for step in range(int(params['steps'])):
            now = time.monotonic()
            if now >= deadline - .025:
                break
            dx = z[I]-z[J]
            dy = z[N+I]-z[N+J]
            d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-14)
            ux, uy = dx/d, dy/d
            A[t, I] = -ux
            A[t, J] = ux
            A[t, N+I] = -uy
            A[t, N+J] = uy
            trust = float(params['trust']) * (1.0 if step < 5 else .65)
            lower = np.r_[np.maximum(z[:2*N]-trust, 0), np.full(N, floor), .08]
            upper = np.r_[np.minimum(z[:2*N]+trust, 2), np.full(N, .9), 1.92]
            bounds = list(zip(lower, upper))
            result = linprog(objective, A_ub=A, b_ub=b, bounds=bounds,
                             method='highs', options={'time_limit': max(.001, deadline-now-.015),
                                                      'primal_feasibility_tolerance': 1e-9,
                                                      'dual_feasibility_tolerance': 1e-9})
            if not result.success:
                break
            z = result.x
            a = repaired(z)
            value = score(a)
            if value > best_score:
                best_score = value
                best = {'circles': a.tolist()}
            current = float(np.sum(z[2*N:3*N]))
            if current-previous < 2e-9:
                stagnant += 1
            else:
                stagnant = 0
            previous = current
            if stagnant >= 2:
                break
    return best
