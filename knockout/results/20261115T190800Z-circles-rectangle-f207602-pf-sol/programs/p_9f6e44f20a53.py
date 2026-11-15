import time
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, OptimizeWarning

PARAMS = {
    'tilt': (0.48, 0.05, 1.5),
    'trust': (0.055, 0.005, 0.2),
    'biased_steps': (7, 2, 20),
    'relax_steps': (17, 3, 40),
    'radius_floor': (0.002, 0.000001, 0.02)
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)


def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    r = a[:, 2]
    dx = a[I, 0] - a[J, 0]
    dy = a[I, 1] - a[J, 1]
    if np.any(np.sqrt(dx * dx + dy * dy) < r[I] + r[J]):
        return -np.inf
    x = list(map(lambda v: Fraction(float(v)), a[:, 0]))
    y = list(map(lambda v: Fraction(float(v)), a[:, 1]))
    q = list(map(lambda v: Fraction(float(v)), r))
    perimeter = (max(u + v for u, v in zip(x, q)) - min(u - v for u, v in zip(x, q))
                 + max(u + v for u, v in zip(y, q)) - min(u - v for u, v in zip(y, q)))
    if perimeter > 2:
        return -np.inf
    return float(np.sum(r))


def repaired(z):
    a = np.column_stack((z[:N], z[N:2*N], z[2*N:3*N])).copy()
    r = a[:, 2]
    if np.any(r <= 0):
        return None
    d = np.sqrt((a[I, 0] - a[J, 0])**2 + (a[I, 1] - a[J, 1])**2)
    factor = min(1.0, float(np.min(d / (r[I] + r[J]))))
    a[:, 2] *= factor * (1.0 - 1e-14)
    lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
    hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
    a[:, :2] -= lo
    scale = min(1.0, 2.0 / float(np.sum(hi - lo))) * (1.0 - 1e-14)
    a *= scale
    return a


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a = np.array([[(k % 5 + .5) / 5, (k // 5 + .5) / 5, .045] for k in range(N)])
        best = {'circles': a.tolist()}
    else:
        a = np.asarray(init['circles'], dtype=float).copy()
        best = init
    best_score = exact_score(a)
    lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
    hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
    scale = 2.0 / np.sum(hi - lo)
    p = (a[:, :2] - lo) * scale
    r = a[:, 2] * scale
    root = np.r_[p[:, 0], p[:, 1], r, (hi[0] - lo[0]) * scale]
    rows = np.arange(M)
    k = np.arange(N)
    A = np.zeros((M + 4*N, 3*N + 1))
    b = np.zeros(M + 4*N)
    A[rows, 2*N + I] = 1
    A[rows, 2*N + J] = 1
    for side in range(4):
        rr = M + side*N + k
        A[rr, 2*N + k] = 1
        if side == 0:
            A[rr, k] = -1
        elif side == 1:
            A[rr, k] = 1
            A[rr, -1] = -1
        elif side == 2:
            A[rr, N+k] = -1
        else:
            A[rr, N+k] = 1
            A[rr, -1] = 1
            b[rr] = 2
    trial = 0
    while time.monotonic() < end:
        z = root.copy()
        # Bias smooth spatial regions rather than isolated contacts.
        x = root[:N]
        y = root[N:2*N]
        angle = rng.uniform(0, 2*np.pi)
        freq = rng.uniform(3.0, 10.0)
        phase = rng.uniform(0, 2*np.pi)
        if trial % 3 == 0:
            field = np.cos(freq*x + phase) + .7*np.cos(freq*(y - (2-root[-1])/2))
        elif trial % 3 == 1:
            field = np.cos(freq*(np.cos(angle)*x + np.sin(angle)*y) + phase)
        else:
            field = rng.normal(size=N)
        field -= field.mean()
        field /= max(np.max(np.abs(field)), 1e-12)
        amplitude = float(params['tilt']) * (1.0 + .35*(trial % 3))
        biased = int(params['biased_steps'])
        total = biased + int(params['relax_steps'])
        previous_sum = -np.inf
        for step in range(total):
            now = time.monotonic()
            if now >= end:
                break
            dx = z[I] - z[J]
            dy = z[N+I] - z[N+J]
            dist = np.maximum(np.sqrt(dx*dx + dy*dy), 1e-15)
            ux, uy = dx/dist, dy/dist
            A[rows, I] = -ux
            A[rows, J] = ux
            A[rows, N+I] = -uy
            A[rows, N+J] = uy
            tilt = amplitude if step < biased else 0.0
            cost = np.zeros(3*N+1)
            cost[2*N:3*N] = -(1 + tilt*field)
            trust = float(params['trust']) * (1.0 if step < biased else .7)
            lower = z - trust
            upper = z + trust
            lower[:2*N] = np.maximum(lower[:2*N], 0)
            upper[:2*N] = np.minimum(upper[:2*N], 2)
            lower[2*N:3*N] = np.maximum(lower[2*N:3*N], float(params['radius_floor']))
            upper[2*N:3*N] = np.minimum(upper[2*N:3*N], .6)
            lower[-1] = max(lower[-1], .15)
            upper[-1] = min(upper[-1], 1.85)
            if np.any(lower > upper):
                break
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', OptimizeWarning)
                result = linprog(cost, A_ub=A, b_ub=b, bounds=np.column_stack((lower, upper)),
                                 method='highs-ds', options={'threads': 1, 'primal_feasibility_tolerance': 1e-9,
                                 'dual_feasibility_tolerance': 1e-9, 'time_limit': max(.001, end-time.monotonic())})
            if not result.success:
                break
            z = result.x
            current_sum = float(np.sum(z[2*N:3*N]))
            if current_sum > best_score - 1e-8:
                candidate = repaired(z)
                if candidate is not None:
                    score = exact_score(candidate)
                    if score > best_score:
                        best_score = score
                        best = {'circles': candidate.tolist()}
            if step >= biased and abs(current_sum - previous_sum) < 2e-12:
                break
            previous_sum = current_sum
        trial += 1
    return best
