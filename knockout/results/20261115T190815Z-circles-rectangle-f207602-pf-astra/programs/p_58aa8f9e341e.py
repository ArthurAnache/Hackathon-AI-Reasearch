import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'lift': (0.105, 0.025, 0.22),
    'middle': (0.30, 0.1, 0.7),
    'contact_band': (0.015, 0.001, 0.05),
    'iterations': (85, 30, 160),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class StopSearch(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.69 * budget_s
    rng = np.random.default_rng(seed)

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def exact_score(a):
        if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -np.inf
        d = np.sqrt((a[I, 0] - a[J, 0])**2 + (a[I, 1] - a[J, 1])**2)
        if np.any(d < a[I, 2] + a[J, 2]):
            return -np.inf
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        perimeter = (max(u + v for u, v in zip(x, r)) - min(u - v for u, v in zip(x, r))
                     + max(u + v for u, v in zip(y, r)) - min(u - v for u, v in zip(y, r)))
        return float(np.sum(a[:, 2])) if perimeter <= 2 else -np.inf

    fallback = np.column_stack(((np.arange(N) % 7 + 0.5) * 0.2,
                                (np.arange(N) // 7 + 0.5) * 0.2,
                                np.full(N, 0.1 - 1e-14)))
    best = {'circles': fallback.tolist()}
    best_score = exact_score(fallback)
    a = fallback.copy()
    if init is not None:
        incoming = np.asarray(init['circles'], dtype=float)
        score = exact_score(incoming)
        if score >= best_score:
            best, best_score = init, score
        if incoming.shape == (N, 3) and np.isfinite(incoming).all():
            a = incoming.copy()

    lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
    sides = np.max(a[:, :2] + a[:, 2, None], axis=0) - lo
    scale = 2.0 / np.sum(sides)
    a[:, :2] = (a[:, :2] - lo) * scale
    a[:, 2] *= scale
    base = np.r_[a[:, 0], a[:, 1], a[:, 2], sides[0] * scale]

    # The last 84 inequalities confine the projected disks to a rectangle.
    B = np.zeros((4 * N, 3 * N + 1))
    k = np.arange(N)
    B[k, k] = 1
    B[k, 2 * N + k] = -1
    B[N + k, k] = -1
    B[N + k, 2 * N + k] = -1
    B[N + k, -1] = 1
    B[2 * N + k, N + k] = 1
    B[2 * N + k, 2 * N + k] = -1
    B[3 * N + k, N + k] = -1
    B[3 * N + k, 2 * N + k] = -1
    B[3 * N + k, -1] = -1
    offset = np.r_[np.zeros(3 * N), np.full(N, 2.0)]
    objective_gradient = np.r_[np.zeros(2 * N), -np.ones(N), 0.0]
    bounds = [(None, None)] * (2 * N) + [(1e-7, 0.6)] * N + [(0.15, 1.85)]

    def objective(z):
        check()
        return -float(np.sum(z[2 * N:3 * N])), objective_gradient

    def consider(z):
        nonlocal best, best_score
        check()
        p = np.column_stack((z[:N], z[N:2 * N]))
        r = np.maximum(z[2 * N:3 * N], 1e-12).copy()
        if not np.isfinite(p).all() or not np.isfinite(r).all():
            return
        d = np.sqrt((p[I, 0] - p[J, 0])**2 + (p[I, 1] - p[J, 1])**2)
        factor = min(1.0, float(np.min(d / (r[I] + r[J]))))
        r *= max(0.0, factor) * (1 - 2e-14)
        if np.any(r <= 0):
            return
        low = np.min(p - r[:, None], axis=0)
        width = np.max(p + r[:, None], axis=0) - low
        s = min(1.0, 2.0 / float(np.sum(width))) * (1 - 2e-14)
        candidate = np.column_stack(((p - low) * s, r * s))
        score = exact_score(candidate)
        if score > best_score:
            best_score = score
            best = {'circles': candidate.tolist()}

    try:
        check()
        distances = np.sqrt((a[I, 0] - a[J, 0])**2 + (a[I, 1] - a[J, 1])**2)
        gap = distances - a[I, 2] - a[J, 2]
        weights = np.exp(-np.maximum(gap, 0) / float(params['contact_band']))
        adjacency = np.zeros((N, N))
        adjacency[I, J] = weights
        adjacency[J, I] = weights
        laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
        _, eigenvectors = np.linalg.eigh(laplacian)
        attempt = 0
        while time.monotonic() < deadline:
            check()
            # Alternating contact-graph modes preferentially lift touching pairs apart.
            coefficients = rng.normal(size=5)
            heights = eigenvectors[:, -5:] @ coefficients
            heights -= heights.mean()
            heights /= max(float(np.max(np.abs(heights))), 1e-12)
            height_difference = heights[I] - heights[J]
            z = base.copy()
            amplitude = float(params['lift']) * (1.0 if attempt == 0 else rng.uniform(0.65, 1.45))
            stages = (1.0, float(params['middle']), 0.0)
            for stage in stages:
                check()
                lift_squared = (stage * amplitude * height_difference)**2
                cache_z = None
                cache_value = None
                cache_jac = None

                def constraints(v):
                    nonlocal cache_z, cache_value, cache_jac
                    check()
                    if cache_z is not None and np.array_equal(v, cache_z):
                        return cache_value
                    dx = v[I] - v[J]
                    dy = v[N + I] - v[N + J]
                    rr = v[2 * N + I] + v[2 * N + J]
                    # Lifted sphere nonintersection, planar disk nonintersection at stage zero.
                    pair = dx * dx + dy * dy + lift_squared - rr * rr
                    jac = np.zeros((M + 4 * N, 3 * N + 1))
                    rows = np.arange(M)
                    jac[rows, I] = 2 * dx
                    jac[rows, J] = -2 * dx
                    jac[rows, N + I] = 2 * dy
                    jac[rows, N + J] = -2 * dy
                    jac[rows, 2 * N + I] = -2 * rr
                    jac[rows, 2 * N + J] = -2 * rr
                    jac[M:] = B
                    cache_z = v.copy()
                    cache_value = np.r_[pair, B @ v + offset]
                    cache_jac = jac
                    return cache_value

                def jacobian(v):
                    constraints(v)
                    return cache_jac

                last = [z.copy()]
                def callback(v):
                    check()
                    last[0] = v.copy()
                    if stage == 0.0 and -objective_gradient @ v > best_score:
                        # Only pay for exact certification near planar feasibility.
                        if float(np.min(constraints(v))) > -1e-8:
                            consider(v)

                remaining = deadline - time.monotonic()
                if remaining < 0.025:
                    raise StopSearch()
                result = minimize(objective, z, jac=True, method='SLSQP', bounds=bounds,
                                  constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                                  callback=callback,
                                  options={'maxiter': int(params['iterations']) if stage else 150,
                                           'ftol': 2e-12, 'disp': False})
                z = result.x
                if stage == 0.0:
                    consider(z)
            attempt += 1
    except StopSearch:
        pass
    return best
