import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'trials': (5, 1, 12),
    'iterations': (75, 20, 180),
    'radius_reset': (0.82, 0.5, 1.0),
    'shear': (0.12, 0.0, 0.35),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class Deadline(Exception):
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
    perimeter = (max(u + v for u, v in zip(x, q)) - min(u - v for u, v in zip(x, q))
                 + max(u + v for u, v in zip(y, q)) - min(u - v for u, v in zip(y, q)))
    if perimeter > 2:
        return -np.inf
    return float(np.sum(r))


def repaired(z):
    p = np.column_stack((z[:N], z[N:2*N])).copy()
    r = np.maximum(z[2*N:3*N], 1e-10).copy()
    d = np.sqrt((p[I, 0] - p[J, 0]) ** 2 + (p[I, 1] - p[J, 1]) ** 2)
    factor = min(1.0, float(np.min(d / (r[I] + r[J]))))
    r *= factor * (1.0 - 3e-14)
    lo = np.min(p - r[:, None], axis=0)
    hi = np.max(p + r[:, None], axis=0)
    scale = (2.0 - 2e-13) / float(np.sum(hi - lo))
    return np.column_stack(((p - lo) * scale, r * scale))


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a = np.array([[(i % 7 + 0.5) / 5, (i // 7 + 0.5) / 5, 0.099999999999]
                      for i in range(N)], dtype=float)
        best = {'circles': a.tolist()}
    else:
        best = init
        a = np.asarray(init['circles'], dtype=float).copy()
    best_score = exact_score(a)
    lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
    side = np.max(a[:, :2] + a[:, 2, None], axis=0) - lo
    factor = 2.0 / side.sum()
    base = a.copy()
    base[:, :2] = (base[:, :2] - lo) * factor
    base[:, 2] *= factor
    side *= factor
    grad = np.r_[np.zeros(2*N), -np.ones(N), 0.0]

    def pack(b, width):
        return np.r_[b[:, 0], b[:, 1], b[:, 2], width]

    def check(deadline):
        if time.monotonic() >= deadline:
            raise Deadline()

    for trial in range(int(params['trials'])):
        if time.monotonic() >= end - 0.045:
            break
        b = base.copy()
        axis = trial % 2
        transverse = 1 - axis
        # Move the exterior seam through the interior of the packing.
        # Halfway cuts and largest centre gaps give substantially different outer faces.
        if trial < 2:
            shift = 0.5 * side[axis]
        else:
            centres = np.sort(base[:, axis])
            gaps = np.diff(centres)
            order = np.argsort(gaps)[::-1]
            index = int(order[min((trial - 2) // 2, len(order) - 1)])
            seam = 0.5 * (centres[index] + centres[index + 1])
            shift = side[axis] - seam
        old = b[:, axis].copy()
        b[:, axis] = np.mod(old + shift, side[axis])
        moved = old + shift >= side[axis]
        if trial >= 2:
            # A flipped transported strip changes contacts across the new seam.
            b[moved, transverse] = side[transverse] - b[moved, transverse]
        if trial >= 4:
            b[:, transverse] = np.mod(b[:, transverse] + float(params['shear']) * side[transverse]
                                           * np.sin(2*np.pi*b[:, axis]/side[axis]), side[transverse])
        b[:, 2] *= float(params['radius_reset'])
        z = pack(b, side[0])
        remaining = end - time.monotonic()
        deadline = min(end - 0.018, time.monotonic() + max(0.11, remaining / max(1, int(params['trials']) - trial)))
        last = [z.copy()]
        cache = [None, None, None]

        def compute(v):
            check(deadline)
            if cache[0] is not None and np.array_equal(v, cache[0]):
                return cache[1], cache[2]
            x, y, r, w = v[:N], v[N:2*N], v[2*N:3*N], v[-1]
            dx, dy = x[I] - x[J], y[I] - y[J]
            dist = np.sqrt(dx*dx + dy*dy + 1e-28)
            g = np.r_[dist-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r]
            jac = np.zeros((M+4*N, 3*N+1))
            rows = np.arange(M)
            jac[rows, I] = dx/dist
            jac[rows, J] = -dx/dist
            jac[rows, N+I] = dy/dist
            jac[rows, N+J] = -dy/dist
            jac[rows, 2*N+I] = -1
            jac[rows, 2*N+J] = -1
            ids = np.arange(N)
            jac[M+ids, ids] = 1
            jac[M+N+ids, ids] = -1
            jac[M+2*N+ids, N+ids] = 1
            jac[M+3*N+ids, N+ids] = -1
            jac[M+ids, 2*N+ids] = -1
            jac[M+N+ids, 2*N+ids] = -1
            jac[M+2*N+ids, 2*N+ids] = -1
            jac[M+3*N+ids, 2*N+ids] = -1
            jac[M+N:M+2*N, -1] = 1
            jac[M+3*N:M+4*N, -1] = -1
            cache[:] = [v.copy(), g, jac]
            return g, jac

        def objective(v):
            check(deadline)
            return -float(np.sum(v[2*N:3*N]))

        def callback(v):
            check(deadline)
            last[0] = v.copy()

        try:
            result = minimize(objective, z, jac=lambda v: grad, method='SLSQP',
                              bounds=[(-1.0, 3.0)]*(2*N) + [(1e-7, 0.7)]*N + [(0.2, 1.8)],
                              constraints={'type': 'ineq', 'fun': lambda v: compute(v)[0],
                                           'jac': lambda v: compute(v)[1]},
                              callback=callback,
                              options={'maxiter': int(params['iterations']), 'ftol': 2e-11, 'disp': False})
            last[0] = result.x
        except Deadline:
            pass
        if time.monotonic() >= end:
            break
        candidate = repaired(last[0])
        score = exact_score(candidate)
        if score > best_score:
            best_score = score
            best = {'circles': candidate.tolist()}
    return best
