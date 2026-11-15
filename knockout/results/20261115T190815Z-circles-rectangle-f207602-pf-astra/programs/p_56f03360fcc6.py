import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'poles': (2500, 200, 6000),
    'angles': (12, 4, 24),
    'starts': (4, 1, 8),
    'iterations': (90, 20, 200),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class StopSearch(Exception):
    pass


def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0] - a[J, 0]) ** 2 + (a[I, 1] - a[J, 1]) ** 2)
    if np.any(d < a[I, 2] + a[J, 2]):
        return -np.inf
    x = list(map(lambda v: Fraction(float(v)), a[:, 0]))
    y = list(map(lambda v: Fraction(float(v)), a[:, 1]))
    r = list(map(lambda v: Fraction(float(v)), a[:, 2]))
    perimeter = (max(u + v for u, v in zip(x, r)) - min(u - v for u, v in zip(x, r))
                 + max(u + v for u, v in zip(y, r)) - min(u - v for u, v in zip(y, r)))
    if perimeter > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))


def repair(a):
    a = np.array(a, dtype=float, copy=True)
    if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return None
    p, r = a[:, :2], a[:, 2]
    d = np.sqrt((p[I, 0] - p[J, 0]) ** 2 + (p[I, 1] - p[J, 1]) ** 2)
    ratio = float(np.min(d / (r[I] + r[J])))
    if ratio <= 0:
        return None
    r *= min(1.0, ratio) * (1.0 - 4e-14)
    lo = np.min(p - r[:, None], axis=0)
    hi = np.max(p + r[:, None], axis=0)
    scale = (2.0 / float(np.sum(hi - lo))) * (1.0 - 4e-14)
    p[:] = (p - lo) * scale
    r *= scale
    return a


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        k = np.arange(N)
        a = np.column_stack(((k % 5 + 0.5) / 5, (k // 5 + 0.5) / 5, np.full(N, 0.095)))
        init = {'circles': a.tolist()}
    original = np.array(init['circles'], dtype=float)
    best = init
    best_score = exact_score(original)

    def consider(a):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        b = repair(a)
        if b is None or float(b[:, 2].sum()) <= best_score:
            return
        score = exact_score(b)
        if score > best_score:
            best_score = score
            best = {'circles': b.tolist()}

    consider(original)
    p = original[:, :2].copy()
    r = original[:, 2].copy()
    lo = np.min(p - r[:, None], axis=0)
    side = np.max(p + r[:, None], axis=0) - lo
    s = 2.0 / float(side.sum())
    p = (p - lo) * s
    r *= s
    side *= s
    if time.monotonic() >= deadline:
        return best

    # Invert around poles lying in uncovered interior gaps. Such inversions
    # change which face of the disk contact graph borders the rectangle.
    poles = rng.uniform(0.0, 1.0, (int(params['poles']), 2)) * side
    delta = p[None, :, :] - poles[:, None, :]
    den = np.sum(delta * delta, axis=2) - r[None, :] ** 2
    keep = np.all(den > 1e-9, axis=1)
    poles, delta, den = poles[keep], delta[keep], den[keep]
    if len(poles) == 0 or time.monotonic() >= deadline:
        return best
    q = delta / den[:, :, None]
    rr = r[None, :] / den
    merits = np.full(len(poles), -np.inf)
    angles = np.zeros(len(poles))
    for angle in np.linspace(0.0, np.pi / 2, int(params['angles']), endpoint=False):
        if time.monotonic() >= deadline:
            return best
        c, sn = np.cos(angle), np.sin(angle)
        xx = q[:, :, 0] * c - q[:, :, 1] * sn
        yy = q[:, :, 0] * sn + q[:, :, 1] * c
        box = (np.max(xx + rr, axis=1) - np.min(xx - rr, axis=1)
               + np.max(yy + rr, axis=1) - np.min(yy - rr, axis=1))
        scores = 2 * rr.sum(axis=1) / box
        improved = scores > merits
        merits[improved] = scores[improved]
        angles[improved] = angle

    seeds = []
    faces = set()
    for idx in np.argsort(-merits):
        if time.monotonic() >= deadline:
            return best
        # The three nearest disk boundaries identify most triangular gaps.
        gap = np.sqrt(np.sum((p - poles[idx]) ** 2, axis=1)) - r
        face = tuple(sorted(np.argsort(gap)[:3].tolist()))
        if face in faces:
            continue
        faces.add(face)
        angle = angles[idx]
        c, sn = np.cos(angle), np.sin(angle)
        rot = np.array([[c, sn], [-sn, c]])
        a = np.column_stack((q[idx] @ rot, rr[idx]))
        a = repair(a)
        if a is not None:
            seeds.append(a)
        if len(seeds) >= int(params['starts']):
            break

    # Variables: all x, all y, all radii, rectangle width; height = 2-width.
    objective_jac = np.r_[np.zeros(42), -np.ones(21), 0.0]
    template = np.zeros((M + 4 * N, 64))
    rows = np.arange(M)
    template[rows, 42 + I] = -1
    template[rows, 42 + J] = -1
    k = np.arange(N)
    template[M + k, k] = 1
    template[M + k, 42 + k] = -1
    template[M + N + k, k] = -1
    template[M + N + k, 42 + k] = -1
    template[M + N + k, 63] = 1
    template[M + 2 * N + k, N + k] = 1
    template[M + 2 * N + k, 42 + k] = -1
    template[M + 3 * N + k, N + k] = -1
    template[M + 3 * N + k, 42 + k] = -1
    template[M + 3 * N + k, 63] = -1
    bounds = [(0.0, 2.0)] * 42 + [(1e-8, 0.99)] * 21 + [(0.02, 1.98)]

    for number, a in enumerate(seeds):
        now = time.monotonic()
        if now >= deadline - 0.025:
            break
        remaining = len(seeds) - number
        stage_end = now + (deadline - now - 0.015) / remaining
        low = np.min(a[:, :2] - a[:, 2, None], axis=0)
        high = np.max(a[:, :2] + a[:, 2, None], axis=0)
        a[:, :2] -= low
        z0 = np.r_[a[:, 0], a[:, 1], a[:, 2], high[0] - low[0]]
        latest = [z0.copy()]

        def check():
            if time.monotonic() >= stage_end:
                raise StopSearch()

        def fun(z):
            check()
            return -float(z[42:63].sum())

        def jac(z):
            check()
            return objective_jac

        def constraints(z):
            check()
            x, y, rad, w = z[:21], z[21:42], z[42:63], z[63]
            dx, dy = x[I] - x[J], y[I] - y[J]
            dist = np.sqrt(dx * dx + dy * dy)
            return np.r_[dist - rad[I] - rad[J], x - rad, w - x - rad,
                         y - rad, 2 - w - y - rad]

        def constraint_jac(z):
            check()
            dx = z[I] - z[J]
            dy = z[21 + I] - z[21 + J]
            dist = np.maximum(np.sqrt(dx * dx + dy * dy), 1e-20)
            mat = template.copy()
            mat[rows, I] = dx / dist
            mat[rows, J] = -dx / dist
            mat[rows, 21 + I] = dy / dist
            mat[rows, 21 + J] = -dy / dist
            return mat

        def callback(z):
            check()
            latest[0] = z.copy()
            if float(z[42:63].sum()) > best_score:
                consider(np.column_stack((z[:21], z[21:42], z[42:63])))

        try:
            result = minimize(fun, z0, jac=jac, method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': constraints, 'jac': constraint_jac},
                              callback=callback,
                              options={'maxiter': int(params['iterations']), 'ftol': 2e-12, 'disp': False})
            latest[0] = result.x
        except StopSearch:
            pass
        if time.monotonic() < deadline:
            z = latest[0]
            consider(np.column_stack((z[:21], z[21:42], z[42:63])))
    return best
