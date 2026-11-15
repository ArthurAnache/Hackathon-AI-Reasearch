import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {
    'angle': (0.85, 0.2, 2.5),
    'cluster_size': (3, 3, 6),
    'iterations': (70, 15, 150),
    'trial_fraction': (0.22, 0.1, 0.35),
}
N = 21
II, JJ = np.triu_indices(N, 1)

class StopSearch(Exception):
    pass


def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[II, 0] - a[JJ, 0]) ** 2 + (a[II, 1] - a[JJ, 1]) ** 2)
    if np.any(d < a[II, 2] + a[JJ, 2]):
        return -np.inf
    x = list(map(lambda v: Fraction(float(v)), a[:, 0]))
    y = list(map(lambda v: Fraction(float(v)), a[:, 1]))
    r = list(map(lambda v: Fraction(float(v)), a[:, 2]))
    perimeter = (max(map(lambda p: p[0] + p[1], zip(x, r)))
                 - min(map(lambda p: p[0] - p[1], zip(x, r)))
                 + max(map(lambda p: p[0] + p[1], zip(y, r)))
                 - min(map(lambda p: p[0] - p[1], zip(y, r))))
    if perimeter > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    fallback = np.array([[(i % 5) * 0.18, (i // 5) * 0.18, 0.04] for i in range(21)])
    best = {'circles': fallback.tolist()}
    best_score = exact_score(fallback)
    if init is not None:
        original = np.asarray(init['circles'], dtype=float)
        score = exact_score(original)
        if score >= best_score:
            best, best_score = init, score
    else:
        original = fallback.copy()
    if time.monotonic() >= end:
        return best

    lo = np.min(original[:, :2] - original[:, 2, None], axis=0)
    hi = np.max(original[:, :2] + original[:, 2, None], axis=0)
    base = original.copy()
    base[:, :2] -= (lo + hi) / 2
    side = hi - lo
    base *= 2.0 / np.sum(side)
    side *= 2.0 / np.sum(side)

    # Infer the reflection axis and the quotient's three fixed circles.
    chosen = None
    for swap in (False, True):
        if time.monotonic() >= end:
            return best
        a = base.copy()
        if swap:
            a[:, :2] = a[:, [1, 0]]
        order = np.argsort(a[:, 1])
        lower, singles, upper = order[:9], order[9:12], order[12:]
        costs = ((a[upper, None, 0] - a[None, lower, 0]) ** 2
                 + (a[upper, None, 1] + a[None, lower, 1]) ** 2
                 + (a[upper, None, 2] - a[None, lower, 2]) ** 2)
        rows, cols = linear_sum_assignment(costs)
        err = float(costs[rows, cols].sum() + np.sum(a[singles, 1] ** 2))
        if chosen is None or err < chosen[0]:
            chosen = (err, a, singles, upper[rows], lower[cols], swap)
    _, a, singles, upper, lower, swap = chosen

    # Variables: 12 representative x coordinates, 12 radii,
    # nine positive y coordinates, and rectangle width.
    dim = 34
    X = np.zeros((N, dim))
    Y = np.zeros((N, dim))
    R = np.zeros((N, dim))
    z = np.zeros(dim)
    for g in range(12):
        if time.monotonic() >= end:
            return best
        if g < 3:
            ids = [g]
            z[g] = a[singles[g], 0]
            z[12 + g] = a[singles[g], 2]
        else:
            k = g - 3
            ids = [3 + 2 * k, 4 + 2 * k]
            z[g] = (a[upper[k], 0] + a[lower[k], 0]) / 2
            z[12 + g] = (a[upper[k], 2] + a[lower[k], 2]) / 2
            z[24 + k] = (a[upper[k], 1] - a[lower[k], 1]) / 2
            Y[ids[0], 24 + k] = 1
            Y[ids[1], 24 + k] = -1
        X[ids, g] = 1
        R[ids, 12 + g] = 1
    z[33] = side[1 if swap else 0]
    objective = -R.sum(axis=0)

    # Remove constraints duplicated by reflection.
    dx, dy, sr = X[II] - X[JJ], Y[II] - Y[JJ], R[II] + R[JJ]
    seen, keep = set(), []
    for k in range(len(II)):
        if time.monotonic() >= end:
            return best
        parts = []
        for v in (dx[k], dy[k], sr[k]):
            if time.monotonic() >= end:
                return best
            nz = np.flatnonzero(v)
            parts.append(tuple(v if len(nz) == 0 or v[nz[0]] > 0 else -v))
        key = tuple(parts)
        if key not in seen:
            seen.add(key)
            keep.append(k)
    dx, dy, sr = dx[keep], dy[keep], sr[keep]
    wall = np.vstack((X - R, -X - R, Y - R, -Y - R))
    wall[:42, 33] += 0.5
    wall[42:, 33] -= 0.5
    offset = np.r_[np.zeros(42), np.ones(42)]
    unique = np.unique(np.column_stack((offset, wall)), axis=0)
    offset, wall = unique[:, 0], unique[:, 1:]
    bounds = [(-1.0, 1.0)] * 12 + [(1e-7, 0.5)] * 12 + [(0.0, 1.0)] * 9 + [(0.1, 1.9)]

    def consider(v):
        nonlocal best, best_score
        if time.monotonic() >= end:
            return
        radii = R @ v
        if np.sum(radii) <= best_score or np.any(radii <= 0):
            return
        out = np.column_stack((X @ v, Y @ v, radii))
        if not np.isfinite(out).all():
            return
        d = np.sqrt((out[II, 0] - out[JJ, 0]) ** 2 + (out[II, 1] - out[JJ, 1]) ** 2)
        ratio = min(1.0, float(np.min(d / (out[II, 2] + out[JJ, 2]))))
        out[:, 2] *= ratio * (1.0 - 3e-15)
        sides = np.max(out[:, :2] + out[:, 2, None], axis=0) - np.min(out[:, :2] - out[:, 2, None], axis=0)
        scale = min(1.0, 2.0 / float(sides.sum()))
        out *= scale * (1.0 - 3e-15)
        score = exact_score(out)
        if score > best_score:
            best_score = score
            best = {'circles': out.tolist()}

    trial_end = end
    last = z.copy()
    feasible_seed = z.copy()

    def check():
        if time.monotonic() >= trial_end:
            raise StopSearch()

    def fun(v):
        check()
        return float(objective @ v)

    def jac(v):
        check()
        return objective

    def cons(v):
        check()
        xx, yy, rr = dx @ v, dy @ v, sr @ v
        return np.r_[xx * xx + yy * yy - rr * rr, offset + wall @ v]

    def cjac(v):
        check()
        xx, yy, rr = dx @ v, dy @ v, sr @ v
        return np.vstack((2 * (xx[:, None] * dx + yy[:, None] * dy - rr[:, None] * sr), wall))

    def callback(v):
        nonlocal last
        check()
        last = v.copy()
        consider(v)

    constraints = {'type': 'ineq', 'fun': cons, 'jac': cjac}
    consider(z)
    trial = 0
    while time.monotonic() < end - 0.015:
        now = time.monotonic()
        if trial == 0:
            v = z.copy()
            allowance = 0.16 * budget_s
        else:
            v = feasible_seed.copy()
            count = int(params['cluster_size'])
            points = np.column_stack((v[3:12], v[24:33]))
            anchor = int(rng.integers(9))
            distances = np.sum((points - points[anchor]) ** 2, axis=1)
            # Alternating local and dispersed surgeries explore different graphs.
            if trial % 2:
                selection = np.argsort(distances + rng.uniform(0, 0.035, 9))[:count]
            else:
                selection = rng.choice(9, count, replace=False)
            cluster = points[selection]
            center = np.mean(cluster, axis=0)
            angle = float(params['angle']) * rng.uniform(0.65, 1.5) * rng.choice([-1, 1])
            co, si = np.cos(angle), np.sin(angle)
            rotated = (cluster - center) @ np.array([[co, -si], [si, co]]) + center
            half_w = v[33] / 2
            half_h = (2 - v[33]) / 2
            v[3 + selection] = np.clip(rotated[:, 0], -half_w + 0.02, half_w - 0.02)
            v[24 + selection] = np.clip(np.abs(rotated[:, 1]), 0.015, half_h - 0.02)
            v[15 + selection] = v[15 + np.roll(selection, 1)]
            allowance = float(params['trial_fraction']) * budget_s
        trial_end = min(end, now + allowance)
        last = v.copy()
        try:
            result = minimize(fun, v, jac=jac, bounds=bounds,
                              constraints=constraints, method='SLSQP', callback=callback,
                              options={'maxiter': int(params['iterations']), 'ftol': 2e-13, 'disp': False})
            last = result.x.copy()
            consider(last)
        except StopSearch:
            pass
        if trial == 0:
            # Refinement establishes a clean symmetric base without committing
            # the retained answer to the symmetry assumption.
            if time.monotonic() < end:
                xx, yy, rr = dx @ last, dy @ last, sr @ last
                violation = min(float(np.min(xx * xx + yy * yy - rr * rr)),
                                float(np.min(offset + wall @ last)))
                if violation > -1e-8:
                    feasible_seed = last.copy()
        trial += 1
    return best
