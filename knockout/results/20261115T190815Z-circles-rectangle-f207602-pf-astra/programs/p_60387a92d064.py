import time
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog

PARAMS = {
    'trust': (0.095, 0.025, 0.22),
    'opening': (0.085, 0.015, 0.18),
    'groups': (3, 2, 3),
    'attempts': (8, 2, 20),
    'radius_floor': (0.002, 0.0001, 0.02, 'log'),
}

N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
K = np.arange(N)
R = np.arange(M)


def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0] - a[J, 0]) ** 2 + (a[I, 1] - a[J, 1]) ** 2)
    if np.any(d < a[I, 2] + a[J, 2]):
        return -np.inf
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    w = max(u + v for u, v in zip(x, r)) - min(u - v for u, v in zip(x, r))
    h = max(u + v for u, v in zip(y, r)) - min(u - v for u, v in zip(y, r))
    return float(np.sum(a[:, 2])) if w + h <= 2 else -np.inf


def normalize(a):
    r = a[:, 2].copy()
    lo = np.min(a[:, :2] - r[:, None], axis=0)
    hi = np.max(a[:, :2] + r[:, None], axis=0)
    s = 2.0 / np.sum(hi - lo)
    p = (a[:, :2] - (lo + hi) * 0.5) * s
    return np.r_[p[:, 0], p[:, 1], r * s, (hi[0] - lo[0]) * s]


def repair(z):
    a = np.column_stack((z[:21], z[21:42], z[42:63])).copy()
    if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return None
    d = np.sqrt((a[I, 0] - a[J, 0]) ** 2 + (a[I, 1] - a[J, 1]) ** 2)
    f = min(1.0, float(np.min(d / (a[I, 2] + a[J, 2]))))
    if f <= 0:
        return None
    a[:, 2] *= f * (1.0 - 2e-14)
    lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
    hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
    a[:, :2] -= (hi + lo) * 0.5
    a *= min(1.0, 2.0 / np.sum(hi - lo)) * (1.0 - 2e-14)
    return a


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    ids = np.arange(21)
    fallback = np.column_stack(((ids % 7 - 3) * 0.2, (ids // 7 - 1) * 0.2, np.full(21, 0.1 - 1e-13)))
    best = {'circles': fallback.tolist()}
    best_score = exact_score(fallback)
    anchor = normalize(fallback)
    if init is not None:
        try:
            a = np.asarray(init['circles'], dtype=float)
            score = exact_score(a)
            if score >= best_score:
                best, best_score = init, score
                anchor = normalize(a)
        except (ValueError, TypeError, KeyError, OverflowError):
            pass
    best_z = anchor.copy()

    def consider(z):
        nonlocal best, best_score, best_z
        if time.monotonic() >= end:
            return
        a = repair(z)
        if a is None or np.sum(a[:, 2]) <= best_score:
            return
        score = exact_score(a)
        if score > best_score:
            best, best_score, best_z = {'circles': a.tolist()}, score, normalize(a)

    base = np.zeros((M + 84, 64))
    base[R, 42 + I] = 1.0
    base[R, 42 + J] = 1.0
    base[M + K, K] = 1.0
    base[M + K, 42 + K] = 1.0
    base[M + K, 63] = -0.5
    base[M + 21 + K, K] = -1.0
    base[M + 21 + K, 42 + K] = 1.0
    base[M + 21 + K, 63] = -0.5
    base[M + 42 + K, 21 + K] = 1.0
    base[M + 42 + K, 42 + K] = 1.0
    base[M + 42 + K, 63] = 0.5
    base[M + 63 + K, 21 + K] = -1.0
    base[M + 63 + K, 42 + K] = 1.0
    base[M + 63 + K, 63] = 0.5
    rhs = np.zeros(M + 84)
    rhs[M + 42:] = 1.0
    objective = np.r_[np.zeros(42), -np.ones(21), 0.0]
    floor = float(params['radius_floor'])
    dual = np.ones(M + 84)

    def matrix(z):
        dx = z[I] - z[J]
        dy = z[21 + I] - z[21 + J]
        dist = np.maximum(np.sqrt(dx * dx + dy * dy), 1e-15)
        a = base.copy()
        a[R, I] = -dx / dist
        a[R, J] = dx / dist
        a[R, 21 + I] = -dy / dist
        a[R, 21 + J] = dy / dist
        return a

    def step(z, gaps, stop, trust):
        nonlocal dual
        remaining = min(stop, end) - time.monotonic()
        if remaining < 0.015:
            return None
        bounds = np.empty((64, 2))
        bounds[:42, 0] = z[:42] - trust
        bounds[:42, 1] = z[:42] + trust
        bounds[42:63, 0] = floor
        bounds[42:63, 1] = 0.45
        bounds[63] = [0.35, 1.65]
        result = linprog(objective, A_ub=matrix(z), b_ub=rhs - gaps,
                         bounds=bounds, method='highs', options={
                             'time_limit': max(0.001, remaining - 0.008),
                             'primal_feasibility_tolerance': 1e-9,
                             'dual_feasibility_tolerance': 1e-9,
                         })
        if not result.success or result.x is None:
            return None
        dual = np.maximum(0.0, -result.ineqlin.marginals)
        consider(result.x)
        return result.x

    zero = np.zeros(M + 84)
    z = anchor.copy()
    polish_end = min(end, start + 0.16 * budget_s)
    for _ in range(4):
        if time.monotonic() >= polish_end - 0.015:
            break
        trial = step(z, zero, polish_end, float(params['trust']))
        if trial is None:
            break
        change = np.max(np.abs(trial - z))
        z = trial
        if change < 1e-9:
            break
    anchor = best_z.copy()
    a0 = matrix(anchor)
    active = np.flatnonzero(rhs - a0 @ anchor < 2e-6)
    if len(active) < 3 or time.monotonic() >= end - 0.025:
        return best

    # Infer the incumbent's reflection rather than imposing a new layout.
    p = np.column_stack((anchor[:21], anchor[21:42]))
    permutations = []
    errors = []
    for axis in range(2):
        if time.monotonic() >= end:
            return best
        reflected = p.copy()
        reflected[:, axis] *= -1
        distances = np.sum((reflected[:, None, :] - p[None, :, :]) ** 2, axis=2)
        distances += (anchor[42:63, None] - anchor[None, 42:63]) ** 2
        perm = np.argmin(distances, axis=1)
        permutations.append(perm)
        errors.append(float(np.sum(distances[K, perm])))
    axis = int(np.argmin(errors))
    perm = permutations[axis]
    lookup = np.full((21, 21), -1, dtype=int)
    lookup[I, J] = R
    lookup[J, I] = R
    active_set = set(map(int, active))
    groups = []
    centers = []
    costs = []
    used = set()
    for row in active:
        if time.monotonic() >= end:
            return best
        row = int(row)
        if row in used:
            continue
        if row < M:
            partner = int(lookup[perm[I[row]], perm[J[row]]])
            center = (p[I[row]] + p[J[row]]) * 0.5
        else:
            wall, circle = divmod(row - M, 21)
            reflected_wall = wall ^ 1 if wall // 2 == axis else wall
            partner = M + reflected_wall * 21 + int(perm[circle])
            center = p[circle].copy()
        group = [row]
        if partner in active_set and partner != row:
            group.append(partner)
        used.update(group)
        groups.append(group)
        centers.append(center)
        costs.append(float(np.sum(dual[group])))
    centers = np.asarray(centers)
    costs = np.asarray(costs)
    trust = float(params['trust'])
    opening = float(params['opening'])

    for attempt in range(int(params['attempts'])):
        if time.monotonic() >= end - 0.045:
            break
        remaining = end - time.monotonic()
        stop = min(end, time.monotonic() + max(0.17, remaining * 0.42))
        # Nearby groups produce a coordinated local rearrangement; later trials
        # distribute the opened contacts across the packing.
        weights = 1.0 / (0.08 + costs)
        weights /= np.sum(weights)
        root = int(rng.choice(len(groups), p=weights))
        spatial = np.linalg.norm(centers - centers[root], axis=1)
        ranking = spatial + 0.09 * rng.random(len(groups)) + 0.025 * costs
        if attempt % 3 == 2:
            ranking = rng.random(len(groups)) + 0.12 * costs
        ranking[root] = -1
        selected = np.argsort(ranking)[:int(params['groups'])]
        selected_rows = np.concatenate([np.asarray(groups[g], dtype=int) for g in selected])
        z = anchor.copy()
        excursion_end = time.monotonic() + 0.43 * (stop - time.monotonic())
        amplitude = opening * (0.65 + 0.35 * (attempt % 3))
        failed = False
        for stage in range(3):
            if time.monotonic() >= excursion_end - 0.015:
                break
            gaps = zero.copy()
            gaps[selected_rows] = amplitude * (stage + 1) / 3.0
            for _ in range(2):
                if time.monotonic() >= excursion_end - 0.015:
                    break
                trial = step(z, gaps, excursion_end, trust)
                if trial is None:
                    failed = True
                    break
                z = trial
            if failed:
                break
        stalled = 0
        for _ in range(28):
            if time.monotonic() >= stop - 0.015:
                break
            old_value = float(np.sum(z[42:63]))
            trial = step(z, zero, stop, trust)
            if trial is None:
                break
            gain = float(np.sum(trial[42:63])) - old_value
            z = trial
            stalled = stalled + 1 if gain < 2e-11 else 0
            if stalled >= 3:
                break
        if best_score > float(np.sum(anchor[42:63])) + 1e-10:
            anchor = best_z.copy()
    return best
