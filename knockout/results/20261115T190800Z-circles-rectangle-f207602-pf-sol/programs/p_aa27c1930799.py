import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    "restart_scale": (0.006, 0.00001, 0.08, "log"),
    "maxiter": (100, 10, 300),
    "time_fraction": (0.69, 0.3, 0.72),
}
N = 21
II, JJ = np.triu_indices(N, 1)

class TimeUp(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[II, 0] - a[JJ, 0]) ** 2 + (a[II, 1] - a[JJ, 1]) ** 2)
    if np.any(d < a[II, 2] + a[JJ, 2]):
        return -np.inf
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    p = (max(u + v for u, v in zip(x, r)) - min(u - v for u, v in zip(x, r))
         + max(u + v for u, v in zip(y, r)) - min(u - v for u, v in zip(y, r)))
    if p > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))

def repair(a, deadline):
    a = a.copy()
    if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return None
    d = np.sqrt((a[II, 0] - a[JJ, 0]) ** 2 + (a[II, 1] - a[JJ, 1]) ** 2)
    ratio = float(np.min(d / (a[II, 2] + a[JJ, 2])))
    a[:, 2] *= min(1.0, ratio) * (1.0 - 8e-16)
    for k in range(8):
        if time.monotonic() >= deadline:
            return None
        score = exact_score(a)
        if np.isfinite(score):
            return a
        # The perimeter excess is affine in the radii as long as the
        # four extremal circles do not change.
        r = a[:, 2]
        lo = np.argmin(a[:, :2] - r[:, None], axis=0)
        hi = np.argmax(a[:, :2] + r[:, None], axis=0)
        p = Fraction(0)
        denom = 0.0
        for axis in range(2):
            if time.monotonic() >= deadline:
                return None
            i, j = int(hi[axis]), int(lo[axis])
            p += Fraction(float(a[i, axis])) + Fraction(float(r[i]))
            p -= Fraction(float(a[j, axis])) - Fraction(float(r[j]))
            denom += r[i] + r[j]
        excess = max(0.0, float(p - 2))
        shrink = max(2e-15, excess / max(denom, 1e-30) + 2e-15)
        a[:, 2] *= max(0.1, 1.0 - shrink)
    return a if np.isfinite(exact_score(a)) else None

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + float(params["time_fraction"]) * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf
    if init is not None:
        original = np.asarray(init["circles"], dtype=float)
        best_score = exact_score(original)
        a = original.copy()
    else:
        a = np.array([[(i % 5 + .5) / 5, (i // 5 + .5) / 5, .075] for i in range(N)])
    low = np.min(a[:, :2] - a[:, 2, None], axis=0)
    high = np.max(a[:, :2] + a[:, 2, None], axis=0)
    sides = high - low
    a[:, :2] -= (low + high) / 2

    # Identify the better of the two possible reflection axes.
    def orbit_partition(axis):
        remaining = set(range(N))
        groups = []
        error = 0.0
        while remaining:
            if time.monotonic() >= end:
                raise TimeUp()
            i = min(remaining)
            remaining.remove(i)
            if abs(a[i, axis]) < 2e-6:
                groups.append((i,))
                error = max(error, abs(a[i, axis]))
                continue
            if not remaining:
                return None, np.inf
            target = a[i].copy()
            target[axis] *= -1
            j = min(remaining, key=lambda j: float(np.sum((a[j] - target) ** 2)))
            error = max(error, float(np.max(np.abs(a[j] - target))))
            remaining.remove(j)
            groups.append((i, j))
        return groups, error

    try:
        g0, e0 = orbit_partition(0)
        g1, e1 = orbit_partition(1)
    except TimeUp:
        return best
    axis = 0 if e0 < e1 else 1
    groups, err = (g0, e0) if axis == 0 else (g1, e1)
    symmetric = groups is not None and err < 2e-5
    if symmetric and axis == 0:
        a[:, [0, 1]] = a[:, [1, 0]]
        sides = sides[::-1].copy()
    if not symmetric:
        groups = [(i,) for i in range(N)]

    # Affine, centered-box coordinates. A reflected pair shares x and r.
    d = 3 * N + 1 if not symmetric else 2 * len(groups) + sum(len(g) == 2 for g in groups) + 1
    X = np.zeros((N, d)); Y = X.copy(); R = X.copy()
    z = np.zeros(d)
    k = 0
    for group in groups:
        if time.monotonic() >= end:
            return best
        ids = list(group)
        X[ids, k] = 1
        z[k] = np.mean(a[ids, 0]); k += 1
        R[ids, k] = 1
        z[k] = np.mean(a[ids, 2]); k += 1
        if not symmetric or len(group) == 2:
            Y[group[0], k] = 1
            if len(group) == 2:
                Y[group[1], k] = -1
                z[k] = (a[group[0], 1] - a[group[1], 1]) / 2
            else:
                z[k] = a[group[0], 1]
            k += 1
    z[-1] = sides[0]
    objective = -np.sum(R, axis=0)
    rows = []; constants = []; seen = set()
    for i in range(N):
        if time.monotonic() >= end:
            return best
        for mat, sign, vertical in ((X, 1, False), (X, -1, False), (Y, 1, True), (Y, -1, True)):
            row = sign * mat[i] - R[i]
            row = row.copy()
            row[-1] += -.5 if vertical else .5
            constant = 1.0 if vertical else 0.0
            key = (tuple(row), constant)
            if key not in seen:
                seen.add(key); rows.append(row); constants.append(constant)
        row = R[i].copy()
        key = (tuple(row), -1e-9)
        if key not in seen:
            seen.add(key); rows.append(row); constants.append(-1e-9)
    C = np.asarray(rows); b = np.asarray(constants)
    dxs = []; dys = []; rss = []; seen = set()
    def canonical(v):
        nonzero = np.flatnonzero(v)
        return tuple(-v if len(nonzero) and v[nonzero[0]] < 0 else v)
    for i, j in zip(II, JJ):
        if time.monotonic() >= end:
            return best
        dx, dy, rs = X[i] - X[j], Y[i] - Y[j], R[i] + R[j]
        key = (canonical(dx), canonical(dy), tuple(rs))
        if key not in seen:
            seen.add(key); dxs.append(dx); dys.append(dy); rss.append(rs)
    DX = np.asarray(dxs); DY = np.asarray(dys); RS = np.asarray(rss)
    cache_z = None; cache_v = None; cache_j = None
    def constraints(v):
        nonlocal cache_z, cache_v, cache_j
        if time.monotonic() >= end:
            raise TimeUp()
        if cache_z is None or not np.array_equal(v, cache_z):
            dx = DX @ v; dy = DY @ v
            dist = np.sqrt(dx * dx + dy * dy)
            inv = 1.0 / np.maximum(dist, 1e-30)
            cache_v = np.concatenate((C @ v + b, dist - RS @ v))
            cache_j = np.vstack((C, (dx * inv)[:, None] * DX + (dy * inv)[:, None] * DY - RS))
            cache_z = v.copy()
        return cache_v
    def jacobian(v):
        constraints(v)
        return cache_j
    def decode(v):
        return np.column_stack((X @ v, Y @ v, R @ v))
    bounds = [(None, None)] * d
    bounds[-1] = (.03, 1.97)
    best_z = z.copy()
    attempt = 0
    while time.monotonic() < end:
        trial = best_z.copy()
        if attempt:
            scale = float(params["restart_scale"]) * (0.25, 1.0, 2.5, 5.0, .05)[(attempt - 1) % 5]
            direction = rng.normal(size=d)
            # Change many centers together, keeping radius perturbations smaller.
            radius_columns = np.any(R != 0, axis=0)
            direction[radius_columns] *= .2
            direction[-1] *= .4
            trial += scale * direction
        last = [trial.copy()]
        def callback(v):
            last[0] = v.copy()
            if time.monotonic() >= end:
                raise TimeUp()
        try:
            result = minimize(lambda v: float(objective @ v), trial, jac=lambda v: objective,
                              constraints={"type": "ineq", "fun": constraints, "jac": jacobian},
                              bounds=bounds, method="SLSQP", callback=callback,
                              options={"maxiter": int(params["maxiter"]), "ftol": 2e-14, "disp": False})
            candidate_z = result.x
        except TimeUp:
            break
        candidate = repair(decode(candidate_z), end)
        if candidate is not None:
            score = exact_score(candidate)
            if score > best_score:
                best_score = score
                best = {"circles": candidate.tolist()}
                best_z = candidate_z.copy()
        attempt += 1
    if best is None:
        # Cold-start fallback has generous separation and perimeter slack.
        fallback = [[(i % 5) * .18, (i // 5) * .18, .035] for i in range(N)]
        return {"circles": fallback}
    return best
