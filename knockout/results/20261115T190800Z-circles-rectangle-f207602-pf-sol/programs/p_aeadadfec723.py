import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment
from scipy.linalg import lstsq

PARAMS = {
    'initial_fraction': (0.57, 0.35, 0.8),
    'iterations': (220, 60, 400),
    'safety': (2e-14, 4e-15, 1e-10, 'log'),
    'perturbation': (0.018, 0.002, 0.12)
}
I, J = np.triu_indices(21, 1)

class StopSearch(Exception):
    pass

def valid(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return False
    d = np.sqrt((a[I, 0] - a[J, 0]) ** 2 + (a[I, 1] - a[J, 1]) ** 2)
    if np.any(d < a[I, 2] + a[J, 2]):
        return False
    x = list(map(Fraction, map(float, a[:, 0])))
    y = list(map(Fraction, map(float, a[:, 1])))
    r = list(map(Fraction, map(float, a[:, 2])))
    return (max(u + v for u, v in zip(x, r)) - min(u - v for u, v in zip(x, r)) +
            max(u + v for u, v in zip(y, r)) - min(u - v for u, v in zip(y, r))) <= 2

def repair(a, eps):
    a = np.asarray(a, dtype=float).copy()
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return None
    distances = np.sqrt((a[I, 0] - a[J, 0]) ** 2 + (a[I, 1] - a[J, 1]) ** 2)
    ratio = min(1., float(np.min(distances / (a[I, 2] + a[J, 2]))))
    if ratio <= 0:
        return None
    a[:, 2] *= ratio * (1 - eps)
    lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
    hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
    scale = (2 - 8 * eps) / np.sum(hi - lo)
    a[:, :2] = (a[:, :2] - (lo + hi) / 2) * scale
    a[:, 2] *= scale
    return a if valid(a) else None

def make_model(symmetric):
    if symmetric:
        n = 34
        ids = np.r_[np.arange(12), np.arange(3, 12)]
        signs = np.r_[np.ones(12), -np.ones(9)]
        x = np.zeros((21, n)); y = x.copy(); r = x.copy()
        x[np.arange(21), ids] = 1
        mask = ids >= 3
        y[np.arange(21)[mask], 12 + ids[mask] - 3] = signs[mask]
        r[np.arange(21), 21 + ids] = 1
        pairs = [(i, j) for i in range(12) for j in range(i + 1, 12)]
        pairs += [(i, 12 + j - 3) for i in range(3, 12) for j in range(i, 12)]
        pi, pj = np.array(pairs).T
        boundary = np.arange(12)
        radius_indices = np.arange(21, 33)
    else:
        n = 64
        x = np.zeros((21, n)); y = x.copy(); r = x.copy()
        x[np.arange(21), np.arange(21)] = 1
        y[np.arange(21), 21 + np.arange(21)] = 1
        r[np.arange(21), 42 + np.arange(21)] = 1
        pi, pj = I, J
        boundary = np.arange(21)
        radius_indices = np.arange(42, 63)
    width = np.zeros(n); width[-1] = 1
    b = np.concatenate((x[boundary] - r[boundary], width - x[boundary] - r[boundary],
                        1e0 * (-.5 * width - y[boundary] - r[boundary]),
                        -.5 * width + y[boundary] - r[boundary]))
    offset = np.r_[np.zeros(2 * len(boundary)), np.ones(2 * len(boundary))]
    return x, y, r, x[pi] - x[pj], y[pi] - y[pj], r[pi] + r[pj], b, offset, -r.sum(axis=0), radius_indices

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + .70 * budget_s
    rng = np.random.default_rng(seed)
    eps = float(params['safety'])
    best = None
    bestscore = -np.inf
    original = None
    try:
        if init is not None:
            if 'circles' in init:
                original = np.asarray(init['circles'], dtype=float)
            else:
                original = np.asarray(init['state'], dtype=float).reshape(3, 21).T
            if valid(original):
                best = init
                bestscore = float(np.sum(original[:, 2]))
    except (ValueError, TypeError, KeyError):
        original = None

    def offer(a):
        nonlocal best, bestscore
        b = repair(a, eps)
        if b is not None:
            score = float(np.sum(b[:, 2]))
            if score > bestscore:
                bestscore = score
                best = {'state': b.T.reshape(-1).tolist()}
        return b

    if original is not None:
        offer(original)
    fallback = []
    for row, count in enumerate((4, 5, 4, 5, 3)):
        if time.monotonic() >= deadline:
            break
        shift = .5 if count == 4 else 0.
        for col in range(count):
            if time.monotonic() >= deadline:
                break
            fallback.append([2 * (col + shift), np.sqrt(3.) * row, 1.])
    if len(fallback) == 21:
        offer(np.asarray(fallback))
    if best is None:
        return {'circles': [[(i % 3) * .2, (i // 3) * .2, .099999999999] for i in range(21)]}

    def best_array():
        if 'circles' in best:
            return np.asarray(best['circles'], dtype=float)
        return np.asarray(best['state'], dtype=float).reshape(3, 21).T.copy()

    full = make_model(False)
    symmetric = make_model(True)

    def unpack(z, model):
        return np.column_stack((model[0] @ z, model[1] @ z, model[2] @ z))

    def full_state(a):
        lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
        hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
        p = a[:, :2] - lo
        p[:, 1] -= .5 * (hi[1] - lo[1])
        return np.r_[p[:, 0], p[:, 1], a[:, 2], hi[0] - lo[0]]

    def optimize(z, model, stop, maxiter):
        x, y, r, dxmat, dymat, rsmat, boundary, offset, objective, ridx = model
        initial_rsum = rsmat @ z
        weights = 1. / np.maximum(2 * initial_rsum, .08)
        cache = [None, None, None]
        last = [z.copy()]
        counter = [0]
        def calc(v):
            if time.monotonic() >= stop:
                raise StopSearch()
            if cache[0] is not None and np.array_equal(v, cache[0]):
                return cache[1], cache[2]
            dx = dxmat @ v; dy = dymat @ v; rs = rsmat @ v
            values = np.r_[(dx * dx + dy * dy - rs * rs) * weights, boundary @ v + offset]
            jac = np.concatenate((2 * weights[:, None] * (dx[:, None] * dxmat + dy[:, None] * dymat - rs[:, None] * rsmat), boundary))
            cache[:] = [v.copy(), values, jac]
            last[0] = v.copy()
            return values, jac
        def fun(v):
            if time.monotonic() >= stop:
                raise StopSearch()
            return float(objective @ v), objective
        def callback(v):
            if time.monotonic() >= stop:
                raise StopSearch()
            counter[0] += 1
            if counter[0] % 12 == 0:
                offer(unpack(v, model))
        bounds = [(None, None)] * len(z)
        for k in ridx:
            if time.monotonic() >= stop:
                return z
            bounds[int(k)] = (1e-7, .5)
        bounds[-1] = (.15, 1.85)
        try:
            result = minimize(fun, z, jac=True, method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': lambda v: calc(v)[0], 'jac': lambda v: calc(v)[1]},
                              callback=callback, options={'maxiter': maxiter, 'ftol': 2e-13, 'disp': False})
            z = result.x
        except StopSearch:
            z = last[0]
        offer(unpack(z, model))
        for _ in range(3):
            if time.monotonic() >= stop - .004:
                break
            try:
                c, jac = calc(z)
            except StopSearch:
                break
            active = np.flatnonzero(np.abs(c) < 1e-7)
            if len(active) < len(z):
                break
            delta, _, rank, _ = lstsq(jac[active], -c[active], lapack_driver='gelsy', check_finite=False)
            if rank < len(z) or np.max(np.abs(delta)) > 1e-4:
                break
            z = z + delta
            offer(unpack(z, model))
        return z

    # Preserve the actual warm-start geometry even when the safety lattice scores higher.
    a = original if original is not None else best_array()
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        a = best_array()
    initial_stop = min(deadline - .025, started + .70 * budget_s * float(params['initial_fraction']))
    z = optimize(full_state(a), full, initial_stop, int(params['iterations']))
    a = best_array()
    if time.monotonic() >= deadline - .035:
        return best

    choices = []
    for swap in (False, True):
        if time.monotonic() >= deadline - .025:
            return best
        b = a.copy()
        if swap:
            b[:, :2] = b[:, [1, 0]]
        lo = np.min(b[:, :2] - b[:, 2, None], axis=0)
        hi = np.max(b[:, :2] + b[:, 2, None], axis=0)
        b[:, 0] -= lo[0]
        b[:, 1] -= .5 * (lo[1] + hi[1])
        reflected = b.copy(); reflected[:, 1] *= -1
        cost = np.sum((b[:, None, :] - reflected[None, :, :]) ** 2, axis=2)
        rows, cols = linear_sum_assignment(cost)
        choices.append((float(cost[rows, cols].sum()), b, hi[0] - lo[0]))
    _, a, width = min(choices, key=lambda item: item[0])
    singles = np.argsort(np.abs(a[:, 1]))[:3]
    mask = np.ones(21, dtype=bool); mask[singles] = False
    remaining = np.flatnonzero(mask)
    ordered = remaining[np.argsort(a[remaining, 1])]
    negative, positive = ordered[:9], ordered[9:]
    cost = ((a[positive, None, 0] - a[None, negative, 0]) ** 2 +
            (a[positive, None, 1] + a[None, negative, 1]) ** 2 +
            (a[positive, None, 2] - a[None, negative, 2]) ** 2)
    _, cols = linear_sum_assignment(cost)
    negative = negative[cols]
    base = np.r_[a[singles, 0], .5 * (a[positive, 0] + a[negative, 0]),
                 .5 * (a[positive, 1] - a[negative, 1]),
                 a[singles, 2], .5 * (a[positive, 2] + a[negative, 2]), width]
    now = time.monotonic()
    base = optimize(base, symmetric, min(deadline - .018, now + .73 * (deadline - now)), int(params['iterations']))
    for attempt in range(8):
        now = time.monotonic()
        if now >= deadline - .035:
            break
        trial = base.copy()
        amplitude = float(params['perturbation']) * (1 + attempt * .5)
        chosen = rng.choice(12, size=4, replace=False)
        for k in chosen:
            if time.monotonic() >= deadline - .025:
                break
            trial[k] += rng.normal(0, amplitude)
            if k >= 3:
                trial[12 + k - 3] = abs(trial[12 + k - 3] + rng.normal(0, amplitude))
        candidate = optimize(trial, symmetric, min(deadline - .012, now + max(.04, .65 * (deadline - now))), 100)
        if float(symmetric[8] @ candidate) < float(symmetric[8] @ base):
            base = candidate
    return best
