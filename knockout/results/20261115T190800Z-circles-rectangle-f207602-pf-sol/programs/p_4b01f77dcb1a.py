import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'iterations': (190, 70, 400),
    'tolerance': (1e-12, 1e-14, 1e-8, 'log'),
    'safety': (3e-14, 1e-14, 1e-10, 'log'),
    'polish_share': (0.19, 0.1, 0.35)
}

I, J = np.triu_indices(21, 1)

class StopSearch(Exception):
    pass


def exact_score(a):
    if a is None or a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    dx = a[I, 0] - a[J, 0]
    dy = a[I, 1] - a[J, 1]
    if np.any(np.sqrt(dx * dx + dy * dy) < a[I, 2] + a[J, 2]):
        return -np.inf
    x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
    extent = max(u + v for u, v in zip(x, r)) - min(u - v for u, v in zip(x, r))
    extent += max(u + v for u, v in zip(y, r)) - min(u - v for u, v in zip(y, r))
    return float(np.sum(a[:, 2])) if extent <= 2 else -np.inf


def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + .70 * budget_s
    safety = float(params['safety'])
    best_obj, best_a, best_score = None, None, -np.inf

    def offer(a, repair=True):
        nonlocal best_obj, best_a, best_score
        a = np.asarray(a, dtype=float).copy()
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return
        if repair:
            delta = a[I, :2] - a[J, :2]
            distance = np.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2)
            ratio = min(1., float(np.min(distance / (a[I, 2] + a[J, 2]))))
            if ratio <= 0:
                return
            a[:, 2] *= ratio * (1 - safety)
            lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
            hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
            scale = (2 - 16 * safety) / float(np.sum(hi - lo))
            a[:, :2] = (a[:, :2] - (hi + lo) / 2) * scale
            a[:, 2] *= scale
        if float(np.sum(a[:, 2])) > best_score:
            score = exact_score(a)
            if score > best_score:
                best_score, best_a = score, a
                best_obj = {'state': a.T.ravel().tolist()}

    if init is not None:
        try:
            original = np.asarray(init['circles'], float) if 'circles' in init else np.asarray(init['state'], float).reshape(3, 21).T
            value = exact_score(original)
            if value > best_score:
                best_score, best_a, best_obj = value, original.copy(), init
            offer(original)
        except (KeyError, ValueError, TypeError):
            pass
    offer(np.array([[2 * (k % 3), 2 * (k // 3), 1.] for k in range(21)], float))

    # Horizontal reflection: three axis circles and nine upper/lower pairs.
    T = np.zeros((64, 34))
    ids = np.r_[np.arange(12), np.arange(3, 12)]
    T[np.arange(21), ids] = 1
    T[21 + np.arange(3, 12), np.arange(12, 21)] = 1
    T[21 + np.arange(12, 21), np.arange(12, 21)] = -1
    T[42 + np.arange(21), 21 + ids] = 1
    T[63, 33] = 1
    partner = np.r_[np.arange(3), np.arange(12, 21), np.arange(3, 12)]
    keep = []
    for q in range(len(I)):
        if time.monotonic() >= deadline:
            return best_obj
        reflected = tuple(sorted((int(partner[I[q]]), int(partner[J[q]]))))
        if (int(I[q]), int(J[q])) <= reflected:
            keep.append(q)
    keep = np.asarray(keep, dtype=int)

    walls = np.zeros((84, 64))
    k = np.arange(21)
    walls[k, k] = 1
    walls[21 + k, k] = -1
    walls[42 + k, 21 + k] = 1
    walls[63 + k, 21 + k] = -1
    walls[np.arange(84), 42 + np.tile(k, 4)] = -1
    walls[:42, 63] = .5
    walls[42:, 63] = -.5
    offset = np.r_[np.zeros(42), np.ones(42)]
    objective = np.r_[np.zeros(42), -np.ones(21), 0.]

    def state(a):
        lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
        hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
        b = a.copy()
        b[:, :2] -= (hi + lo) / 2
        return np.r_[b.T.ravel(), hi[0] - lo[0]]

    def solve(a, symmetric, until):
        if time.monotonic() >= until - .004:
            return
        full = state(a)
        if symmetric:
            z = np.r_[full[:12], full[24:33], full[42:54], full[63]]
            transform = T
            selected = keep
            # Upper-pair y coordinates are constrained positive, so only
            # their upper wall is needed; axis walls are identical.
            wall_indices = np.r_[np.arange(12), 21 + np.arange(12), 63 + np.arange(12)]
            bounds = [(-1., 1.)] * 12 + [(1e-8, 1.)] * 9 + [(1e-7, .6)] * 12 + [(.04, 1.96)]
        else:
            z = full
            transform = np.eye(64)
            selected = np.arange(len(I))
            wall_indices = np.arange(84)
            bounds = [(-1., 1.)] * 42 + [(1e-7, .6)] * 21 + [(.04, 1.96)]
        ii, jj = I[selected], J[selected]
        X = transform[ii] - transform[jj]
        Y = transform[21 + ii] - transform[21 + jj]
        R = transform[42 + ii] + transform[42 + jj]
        W = walls[wall_indices] @ transform
        off = offset[wall_indices]
        grad = objective @ transform
        cache = [None, None, None]
        last = [z.copy()]
        ticks = [0]

        def check():
            if time.monotonic() >= until:
                raise StopSearch()

        def calc(v):
            check()
            if cache[0] is not None and np.array_equal(cache[0], v):
                return cache[1], cache[2]
            dx, dy = X @ v, Y @ v
            d = np.sqrt(dx * dx + dy * dy)
            safe = np.maximum(d, 1e-20)
            c = np.r_[d - R @ v, W @ v + off]
            jac = np.vstack(((dx / safe)[:, None] * X + (dy / safe)[:, None] * Y - R, W))
            cache[:] = [v.copy(), c, jac]
            return c, jac

        def fun(v):
            check()
            return float(grad @ v), grad

        def callback(v):
            check()
            last[0] = v.copy()
            ticks[0] += 1
            if ticks[0] % 16 == 0:
                offer((transform @ v)[:63].reshape(3, 21).T)

        try:
            result = minimize(fun, z, jac=True, method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': lambda v: calc(v)[0], 'jac': lambda v: calc(v)[1]},
                              callback=callback,
                              options={'maxiter': int(params['iterations']), 'ftol': float(params['tolerance']), 'disp': False})
            last[0] = result.x
        except (StopSearch, ValueError, FloatingPointError):
            pass
        offer((transform @ last[0])[:63].reshape(3, 21).T)

    def horizontal(rows):
        a = np.zeros((21, 3))
        a[:3, 0] = [-2., 0., 2.]
        a[:, 2] = 1.
        cursor = 3
        for count, height, shift in rows:
            if time.monotonic() >= deadline:
                return None
            a[cursor:cursor + count, 0] = 2 * np.arange(count) - count + 1 + shift
            a[cursor:cursor + count, 1] = height
            cursor += count
        a[12:] = a[3:12]
        a[12:, 1] *= -1
        lo = np.min(a[:, :2] - 1, axis=0)
        hi = np.max(a[:, :2] + 1, axis=0)
        a *= 2 / np.sum(hi - lo)
        return a

    def vertical_counts(counts):
        singles, pairs = [], []
        for row, count in enumerate(counts):
            if time.monotonic() >= deadline:
                return None
            x = (row - (len(counts) - 1) / 2) * np.sqrt(3)
            if count % 2:
                singles.append([x, 0., 1.])
            for j in range(count // 2):
                pairs.append([x, float(2 * j + (1 if count % 2 == 0 else 2)), 1.])
        if len(singles) != 3 or len(pairs) != 9:
            return None
        a = np.array(singles + pairs + [[x, -y, r] for x, y, r in pairs])
        lo = np.min(a[:, :2] - 1, axis=0)
        hi = np.max(a[:, :2] + 1, axis=0)
        a *= 2 / np.sum(hi - lo)
        return a

    if best_score > 2.34:
        solve(best_a, False, deadline - .008)
        return best_obj

    root3 = np.sqrt(3)
    seeds = [
        horizontal([(5, 2., 0.), (4, 2. + root3, 0.)]),
        vertical_counts([2, 3, 4, 3, 4, 3, 2]),
        vertical_counts([3, 4, 3, 4, 3, 4]),
        horizontal([(4, root3, 0.), (5, 2 * root3, 0.)])
    ]
    search_end = started + (1 - float(params['polish_share'])) * (deadline - started)
    fractions = [.32, .59, .83, 1.]
    for a, fraction in zip(seeds, fractions):
        if time.monotonic() >= search_end - .006:
            break
        if a is not None:
            offer(a)
            solve(a, True, started + fraction * (search_end - started))
    if best_a is not None and time.monotonic() < deadline - .012:
        solve(best_a, False, deadline - .008)
    return best_obj
