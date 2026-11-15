import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"ftol": (2e-13, 1e-14, 1e-8, "log"), "iterations": (180, 30, 350)}


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    best = init
    best_score = -float('inf')

    def exact_score(obj):
        if obj is None:
            return -float('inf')
        try:
            if 'circles' in obj:
                a = np.asarray(obj['circles'], dtype=float)
            else:
                z = np.asarray(obj['state'], dtype=float)
                a = np.column_stack((z[:21], z[21:42], z[42:63]))
            if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
                return -float('inf')
            for i in range(21):
                if time.monotonic() > deadline:
                    return -float('inf')
                for j in range(i):
                    dx = a[i, 0] - a[j, 0]
                    dy = a[i, 1] - a[j, 1]
                    if np.sqrt(dx * dx + dy * dy) < a[i, 2] + a[j, 2]:
                        return -float('inf')
            x = [Fraction(float(v)) for v in a[:, 0]]
            y = [Fraction(float(v)) for v in a[:, 1]]
            r = [Fraction(float(v)) for v in a[:, 2]]
            perimeter = max(u + v for u, v in zip(x, r)) - min(u - v for u, v in zip(x, r))
            perimeter += max(u + v for u, v in zip(y, r)) - min(u - v for u, v in zip(y, r))
            if perimeter > 2:
                return -float('inf')
            return float(np.sum(a[:, 2]))
        except Exception:
            return -float('inf')

    best_score = exact_score(init)

    def keep(a):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        a = np.array(a, dtype=float, copy=True)
        if not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return
        # Repair overlap by reducing radii uniformly before perimeter scaling.
        ratio = 1.0
        for i in range(21):
            if time.monotonic() >= deadline:
                return
            for j in range(i):
                d = np.sqrt(np.sum((a[i, :2] - a[j, :2]) ** 2))
                ratio = min(ratio, d / (a[i, 2] + a[j, 2]))
        a[:, 2] *= max(0.0, ratio * (1.0 - 8e-15))
        lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
        hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
        a[:, :2] -= (lo + hi) / 2
        factor = (2.0 / float(np.sum(hi - lo))) * (1.0 - 8e-15)
        a *= factor
        obj = {'circles': a.tolist()}
        score = exact_score(obj)
        if score > best_score:
            best_score, best = score, obj

    class EndSearch(Exception):
        pass

    def model(rows):
        orbits = []
        initial = []
        circle = 0
        nr = len(rows)
        for row, count in enumerate(rows):
            if time.monotonic() >= deadline:
                raise EndSearch
            yy = (row - (nr - 1) / 2) * np.sqrt(3.0)
            if count % 2:
                orbits.append(([(circle, 0)], len(initial), False))
                initial.extend([yy, 1.0])
                circle += 1
            for k in range(count // 2):
                xx = 2 * k + (2 if count % 2 else 1)
                orbits.append(([(circle, 1), (circle + 1, -1)], len(initial), True))
                initial.extend([float(xx), yy, 1.0])
                circle += 2
        d = len(initial) + 1
        T = np.zeros((64, d))
        lower = np.full(d, -np.inf)
        upper = np.full(d, np.inf)
        for members, p, paired in orbits:
            if time.monotonic() >= deadline:
                raise EndSearch
            for idx, sign in members:
                if paired:
                    T[idx, p] = sign
                    T[21 + idx, p + 1] = 1
                    T[42 + idx, p + 2] = 1
                else:
                    T[21 + idx, p] = 1
                    T[42 + idx, p + 1] = 1
            lower[p + (2 if paired else 1)] = 1e-8
            if paired:
                lower[p] = 0
        T[63, -1] = 1
        lower[-1], upper[-1] = 0.08, 1.92
        initial.append(1.0)
        q = np.array(initial)
        z = T @ q
        a = np.column_stack((z[:21], z[21:42], z[42:63]))
        wh = np.max(a[:, :2] + a[:, 2, None], axis=0) - np.min(a[:, :2] - a[:, 2, None], axis=0)
        scale = 2 / np.sum(wh)
        q[:-1] *= scale
        q[-1] = wh[0] * scale
        ix, jx = np.triu_indices(21, 1)
        DX = T[ix] - T[jx]
        DY = T[21 + ix] - T[21 + jx]
        DR = T[42 + ix] + T[42 + jx]
        # Identify pair constraints which are identical under reflection.
        keys = np.concatenate((np.abs(DX), DY, DR), axis=1)
        _, unique = np.unique(keys, axis=0, return_index=True)
        DX, DY, DR = DX[unique], DY[unique], DR[unique]
        X, Y, R, W = T[:21], T[21:42], T[42:63], T[63]
        B = np.concatenate((W / 2 - X - R, W / 2 + X - R,
                            -W / 2 - Y - R, -W / 2 + Y - R))
        offset = np.concatenate((np.zeros(42), np.ones(42)))
        _, unique = np.unique(np.column_stack((B, offset)), axis=0, return_index=True)
        B, offset = B[unique], offset[unique]
        objective = -np.sum(R, axis=0)
        last_q = None
        last_c = None
        last_j = None

        def compute(q):
            nonlocal last_q, last_c, last_j
            if time.monotonic() >= deadline:
                raise EndSearch
            if last_q is None or not np.array_equal(q, last_q):
                dx, dy, dr = DX @ q, DY @ q, DR @ q
                last_c = np.concatenate((dx * dx + dy * dy - dr * dr, B @ q + offset))
                last_j = np.vstack((2 * (dx[:, None] * DX + dy[:, None] * DY - dr[:, None] * DR), B))
                last_q = q.copy()
            return last_c, last_j

        def unpack(q):
            z = T @ q
            return np.column_stack((z[:21], z[21:42], z[42:63]))

        return q, lower, upper, objective, compute, unpack

    # Always have a robust, reasonably dense candidate even on very short calls.
    fallback = []
    for row in range(7):
        if time.monotonic() >= deadline:
            break
        for col in range(3):
            fallback.append([(col - 1) * 2.0, (row - 3) * 2.0, 1.0])
    if len(fallback) == 21:
        keep(fallback)

    layouts = [[3, 4, 3, 4, 3, 4], [3, 3, 3, 3, 3, 3, 3]]
    for layout in layouts:
        if time.monotonic() >= deadline - 0.04:
            break
        try:
            q, lower, upper, objective, compute, unpack = model(layout)
            keep(unpack(q))
            latest = [q.copy()]
            local_end = min(deadline - 0.025, time.monotonic() + (0.87 * max(0.0, deadline - time.monotonic()) if layout == layouts[0] else 10))

            def callback(v):
                latest[0] = v.copy()
                if time.monotonic() >= local_end:
                    raise EndSearch

            try:
                result = minimize(lambda v: float(objective @ v), q, jac=lambda v: objective,
                                  bounds=list(zip(lower, upper)), method='SLSQP',
                                  constraints={'type': 'ineq', 'fun': lambda v: compute(v)[0],
                                               'jac': lambda v: compute(v)[1]},
                                  callback=callback,
                                  options={'ftol': float(params['ftol']), 'maxiter': int(params['iterations']), 'disp': False})
                q = result.x
            except EndSearch:
                q = latest[0]
            keep(unpack(q))
            # The limiting configuration is commonly isostatic. Solve its
            # contact equations directly to remove nonlinear stopping error.
            if time.monotonic() < deadline - 0.025:
                c, jac = compute(q)
                active = np.flatnonzero(c < 2e-7)
                for iteration in range(5):
                    if time.monotonic() >= deadline - 0.018:
                        break
                    c, jac = compute(q)
                    if len(active) == 0:
                        break
                    delta = np.linalg.lstsq(jac[active], -c[active], rcond=1e-11)[0]
                    if np.max(np.abs(delta)) > 0.02:
                        break
                    q = q + delta
                    keep(unpack(q))
                    if np.max(np.abs(delta)) < 1e-15:
                        break
        except EndSearch:
            break
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
    return best
