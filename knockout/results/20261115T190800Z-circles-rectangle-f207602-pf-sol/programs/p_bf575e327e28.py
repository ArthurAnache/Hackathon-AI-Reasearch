import time
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog

PARAMS = {
    "fraction": (0.70, 0.50, 0.75),
    "perturbation": (0.009, 0.0001, 0.08, "log"),
    "tilt": (0.018, 0.0001, 0.15, "log"),
    "steps": (9, 4, 20),
}


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + params["fraction"] * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    best = init
    best_score = -float("inf")

    def exact_score(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float("inf")
        d = a[ii, :2] - a[jj, :2]
        dist = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
        if np.any(dist < a[ii, 2] + a[jj, 2]):
            return -float("inf")
        xs = [Fraction(float(v)) for v in a[:, 0]]
        ys = [Fraction(float(v)) for v in a[:, 1]]
        rs = [Fraction(float(v)) for v in a[:, 2]]
        w = max(x+r for x, r in zip(xs, rs)) - min(x-r for x, r in zip(xs, rs))
        h = max(y+r for y, r in zip(ys, rs)) - min(y-r for y, r in zip(ys, rs))
        if w + h > 2:
            return -float("inf")
        return float(np.sum(a[:, 2]))

    if init is not None:
        initial = np.array(init["circles"], dtype=float)
        best_score = exact_score(initial)
    else:
        initial = np.array([[(k % 7 + 0.5)*0.2, (k//7 + 0.5)*0.2, 0.1-1e-14] for k in range(21)])

    def repair(a):
        a = np.array(a, dtype=float, copy=True)
        if not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return None
        delta = a[ii, :2] - a[jj, :2]
        distances = np.sqrt(delta[:, 0]**2 + delta[:, 1]**2)
        factor = min(1.0, float(np.min(distances / (a[ii, 2] + a[jj, 2]))))
        a[:, 2] *= factor * (1.0 - 7e-16)
        xs = [Fraction(float(v)) for v in a[:, 0]]
        ys = [Fraction(float(v)) for v in a[:, 1]]
        rs = [Fraction(float(v)) for v in a[:, 2]]
        total = (max(x+r for x,r in zip(xs,rs))-min(x-r for x,r in zip(xs,rs))
                 + max(y+r for y,r in zip(ys,rs))-min(y-r for y,r in zip(ys,rs)))
        if total > 2:
            a[:, 2] -= float((total-2)/4) + 3e-16
        return a

    def consider(a):
        nonlocal best, best_score
        a = repair(a)
        if a is None or float(np.sum(a[:, 2])) <= best_score:
            return
        score = exact_score(a)
        if score > best_score:
            best_score = score
            best = {"circles": a.tolist()}

    consider(initial)

    # Variables are x[21], y[21], radius[21], and rectangle width.
    A = np.zeros((84 + m, 64))
    b = np.zeros(84 + m)
    ids = np.arange(n)
    A[ids, ids] = -1
    A[ids, 42+ids] = 1
    A[21+ids, ids] = 1
    A[21+ids, 42+ids] = 1
    A[21+ids, 63] = -1
    A[42+ids, 21+ids] = -1
    A[42+ids, 42+ids] = 1
    A[63+ids, 21+ids] = 1
    A[63+ids, 42+ids] = 1
    A[63+ids, 63] = 1
    b[63:84] = 2
    rows = 84 + np.arange(m)
    A[rows, 42+ii] = 1
    A[rows, 42+jj] = 1
    objective = np.zeros(64)
    objective[42:63] = -1
    bounds = [(0, 2)]*42 + [(1e-6, 1)]*21 + [(0.15, 1.85)]

    def normalized(a):
        a = a.copy()
        a[:, 0] -= np.min(a[:, 0]-a[:, 2])
        a[:, 1] -= np.min(a[:, 1]-a[:, 2])
        return a

    anchor = normalized(initial)
    restart = 0
    while time.monotonic() < deadline - 0.035:
        if best is not None:
            anchor = normalized(np.array(best["circles"], dtype=float))
        current = anchor.copy()
        height = float(np.max(current[:, 1]+current[:, 2]))
        width = float(np.max(current[:, 0]+current[:, 2]))
        trial_obj = objective.copy()
        if restart:
            reflected_y = current[:, :2].copy()
            reflected_y[:, 1] = height - reflected_y[:, 1]
            reflected_x = current[:, :2].copy()
            reflected_x[:, 0] = width - reflected_x[:, 0]
            dy = ((reflected_y[:, None, :] - current[None, :, :2])**2).sum(axis=2)
            dx = ((reflected_x[:, None, :] - current[None, :, :2])**2).sum(axis=2)
            axis = 1 if np.min(dy, axis=1).sum() < np.min(dx, axis=1).sum() else 0
            mate = np.argmin(dy if axis == 1 else dx, axis=1)
            noise = rng.normal(size=(21, 2))
            mirrored = noise[mate].copy()
            mirrored[:, axis] *= -1
            noise = (noise + mirrored) * 0.5
            amplitude = params["perturbation"] * (0.35, 1.0, 2.5, 0.08)[(restart-1) % 4]
            current[:, :2] += amplitude * noise
            if restart % 3 != 0:
                tilt = rng.normal(size=21)
                tilt = (tilt + tilt[mate])*0.5
                trial_obj[42:63] -= params["tilt"] * tilt
        previous = -1.0
        for step in range(params["steps"]):
            if time.monotonic() >= deadline - 0.035:
                break
            diff = current[ii, :2] - current[jj, :2]
            lengths = np.sqrt(np.sum(diff*diff, axis=1))
            normals = diff / np.maximum(lengths[:, None], 1e-30)
            A[rows, ii] = -normals[:, 0]
            A[rows, jj] = normals[:, 0]
            A[rows, 21+ii] = -normals[:, 1]
            A[rows, 21+jj] = normals[:, 1]
            c = trial_obj if restart and step < 2 else objective
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs-ds",
                                 options={"threads": 1, "primal_feasibility_tolerance": 1e-10,
                                          "dual_feasibility_tolerance": 1e-10,
                                          "time_limit": max(0.001, deadline-time.monotonic()-0.015)})
            if not result.success:
                break
            z = result.x
            current = np.column_stack((z[:21], z[21:42], z[42:63]))
            consider(current)
            value = float(np.sum(current[:, 2]))
            if step >= 3 and abs(value-previous) < 2e-13:
                break
            previous = value
        restart += 1
    return best if best is not None else {"circles": initial.tolist()}
