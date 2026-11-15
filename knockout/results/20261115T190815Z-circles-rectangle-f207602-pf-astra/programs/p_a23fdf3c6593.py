import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    "iterations": (100, 30, 200),
    "restart_jitter": (0.025, 0.0, 0.1),
    "time_fraction": (0.70, 0.5, 0.8),
}

class StopSearch(Exception):
    pass


def construct(params: dict, seed: int, budget_s: float, init=None):
    started = time.monotonic()
    deadline = started + float(params["time_fraction"]) * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    rows = np.arange(21)
    best = None
    best_score = -float("inf")
    best_array = None

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def exact_score(a):
        check()
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float("inf")
        dx = a[ii, 0] - a[jj, 0]
        dy = a[ii, 1] - a[jj, 1]
        if np.any(np.sqrt(dx ** 2 + dy ** 2) < a[ii, 2] + a[jj, 2]):
            return -float("inf")
        lx, ux, ly, uy = [], [], [], []
        for x, y, r in a:
            check()
            x, y, r = Fraction(float(x)), Fraction(float(y)), Fraction(float(r))
            lx.append(x-r)
            ux.append(x+r)
            ly.append(y-r)
            uy.append(y+r)
        if max(ux)-min(lx)+max(uy)-min(ly) > Fraction(2):
            return -float("inf")
        return float(np.sum(a[:, 2]))

    def consider(a, original=None):
        nonlocal best, best_score, best_array
        score = exact_score(a)
        if score > best_score:
            best_score = score
            best_array = a.copy()
            best = original if original is not None else {"circles": a.tolist()}

    def repair(a):
        check()
        a = np.asarray(a, dtype=float).copy()
        if a.shape != (21, 3) or not np.isfinite(a).all():
            return
        a[:, 2] = np.maximum(a[:, 2], 1e-10)
        d = np.sqrt((a[ii, 0]-a[jj, 0])**2 + (a[ii, 1]-a[jj, 1])**2)
        shrink = min(1.0, float(np.min(d/(a[ii, 2]+a[jj, 2]))))
        if shrink <= 0:
            return
        a[:, 2] *= shrink * (1.0-4e-14)
        lower = np.min(a[:, :2]-a[:, 2, None], axis=0)
        upper = np.max(a[:, :2]+a[:, 2, None], axis=0)
        span = float(np.sum(upper-lower))
        if span <= 0:
            return
        a[:, :2] -= lower
        a *= (2.0/span) * (1.0-4e-14)
        consider(a)

    def lattice(counts):
        out = []
        for k, n in enumerate(counts):
            check()
            for j in range(n):
                check()
                out.append([2*j-(n-1), k*np.sqrt(3.0), 1.0])
        return np.asarray(out, dtype=float)

    def optimize(a):
        check()
        low = np.min(a[:, :2]-a[:, 2, None], axis=0)
        a = a.copy()
        a[:, :2] -= low
        width = float(np.max(a[:, 0]+a[:, 2]))
        z = np.concatenate((a[:, 0], a[:, 1], a[:, 2], [width]))
        grad = np.zeros(64)
        grad[42:63] = -1.0
        cached_z = None
        cached_c = None
        cached_j = None

        def compute(v):
            nonlocal cached_z, cached_c, cached_j
            check()
            if cached_z is not None and np.array_equal(v, cached_z):
                return
            x, y, r, w = v[:21], v[21:42], v[42:63], v[63]
            dx, dy = x[ii]-x[jj], y[ii]-y[jj]
            dist = np.sqrt(dx*dx+dy*dy)
            safe = np.maximum(dist, 1e-15)
            c = np.concatenate((dist-r[ii]-r[jj], x-r, w-x-r, y-r, 2-w-y-r))
            j = np.zeros((294, 64))
            p = np.arange(210)
            j[p, ii] = dx/safe
            j[p, jj] = -dx/safe
            j[p, 21+ii] = dy/safe
            j[p, 21+jj] = -dy/safe
            j[p, 42+ii] = -1.0
            j[p, 42+jj] = -1.0
            j[210+rows, rows] = 1.0
            j[210+rows, 42+rows] = -1.0
            j[231+rows, rows] = -1.0
            j[231+rows, 42+rows] = -1.0
            j[231+rows, 63] = 1.0
            j[252+rows, 21+rows] = 1.0
            j[252+rows, 42+rows] = -1.0
            j[273+rows, 21+rows] = -1.0
            j[273+rows, 42+rows] = -1.0
            j[273+rows, 63] = -1.0
            cached_z, cached_c, cached_j = v.copy(), c, j

        def fun(v):
            check()
            return -float(np.sum(v[42:63]))

        def jac(v):
            check()
            return grad

        def constraints(v):
            compute(v)
            return cached_c

        def constraint_jac(v):
            compute(v)
            return cached_j

        def callback(v):
            check()
            repair(np.column_stack((v[:21], v[21:42], v[42:63])))

        result = minimize(fun, z, jac=jac, method="SLSQP",
            bounds=[(0.0, 2.0)]*42+[(1e-5, 0.8)]*21+[(0.2, 1.8)],
            constraints={"type": "ineq", "fun": constraints, "jac": constraint_jac},
            callback=callback,
            options={"maxiter": int(params["iterations"]), "ftol": 2e-11, "disp": False})
        callback(result.x)

    try:
        if init is not None:
            try:
                consider(np.asarray(init["circles"], dtype=float), init)
            except (KeyError, ValueError, TypeError):
                pass
        patterns = ((4, 5, 4, 5, 3), (4, 5, 3, 5, 4), (3, 4, 5, 4, 5), (5, 6, 5, 5))
        for counts in patterns:
            check()
            a = lattice(counts)
            a[:, :2] -= np.min(a[:, :2]-a[:, 2, None], axis=0)
            span = np.sum(np.max(a[:, :2]+a[:, 2, None], axis=0))
            a *= (2.0/span)*(1-1e-12)
            repair(a)
            optimize(a)
        while time.monotonic() < deadline:
            check()
            a = best_array.copy()
            a[:, :2] += rng.normal(0, float(params["restart_jitter"]), (21, 2))
            optimize(a)
    except StopSearch:
        pass
    if best is not None:
        return best
    # An explicit safe fallback, independent of any optimizer result.
    k = np.arange(21)
    a = np.column_stack(((k % 7)*0.2+0.1, (k // 7)*0.2+0.1, np.full(21, 0.099999999999)))
    return {"circles": a.tolist()}
