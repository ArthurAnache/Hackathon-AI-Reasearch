import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    "search_fraction": (0.70, 0.50, 0.78),
    "ftol": (2e-11, 1e-13, 1e-7, "log"),
    "shake": (0.018, 0.002, 0.07, "log"),
    "max_iterations": (180, 40, 400),
}


def construct(params: dict, seed: int, budget_s: float, init=None):
    start = time.monotonic()
    deadline = start + float(params["search_fraction"]) * budget_s
    final_deadline = start + 0.92 * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    k = np.arange(21)
    rows = np.arange(len(ii))
    best = None
    best_score = -float("inf")
    best_array = None

    class Finished(Exception):
        pass

    def check():
        if time.monotonic() >= deadline:
            raise Finished()

    def exact_score(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float("inf")
        d = a[ii, :2] - a[jj, :2]
        dist = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
        if np.any(dist < a[ii, 2] + a[jj, 2]):
            return -float("inf")
        xl, xh, yl, yh = [], [], [], []
        for x, y, r in a:
            if time.monotonic() >= final_deadline:
                return -float("inf")
            x, y, r = Fraction(float(x)), Fraction(float(y)), Fraction(float(r))
            xl.append(x-r)
            xh.append(x+r)
            yl.append(y-r)
            yh.append(y+r)
        if max(xh)-min(xl)+max(yh)-min(yl) > Fraction(2):
            return -float("inf")
        return float(np.sum(a[:, 2]))

    def consider(a, repair=True, original=None):
        nonlocal best, best_score, best_array
        if time.monotonic() >= final_deadline:
            return
        a = np.asarray(a, dtype=float).copy()
        if a.shape != (21, 3) or not np.isfinite(a).all():
            return
        if repair:
            a[:, 2] = np.maximum(a[:, 2], 1e-10)
            delta = a[ii, :2] - a[jj, :2]
            dist = np.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2)
            factor = min(1.0, float(np.min(dist / (a[ii, 2]+a[jj, 2]))))
            if factor <= 0:
                return
            a[:, 2] *= factor * (1.0-8e-15)
            lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
            a[:, :2] -= lo
            extent = np.max(a[:, :2]+a[:, 2, None], axis=0)
            perimeter = float(np.sum(extent))
            if perimeter <= 0:
                return
            a *= (2.0/perimeter)*(1.0-8e-15)
        approximate = float(np.sum(a[:, 2]))
        if approximate <= best_score:
            return
        score = exact_score(a)
        if score > best_score:
            best_score = score
            best_array = a.copy()
            best = original if original is not None else {"circles": a.tolist()}

    supplied = None
    if init is not None:
        try:
            supplied = np.asarray(init["circles"], dtype=float)
            consider(supplied, False, init)
            consider(supplied)
        except (ValueError, TypeError, KeyError, OverflowError):
            supplied = None

    fallback = np.array([[0.1+0.2*(i % 5), 0.1+0.2*(i // 5), 0.09999999999999] for i in range(21)])
    consider(fallback)
    if supplied is None or supplied.shape != (21, 3) or not np.isfinite(supplied).all():
        supplied = best_array.copy()

    def vectorize(a):
        a = a.copy()
        a[:, 2] = np.maximum(a[:, 2], 1e-5)
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        extent = hi-lo
        scale = 2.0/max(float(np.sum(extent)), 1e-8)
        a[:, :2] = (a[:, :2]-lo)*scale
        a[:, 2] *= scale
        return np.concatenate((a.ravel(), [float(extent[0]*scale)]))

    objective_gradient = np.zeros(64)
    objective_gradient[2:63:3] = -1.0

    def objective(z):
        check()
        return -float(np.sum(z[2:63:3]))

    def gradient(z):
        check()
        return objective_gradient

    cached_z = None
    cached_c = None
    cached_j = None

    def calculate(z):
        nonlocal cached_z, cached_c, cached_j
        check()
        if cached_z is not None and np.array_equal(z, cached_z):
            return
        a = z[:63].reshape(21, 3)
        x, y, r = a.T
        dx, dy = x[ii]-x[jj], y[ii]-y[jj]
        d = np.sqrt(dx*dx+dy*dy)
        safe = np.maximum(d, 1e-15)
        c = np.concatenate((d-r[ii]-r[jj], x-r, z[63]-x-r, y-r, 2-z[63]-y-r))
        jac = np.zeros((294, 64))
        jac[rows, 3*ii] = dx/safe
        jac[rows, 3*jj] = -dx/safe
        jac[rows, 3*ii+1] = dy/safe
        jac[rows, 3*jj+1] = -dy/safe
        jac[rows, 3*ii+2] = -1
        jac[rows, 3*jj+2] = -1
        jac[210+k, 3*k] = 1
        jac[210+k, 3*k+2] = -1
        jac[231+k, 3*k] = -1
        jac[231+k, 3*k+2] = -1
        jac[231+k, 63] = 1
        jac[252+k, 3*k+1] = 1
        jac[252+k, 3*k+2] = -1
        jac[273+k, 3*k+1] = -1
        jac[273+k, 3*k+2] = -1
        jac[273+k, 63] = -1
        cached_z, cached_c, cached_j = z.copy(), c, jac

    def constraints(z):
        calculate(z)
        return cached_c

    def jacobian(z):
        calculate(z)
        return cached_j

    callback_count = 0
    latest = None

    def callback(z):
        nonlocal callback_count, latest
        check()
        latest = z.copy()
        callback_count += 1
        if callback_count % 3 == 0:
            consider(z[:63].reshape(21, 3))

    bounds = [(0.0, 2.0), (0.0, 2.0), (1e-6, 0.7)]*21 + [(0.05, 1.95)]
    attempt = 0
    current = supplied.copy()
    while time.monotonic() < deadline:
        z = vectorize(current)
        latest = z.copy()
        cached_z = None
        try:
            result = minimize(objective, z, jac=gradient, method="SLSQP", bounds=bounds,
                              constraints={"type": "ineq", "fun": constraints, "jac": jacobian},
                              callback=callback,
                              options={"maxiter": int(params["max_iterations"]), "ftol": float(params["ftol"]), "disp": False})
            consider(result.x[:63].reshape(21, 3))
        except Finished:
            if latest is not None:
                consider(latest[:63].reshape(21, 3))
            break
        attempt += 1
        if time.monotonic() >= deadline or best_array is None:
            break
        current = best_array.copy()
        amplitude = float(params["shake"]) * (1.0 if attempt % 3 else 2.0)
        displacement = rng.normal(0.0, amplitude, (21, 2))
        displacement -= np.mean(displacement, axis=0)
        current[:, :2] += displacement
        current[:, 2] *= 0.97
    return best
