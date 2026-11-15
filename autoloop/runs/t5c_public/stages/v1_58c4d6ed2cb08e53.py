def refine(v, score, budget_s, rng):
    import time
    import numpy as np
    from scipy.optimize import minimize

    start = time.monotonic()
    original = np.asarray(v, dtype=np.float64).reshape(63).copy()
    if budget_s < 0.08:
        return original
    deadline = start + max(0.0, budget_s - max(0.025, 0.035 * budget_s))
    best = original.copy()
    best_value = float(score(best))
    n = 21
    ia, ib = np.triu_indices(n, 1)
    m = len(ia)
    rows = np.arange(m)

    class Finished(Exception):
        pass

    def check_time():
        if time.monotonic() >= deadline:
            raise Finished()

    def encode(vec):
        c = np.asarray(vec).reshape(n, 3).copy()
        lo = np.min(c[:, :2] - c[:, 2, None], axis=0)
        hi = np.max(c[:, :2] + c[:, 2, None], axis=0)
        c[:, :2] -= (lo + hi) * 0.5
        wh = hi - lo
        # Divide any unused perimeter equally between the two dimensions.
        w = wh[0] + 0.5 * max(0.0, 2.0 - wh.sum())
        return np.r_[c.ravel(), np.clip(w, 0.00001, 1.99999)]

    def consider(z):
        nonlocal best, best_value
        check_time()
        c = np.asarray(z[:63], dtype=float).reshape(n, 3).copy()
        if not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return
        # Repair with strict slack, using exactly the verifier's distance formula.
        dx = c[ia, 0] - c[ib, 0]
        dy = c[ia, 1] - c[ib, 1]
        d = np.sqrt(dx * dx + dy * dy)
        ratios = d / (c[ia, 2] + c[ib, 2])
        shrink = min(1.0, float(np.min(ratios)))
        if shrink <= 0:
            return
        c[:, 2] *= shrink * (1.0 - 1.5e-13)
        lo = np.min(c[:, :2] - c[:, 2, None], axis=0)
        hi = np.max(c[:, :2] + c[:, 2, None], axis=0)
        c[:, :2] -= 0.5 * (lo + hi)
        t = c.astype(np.longdouble)
        perimeter = np.sum(np.max(t[:, :2] + t[:, 2, None], axis=0) -
                           np.min(t[:, :2] - t[:, 2, None], axis=0))
        if perimeter <= 0:
            return
        c *= float(np.longdouble(2.0) / perimeter) * (1.0 - 1.5e-13)
        if -float(np.sum(c[:, 2])) >= best_value:
            return
        check_time()
        candidate = c.ravel().copy()
        value = float(score(candidate))
        if value < best_value:
            best_value = value
            best = candidate

    objective_jac = np.zeros(64)
    objective_jac[2:63:3] = -1.0

    def fun(z):
        check_time()
        return -np.sum(z[2:63:3])

    def jac(z):
        return objective_jac

    def constraints(z):
        check_time()
        c = z[:63].reshape(n, 3)
        x, y, r = c.T
        w = z[63]
        dx = x[ia] - x[ib]
        dy = y[ia] - y[ib]
        dist = np.sqrt(dx * dx + dy * dy)
        return np.r_[dist - r[ia] - r[ib],
                     0.5*w + x - r, 0.5*w - x - r,
                     1.0 - 0.5*w + y - r, 1.0 - 0.5*w - y - r]

    wall_jac = np.zeros((4*n, 64))
    k = np.arange(n)
    wall_jac[k, 3*k] = 1.0
    wall_jac[n+k, 3*k] = -1.0
    wall_jac[2*n+k, 3*k+1] = 1.0
    wall_jac[3*n+k, 3*k+1] = -1.0
    for block in range(4):
        wall_jac[block*n+k, 3*k+2] = -1.0
    wall_jac[:2*n, 63] = 0.5
    wall_jac[2*n:, 63] = -0.5

    def constraint_jac(z):
        check_time()
        c = z[:63].reshape(n, 3)
        dx = c[ia, 0] - c[ib, 0]
        dy = c[ia, 1] - c[ib, 1]
        dist = np.maximum(np.sqrt(dx*dx + dy*dy), 1e-30)
        a = np.zeros((m + 4*n, 64))
        a[rows, 3*ia] = dx / dist
        a[rows, 3*ib] = -dx / dist
        a[rows, 3*ia+1] = dy / dist
        a[rows, 3*ib+1] = -dy / dist
        a[rows, 3*ia+2] = -1.0
        a[rows, 3*ib+2] = -1.0
        a[m:] = wall_jac
        return a

    bounds = [(-1.0, 1.0), (-1.0, 1.0), (1e-9, 0.5)] * n + [(1e-6, 2.0-1e-6)]
    cons = {'type': 'ineq', 'fun': constraints, 'jac': constraint_jac}
    attempt = 0
    last = None
    try:
        consider(encode(best))
        while time.monotonic() < deadline:
            z = encode(best)
            if attempt:
                # A mixture of localized contact changes and collective rearrangements.
                if last is not None and rng.random() < 0.18:
                    z = last.copy()
                c = z[:63].reshape(n, 3)
                amplitude = (0.0015, 0.004, 0.009, 0.018, 0.035, 0.065)[(attempt-1) % 6]
                if rng.random() < 0.65:
                    count = int(rng.integers(2, 9)) if hasattr(rng, 'integers') else int(rng.randint(2, 9))
                    ids = rng.choice(n, count, replace=False)
                else:
                    ids = np.arange(n)
                c[ids, :2] += rng.normal(size=(len(ids), 2)) * amplitude
                c[ids, 2] *= np.exp(rng.normal(size=len(ids)) * min(0.45, amplitude * 9.0))
                c[:, 2] = np.clip(c[:, 2], 1e-8, 0.499999)
                z[63] = np.clip(z[63] + rng.normal() * amplitude, 0.02, 1.98)
            iterations = [0]

            def callback(current):
                check_time()
                iterations[0] += 1
                if iterations[0] % 4 == 0:
                    consider(current)

            result = minimize(fun, z, jac=jac, method='SLSQP', bounds=bounds,
                              constraints=cons, callback=callback,
                              options={'maxiter': 220, 'ftol': 2e-13, 'disp': False})
            last = result.x.copy()
            consider(result.x)
            attempt += 1
    except Finished:
        pass
    except (ValueError, FloatingPointError, ArithmeticError):
        pass
    return best
