def refine(v, score, budget_s, rng):
    import time
    import numpy as np
    from scipy.optimize import minimize

    start = time.monotonic()
    deadline = start + max(0.0, float(budget_s))
    n = 21
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    rows = np.arange(n)
    best = None
    best_value = float('inf')
    best_sum = -1.0

    def repair(a):
        a = np.asarray(a, dtype=float).reshape(-1)[:63].reshape(n, 3).copy()
        if not np.isfinite(a).all():
            return None
        a[:, 2] = np.maximum(a[:, 2], 1e-12)
        a[:, :2] -= np.mean(a[:, :2], axis=0)
        d = np.hypot(a[ii, 0] - a[jj, 0], a[ii, 1] - a[jj, 1])
        factor = min(1.0, float(np.min(d / (a[ii, 2] + a[jj, 2]))))
        if factor <= 0:
            return None
        a[:, 2] *= factor * (1.0 - 2e-10)
        lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
        hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
        size = float(np.sum(hi - lo))
        if size <= 0:
            return None
        a[:, :2] -= (lo + hi) * 0.5
        a *= (2.0 / size) * (1.0 - 2e-11)
        return a.reshape(-1)

    def consider(a):
        nonlocal best, best_value, best_sum
        c = repair(a)
        if c is None:
            return
        s = float(c[2::3].sum())
        if s <= best_sum:
            return
        val = score(c)
        if np.isfinite(val) and val < best_value:
            best, best_value, best_sum = c.copy(), val, s

    if v is not None:
        vv = np.asarray(v, dtype=float).reshape(-1)
        if len(vv) == 63:
            val = score(vv)
            if np.isfinite(val):
                best, best_value, best_sum = vv.copy(), val, float(vv[2::3].sum())
            consider(vv)

    def lattice(k):
        layouts = [[5, 4, 5, 4, 3], [4, 4, 5, 4, 4],
                   [5, 6, 5, 5], [3, 4, 4, 4, 3, 3], [7, 7, 7]]
        counts = layouts[k % len(layouts)]
        pts = []
        for row, count in enumerate(counts):
            offset = (row % 2) * 1.0
            if k % len(layouts) in (0, 1, 3):
                offset = (max(counts) - count)
            for col in range(count):
                pts.append([2.0 * col + offset, np.sqrt(3.0) * row, 1.0])
        a = np.array(pts, dtype=float)
        if k >= len(layouts):
            a[:, :2] += rng.normal(0, 0.16, (n, 2))
        return repair(a.reshape(-1))

    consider(lattice(0))
    if time.monotonic() >= deadline - 0.015:
        return best

    grad = np.zeros(64)
    grad[2:63:3] = -1.0
    base_jac = np.zeros((4*n + m, 64))
    base_jac[rows, 3*rows] = 1
    base_jac[rows, 3*rows+2] = -1
    base_jac[n+rows, 3*rows] = -1
    base_jac[n+rows, 3*rows+2] = -1
    base_jac[n+rows, 63] = 1
    base_jac[2*n+rows, 3*rows+1] = 1
    base_jac[2*n+rows, 3*rows+2] = -1
    base_jac[3*n+rows, 3*rows+1] = -1
    base_jac[3*n+rows, 3*rows+2] = -1
    base_jac[3*n+rows, 63] = -1
    pr = 4*n + np.arange(m)
    base_jac[pr, 3*ii+2] = -1
    base_jac[pr, 3*jj+2] = -1

    def constraints(z):
        a = z[:63].reshape(n, 3)
        x, y, r = a.T
        w = z[63]
        d = np.hypot(x[ii]-x[jj], y[ii]-y[jj])
        return np.concatenate((x-r, w-x-r, y-r, 2-w-y-r, d-r[ii]-r[jj]))

    def jacobian(z):
        a = z[:63].reshape(n, 3)
        dx = a[ii, 0] - a[jj, 0]
        dy = a[ii, 1] - a[jj, 1]
        d = np.maximum(np.hypot(dx, dy), 1e-30)
        j = base_jac.copy()
        j[pr, 3*ii] = dx/d
        j[pr, 3*jj] = -dx/d
        j[pr, 3*ii+1] = dy/d
        j[pr, 3*jj+1] = -dy/d
        return j

    bounds = [(0.0, 2.0), (0.0, 2.0), (1e-9, 0.65)] * n + [(0.08, 1.92)]
    class OutOfTime(Exception):
        pass

    trial = 0
    while time.monotonic() < deadline - 0.04:
        if trial == 0 or (trial < 8 and trial % 2 == 0) or rng.random() < 0.22:
            seed = lattice(trial // 2 if trial < 8 else int(rng.integers(0, 40)))
        else:
            a = best.reshape(n, 3).copy()
            amplitude = float(rng.choice([0.008, 0.018, 0.04, 0.075]))
            a[:, :2] += rng.normal(0, amplitude, (n, 2))
            if rng.random() < 0.25:
                ids = rng.choice(n, 2, replace=False)
                a[ids, :2] = a[ids[::-1], :2]
            seed = a.reshape(-1)
        a = seed.reshape(n, 3).copy()
        lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
        hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
        scale = min(1.0, 1.999999 / float(np.sum(hi-lo)))
        a *= scale
        lo *= scale
        hi *= scale
        a[:, :2] -= lo
        w = float(hi[0]-lo[0])
        z0 = np.r_[a.reshape(-1), np.clip(w, 0.08, 1.92)]
        z0 = np.minimum(np.maximum(z0, np.array(bounds)[:, 0]), np.array(bounds)[:, 1])
        last = [z0]
        calls = [0]
        def callback(z):
            last[0] = z.copy()
            calls[0] += 1
            if time.monotonic() >= deadline - 0.025:
                raise OutOfTime()
            if calls[0] % 15 == 0:
                consider(z[:63])
        try:
            result = minimize(lambda z: -float(np.sum(z[2:63:3])), z0,
                              jac=lambda z: grad, method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                              callback=callback,
                              options={'maxiter': 230, 'ftol': 2e-11, 'disp': False})
            consider(result.x[:63])
        except OutOfTime:
            consider(last[0][:63])
            break
        except (ValueError, FloatingPointError):
            consider(last[0][:63])
        trial += 1
    return best
