def refine(v, score, budget_s, rng):
    import time
    import numpy as np
    from scipy.optimize import minimize

    start = time.monotonic()
    deadline = start + max(0.0, float(budget_s)) - min(0.08, max(0.005, float(budget_s) * 0.025))
    n = 21
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    best = None
    best_score = float('inf')
    best_sum = -1.0

    class OutOfTime(Exception):
        pass

    def check_time():
        if time.monotonic() >= deadline:
            raise OutOfTime()

    def repair(c):
        c = np.asarray(c, dtype=float).reshape(n, 3).copy()
        if not np.isfinite(c).all():
            return None
        c[:, 2] = np.maximum(c[:, 2], 1e-10)
        dx = c[ii, 0] - c[jj, 0]
        dy = c[ii, 1] - c[jj, 1]
        d = np.hypot(dx, dy)
        f = min(1.0, float(np.min(d / (c[ii, 2] + c[jj, 2]))))
        if f <= 0:
            return None
        c[:, 2] *= f * (1.0 - 3e-12)
        lo = np.min(c[:, :2] - c[:, 2, None], axis=0)
        hi = np.max(c[:, :2] + c[:, 2, None], axis=0)
        size = float(np.sum(hi - lo))
        if not size > 0:
            return None
        c[:, :2] -= (hi + lo) * 0.5
        c *= (2.0 / size) * (1.0 - 3e-12)
        return c

    def consider(c, force=False):
        nonlocal best, best_score, best_sum
        c = repair(c)
        if c is None:
            return
        s = float(np.sum(c[:, 2]))
        if not force and s <= best_sum + 1e-11:
            return
        val = score(c.ravel())
        if np.isfinite(val) and val < best_score:
            best = c.copy()
            best_score = val
            best_sum = s

    if v is not None:
        vv = np.asarray(v, dtype=float).reshape(n, 3)
        val = score(vv.ravel())
        if np.isfinite(val):
            best = vv.copy()
            best_score = val
            best_sum = float(np.sum(vv[:, 2]))
        consider(vv)

    seeds = []
    for rows in (5, 4, 6, 3, 7):
        counts = np.full(rows, n // rows, dtype=int)
        order = sorted(range(rows), key=lambda k: (k % 2 == 0, abs(k - (rows - 1) / 2)))
        for k in order[:n % rows]:
            counts[k] += 1
        pts = []
        for k, count in enumerate(counts):
            for j in range(count):
                pts.append((2.0 * j - (count - 1) + 0.18 * (k % 2), np.sqrt(3.0) * k, 1.0))
        c = repair(np.array(pts))
        seeds.append(c)
    seeds.sort(key=lambda c: -np.sum(c[:, 2]))
    if best is None:
        consider(seeds[0], True)
    if time.monotonic() >= deadline:
        return best.ravel()

    # Internal coordinates are x[21], y[21], r[21], rectangle width.
    A = np.zeros((4 * n, 3 * n + 1))
    b = np.zeros(4 * n)
    ids = np.arange(n)
    A[ids, ids] = 1
    A[ids, 2*n + ids] = -1
    A[n + ids, ids] = -1
    A[n + ids, 2*n + ids] = -1
    A[n + ids, -1] = 1
    A[2*n + ids, n + ids] = 1
    A[2*n + ids, 2*n + ids] = -1
    A[3*n + ids, n + ids] = -1
    A[3*n + ids, 2*n + ids] = -1
    A[3*n + ids, -1] = -1
    b[3*n:] = 2
    grad = np.zeros(3*n + 1)
    grad[2*n:3*n] = -1
    bounds = [(0.0, 2.0)] * (2*n) + [(1e-8, 1.0)] * n + [(0.03, 1.97)]
    pair_rows = np.arange(m)

    def unpack(z):
        return np.column_stack((z[:n], z[n:2*n], z[2*n:3*n]))

    def pack(c):
        c = repair(c)
        lo = np.min(c[:, :2] - c[:, 2, None], axis=0)
        hi = np.max(c[:, :2] + c[:, 2, None], axis=0)
        span = hi - lo
        c[:, :2] -= lo
        slack = max(0.0, 2.0 - float(np.sum(span)))
        c[:, :2] += slack / 4
        return np.r_[c[:, 0], c[:, 1], c[:, 2], span[0] + slack / 2]

    attempt = 0
    try:
        while time.monotonic() < deadline:
            if attempt < len(seeds):
                c = seeds[attempt].copy()
                if attempt and best is not None and attempt % 2 == 0:
                    c = best.copy()
                    c[:, :2] += rng.normal(0, 0.025, (n, 2))
            else:
                c = best.copy()
                mode = (attempt - len(seeds)) % 5
                if mode == 0:
                    # Reinsert a weak circle into a different interstice.
                    k = int(np.argmin(c[:, 2]))
                    lower = np.min(c[:, :2] - c[:, 2, None], axis=0)
                    upper = np.max(c[:, :2] + c[:, 2, None], axis=0)
                    probes = rng.uniform(lower, upper, size=(100, 2))
                    keep = np.arange(n) != k
                    clearance = np.min(np.linalg.norm(probes[:, None, :] - c[None, keep, :2], axis=2) - c[None, keep, 2], axis=1)
                    clearance = np.minimum(clearance, np.min(np.minimum(probes-lower, upper-probes), axis=1))
                    p = int(np.argmax(clearance))
                    c[k, :2] = probes[p]
                    c[k, 2] = max(1e-6, clearance[p])
                else:
                    amplitude = (0.009, 0.018, 0.04, 0.075, 0.12)[mode]
                    c[:, :2] += rng.normal(0, amplitude, (n, 2))
                    c[:, 2] *= np.exp(rng.normal(0, 0.1, n))
            attempt += 1
            z0 = pack(c)
            cached_z = None
            cached_c = None
            cached_j = None

            def constraints(z):
                nonlocal cached_z, cached_c, cached_j
                check_time()
                if cached_z is None or not np.array_equal(z, cached_z):
                    dx = z[ii] - z[jj]
                    dy = z[n + ii] - z[n + jj]
                    dist = np.maximum(np.hypot(dx, dy), 1e-30)
                    pair = dist - z[2*n + ii] - z[2*n + jj]
                    cached_c = np.concatenate((A @ z + b, pair))
                    J = np.zeros((m, 3*n + 1))
                    ux, uy = dx / dist, dy / dist
                    J[pair_rows, ii] = ux
                    J[pair_rows, jj] = -ux
                    J[pair_rows, n + ii] = uy
                    J[pair_rows, n + jj] = -uy
                    J[pair_rows, 2*n + ii] = -1
                    J[pair_rows, 2*n + jj] = -1
                    cached_j = np.vstack((A, J))
                    cached_z = z.copy()
                return cached_c

            def jacobian(z):
                constraints(z)
                return cached_j

            def objective(z):
                check_time()
                return -np.sum(z[2*n:3*n])

            def callback(z):
                check_time()
                if np.sum(z[2*n:3*n]) > best_sum + 1e-10:
                    consider(unpack(z))

            result = minimize(objective, z0, jac=lambda z: grad,
                              bounds=bounds,
                              constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                              method='SLSQP', callback=callback,
                              options={'maxiter': 180, 'ftol': 3e-11, 'disp': False})
            consider(unpack(result.x))
    except OutOfTime:
        pass
    return best.ravel()
