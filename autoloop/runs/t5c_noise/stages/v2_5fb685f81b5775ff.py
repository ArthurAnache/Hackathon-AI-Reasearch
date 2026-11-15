def refine(v, score, budget_s, rng):
    import time
    import numpy as np
    from scipy.optimize import minimize

    t0 = time.monotonic()
    budget = max(0.0, float(budget_s))
    reserve = min(0.25, max(0.015, budget * 0.015))
    deadline = t0 + budget - reserve
    original = np.asarray(v, dtype=np.float64).reshape(63).copy()
    best = original.copy()
    best_score = float(score(best))
    best_sum = float(best[2::3].sum())
    if budget < 0.08:
        return best

    n = 21
    ii, jj = np.triu_indices(n, 1)
    p = np.arange(n)
    q = 84 + np.arange(len(ii))
    archive = []

    class Timeout(Exception):
        pass

    def check():
        if time.monotonic() >= deadline:
            raise Timeout()

    def repair(x):
        a = np.asarray(x[:63], dtype=float).reshape(n, 3).copy()
        if not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return None
        a[:, :2] -= np.mean(a[:, :2], axis=0)
        d = np.hypot(a[ii, 0] - a[jj, 0], a[ii, 1] - a[jj, 1])
        f = min(1.0, float(np.min(d / (a[ii, 2] + a[jj, 2]))))
        if f <= 0:
            return None
        a[:, 2] *= f * (1.0 - 8e-14)
        lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
        hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
        size = float(np.sum(hi - lo))
        if not np.isfinite(size) or size <= 0:
            return None
        a[:, :2] -= (lo + hi) * 0.5
        a *= (2.0 / size) * (1.0 - 8e-14)
        return a.reshape(63)

    def remember(c, s):
        nonlocal archive
        if s < best_sum - 0.025:
            return
        if any(np.linalg.norm(c - x[1]) < 0.004 for x in archive):
            return
        archive.append((s, c.copy()))
        archive.sort(key=lambda x: -x[0])
        archive = archive[:10]

    def consider(x, keep=True):
        nonlocal best, best_sum, best_score
        c = repair(x)
        if c is None:
            return
        s = float(c[2::3].sum())
        if keep:
            remember(c, s)
        if s <= best_sum:
            return
        val = float(score(c))
        if np.isfinite(val) and val < best_score:
            best, best_sum, best_score = c.copy(), s, val

    remember(best, best_sum)
    consider(best)

    # Coordinates lie in [0,w] x [0,2-w].  The extra variable w
    # makes all four sets of wall constraints linear.
    constant_jac = np.zeros((84 + len(ii), 64))
    constant_jac[p, 3*p] = 1
    constant_jac[p, 3*p+2] = -1
    constant_jac[21+p, 3*p] = -1
    constant_jac[21+p, 3*p+2] = -1
    constant_jac[21+p, 63] = 1
    constant_jac[42+p, 3*p+1] = 1
    constant_jac[42+p, 3*p+2] = -1
    constant_jac[63+p, 3*p+1] = -1
    constant_jac[63+p, 3*p+2] = -1
    constant_jac[63+p, 63] = -1
    constant_jac[q, 3*ii+2] = -1
    constant_jac[q, 3*jj+2] = -1

    def con(z):
        check()
        a = z[:63].reshape(n, 3)
        x, y, r = a.T
        d = np.hypot(x[ii]-x[jj], y[ii]-y[jj])
        return np.concatenate((x-r, z[63]-x-r, y-r,
                               2-z[63]-y-r, d-r[ii]-r[jj]))

    def jac(z):
        check()
        a = z[:63].reshape(n, 3)
        dx = a[ii, 0]-a[jj, 0]
        dy = a[ii, 1]-a[jj, 1]
        d = np.maximum(np.hypot(dx, dy), 1e-100)
        j = constant_jac.copy()
        j[q, 3*ii] = dx/d
        j[q, 3*jj] = -dx/d
        j[q, 3*ii+1] = dy/d
        j[q, 3*jj+1] = -dy/d
        return j

    bounds = [(0., 2.), (0., 2.), (1e-8, 0.9)] * n + [(0.025, 1.975)]

    def to_z(a):
        a = np.asarray(a).reshape(n, 3).copy()
        a[:, 2] = np.maximum(a[:, 2], 1.1e-8)
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        f = min(1.0, 1.999999 / float(np.sum(hi-lo)))
        a *= f
        lo *= f
        hi *= f
        a[:, :2] -= lo
        z = np.r_[a.ravel(), np.clip(hi[0]-lo[0], 0.025, 1.975)]
        z[:63:3] = np.clip(z[:63:3], 0., 2.)
        z[1:63:3] = np.clip(z[1:63:3], 0., 2.)
        z[2:63:3] = np.clip(z[2:63:3], 1e-8, 0.9)
        return z

    def optimize(seed, weights, maxiter):
        z = to_z(seed)
        gradient = np.zeros(64)
        gradient[2:63:3] = -weights
        last = [z]
        iterations = [0]
        def fun(x):
            check()
            return float(gradient @ x)
        def cb(x):
            last[0] = x.copy()
            iterations[0] += 1
            check()
            if iterations[0] % 20 == 0:
                consider(x)
        try:
            result = minimize(fun, z, jac=lambda x: gradient, method='SLSQP',
                              bounds=bounds,
                              constraints={'type': 'ineq', 'fun': con, 'jac': jac},
                              callback=cb,
                              options={'maxiter': maxiter, 'ftol': 8e-13, 'disp': False})
            last[0] = result.x
        except Timeout:
            consider(last[0])
            raise
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            pass
        consider(last[0])
        return last[0][:63].copy()

    def relocate(a):
        # Move a redundant/small disk into a different interstice; subsequent
        # nonlinear relaxation is allowed to change every radius and the box.
        r = a[:, 2]
        probs = 1.0 / np.maximum(r, 0.012)**1.5
        k = int(rng.choice(n, p=probs/probs.sum()))
        others = np.arange(n) != k
        lo = np.min(a[:, :2]-r[:, None], axis=0)
        hi = np.max(a[:, :2]+r[:, None], axis=0)
        points = rng.uniform(lo, hi, size=(350, 2))
        clearance = np.minimum(points-lo, hi-points).min(axis=1)
        dist = np.linalg.norm(points[:, None, :] - a[None, others, :2], axis=2)
        clearance = np.minimum(clearance, np.min(dist-r[others], axis=1))
        clearance[np.linalg.norm(points-a[k, :2], axis=1) < 0.09] = -1
        if np.max(clearance) > 0.006:
            # Randomness helps avoid repeatedly choosing the same empty hole.
            quality = clearance * rng.uniform(0.85, 1.15, len(points))
            j = int(np.argmax(quality))
            a[k, :2] = points[j]
            a[k, 2] = max(1e-5, clearance[j]*0.98)
        return a

    def lattice():
        layouts = [[5,4,5,4,3], [4,4,5,4,4], [5,6,5,5], [7,7,7], [4,3,4,3,4,3]]
        counts = layouts[int(rng.integers(len(layouts)))]
        pts = []
        for j, count in enumerate(counts):
            for k in range(count):
                pts.append([2*k + (max(counts)-count), np.sqrt(3.)*j, 1.])
        a = np.asarray(pts, dtype=float)
        a[:, :2] += rng.normal(0., 0.07, (n, 2))
        return repair(a.ravel()).reshape(n, 3)

    trial = 0
    stale = 0
    try:
        # Always polish the supplied packing first, rather than spending the
        # first restart on an unrelated lattice.
        optimize(best, np.ones(n), 350)
        while time.monotonic() < deadline:
            before = best_sum
            trial += 1
            if trial > 15 and rng.random() < 0.06:
                a = lattice()
                mode = 0
            else:
                if len(archive) > 1 and rng.random() < 0.22:
                    idx = int(rng.integers(min(6, len(archive))))
                    a = archive[idx][1].reshape(n, 3).copy()
                else:
                    a = best.reshape(n, 3).copy()
                mode = int(rng.integers(6))
                if mode <= 1:
                    amplitude = float(rng.choice([0.002, 0.006, 0.014, 0.028, 0.05]))
                    if stale > 18:
                        amplitude *= 1.5
                    a[:, :2] += rng.normal(0., amplitude, (n, 2))
                    if rng.random() < 0.4:
                        a[:, 2] *= np.exp(rng.normal(0., 0.12, n))
                elif mode == 2:
                    a = relocate(a)
                elif mode == 3:
                    # A low-frequency shear/aspect perturbation changes
                    # contacts without destroying the entire packing.
                    shear = rng.normal(0., 0.07)
                    a[:, 0] += shear*a[:, 1]
                    f = np.exp(rng.normal(0., 0.055))
                    a[:, 0] *= f
                    a[:, 1] /= f
                    a[:, :2] += rng.normal(0., 0.003, (n, 2))
                elif mode == 4:
                    k = int(rng.integers(n))
                    d = np.linalg.norm(a[:, :2]-a[k, :2], axis=1)
                    ids = np.argsort(d)[:int(rng.integers(3, 9))]
                    a[ids, :2] += rng.normal(0., 0.025, (len(ids), 2))
                    a[ids, 2] *= rng.uniform(0.85, 1.08, len(ids))

            if mode == 5 or (trial % 5 == 0):
                # Tilt the radius objective to cross a contact-graph boundary,
                # then remove the tilt and optimize the actual objective.
                sigma = float(rng.choice([0.12, 0.25, 0.45]))
                weights = np.exp(rng.normal(0., sigma, n))
                weights /= weights.mean()
                seed = optimize(a, weights, 65)
                optimize(seed, np.ones(n), 180)
            else:
                optimize(a, np.ones(n), 210)
            if best_sum > before + 1e-9:
                stale = 0
            else:
                stale += 1
    except Timeout:
        pass
    return best
