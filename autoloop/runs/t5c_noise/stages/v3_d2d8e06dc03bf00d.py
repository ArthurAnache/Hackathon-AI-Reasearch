def refine(v, score, budget_s, rng):
    import time
    import numpy as np
    from scipy.optimize import minimize, linprog

    started = time.monotonic()
    budget = max(0.0, float(budget_s))
    deadline = started + budget - min(0.6, max(0.025, 0.01 * budget))
    best = np.asarray(v, dtype=np.float64).reshape(63).copy()
    best_value = float(score(best))
    best_sum = best[2::3].sum()
    if budget < 0.1:
        return best
    n = 21
    ii, jj = np.triu_indices(n, 1)
    p = np.arange(n)
    rows = 84 + np.arange(len(ii))
    archive = []

    class Stop(Exception):
        pass

    def check():
        if time.monotonic() >= deadline:
            raise Stop()

    def repair(z):
        a = np.asarray(z[:63], dtype=float).reshape(n, 3).copy()
        if not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return None
        a[:, :2] -= a[:, :2].mean(axis=0)
        d = np.hypot(a[ii, 0] - a[jj, 0], a[ii, 1] - a[jj, 1])
        f = min(1., np.min(d / (a[ii, 2] + a[jj, 2])))
        if f <= 0:
            return None
        a[:, 2] *= f * (1 - 1e-13)
        lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
        hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
        wh = np.sum(hi - lo)
        if wh <= 0 or not np.isfinite(wh):
            return None
        a[:, :2] -= (lo + hi) / 2
        a *= (2 / wh) * (1 - 1e-13)
        return a.ravel()

    def signature(c):
        a = c.reshape(n, 3)
        return np.r_[np.sort(a[:, 2]), np.sort(np.hypot(a[ii, 0]-a[jj, 0], a[ii, 1]-a[jj, 1]))[::5]]

    def consider(z, store=True):
        nonlocal best, best_value, best_sum, archive
        c = repair(z)
        if c is None:
            return None
        s = float(c[2::3].sum())
        if s > best_sum:
            val = float(score(c))
            if np.isfinite(val) and val < best_value:
                best, best_value, best_sum = c.copy(), val, s
        if store and s > best_sum - 0.22:
            sig = signature(c)
            if not any(np.linalg.norm(sig - item[2]) < 0.0015 for item in archive):
                archive.append((s, c.copy(), sig))
                archive.sort(key=lambda t: -t[0])
                archive = archive[:32]
        return c

    consider(best)
    # w+h=2 eliminates one box variable. Distances are the only
    # nonlinear constraints; their tangent halfspaces are inner bounds.
    J0 = np.zeros((294, 64))
    J0[p, 3*p] = 1
    J0[p, 3*p+2] = -1
    J0[21+p, 3*p] = -1
    J0[21+p, 3*p+2] = -1
    J0[21+p, 63] = 1
    J0[42+p, 3*p+1] = 1
    J0[42+p, 3*p+2] = -1
    J0[63+p, 3*p+1] = -1
    J0[63+p, 3*p+2] = -1
    J0[63+p, 63] = -1
    J0[rows, 3*ii+2] = -1
    J0[rows, 3*jj+2] = -1
    offset = np.zeros(294)
    offset[63:84] = 2
    bounds = [(0., 2.), (0., 2.), (1e-7, 0.8)] * n + [(0.03, 1.97)]

    def to_z(c):
        a = np.asarray(c[:63]).reshape(n, 3).copy()
        a[:, 2] = np.maximum(a[:, 2], 1.1e-7)
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        f = min(1., 1.99999 / np.sum(hi-lo))
        a *= f
        a[:, :2] -= lo*f
        return np.r_[a.ravel(), np.clip((hi[0]-lo[0])*f, .03, 1.97)]

    def jac(z):
        check()
        a = z[:63].reshape(n, 3)
        dx = a[ii, 0]-a[jj, 0]
        dy = a[ii, 1]-a[jj, 1]
        d = np.maximum(np.hypot(dx, dy), 1e-100)
        j = J0.copy()
        j[rows, 3*ii] = dx/d
        j[rows, 3*jj] = -dx/d
        j[rows, 3*ii+1] = dy/d
        j[rows, 3*jj+1] = -dy/d
        return j

    def con(z):
        check()
        a = z[:63].reshape(n, 3)
        x, y, r = a.T
        return np.r_[x-r, z[63]-x-r, y-r, 2-z[63]-y-r,
                     np.hypot(x[ii]-x[jj], y[ii]-y[jj])-r[ii]-r[jj]]

    def polish(c, iterations=160, weights=None):
        z = to_z(c)
        g = np.zeros(64)
        g[2:63:3] = -1 if weights is None else -weights
        last = [z]
        count = [0]
        def fun(x):
            check()
            return float(g @ x)
        def cb(x):
            check()
            last[0] = x.copy()
            count[0] += 1
            if count[0] % 25 == 0:
                consider(x, False)
        try:
            res = minimize(fun, z, jac=lambda x: g, method='SLSQP', bounds=bounds,
                           constraints={'type': 'ineq', 'fun': con, 'jac': jac},
                           callback=cb, options={'maxiter': iterations, 'ftol': 3e-13})
            last[0] = res.x
        except Stop:
            consider(last[0])
            raise
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            pass
        out = consider(last[0])
        return out if out is not None else np.asarray(c[:63]).copy()

    def linear_relax(c, weights=None, steps=30, trust=None):
        z = to_z(c)
        g = np.zeros(64)
        g[2:63:3] = -1 if weights is None else -weights
        oldobj = float(g @ z)
        for k in range(steps):
            check()
            j = jac(z)
            bb = bounds
            if trust is not None:
                bb = list(bounds)
                for t in range(63):
                    if t % 3 != 2:
                        bb[t] = (max(0., z[t]-trust), min(2., z[t]+trust))
            remaining = max(.001, deadline-time.monotonic())
            res = linprog(g, A_ub=-j, b_ub=offset, bounds=bb, method='highs',
                          options={'time_limit': remaining, 'dual_feasibility_tolerance': 1e-8,
                                   'primal_feasibility_tolerance': 1e-8})
            if not res.success:
                break
            nz = res.x
            obj = float(g @ nz)
            delta = np.max(np.abs(nz-z))
            z = nz
            if k % 12 == 11:
                consider(z, False)
            if delta < 1e-8 or (k > 4 and abs(obj-oldobj) < 2e-11):
                break
            oldobj = obj
        out = consider(z, False)
        return out if out is not None else np.asarray(c[:63]).copy()

    def holes(a, count=1):
        # Delete and reinsert disks, not just perturb their positions: this
        # changes the planar contact graph even at a very rigid local optimum.
        a = a.copy()
        prob = 1 / np.maximum(a[:, 2], .015)**1.1
        ids = rng.choice(n, size=count, replace=False, p=prob/prob.sum())
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        a[ids, 2] = 1e-7
        for k in ids:
            others = np.arange(n) != k
            pts = rng.uniform(lo, hi, (1600, 2))
            wall = np.minimum(pts-lo, hi-pts).min(axis=1)
            gap = np.min(np.linalg.norm(pts[:, None, :]-a[None, others, :2], axis=2)-a[others, 2], axis=1)
            gap = np.minimum(gap, wall)
            far = np.linalg.norm(pts-a[k, :2], axis=1) > .07
            gap[~far] = -1
            idx = np.argmax(gap * rng.uniform(.9, 1.1, len(gap)))
            if gap[idx] > 1e-6:
                a[k, :2] = pts[idx]
                a[k, 2] = gap[idx] * .95
        return a

    def fresh():
        # Cropped triangular lattices at many orientations/aspect ratios.
        aspect = np.exp(rng.uniform(-.7, .7))
        pts = []
        for y in range(-5, 6):
            for x in range(-5, 6):
                pts.append([2*x+(y % 2), np.sqrt(3.)*y])
        pts = np.asarray(pts, dtype=float)
        shift = rng.uniform(-.8, .8, 2)
        rank = np.maximum(np.abs((pts[:, 0]-shift[0])/aspect), np.abs((pts[:, 1]-shift[1])*aspect))
        pts = pts[np.argsort(rank)[:n]]
        a = np.c_[pts, np.ones(n)]
        a[:, :2] += rng.normal(0, .08, (n, 2))
        return repair(a.ravel()).reshape(n, 3)

    try:
        # Preserve the high-accuracy local improvement of the previous stage.
        polish(best, 400)
        chain = best.copy()
        chain_sum = float(chain[2::3].sum())
        trial = 0
        while True:
            check()
            trial += 1
            u = rng.random()
            if u < .10:
                a = fresh()
            else:
                if u < .38:
                    source = chain
                elif u < .65 and len(archive) > 1:
                    ranks = np.arange(len(archive))
                    prob = np.exp(-ranks / 9.)
                    source = archive[int(rng.choice(len(archive), p=prob/prob.sum()))][1]
                else:
                    source = best
                a = source.reshape(n, 3).copy()
            mode = int(rng.integers(9))
            if mode == 0:
                a = holes(a, int(rng.integers(1, 5)))
            elif mode == 1:
                ids = rng.choice(n, int(rng.integers(2, 6)), replace=False)
                a[ids, 2] = a[rng.permutation(ids), 2]
                a[ids, :2] += rng.normal(0, .035, (len(ids), 2))
            elif mode == 2:
                amp = float(rng.choice([.015, .035, .07, .12]))
                a[:, :2] += rng.normal(0, amp, (n, 2))
                a[:, 2] *= np.exp(rng.normal(0, .25, n))
            elif mode == 3:
                k = int(rng.integers(n))
                ids = np.argsort(np.linalg.norm(a[:, :2]-a[k, :2], axis=1))[:int(rng.integers(4, 10))]
                angle = rng.normal(0, .7)
                rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                center = a[ids, :2].mean(axis=0)
                a[ids, :2] = (a[ids, :2]-center) @ rot + center
            elif mode == 4:
                f = np.exp(rng.normal(0, .22))
                a[:, 0] *= f
                a[:, 1] /= f
                a[:, 0] += rng.normal(0, .15)*a[:, 1]
            elif mode == 5:
                a = holes(a, 1)
                a[:, :2] += rng.normal(0, .012, (n, 2))

            c = a.ravel()
            if mode >= 6 or trial % 4 == 0:
                # Large objective tilts traverse contact-graph boundaries.
                # Unlike tiny coordinate jitter they can cross a rigid basin.
                sigma = float(rng.choice([.18, .4, .7, 1.1]))
                weights = np.exp(rng.normal(0, sigma, n))
                weights /= weights.mean()
                c = linear_relax(c, weights, steps=18)
                if rng.random() < .35:
                    c = polish(c, 55, weights)
                c = linear_relax(c, steps=45)
            else:
                c = linear_relax(c, steps=55)
            c = polish(c, 180)
            s = float(c[2::3].sum())
            # Retain a walking basin as well as the elitist best. A strictly
            # elitist restart policy repeatedly falls into the same packing.
            temperature = float(rng.choice([.012, .035, .07]))
            if s >= chain_sum or rng.random() < np.exp(max(-700., (s-chain_sum)/temperature)):
                chain, chain_sum = c.copy(), s
            if trial % 17 == 0:
                chain, chain_sum = best.copy(), best_sum
    except Stop:
        pass
    return best
