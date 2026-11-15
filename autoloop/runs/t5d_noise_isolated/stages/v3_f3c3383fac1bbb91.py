def refine(v, score, budget_s, rng):
    import time
    import numpy as np
    from scipy.optimize import minimize, linprog

    start = time.monotonic()
    deadline = start + max(0., budget_s) - min(0.5, max(0., budget_s)*.015)
    n = 21
    d = 64
    ii, jj = np.triu_indices(n, 1)
    rr = np.arange(len(ii))
    ids = np.arange(n)
    best = np.asarray(v, dtype=float).reshape(n, 3).copy()
    best_score = score(best.ravel())
    best_sum = best[:, 2].sum()

    class Stop(Exception):
        pass

    def check(end=None):
        if time.monotonic() >= (deadline if end is None else min(deadline, end)):
            raise Stop()

    def repair(c):
        c = np.asarray(c).reshape(n, 3).copy()
        if not np.isfinite(c).all() or c[:, 2].min() <= 0:
            return None
        ds = np.sqrt(np.sum((c[ii, :2]-c[jj, :2])**2, axis=1))
        f = min(1., np.min(ds/(c[ii, 2]+c[jj, 2])))
        if f <= 0:
            return None
        c[:, 2] *= f*(1.-3e-14)
        lo = (c[:, :2]-c[:, 2, None]).min(axis=0)
        hi = (c[:, :2]+c[:, 2, None]).max(axis=0)
        c[:, :2] -= (lo+hi)/2
        c *= (2./sum(hi-lo))*(1.-3e-14)
        return c

    def consider(c):
        nonlocal best, best_score, best_sum
        c = repair(c)
        if c is None:
            return None
        if c[:, 2].sum() > best_sum + 2e-15:
            check()
            val = score(c.ravel())
            if val < best_score:
                best, best_score = c.copy(), val
                best_sum = c[:, 2].sum()
        return c

    def pack(c):
        c = c.copy()
        lo = (c[:, :2]-c[:, 2, None]).min(axis=0)
        hi = (c[:, :2]+c[:, 2, None]).max(axis=0)
        s = 2./sum(hi-lo)
        c[:, :2] -= lo
        c *= s
        return np.r_[c[:, 0], c[:, 1], c[:, 2], (hi[0]-lo[0])*s]

    def unpack(z):
        return np.column_stack((z[:n], z[n:2*n], z[2*n:3*n]))

    A = np.zeros((4*n, d))
    b = np.zeros(4*n)
    A[ids, ids] = 1
    A[ids, 2*n+ids] = -1
    A[n+ids, ids] = -1
    A[n+ids, 2*n+ids] = -1
    A[n+ids, -1] = 1
    A[2*n+ids, n+ids] = 1
    A[2*n+ids, 2*n+ids] = -1
    A[3*n+ids, n+ids] = -1
    A[3*n+ids, 2*n+ids] = -1
    A[3*n+ids, -1] = -1
    b[3*n:] = 2
    obj = np.zeros(d)
    obj[2*n:3*n] = -1
    bounds = [(0., 2.)]*(2*n)+[(2e-8, 1.)]*n+[(.04, 1.96)]

    def geom(z):
        dx, dy = z[ii]-z[jj], z[n+ii]-z[n+jj]
        ds = np.maximum(np.hypot(dx, dy), 1e-25)
        J = np.zeros((len(ii), d))
        J[rr, ii], J[rr, jj] = dx/ds, -dx/ds
        J[rr, n+ii], J[rr, n+jj] = dy/ds, -dy/ds
        J[rr, 2*n+ii] = -1
        J[rr, 2*n+jj] = -1
        return ds-z[2*n+ii]-z[2*n+jj], J

    def local(z, weights=None, seconds=3., nit=120):
        end = min(deadline, time.monotonic()+seconds)
        g = obj.copy()
        if weights is not None:
            g[2*n:3*n] = -weights
        cache = [None, None, None]
        last = [z.copy()]
        def con(x):
            check(end)
            if cache[0] is None or not np.array_equal(x, cache[0]):
                h, J = geom(x)
                cache[:] = [x.copy(), np.r_[A@x+b, h], np.vstack((A, J))]
            return cache[1]
        def jac(x):
            con(x)
            return cache[2]
        def callback(x):
            last[0] = x.copy()
            if x[2*n:3*n].sum() > best_sum:
                consider(unpack(x))
            check(end)
        try:
            res = minimize(lambda x: g@x, z, jac=lambda x: g,
                           method='SLSQP', bounds=bounds,
                           constraints={'type':'ineq','fun':con,'jac':jac},
                           callback=callback,
                           options={'ftol':2e-11,'maxiter':nit,'disp':False})
            last[0] = res.x
        except Stop:
            pass
        check()
        cc = consider(unpack(last[0]))
        return last[0], cc

    def polish(c, steps=18):
        z = pack(c)
        old = -1.
        for k in range(steps):
            check()
            _, J = geom(z)
            M = np.vstack((A, J))
            rad = min(.07, .012*1.3**k)
            bb = [(max(a, x-rad), min(bi, x+rad)) for x, (a, bi) in zip(z, bounds)]
            res = linprog(obj, A_ub=-M, b_ub=np.r_[b, np.zeros(len(ii))],
                          bounds=bb, method='highs', options={'primal_feasibility_tolerance':1e-9,
                          'dual_feasibility_tolerance':1e-9,
                          'time_limit':max(.001, min(.4, deadline-time.monotonic()))})
            if res.x is None:
                break
            z = res.x
            consider(unpack(z))
            val = z[2*n:3*n].sum()
            if k > 2 and val-old < 1e-12:
                break
            old = val
        return z

    def relocate(c, count):
        c = c.copy()
        # Remove a cluster, or several independently chosen disks. Refill the holes
        # sequentially, rather than repeatedly inserting into the same vacancy.
        if rng.random() < .5:
            seed = int(rng.integers(n))
            order = np.argsort(np.linalg.norm(c[:, :2]-c[seed, :2], axis=1))
            removed = order[:count]
        else:
            p = 1./np.sqrt(c[:, 2]+.005)
            removed = rng.choice(n, count, replace=False, p=p/p.sum())
        lo = (c[:, :2]-c[:, 2, None]).min(axis=0)
        hi = (c[:, :2]+c[:, 2, None]).max(axis=0)
        keep = np.ones(n, dtype=bool)
        keep[removed] = False
        for k in removed:
            pts = rng.uniform(lo, hi, (900, 2))
            ds = np.linalg.norm(pts[:, None, :]-c[None, keep, :2], axis=2)-c[None, keep, 2]
            clear = np.minimum(ds.min(axis=1), np.minimum(pts-lo, hi-pts).min(axis=1))
            top = np.argsort(clear)[-8:]
            j = top[-1] if rng.random() < .6 else rng.choice(top)
            c[k, :2] = pts[j]
            c[k, 2] = max(1e-5, clear[j]*.95)
            keep[k] = True
        return c

    def lattice():
        # Independent basins, including staggered and strongly unequal row lengths.
        nr = int(rng.integers(3, 7))
        counts = np.full(nr, n//nr)
        counts[:n % nr] += 1
        rng.shuffle(counts)
        if rng.random() < .5:
            for _ in range(3):
                a, b0 = rng.choice(nr, 2, replace=False)
                if counts[a] > 2:
                    counts[a] -= 1
                    counts[b0] += 1
        c = []
        for row, ct in enumerate(counts):
            offset = .45*(row % 2) if rng.random() < .6 else 0.
            for col in range(ct):
                c.append([2*col+offset, 1.76*row, 1.])
        c = np.array(c)
        c[:, :2] += rng.normal(0, .06, (n, 2))
        c[:, 2] *= np.exp(rng.normal(0, .12, n))
        return c

    try:
        z = polish(best)
        _, current = local(z, seconds=4., nit=180)
        if current is None:
            current = best.copy()
        pool = [best.copy()]
        attempt = 0
        stale = 0
        while time.monotonic() < deadline:
            attempt += 1
            prev_best = best_sum
            mode = attempt % 12
            # Basin hopping accepts genuinely different, slightly worse packings.
            if mode == 0:
                seed = lattice()
            elif mode == 1:
                seed = best.copy()
            elif rng.random() < .20:
                seed = pool[int(rng.integers(len(pool)))].copy()
            else:
                seed = current.copy()
            if mode in (2, 3, 7, 9):
                seed = relocate(seed, int(rng.integers(1, 5)))
            elif mode in (4, 10):
                # Strong weighting crosses contact-network barriers which tiny
                # perturbations cannot cross; immediately restore the real objective.
                weights = np.exp(rng.normal(0., .65 if mode == 4 else 1., n))
                weights = np.clip(weights, .08, 6.)
                zz, _ = local(pack(seed), weights, seconds=2.8, nit=95)
                seed = unpack(zz)
            elif mode in (5, 8):
                amp = float(rng.choice([.012, .035, .075, .13]))
                take = rng.choice(n, int(rng.integers(4, n+1)), replace=False)
                seed[take, :2] += rng.normal(0, amp, (len(take), 2))
                seed[take, 2] *= np.exp(rng.normal(0, .3, len(take)))
            elif mode == 6:
                s = np.exp(rng.normal(0, .3))
                seed[:, 0] *= s
                seed[:, 1] /= s
            elif mode == 11:
                # Exchange large/small roles without preserving the old tangencies.
                a, b0 = rng.choice(n, 2, replace=False)
                seed[a, 2], seed[b0, 2] = seed[b0, 2], seed[a, 2]
                seed[:, :2] += rng.normal(0, .008, (n, 2))
            zz, cand = local(pack(seed), seconds=4.5 if mode == 0 else 3.2, nit=160)
            if cand is not None:
                cs = cand[:, 2].sum()
                temp = .003 + .012*(1+np.sin(attempt*.43))*.5
                delta = cs-current[:, 2].sum()
                if delta >= 0 or rng.random() < np.exp(max(-700., delta/temp)):
                    current = cand
                if cs > best_sum-.045 and all(np.max(np.abs(np.sort(cand[:, 2])-np.sort(p[:, 2]))) > 3e-4 for p in pool):
                    pool.append(cand.copy())
                    if len(pool) > 14:
                        pool.pop(0)
            if best_sum > prev_best+1e-8:
                stale = 0
                polish(best, 12)
                current = best.copy()
            else:
                stale += 1
            if stale >= 18:
                current = pool[int(rng.integers(len(pool)))].copy()
                stale = 0
    except Stop:
        pass
    return best.ravel()
