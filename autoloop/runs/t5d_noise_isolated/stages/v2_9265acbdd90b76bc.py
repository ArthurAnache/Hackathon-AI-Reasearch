def refine(v, score, budget_s, rng):
    import time
    import numpy as np
    from scipy.optimize import minimize, linprog

    t0 = time.monotonic()
    deadline = t0 + max(0., float(budget_s)) - min(.15, .02 * max(0., float(budget_s)))
    n = 21
    d = 64
    ii, jj = np.triu_indices(n, 1)
    q = len(ii)
    rows = np.arange(q)
    ids = np.arange(n)
    best = np.asarray(v, dtype=float).reshape(n, 3).copy()
    best_value = score(best.ravel())
    best_sum = float(best[:, 2].sum())

    class Timeout(Exception):
        pass

    def clock():
        if time.monotonic() >= deadline:
            raise Timeout()

    def repair(c):
        c = np.asarray(c).reshape(n, 3).copy()
        if not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return None
        dist = np.hypot(c[ii, 0]-c[jj, 0], c[ii, 1]-c[jj, 1])
        fac = min(1., float(np.min(dist/(c[ii, 2]+c[jj, 2]))))
        if fac <= 0:
            return None
        c[:, 2] *= fac * (1.-2e-13)
        lo = np.min(c[:, :2]-c[:, 2, None], axis=0)
        hi = np.max(c[:, :2]+c[:, 2, None], axis=0)
        c[:, :2] -= (hi+lo)/2
        c *= (2./np.sum(hi-lo))*(1.-2e-13)
        return c

    def consider(c):
        nonlocal best, best_value, best_sum
        c = repair(c)
        if c is None or c[:, 2].sum() <= best_sum + 2e-14:
            return
        clock()
        val = score(c.ravel())
        if val < best_value:
            best, best_value = c.copy(), val
            best_sum = float(c[:, 2].sum())

    def unpack(z):
        return np.column_stack((z[:n], z[n:2*n], z[2*n:3*n]))

    def pack(c):
        c = c.copy()
        lo = np.min(c[:, :2]-c[:, 2, None], axis=0)
        hi = np.max(c[:, :2]+c[:, 2, None], axis=0)
        spans = hi-lo
        c[:, :2] -= lo
        scale = 2./sum(spans)
        c *= scale
        return np.r_[c[:, 0], c[:, 1], c[:, 2], spans[0]*scale]

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
    g = np.zeros(d)
    g[2*n:3*n] = -1
    bounds = [(0., 2.)]*(2*n) + [(1e-9, 1.)]*n + [(.02, 1.98)]

    def geometry(z):
        dx = z[ii]-z[jj]
        dy = z[n+ii]-z[n+jj]
        dist = np.maximum(np.hypot(dx, dy), 1e-30)
        J = np.zeros((q, d))
        J[rows, ii] = dx/dist
        J[rows, jj] = -dx/dist
        J[rows, n+ii] = dy/dist
        J[rows, n+jj] = -dy/dist
        J[rows, 2*n+ii] = -1
        J[rows, 2*n+jj] = -1
        return dist-z[2*n+ii]-z[2*n+jj], J

    def slsqp(z, weights=None, iterations=160, slice_s=8.):
        end = min(deadline, time.monotonic()+slice_s)
        gg = g.copy()
        if weights is not None:
            gg[2*n:3*n] = -weights
        cache = [None, None, None]
        last = [z.copy()]

        def funcon(x):
            clock()
            if time.monotonic() >= end:
                raise Timeout()
            if cache[0] is None or not np.array_equal(x, cache[0]):
                h, J = geometry(x)
                cache[0] = x.copy()
                cache[1] = np.r_[A@x+b, h]
                cache[2] = np.vstack((A, J))
            return cache[1]

        def jaccon(x):
            funcon(x)
            return cache[2]

        def cb(x):
            last[0] = x.copy()
            if x[2*n:3*n].sum() > best_sum:
                consider(unpack(x))
            clock()
            if time.monotonic() >= end:
                raise Timeout()
        try:
            res = minimize(lambda x: gg@x, z, jac=lambda x: gg,
                           bounds=bounds,
                           constraints={'type':'ineq', 'fun':funcon, 'jac':jaccon},
                           method='SLSQP', callback=cb,
                           options={'maxiter':iterations, 'ftol':8e-13, 'disp':False})
            last[0] = res.x
        except Timeout:
            pass
        if time.monotonic() < deadline:
            consider(unpack(last[0]))
        return last[0]

    def linear_polish(z, steps=45):
        radius = .025
        prev = -float('inf')
        for k in range(steps):
            clock()
            h, J = geometry(z)
            M = np.vstack((A, J))
            rhs = np.r_[b, np.zeros(q)]
            lb = np.array([p[0] for p in bounds])
            ub = np.array([p[1] for p in bounds])
            lb = np.maximum(lb, z-radius)
            ub = np.minimum(ub, z+radius)
            res = linprog(g, A_ub=-M, b_ub=rhs,
                          bounds=list(zip(lb, ub)), method='highs',
                          options={'primal_feasibility_tolerance':1e-9,
                                   'dual_feasibility_tolerance':1e-9,
                                   'time_limit':max(.01, min(1., deadline-time.monotonic()))})
            if res.x is None:
                break
            z = res.x
            val = float(z[2*n:3*n].sum())
            consider(unpack(z))
            if k > 2 and val-prev < 3e-12:
                break
            prev = val
            radius = min(.10, radius*1.25)
        return z

    try:
        clock()
        z = linear_polish(pack(best), 45)
        z = slsqp(z, iterations=220)
        pool = [best.copy()]
        attempt = 0
        while time.monotonic() < deadline:
            clock()
            attempt += 1
            c = best.copy() if attempt % 4 else pool[int(rng.integers(len(pool)))].copy()
            mode = attempt % 8
            if mode == 1:
                # Follow a nearby weighted-radius optimum, then restore the true objective.
                weights = np.exp(rng.normal(0., .12 if attempt % 3 else .30, n))
                z = slsqp(pack(c), weights, iterations=85, slice_s=4.)
            else:
                if mode in (2, 6):
                    # Move a small circle to the most promising vacant pocket.
                    small = np.argsort(c[:, 2])[:7]
                    k = int(rng.choice(small))
                    lo = np.min(c[:, :2]-c[:, 2, None], axis=0)
                    hi = np.max(c[:, :2]+c[:, 2, None], axis=0)
                    probes = rng.uniform(lo, hi, (700, 2))
                    keep = np.arange(n) != k
                    clear = np.min(np.linalg.norm(probes[:, None, :]-c[None, keep, :2], axis=2)-c[None, keep, 2], axis=1)
                    clear = np.minimum(clear, np.min(np.minimum(probes-lo, hi-probes), axis=1))
                    order = np.argsort(clear)[-10:]
                    j = int(order[-1] if mode == 2 else rng.choice(order))
                    c[k, :2] = probes[j]
                    c[k, 2] = max(1e-6, clear[j]*.97)
                elif mode == 3:
                    c[:, 2] *= np.exp(rng.normal(0., .18, n))
                elif mode == 4:
                    # Distort the rectangle without changing its orientation.
                    s = np.exp(rng.normal(0., .10))
                    c[:, 0] *= s
                    c[:, 1] /= s
                else:
                    amp = {0:.045, 5:.008, 7:.022}.get(mode, .015)
                    if attempt % 3 == 0:
                        take = rng.choice(n, size=6, replace=False)
                        c[take, :2] += rng.normal(0., amp*2, (6, 2))
                    else:
                        c[:, :2] += rng.normal(0., amp, (n, 2))
                # Do not shrink all radii to repair an intentionally overlapping seed.
                z = pack(c)
            z = slsqp(z, iterations=170, slice_s=7.)
            clock()
            cc = repair(unpack(z))
            if cc is not None and cc[:, 2].sum() > best_sum-.025:
                pool.append(cc)
                if len(pool) > 8:
                    pool.pop(0)
            if attempt % 5 == 0:
                linear_polish(pack(best), 20)
    except Timeout:
        pass
    return best.ravel()
