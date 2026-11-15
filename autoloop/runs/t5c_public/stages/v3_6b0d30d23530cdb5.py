def refine(v, score, budget_s, rng):
    import time
    import numpy as np
    from scipy.optimize import minimize, linprog

    start = time.monotonic()
    original = np.asarray(v, dtype=np.float64).reshape(63).copy()
    if budget_s < .2:
        return original
    end = start + budget_s - max(.08, .015 * budget_s)
    best = original.copy()
    best_score = float(score(best))
    n = 21
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    ar = np.arange(m)
    k = np.arange(n)
    objective = np.zeros(64)
    objective[2:63:3] = -1.

    class Timeout(Exception):
        pass

    def check():
        if time.monotonic() >= end:
            raise Timeout()

    def encode(x):
        c = np.asarray(x[:63]).reshape(n, 3).copy()
        lo = np.min(c[:, :2] - c[:, 2, None], axis=0)
        hi = np.max(c[:, :2] + c[:, 2, None], axis=0)
        c[:, :2] -= (lo + hi) / 2
        wh = hi - lo
        return np.r_[c.ravel(), np.clip(wh[0] + max(0., 2.-wh.sum())/2, .001, 1.999)]

    def repair(z, eps=4e-15):
        c = np.asarray(z[:63]).reshape(n, 3).copy()
        if not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return None
        d = c[ii, :2] - c[jj, :2]
        d = np.sqrt(np.sum(d*d, axis=1))
        f = min(1., np.min(d/(c[ii, 2]+c[jj, 2])))
        if f <= 0:
            return None
        c[:, 2] *= f*(1-eps)
        lo = np.min(c[:, :2]-c[:, 2, None], axis=0)
        hi = np.max(c[:, :2]+c[:, 2, None], axis=0)
        c[:, :2] -= (lo+hi)/2
        t = c.astype(np.longdouble)
        p = np.sum(np.max(t[:, :2]+t[:, 2, None], axis=0)-np.min(t[:, :2]-t[:, 2, None], axis=0))
        if p <= 0:
            return None
        c *= float(np.longdouble(2)/p)*(1-eps)
        return c.ravel()

    def consider(z):
        nonlocal best, best_score
        check()
        x = repair(z)
        if x is None:
            return None
        val = -float(np.sum(x[2::3]))
        if val < best_score:
            s = float(score(x))
            if s < best_score:
                best, best_score = x.copy(), s
            elif val < best_score - 1e-12:
                x = repair(z, 4e-14)
                s = float(score(x))
                if s < best_score:
                    best, best_score = x.copy(), s
        return x

    # All separation planes are inner approximations: a feasible LP solution
    # is a genuine circle packing, not just a feasible linearized distance.
    wall = np.zeros((84, 64))
    wall[k, 3*k] = 1
    wall[n+k, 3*k] = -1
    wall[2*n+k, 3*k+1] = 1
    wall[3*n+k, 3*k+1] = -1
    for b in range(4):
        wall[b*n+k, 3*k+2] = -1
    wall[:2*n, 63] = .5
    wall[2*n:, 63] = -.5
    offsets = np.r_[np.zeros(2*n), np.ones(2*n)]
    rhs = np.r_[np.zeros(m+2*n), np.ones(2*n)]
    A = np.zeros((m+84, 64))
    A[m:] = -wall
    A[ar, 3*ii+2] = 1
    A[ar, 3*jj+2] = 1
    base_bounds = [(-1., 1.), (-1., 1.), (1e-8, .5)]*n + [(.002, 1.998)]

    def planes(z):
        c = z[:63].reshape(n, 3)
        d = c[ii, :2]-c[jj, :2]
        norms = np.sqrt(np.sum(d*d, axis=1))
        bad = norms < 1e-12
        if np.any(bad):
            d[bad] = rng.normal(size=(int(bad.sum()), 2))
            norms = np.sqrt(np.sum(d*d, axis=1))
        d /= norms[:, None]
        A[ar, 3*ii] = -d[:, 0]
        A[ar, 3*jj] = d[:, 0]
        A[ar, 3*ii+1] = -d[:, 1]
        A[ar, 3*jj+1] = d[:, 1]

    def slp(z, iterations=40, weights=None, floor=1e-8):
        z = z.copy()
        g = objective.copy()
        if weights is not None:
            g[2:63:3] = -weights
        bounds = base_bounds if floor == 1e-8 else [(-1., 1.), (-1., 1.), (floor, .5)]*n + [(.002, 1.998)]
        prev = np.inf
        slow = 0
        for it in range(iterations):
            check()
            planes(z)
            res = linprog(g, A_ub=A, b_ub=rhs, bounds=bounds, method='highs',
                          options={'dual_feasibility_tolerance': 1e-8,
                                   'primal_feasibility_tolerance': 1e-8,
                                   'time_limit': max(.001, min(1., end-time.monotonic()))})
            if not res.success:
                break
            zz = res.x
            val = float(g @ zz)
            improvement = prev-val
            slow = slow+1 if improvement < 2e-9 else 0
            z = zz
            prev = val
            if slow >= 3:
                break
        return z

    def constraints(z):
        check()
        c = z[:63].reshape(n, 3)
        d = c[ii, :2]-c[jj, :2]
        return np.r_[np.sqrt(np.sum(d*d, axis=1))-c[ii, 2]-c[jj, 2], wall@z+offsets]

    def jacobian(z):
        check()
        c = z[:63].reshape(n, 3)
        d = c[ii, :2]-c[jj, :2]
        d /= np.maximum(np.sqrt(np.sum(d*d, axis=1)), 1e-30)[:, None]
        J = np.zeros((m+84, 64))
        J[ar, 3*ii] = d[:, 0]
        J[ar, 3*jj] = -d[:, 0]
        J[ar, 3*ii+1] = d[:, 1]
        J[ar, 3*jj+1] = -d[:, 1]
        J[ar, 3*ii+2] = -1
        J[ar, 3*jj+2] = -1
        J[m:] = wall
        return J

    def polish(z, seconds=3., iterations=150):
        local_end = min(end, time.monotonic()+seconds)
        last = [z.copy()]
        class SliceEnd(Exception):
            pass
        def fun(x):
            check()
            if time.monotonic() >= local_end:
                raise SliceEnd()
            return float(objective@x)
        def cb(x):
            last[0] = x.copy()
        try:
            r = minimize(fun, z, jac=lambda x: objective, method='SLSQP',
                         bounds=base_bounds,
                         constraints={'type':'ineq', 'fun':constraints, 'jac':jacobian},
                         callback=cb, options={'maxiter':iterations, 'ftol':3e-13})
            return r.x
        except SliceEnd:
            return last[0]

    def randint(a, b):
        if hasattr(rng, 'integers'):
            return int(rng.integers(a, b))
        return int(rng.randint(a, b))

    def relocate(z, number):
        z = z.copy()
        c = z[:63].reshape(n, 3)
        w, h = z[63], 2-z[63]
        probs = 1/np.maximum(c[:, 2], .01)
        probs /= probs.sum()
        ids = rng.choice(n, size=number, replace=False, p=probs if rng.random()<.7 else None)
        keep = np.ones(n, dtype=bool)
        keep[ids] = False
        for j in ids:
            pts = rng.uniform(-.5, .5, size=(450, 2))*[w, h]
            good = c[keep]
            clr = np.minimum(w/2-np.abs(pts[:, 0]), h/2-np.abs(pts[:, 1]))
            if len(good):
                dist = np.sqrt(np.sum((pts[:, None, :]-good[None, :, :2])**2, axis=2))
                clr = np.minimum(clr, np.min(dist-good[None, :, 2], axis=1))
            q = int(np.argmax(clr))
            c[j, :2] = pts[q]
            c[j, 2] = max(1e-7, clr[q]*.9)
            keep[j] = True
        return z

    patterns = [[4,4,5,4,4], [5,5,6,5], [5,6,5,5], [4,5,4,4,4],
                [3,4,3,4,3,4], [7,7,7], [4,4,4,4,5], [6,5,5,5],
                [3,3,3,3,3,3,3], [5,4,3,4,5]]
    def lattice(index):
        counts = list(patterns[index % len(patterns)])
        if rng.random() < .6:
            rng.shuffle(counts)
        c = np.zeros((n, 3))
        p = 0
        for row, count in enumerate(counts):
            c[p:p+count, 0] = 2*np.arange(count)-(count-1)
            c[p:p+count, 0] += (.5 if row % 2 else -.5)*rng.choice([0., 1., 2.])
            c[p:p+count, 1] = row*np.sqrt(3.)
            p += count
        c[:, :2] += rng.normal(size=(n, 2))*.06
        c[:, 2] = .95
        x = repair(c.ravel())
        return encode(x)

    pool = []
    current = encode(best)
    current_value = -best_score
    attempt = 0
    stale = 0
    try:
        x = consider(polish(current, seconds=min(5., .08*budget_s)))
        if x is not None:
            current = encode(x)
            current_value = float(np.sum(x[2::3]))
        pool.append((current_value, current.copy()))
        while time.monotonic() < end:
            check()
            attempt += 1
            oldbest = best_score
            mode = attempt % 12
            if mode in (0, 6):
                z = lattice(attempt//6)
            else:
                u = rng.random()
                if u < .35:
                    z = encode(best)
                elif u < .65 and pool:
                    z = pool[randint(0, len(pool))][1].copy()
                else:
                    z = current.copy()
                c = z[:63].reshape(n, 3)
                if mode in (1, 7):
                    # Sacrifice several circles and insert them into new voids.
                    z = relocate(z, randint(1, 5))
                elif mode in (2, 8):
                    # Broad, temporary objectives let circles exchange area;
                    # this is deliberately much stronger than local jitter.
                    sigma = float(rng.choice([.35, .65, 1., 1.5]))
                    for sweep in range(randint(1, 4)):
                        weights = np.exp(np.clip(rng.normal(size=n)*sigma, -3., 3.))
                        z = slp(z, iterations=randint(5, 16), weights=weights)
                    if rng.random() < .35:
                        z = relocate(z, randint(1, 4))
                elif mode in (3, 9):
                    amp = float(rng.choice([.008, .025, .055, .11]))
                    ids = rng.choice(n, size=randint(3, n+1), replace=False)
                    c[ids, :2] += rng.normal(size=(len(ids), 2))*amp
                    c[ids, 2] *= np.exp(rng.normal(size=len(ids))*.5)
                    c[:, 2] = np.clip(c[:, 2], 1e-7, .49)
                elif mode in (4, 10):
                    # Distort the aspect ratio while preserving normalized
                    # positions. Rectangle shape is an optimization variable.
                    nw = np.clip(z[63]+rng.normal()*.22, .2, 1.8)
                    c[:, 0] *= nw/z[63]
                    c[:, 1] *= (2-nw)/(2-z[63])
                    z[63] = nw
                    c[:, :2] += rng.normal(size=(n, 2))*.012
                else:
                    ids = rng.choice(n, size=randint(2, 8), replace=False)
                    rr = c[ids, 2].copy()
                    rng.shuffle(rr)
                    c[ids, 2] = rr
                    c[ids, :2] += rng.normal(size=(len(ids), 2))*.035
            z[:63].reshape(n, 3)[:, :2] = np.clip(z[:63].reshape(n, 3)[:, :2], -.999, .999)
            # A radius floor occasionally avoids the degenerate boundary of
            # packing space. Release it before final optimization.
            if attempt % 7 == 0:
                z = slp(z, iterations=18, floor=float(rng.choice([.02, .04, .06])))
            z = slp(z, iterations=55)
            x = consider(z)
            if x is None:
                continue
            value = float(np.sum(x[2::3]))
            # Polish promising contact graphs using the actual nonlinear
            # constraints, rather than spending most of the budget on them.
            threshold = .006 if stale < 60 else .018
            if value > -best_score-threshold or attempt % 15 == 0:
                z = polish(encode(x), seconds=min(3., max(.1, end-time.monotonic())), iterations=140)
                xx = consider(z)
                if xx is not None and np.sum(xx[2::3]) > value:
                    x = xx
                    value = float(np.sum(x[2::3]))
            zz = encode(x)
            temp = float([.0015, .004, .009, .018][(attempt//30) % 4])
            if value >= current_value or rng.random() < np.exp(max(-60., (value-current_value)/temp)):
                current, current_value = zz.copy(), value
            # Keep geometrically distinct near-optimal basins, including
            # configurations with different radius distributions.
            signature = np.sort(zz[2:63:3])
            duplicate = False
            for pv, pz in pool:
                if abs(value-pv) < 2e-6 and abs(zz[63]-pz[63]) < 2e-3 and np.max(np.abs(signature-np.sort(pz[2:63:3]))) < 1e-3:
                    duplicate = True
                    break
            if not duplicate:
                pool.append((value, zz.copy()))
                pool.sort(key=lambda t: -t[0])
                pool = pool[:20]
            stale = stale+1 if best_score >= oldbest-1e-9 else 0
    except Timeout:
        pass
    except (ValueError, FloatingPointError, ArithmeticError):
        pass
    return best
