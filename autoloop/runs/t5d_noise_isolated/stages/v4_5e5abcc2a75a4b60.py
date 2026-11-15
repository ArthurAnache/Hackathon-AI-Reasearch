def refine(v, score, budget_s, rng):
    import time
    import numpy as np
    from scipy.optimize import minimize, linprog

    start = time.monotonic()
    deadline = start + max(0., budget_s) - min(1., .02*max(0., budget_s))
    n = 21
    original = np.asarray(v, dtype=np.float64).reshape(n, 3).copy()
    best = original.copy()
    best_value = score(best.ravel())
    best_sum = best[:, 2].sum()

    class Stop(Exception):
        pass

    def check():
        if time.monotonic() >= deadline:
            raise Stop()

    def repair(c):
        c = np.array(c, dtype=float, copy=True)
        if not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return None
        m = len(c)
        i, j = np.triu_indices(m, 1)
        dist = np.hypot(c[i, 0]-c[j, 0], c[i, 1]-c[j, 1])
        f = min(1., np.min(dist/(c[i, 2]+c[j, 2])))
        if f <= 0:
            return None
        c[:, 2] *= f*(1.-5e-15)
        lo = np.min(c[:, :2]-c[:, 2, None], axis=0)
        hi = np.max(c[:, :2]+c[:, 2, None], axis=0)
        c[:, :2] -= (lo+hi)*.5
        c *= (2./np.sum(hi-lo))*(1.-5e-15)
        return c

    def consider(c):
        nonlocal best, best_sum, best_value
        if len(c) != n:
            return
        c = repair(c)
        if c is None or c[:, 2].sum() <= best_sum:
            return
        check()
        val = score(c.ravel())
        if val < best_value:
            best, best_value = c.copy(), val
            best_sum = c[:, 2].sum()
        elif not np.isfinite(val):
            c[:, 2] *= 1.-8e-14
            c = repair(c)
            if c is not None and c[:, 2].sum() > best_sum:
                val = score(c.ravel())
                if val < best_value:
                    best, best_value = c.copy(), val
                    best_sum = c[:, 2].sum()

    models = {}
    def model(m):
        if m in models:
            return models[m]
        i, j = np.triu_indices(m, 1)
        k = np.arange(len(i))
        a = np.arange(m)
        d = 3*m+1
        A = np.zeros((4*m, d))
        A[a, a] = 1.
        A[a, 2*m+a] = -1.
        A[m+a, a] = -1.
        A[m+a, 2*m+a] = -1.
        A[m+a, -1] = 1.
        A[2*m+a, m+a] = 1.
        A[2*m+a, 2*m+a] = -1.
        A[3*m+a, m+a] = -1.
        A[3*m+a, 2*m+a] = -1.
        A[3*m+a, -1] = -1.
        b = np.zeros(4*m)
        b[3*m:] = 2.
        bounds = [(0., 2.)]*(2*m)+[(1e-9, .999999)]*m+[(.025, 1.975)]
        models[m] = (i, j, k, A, b, bounds)
        return models[m]

    def pack(c):
        m = len(c)
        c = c.copy()
        lo = np.min(c[:, :2]-c[:, 2, None], axis=0)
        hi = np.max(c[:, :2]+c[:, 2, None], axis=0)
        scale = 2./np.sum(hi-lo)
        c[:, :2] -= lo
        c *= scale
        return np.r_[c[:, 0], c[:, 1], c[:, 2], (hi[0]-lo[0])*scale]

    def unpack(z):
        m = (len(z)-1)//3
        return np.column_stack((z[:m], z[m:2*m], z[2*m:3*m]))

    def geometry(z, deriv=True):
        m = (len(z)-1)//3
        i, j, k, A, b, bounds = model(m)
        dx, dy = z[i]-z[j], z[m+i]-z[m+j]
        ds = np.maximum(np.hypot(dx, dy), 1e-24)
        g = ds-z[2*m+i]-z[2*m+j]
        wall = np.r_[z[:m]-z[2*m:3*m], z[-1]-z[:m]-z[2*m:3*m],
                     z[m:2*m]-z[2*m:3*m], 2.-z[-1]-z[m:2*m]-z[2*m:3*m]]
        if not deriv:
            return np.r_[wall, g]
        J = np.zeros((len(i), 3*m+1))
        J[k, i] = dx/ds
        J[k, j] = -dx/ds
        J[k, m+i] = dy/ds
        J[k, m+j] = -dy/ds
        J[k, 2*m+i] = -1.
        J[k, 2*m+j] = -1.
        return np.r_[wall, g], np.vstack((A, J))

    def solve(c, weights=None, seconds=2., nit=110, tight=False):
        check()
        end = min(deadline, time.monotonic()+seconds)
        m = len(c)
        z = pack(c)
        i, j, k, A, b, bounds = model(m)
        g = np.zeros(3*m+1)
        g[2*m:3*m] = -1. if weights is None else -np.asarray(weights)
        last = [z.copy()]
        feasible = [None, float('inf')]
        cache = [None, None, None]
        def cons(x):
            if time.monotonic() >= end:
                raise Stop()
            if cache[0] is None or not np.array_equal(x, cache[0]):
                h, J = geometry(x)
                cache[:] = [x.copy(), h, J]
            return cache[1]
        def jac(x):
            cons(x)
            return cache[2]
        def callback(x):
            last[0] = x.copy()
            h = cons(x)
            value = np.dot(g, x)
            if h.min() >= -2e-9 and value < feasible[1]:
                feasible[:] = [x.copy(), value]
            if m == n and x[2*m:3*m].sum() > best_sum and h.min() >= -2e-8:
                consider(unpack(x))
        try:
            res = minimize(lambda x: np.dot(g, x), z, jac=lambda x: g,
                           method='SLSQP', bounds=bounds,
                           constraints={'type': 'ineq', 'fun': cons, 'jac': jac},
                           callback=callback,
                           options={'maxiter': nit, 'ftol': 8e-14 if tight else 2e-10, 'disp': False})
            last[0] = res.x.copy()
        except Stop:
            pass
        check()
        choices = [repair(unpack(last[0]))]
        if feasible[0] is not None:
            choices.append(repair(unpack(feasible[0])))
        choices = [x for x in choices if x is not None]
        if not choices:
            return repair(c)
        if weights is None:
            out = max(choices, key=lambda x: x[:, 2].sum())
        else:
            out = max(choices, key=lambda x: np.dot(weights, x[:, 2]))
        if m == n:
            consider(out)
        return out

    def polish(c, steps=12):
        z = pack(c)
        m = len(c)
        i, j, k, A, b, bounds = model(m)
        obj = np.zeros(len(z))
        obj[2*m:3*m] = -1.
        prev = -1.
        for t in range(steps):
            check()
            h, J = geometry(z)
            # The norm supporting planes are global inner approximations.
            rhs = np.einsum('ij,j->i', J, z)-h
            trust = min(.08, .018*1.4**t)
            bb = [(max(a, x-trust), min(b0, x+trust)) for x, (a, b0) in zip(z, bounds)]
            res = linprog(obj, A_ub=-J, b_ub=-rhs, bounds=bb, method='highs',
                          options={'primal_feasibility_tolerance': 1e-9,
                                   'dual_feasibility_tolerance': 1e-9,
                                   'time_limit': max(.001, min(.35, deadline-time.monotonic()))})
            if res.x is None:
                break
            z = res.x
            consider(unpack(z))
            val = z[2*m:3*m].sum()
            if t > 2 and val-prev < 2e-13:
                break
            prev = val
        return repair(unpack(z))

    def holes(c, count, randomize=True):
        # Optimize actual empty circles, rather than inserting at a random grid point.
        lo = np.min(c[:, :2]-c[:, 2, None], axis=0)
        hi = np.max(c[:, :2]+c[:, 2, None], axis=0)
        for step in range(count):
            check()
            pts = rng.uniform(lo, hi, (1500, 2))
            grid = np.linspace(.025, .975, 18)
            xx, yy = np.meshgrid(grid, grid)
            pts = np.vstack((pts, lo+np.column_stack((xx.ravel(), yy.ravel()))*(hi-lo)))
            ds = np.linalg.norm(pts[:, None, :]-c[None, :, :2], axis=2)-c[None, :, 2]
            clear = np.minimum(ds.min(axis=1), np.minimum(pts-lo, hi-pts).min(axis=1))
            order = np.argsort(clear)[::-1]
            seeds = []
            for q in order:
                if clear[q] <= 0:
                    break
                if all(np.linalg.norm(pts[q]-s[:2]) > .055 for s in seeds):
                    seeds.append(np.r_[pts[q], clear[q]])
                    if len(seeds) >= 7:
                        break
            if not seeds:
                seeds = [np.r_[pts[order[0]], max(1e-8, clear[order[0]])]]
            def fun(z):
                dx = z[:2]-c[:, :2]
                dist = np.maximum(np.linalg.norm(dx, axis=1), 1e-25)
                return np.r_[dist-c[:, 2]-z[2], z[:2]-lo-z[2], hi-z[:2]-z[2]]
            def jac(z):
                dx = z[:2]-c[:, :2]
                dist = np.maximum(np.linalg.norm(dx, axis=1), 1e-25)
                return np.vstack((np.column_stack((dx/dist[:, None], -np.ones(len(c)))),
                                  [[1, 0, -1], [0, 1, -1], [-1, 0, -1], [0, -1, -1]]))
            found = []
            for seed in seeds:
                check()
                r = minimize(lambda z: -z[2], seed, jac=lambda z: np.array([0., 0., -1.]),
                             method='SLSQP', bounds=[(lo[0], hi[0]), (lo[1], hi[1]), (1e-10, 1.)],
                             constraints={'type': 'ineq', 'fun': fun, 'jac': jac},
                             options={'maxiter': 45, 'ftol': 2e-12, 'disp': False})
                z = r.x
                radius = min(np.min(np.linalg.norm(z[:2]-c[:, :2], axis=1)-c[:, 2]),
                             np.min(z[:2]-lo), np.min(hi-z[:2]))
                if radius > 1e-8 and all(np.linalg.norm(z[:2]-s[:2]) > 1e-4 for s in found):
                    found.append(np.r_[z[:2], radius*(1.-1e-10)])
            if found:
                found.sort(key=lambda z: z[2], reverse=True)
                q = 0
                if randomize and len(found) > 1 and rng.random() < .3:
                    eligible = [s for s in found if s[2] > .65*found[0][2]]
                    q = int(rng.integers(len(eligible)))
                new = found[q]
            else:
                new = seeds[0]
                new[2] = max(1e-9, new[2]*.8)
            c = np.vstack((c, new))
        return c

    def delete_relax(c):
        count = int(rng.choice([1, 2, 2, 3, 4, 5]))
        if rng.random() < .45:
            p = 1./(c[:, 2]+.015)
            removed = rng.choice(n, count, replace=False, p=p/p.sum())
        else:
            q = int(rng.integers(n))
            removed = np.argsort(np.linalg.norm(c[:, :2]-c[q, :2], axis=1))[:count]
        keep = np.ones(n, bool)
        keep[removed] = False
        small = c[keep].copy()
        # Let the surviving contact network collapse into the vacancies first.
        small = solve(small, seconds=1.6, nit=85)
        if small is None:
            return c
        return holes(small, count)

    def split_move(c):
        c = c.copy()
        count = int(rng.integers(1, 4))
        p = 1./(c[:, 2]+.005)**1.5
        removed = rng.choice(n, count, replace=False, p=p/p.sum())
        c = np.delete(c, removed, axis=0)
        for k in range(count):
            p = c[:, 2]**2
            j = int(rng.choice(len(c), p=p/p.sum()))
            x, y, r = c[j]
            angle = rng.uniform(0, 2*np.pi)
            u = np.array([np.cos(angle), np.sin(angle)])
            f = rng.uniform(.35, .65)
            c[j, :2] = np.array([x, y])+u*r*(1.-f)
            c[j, 2] = r*f
            c = np.vstack((c, np.r_[np.array([x, y])-u*r*f, r*(1.-f)]))
        return c

    def fresh():
        nr = int(rng.integers(3, 8))
        counts = np.full(nr, n//nr)
        counts[:n % nr] += 1
        for _ in range(int(rng.integers(0, 6))):
            a, b = rng.choice(nr, 2, replace=False)
            if counts[a] > 1:
                counts[a] -= 1
                counts[b] += 1
        rng.shuffle(counts)
        c = []
        for row, ct in enumerate(counts):
            offset = (max(counts)-ct)*rng.uniform(.3, 1.)
            for col in range(ct):
                c.append([2.*col+offset+.7*(row % 2), 1.73*row, 1.])
        c = np.array(c)
        c[:, :2] += rng.normal(0, .14, (n, 2))
        c[:, 2] *= np.exp(rng.normal(0, .28, n))
        return c

    try:
        consider(best)
        c = polish(best, 15)
        if c is not None:
            solve(c, seconds=3., nit=180, tight=True)
        current = best.copy()
        pool = [best.copy()]
        attempt = 0
        stagnant = 0
        while time.monotonic() < deadline:
            attempt += 1
            previous = best_sum
            mode = attempt % 10
            if rng.random() < .25:
                seed = best.copy()
            elif rng.random() < .2:
                seed = pool[int(rng.integers(len(pool)))].copy()
            else:
                seed = current.copy()
            if mode in (0, 1, 6):
                seed = delete_relax(seed)
            elif mode in (2, 7):
                seed = split_move(seed)
            elif mode == 3:
                # An entire weighted optimization, not just a noisy starting radius.
                weights = np.exp(rng.normal(0., rng.uniform(.45, 1.15), n))
                weights = np.clip(weights, .07, 8.)
                seed = solve(seed, weights, seconds=2., nit=100)
            elif mode == 4:
                # Reflect a disk across a nearby contact: a discrete triangulation flip.
                dist = np.linalg.norm(seed[:, None, :2]-seed[None, :, :2], axis=2)
                q = int(rng.integers(n))
                order = np.argsort(dist[q]-seed[:, 2]-seed[q, 2])
                near = [int(x) for x in order if x != q][:6]
                a, b = rng.choice(near, 2, replace=False)
                u = seed[b, :2]-seed[a, :2]
                u /= max(np.linalg.norm(u), 1e-20)
                p = seed[q, :2]-seed[a, :2]
                seed[q, :2] = seed[a, :2]+2.*u*np.dot(p, u)-p
                seed[q, 2] *= .8
            elif mode == 5:
                seed = fresh()
            elif mode == 8:
                s = np.exp(rng.normal(0., .24))
                seed[:, 0] *= s
                seed[:, 1] /= s
                seed[:, 0] += rng.normal(0, .07)*seed[:, 1]
            else:
                take = rng.choice(n, int(rng.integers(3, n+1)), replace=False)
                amp = float(rng.choice([.015, .035, .065, .12]))
                seed[take, :2] += rng.normal(0, amp, (len(take), 2))
                seed[take, 2] *= np.exp(rng.normal(0, .35, len(take)))
            if seed is None:
                continue
            cand = solve(seed, seconds=3.5, nit=160)
            if cand is not None:
                cs = cand[:, 2].sum()
                delta = cs-current[:, 2].sum()
                temperature = .006+.025*min(1., stagnant/25.)
                if delta >= 0 or rng.random() < np.exp(max(-700., delta/temperature)):
                    current = cand.copy()
                if cs > best_sum-.10 and all(np.max(np.abs(np.sort(cand[:, 2])-np.sort(p[:, 2]))) > 5e-4 for p in pool):
                    pool.append(cand.copy())
                    if len(pool) > 18:
                        pool.pop(int(rng.integers(len(pool))))
            if best_sum > previous+2e-9:
                stagnant = 0
                c = polish(best, 14)
                if c is not None:
                    solve(c, seconds=2., nit=110, tight=True)
                current = best.copy()
            else:
                stagnant += 1
            if stagnant and stagnant % 30 == 0:
                current = pool[int(rng.integers(len(pool)))].copy()
    except Stop:
        pass
    return best.ravel()
