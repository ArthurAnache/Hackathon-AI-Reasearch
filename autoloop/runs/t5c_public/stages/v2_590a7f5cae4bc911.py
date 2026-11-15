def refine(v, score, budget_s, rng):
    import time
    import numpy as np
    from scipy.optimize import minimize

    start = time.monotonic()
    original = np.asarray(v, dtype=np.float64).reshape(63).copy()
    if budget_s < 0.12:
        return original
    deadline = start + budget_s - max(0.04, 0.025 * budget_s)
    best = original.copy()
    best_value = float(score(best))
    n = 21
    ia, ib = np.triu_indices(n, 1)
    m = len(ia)
    rows = np.arange(m)
    k = np.arange(n)

    class Finished(Exception):
        pass

    class SliceFinished(Exception):
        pass

    def check():
        if time.monotonic() >= deadline:
            raise Finished()

    def encode(vec):
        c = np.asarray(vec).reshape(n, 3).copy()
        lo = np.min(c[:, :2] - c[:, 2, None], axis=0)
        hi = np.max(c[:, :2] + c[:, 2, None], axis=0)
        c[:, :2] -= (lo + hi) * 0.5
        wh = hi - lo
        w = wh[0] + max(0.0, 2.0 - wh.sum()) * 0.5
        return np.r_[c.ravel(), np.clip(w, 1e-5, 2-1e-5)]

    def repair(z, eps=1.5e-14):
        c = np.asarray(z[:63], dtype=float).reshape(n, 3).copy()
        if not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return None
        dx = c[ia, 0] - c[ib, 0]
        dy = c[ia, 1] - c[ib, 1]
        d = np.sqrt(dx*dx + dy*dy)
        s = min(1.0, float(np.min(d / (c[ia, 2] + c[ib, 2]))))
        if s <= 0:
            return None
        c[:, 2] *= s * (1-eps)
        lo = np.min(c[:, :2] - c[:, 2, None], axis=0)
        hi = np.max(c[:, :2] + c[:, 2, None], axis=0)
        c[:, :2] -= (lo + hi) * 0.5
        t = c.astype(np.longdouble)
        p = np.sum(np.max(t[:, :2] + t[:, 2, None], axis=0) -
                   np.min(t[:, :2] - t[:, 2, None], axis=0))
        if p <= 0:
            return None
        c *= float(np.longdouble(2) / p) * (1-eps)
        return c.ravel()

    def consider(z, precise=False):
        nonlocal best, best_value
        check()
        candidate = repair(z, 2e-15 if precise else 1.5e-14)
        if candidate is not None and -np.sum(candidate[2::3]) < best_value:
            value = float(score(candidate))
            if value < best_value:
                best, best_value = candidate.copy(), value
        return candidate

    wall_jac = np.zeros((4*n, 64))
    wall_jac[k, 3*k] = 1
    wall_jac[n+k, 3*k] = -1
    wall_jac[2*n+k, 3*k+1] = 1
    wall_jac[3*n+k, 3*k+1] = -1
    for b in range(4):
        wall_jac[b*n+k, 3*k+2] = -1
    wall_jac[:2*n, 63] = .5
    wall_jac[2*n:, 63] = -.5

    def constraints(z):
        check()
        c = z[:63].reshape(n, 3)
        x, y, r = c.T
        w = z[63]
        dx, dy = x[ia]-x[ib], y[ia]-y[ib]
        return np.r_[np.sqrt(dx*dx+dy*dy)-r[ia]-r[ib],
                     .5*w+x-r, .5*w-x-r,
                     1-.5*w+y-r, 1-.5*w-y-r]

    def constraint_jac(z):
        check()
        c = z[:63].reshape(n, 3)
        dx = c[ia, 0]-c[ib, 0]
        dy = c[ia, 1]-c[ib, 1]
        d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-30)
        a = np.zeros((m+4*n, 64))
        a[rows, 3*ia] = dx/d
        a[rows, 3*ib] = -dx/d
        a[rows, 3*ia+1] = dy/d
        a[rows, 3*ib+1] = -dy/d
        a[rows, 3*ia+2] = -1
        a[rows, 3*ib+2] = -1
        a[m:] = wall_jac
        return a

    bounds = [(-1., 1.), (-1., 1.), (1e-8, .5)]*n + [(1e-5, 2-1e-5)]
    cons = {'type': 'ineq', 'fun': constraints, 'jac': constraint_jac}

    def solve(z, weights=None, seconds=4., maxiter=180):
        check()
        local_end = min(deadline, time.monotonic()+seconds)
        g = np.zeros(64)
        g[2:63:3] = -1 if weights is None else -weights
        state = [z.copy(), 0]
        def fun(x):
            check()
            if time.monotonic() >= local_end:
                raise SliceFinished()
            return float(g @ x)
        def jac(x):
            return g
        def callback(x):
            check()
            state[0] = x.copy()
            state[1] += 1
            if weights is None and state[1] % 12 == 0:
                consider(x)
            if time.monotonic() >= local_end:
                raise SliceFinished()
        try:
            res = minimize(fun, z, jac=jac, method='SLSQP', bounds=bounds,
                           constraints=cons, callback=callback,
                           options={'maxiter': maxiter, 'ftol': 8e-13, 'disp': False})
            return res.x
        except SliceFinished:
            return state[0]
        except (ValueError, FloatingPointError):
            return state[0]

    def random_int(a, b):
        return int(rng.integers(a, b)) if hasattr(rng, 'integers') else int(rng.randint(a, b))

    def lattice(index):
        patterns = [[4,4,5,4,4], [5,5,6,5], [5,6,5,5],
                    [4,5,4,4,4], [3,4,3,4,3,4], [7,7,7],
                    [4,4,4,4,5], [3,3,3,3,3,3,3], [6,5,5,5]]
        counts = list(patterns[index % len(patterns)])
        if rng.random() < .5:
            rng.shuffle(counts)
        c = np.zeros((n, 3))
        p = 0
        shift = 0.
        for row, count in enumerate(counts):
            if row and ((count-counts[row-1]) % 2 == 0):
                shift += 1 if row % 2 else -1
            c[p:p+count, 0] = 2*np.arange(count)-(count-1)+shift
            c[p:p+count, 1] = row*np.sqrt(3.)
            p += count
        c[:, :2] += rng.normal(size=(n, 2)) * .035
        c[:, 2] = .98
        if rng.random() < .35:
            c[:, 0] *= rng.uniform(.85, 1.2)
        out = repair(c.ravel())
        return encode(out)

    def relocate(z, count):
        c = z[:63].reshape(n, 3)
        w, h = z[63], 2-z[63]
        if rng.random() < .65:
            prob = 1 / np.maximum(c[:, 2], .005)
            prob /= prob.sum()
            ids = rng.choice(n, size=count, replace=False, p=prob)
        else:
            ids = rng.choice(n, size=count, replace=False)
        keep = np.ones(n, dtype=bool)
        keep[ids] = False
        for j in ids:
            pts = rng.uniform(-.5, .5, size=(650, 2))*[w, h]
            # Include points near contacts, where many useful interstices lie.
            good = c[keep]
            if len(good):
                clr = np.minimum(.5*w-np.abs(pts[:, 0]), .5*h-np.abs(pts[:, 1]))
                dist = np.sqrt(np.sum((pts[:, None, :]-good[None, :, :2])**2, axis=2))
                clr = np.minimum(clr, np.min(dist-good[None, :, 2], axis=1))
            else:
                clr = np.minimum(.5*w-np.abs(pts[:, 0]), .5*h-np.abs(pts[:, 1]))
            # Usually use a large hole, occasionally choose among several holes.
            order = np.argsort(clr)[-12:]
            q = order[-1] if rng.random() < .65 else rng.choice(order)
            c[j, :2] = pts[q]
            c[j, 2] = max(1e-6, clr[q]*.95)
            keep[j] = True
        return z

    pool = []
    current = encode(best)
    current_sum = -best_value
    attempt = 0
    lattice_index = 0
    try:
        consider(current, True)
        polished = solve(current, seconds=min(5., budget_s*.12), maxiter=160)
        feasible = consider(polished, True)
        if feasible is not None:
            current = encode(feasible)
            current_sum = float(np.sum(feasible[2::3]))
        pool.append((current_sum, current.copy()))
        while time.monotonic() < deadline:
            check()
            attempt += 1
            mode = attempt % 10
            if mode in (0, 5):
                z = lattice(lattice_index)
                lattice_index += 1
            else:
                u = rng.random()
                if u < .35:
                    z = encode(best)
                elif u < .60 and pool:
                    z = pool[random_int(0, len(pool))][1].copy()
                else:
                    z = current.copy()
                c = z[:63].reshape(n, 3)
                if mode in (1, 6):
                    z = relocate(z, random_int(1, 5))
                    c[:, :2] += rng.normal(size=(n, 2))*.002
                elif mode in (2, 7):
                    # Radius exchanges change the contact topology much more
                    # effectively than repeated microscopic coordinate noise.
                    ids = rng.choice(n, size=random_int(2, 6), replace=False)
                    rr = c[ids, 2].copy()
                    rng.shuffle(rr)
                    c[ids, 2] = rr
                    c[ids, :2] += rng.normal(size=(len(ids), 2))*.025
                elif mode in (3, 8):
                    amp = float(rng.choice([.018, .04, .075, .13]))
                    ids = rng.choice(n, size=random_int(4, n+1), replace=False)
                    c[ids, :2] += rng.normal(size=(len(ids), 2))*amp
                    c[ids, 2] *= np.exp(rng.normal(size=len(ids))*.25)
                    c[:, 2] = np.clip(c[:, 2], 1e-7, .49)
                    z[63] = np.clip(z[63]+rng.normal()*.12, .12, 1.88)
                else:
                    # A temporary random objective crosses jammed contact
                    # boundaries; the true objective is restored afterwards.
                    weights = np.exp(rng.normal(size=n)*float(rng.choice([.12, .25, .45])))
                    z = solve(z, weights, seconds=min(2.5, budget_s*.025), maxiter=85)
                    z[2:63:3] = np.clip(z[2:63:3], 1e-8, .5)
            c = z[:63].reshape(n, 3)
            c[:, :2] = np.clip(c[:, :2], -.999, .999)
            c[:, 2] = np.clip(c[:, 2], 1e-8, .5)
            remaining = deadline-time.monotonic()
            result = solve(z, seconds=min(5., max(.15, remaining*.35)), maxiter=210)
            candidate = consider(result)
            if candidate is None:
                continue
            val = float(np.sum(candidate[2::3]))
            zz = encode(candidate)
            # Maintain multiple basins, including slightly inferior ones.
            temperature = (.001, .003, .008, .016)[(attempt//12) % 4]
            if val >= current_sum or rng.random() < np.exp(max(-60., (val-current_sum)/temperature)):
                current, current_sum = zz.copy(), val
            duplicate = any(abs(val-p[0]) < 2e-7 and
                            abs(zz[63]-p[1][63]) < 2e-4 for p in pool)
            if not duplicate:
                pool.append((val, zz.copy()))
                pool.sort(key=lambda t: -t[0])
                pool = pool[:12]
            if attempt % 9 == 0:
                consider(encode(best), True)
    except Finished:
        pass
    except (ValueError, FloatingPointError, ArithmeticError):
        pass
    return best
