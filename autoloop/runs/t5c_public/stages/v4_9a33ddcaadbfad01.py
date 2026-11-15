def refine(v, score, budget_s, rng):
    import time
    import numpy as np
    from scipy.optimize import minimize, linprog

    t0 = time.monotonic()
    original = np.asarray(v, dtype=np.float64).reshape(63).copy()
    if budget_s < .3:
        return original
    deadline = t0 + budget_s - max(.12, .015 * budget_s)
    best = original.copy()
    best_score = float(score(best))
    cache = {}

    class Stop(Exception):
        pass

    def check():
        if time.monotonic() >= deadline:
            raise Stop()

    def integer(a, b):
        return int(rng.integers(a, b)) if hasattr(rng, 'integers') else int(rng.randint(a, b))

    def matrices(n):
        if n in cache:
            return cache[n]
        i, j = np.triu_indices(n, 1)
        a = np.arange(len(i))
        k = np.arange(n)
        dim = 3*n+1
        W = np.zeros((4*n, dim))
        W[k, 3*k] = 1
        W[n+k, 3*k] = -1
        W[2*n+k, 3*k+1] = 1
        W[3*n+k, 3*k+1] = -1
        for b in range(4):
            W[b*n+k, 3*k+2] = -1
        W[:2*n, -1] = .5
        W[2*n:, -1] = -.5
        off = np.r_[np.zeros(2*n), np.ones(2*n)]
        g = np.zeros(dim)
        g[2:3*n:3] = -1
        bounds = [(-1., 1.), (-1., 1.), (1e-8, .6)]*n + [(.04, 1.96)]
        cache[n] = i, j, a, W, off, g, bounds
        return cache[n]

    def encode(x):
        c = np.asarray(x).reshape(-1, 3).copy()
        lo = np.min(c[:, :2]-c[:, 2, None], axis=0)
        hi = np.max(c[:, :2]+c[:, 2, None], axis=0)
        c[:, :2] -= (lo+hi)/2
        wh = hi-lo
        return np.r_[c.ravel(), np.clip(wh[0]+(2-wh.sum())/2, .04, 1.96)]

    def repair(z, eps=1.4e-15):
        c = np.asarray(z[:63]).reshape(21, 3).copy()
        if not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return None
        i, j, _, _, _, _, _ = matrices(21)
        d = c[i, :2]-c[j, :2]
        d = np.sqrt(np.sum(d*d, axis=1))
        f = min(1., float(np.min(d/(c[i, 2]+c[j, 2]))))
        if f <= 0:
            return None
        c[:, 2] *= f*(1-eps)
        lo = np.min(c[:, :2]-c[:, 2, None], axis=0)
        hi = np.max(c[:, :2]+c[:, 2, None], axis=0)
        c[:, :2] -= (lo+hi)/2
        q = c.astype(np.longdouble)
        p = np.sum(np.max(q[:, :2]+q[:, 2, None], axis=0)-np.min(q[:, :2]-q[:, 2, None], axis=0))
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
        value = -float(np.sum(x[2::3]))
        if value < best_score:
            s = float(score(x))
            if s < best_score:
                best, best_score = x.copy(), s
            elif value < best_score-2e-14:
                for eps in (6e-15, 3e-14, 2e-13):
                    xx = repair(z, eps)
                    if xx is None or -np.sum(xx[2::3]) >= best_score:
                        break
                    ss = float(score(xx))
                    if ss < best_score:
                        best, best_score = xx.copy(), ss
                        x = xx
                        break
        return x

    def optimize(z, weights=None, iterations=100, seconds=2., ftol=8e-12):
        check()
        n = (len(z)-1)//3
        i, j, a, W, off, g0, bounds = matrices(n)
        g = g0.copy()
        if weights is not None:
            g[2:3*n:3] = -np.asarray(weights)
        until = min(deadline, time.monotonic()+seconds)
        last = [z.copy()]
        class Slice(Exception):
            pass
        def timer():
            check()
            if time.monotonic() >= until:
                raise Slice()
        def fun(x):
            timer()
            return float(g@x)
        def constraints(x):
            timer()
            c = x[:-1].reshape(n, 3)
            d = c[i, :2]-c[j, :2]
            return np.r_[np.sqrt(np.sum(d*d, axis=1))-c[i, 2]-c[j, 2], W@x+off]
        def jac(x):
            c = x[:-1].reshape(n, 3)
            d = c[i, :2]-c[j, :2]
            d /= np.maximum(np.sqrt(np.sum(d*d, axis=1)), 1e-25)[:, None]
            J = np.zeros((len(i)+4*n, 3*n+1))
            J[a, 3*i] = d[:, 0]
            J[a, 3*j] = -d[:, 0]
            J[a, 3*i+1] = d[:, 1]
            J[a, 3*j+1] = -d[:, 1]
            J[a, 3*i+2] = -1
            J[a, 3*j+2] = -1
            J[len(i):] = W
            return J
        def cb(x):
            last[0] = x.copy()
            timer()
        z = np.asarray(z).copy()
        for k, (lo, hi) in enumerate(bounds):
            z[k] = np.clip(z[k], lo, hi)
        try:
            r = minimize(fun, z, jac=lambda x:g, method='SLSQP', bounds=bounds,
                         constraints={'type':'ineq', 'fun':constraints, 'jac':jac},
                         callback=cb, options={'maxiter':iterations, 'ftol':ftol})
            return r.x
        except Slice:
            return last[0]

    def slp(z, iterations=12):
        n = (len(z)-1)//3
        i, j, a, W, off, g, bounds = matrices(n)
        m = len(i)
        A = np.zeros((m+4*n, 3*n+1))
        A[m:] = -W
        A[a, 3*i+2] = 1
        A[a, 3*j+2] = 1
        rhs = np.r_[np.zeros(m), off]
        prev = 1e9
        for _ in range(iterations):
            check()
            c = z[:-1].reshape(n, 3)
            d = c[i, :2]-c[j, :2]
            d /= np.maximum(np.sqrt(np.sum(d*d, axis=1)), 1e-20)[:, None]
            A[a, 3*i] = -d[:, 0]
            A[a, 3*j] = d[:, 0]
            A[a, 3*i+1] = -d[:, 1]
            A[a, 3*j+1] = d[:, 1]
            r = linprog(g, A_ub=A, b_ub=rhs, bounds=bounds, method='highs',
                        options={'time_limit':max(.001, min(.5, deadline-time.monotonic())),
                                 'primal_feasibility_tolerance':1e-9,
                                 'dual_feasibility_tolerance':1e-9})
            if not r.success:
                break
            z = r.x
            val = float(g@z)
            if prev-val < 1e-10:
                break
            prev = val
        return z

    def insert(z, extra=None):
        # Maximal empty disks, rather than random relocations. Optimizing the
        # smaller packing first lets its contact graph reorganize around a vacancy.
        check()
        c = z[:-1].reshape(-1, 3)
        w = float(z[-1]); h = 2-w
        pts = rng.uniform(-.5, .5, size=(1100, 2))*[w, h]
        q = np.linspace(-.499, .499, 34)
        edge = np.vstack((np.c_[q*w, np.full(len(q), -.499*h)],
                          np.c_[q*w, np.full(len(q), .499*h)],
                          np.c_[np.full(len(q), -.499*w), q*h],
                          np.c_[np.full(len(q), .499*w), q*h]))
        pts = np.vstack((pts, edge))
        if extra is not None:
            pts = np.vstack((pts, np.asarray(extra).reshape(-1, 2)))
        dist = np.sqrt(np.sum((pts[:, None, :]-c[None, :, :2])**2, axis=2))
        clear = np.minimum(np.minimum(w/2-np.abs(pts[:, 0]), h/2-np.abs(pts[:, 1])),
                           np.min(dist-c[None, :, 2], axis=1))
        order = np.argsort(clear)[::-1]
        seeds = []
        for k in order:
            p = pts[k]
            if all(np.linalg.norm(p-s[:2]) > .055 for s in seeds):
                seeds.append(np.r_[p, max(1e-7, clear[k])])
            if len(seeds) == 7:
                break
        def con(x):
            check()
            d = x[:2]-c[:, :2]
            return np.r_[np.sqrt(np.sum(d*d, axis=1))-c[:, 2]-x[2],
                         w/2-x[0]-x[2], w/2+x[0]-x[2],
                         h/2-x[1]-x[2], h/2+x[1]-x[2]]
        def jac(x):
            d = x[:2]-c[:, :2]
            d /= np.maximum(np.sqrt(np.sum(d*d, axis=1)), 1e-25)[:, None]
            return np.vstack((np.c_[d, -np.ones(len(c))],
                              [-1,0,-1], [1,0,-1], [0,-1,-1], [0,1,-1]))
        winner = seeds[0].copy()
        winrad = -1.
        for seed in seeds:
            check()
            r = minimize(lambda x:-x[2], seed, jac=lambda x:np.array([0.,0.,-1.]),
                         method='SLSQP', bounds=[(-w/2,w/2),(-h/2,h/2),(1e-8,.6)],
                         constraints={'type':'ineq','fun':con,'jac':jac},
                         options={'maxiter':32,'ftol':2e-11})
            x = r.x
            feasible_radius = min(w/2-abs(x[0]), h/2-abs(x[1]),
                                  np.min(np.linalg.norm(c[:, :2]-x[:2], axis=1)-c[:, 2]))
            if feasible_radius > winrad:
                winner = np.r_[x[:2], max(1e-8, feasible_radius*.999999)]
                winrad = feasible_radius
        return np.r_[c.ravel(), winner, w]

    def vacancy_move(z, count):
        c = z[:-1].reshape(21,3).copy()
        if rng.random() < .65:
            p = 1/np.maximum(c[:,2], .018)**float(rng.uniform(.3, 1.7))
            p /= p.sum()
            ids = rng.choice(21, size=count, replace=False, p=p)
        else:
            first = integer(0,21)
            ds = np.linalg.norm(c[:,:2]-c[first,:2], axis=1)
            ids = np.argsort(ds)[:count]
        keep = np.ones(21, dtype=bool)
        keep[ids] = False
        holes = c[ids,:2].copy()
        reduced = np.r_[c[keep].ravel(), z[-1]]
        reduced = optimize(reduced, iterations=70, seconds=1.2)
        for _ in ids:
            reduced = insert(reduced, holes)
        return reduced

    def perturb(z, mode):
        z = z.copy()
        c = z[:-1].reshape(21,3)
        if mode == 0:
            amp = float(rng.choice([.002,.005,.012,.025,.045]))
            anchor = integer(0,21)
            ds = np.linalg.norm(c[:,:2]-c[anchor,:2], axis=1)
            ids = np.argsort(ds)[:integer(3,14)]
            c[ids,:2] += rng.normal(size=(len(ids),2))*amp
            c[ids,2] *= np.exp(rng.normal(size=len(ids))*.1)
        elif mode == 1:
            # Infeasible tangential kicks can cross a contact barrier that
            # successive supporting-plane LPs cannot cross.
            i,j,_,_,_,_,_ = matrices(21)
            d = c[i,:2]-c[j,:2]
            norm = np.linalg.norm(d,axis=1)
            gap = norm-c[i,2]-c[j,2]
            active = np.flatnonzero(gap < .0015)
            if len(active):
                e = int(rng.choice(active))
                u = d[e]/max(norm[e],1e-20)
                tangent = np.array([-u[1],u[0]])
                amp = rng.uniform(.008,.065)
                c[i[e],:2] += tangent*amp
                c[j[e],:2] -= tangent*amp
                midpoint = (c[i[e],:2]+c[j[e],:2])/2
                nearby = np.linalg.norm(c[:,:2]-midpoint,axis=1) < .26
                c[nearby,:2] += rng.normal(size=(int(nearby.sum()),2))*.006
        elif mode == 2:
            nw = float(np.clip(z[-1]+rng.normal()*rng.choice([.025,.07,.18]), .25,1.75))
            c[:,0] *= nw/z[-1]
            c[:,1] *= (2-nw)/(2-z[-1])
            z[-1] = nw
        else:
            ids = rng.choice(21, size=integer(2,6), replace=False)
            rr = c[ids,2].copy()
            rng.shuffle(rr)
            c[ids,2] = rr
            c[ids,:2] += rng.normal(size=(len(ids),2))*.018
        c[:,2] = np.clip(c[:,2], 1e-7, .55)
        c[:,:2] = np.clip(c[:,:2], -.97,.97)
        return z

    def seed_lattice(attempt):
        patterns = [[4,4,5,4,4],[5,5,6,5],[3,4,3,4,3,4],
                    [5,4,4,4,4],[3,3,3,3,3,3,3],[6,5,5,5],
                    [5,4,3,4,5],[7,7,7],[2,3,4,3,4,3,2]]
        counts = list(patterns[(attempt//11) % len(patterns)])
        if rng.random()<.5:
            rng.shuffle(counts)
        c = np.zeros((21,3)); p = 0
        for row, k in enumerate(counts):
            c[p:p+k,0] = 2*np.arange(k)-(k-1)+(.5 if row%2 else -.5)*rng.choice([0.,1.])
            c[p:p+k,1] = row*np.sqrt(3.)
            p += k
        c[:,:2] += rng.normal(size=(21,2))*.07
        c[:,2] = .96
        x = repair(c.ravel())
        return encode(x)

    archive = []
    current = encode(best)
    current_val = -best_score
    attempt = 0
    try:
        z = optimize(current, iterations=160, seconds=min(5., budget_s*.06), ftol=2e-14)
        x = consider(z)
        if x is not None:
            current = encode(x)
            current_val = float(np.sum(x[2::3]))
        archive.append((current_val,current.copy()))
        while time.monotonic() < deadline:
            check()
            attempt += 1
            u = rng.random()
            if u < .48:
                base = encode(best)
            elif u < .76:
                base = current.copy()
            else:
                base = archive[integer(0,len(archive))][1].copy()
            mode = attempt % 11
            if mode in (0, 4, 8):
                z = vacancy_move(base, integer(1, 4) if mode != 0 else 1)
            elif mode in (2, 6, 10):
                # Mild tilting changes contacts without eliminating most of
                # the small circles, as broad lognormal weights tend to do.
                sigma = float(rng.choice([.025,.06,.12,.23,.4]))
                weights = np.exp(rng.normal(size=21)*sigma)
                z = optimize(base, weights=weights, iterations=65, seconds=1.2)
                if rng.random() < .3:
                    z = perturb(z, 1)
            elif mode == 9 and attempt % 33 == 9:
                z = seed_lattice(attempt)
                z = slp(z, iterations=25)
            else:
                z = perturb(base, integer(0,4))
            z = optimize(z, iterations=110, seconds=2.0)
            x = consider(z)
            if x is None:
                continue
            value = float(np.sum(x[2::3]))
            # Finish near-optimal basins accurately, but don't spend the
            # whole budget converging obviously inferior topologies.
            if value > -best_score-.0035:
                z = optimize(encode(x), iterations=180, seconds=2.5, ftol=3e-14)
                xx = consider(z)
                if xx is not None and np.sum(xx[2::3]) > value:
                    x = xx
                    value = float(np.sum(x[2::3]))
            elif attempt % 8 == 0:
                z = slp(encode(x), iterations=15)
                xx = consider(z)
                if xx is not None and np.sum(xx[2::3]) > value:
                    x = xx
                    value = float(np.sum(x[2::3]))
            zz = encode(x)
            temperature = [.001,.003,.007,.014][(attempt//24)%4]
            if value >= current_val or rng.random() < np.exp(max(-60., (value-current_val)/temperature)):
                current, current_val = zz.copy(), value
            sig = np.sort(x[2::3])
            duplicate = False
            for av, az in archive:
                if abs(av-value)<1e-6 and abs(az[-1]-zz[-1])<.002 and np.max(np.abs(np.sort(az[2:-1:3])-sig))<.001:
                    duplicate = True
                    break
            if not duplicate:
                archive.append((value,zz.copy()))
                archive.sort(key=lambda p:-p[0])
                archive = archive[:16]
            if attempt % 45 == 0:
                current = encode(best)
                current_val = -best_score
    except Stop:
        pass
    except (ValueError, FloatingPointError, ArithmeticError):
        pass
    return best
