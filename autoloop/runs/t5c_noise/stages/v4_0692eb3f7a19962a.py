def refine(v, score, budget_s, rng):
    import time
    import itertools
    import numpy as np
    from fractions import Fraction
    from scipy.optimize import minimize, nnls
    from scipy.linalg import lstsq

    start = time.monotonic()
    end = start + max(0., budget_s) - min(.8, max(.03, .015 * budget_s))
    best = np.asarray(v, dtype=np.float64).reshape(63).copy()
    best_value = float(score(best))
    best_sum = float(best[2::3].sum())
    if budget_s < .15:
        return best
    n = 21
    ii, jj = np.triu_indices(n, 1)
    p = np.arange(n)
    rr = 84 + np.arange(len(ii))
    g = np.zeros(64)
    g[2:63:3] = -1.
    base = np.zeros((294, 64))
    base[p, 3*p] = 1
    base[p, 3*p+2] = -1
    base[21+p, 3*p] = -1
    base[21+p, 3*p+2] = -1
    base[21+p, 63] = 1
    base[42+p, 3*p+1] = 1
    base[42+p, 3*p+2] = -1
    base[63+p, 3*p+1] = -1
    base[63+p, 3*p+2] = -1
    base[63+p, 63] = -1
    base[rr, 3*ii+2] = -1
    base[rr, 3*jj+2] = -1
    bounds = [(0., 2.), (0., 2.), (1.e-11, 1.)] * n + [(.015, 1.985)]
    archive = []

    class Stop(Exception):
        pass

    def check():
        if time.monotonic() >= end:
            raise Stop()

    def constraints(z):
        a = z[:63].reshape(n, 3)
        x, y, r = a.T
        return np.r_[x-r, z[63]-x-r, y-r, 2-z[63]-y-r,
                     np.sqrt((x[ii]-x[jj])**2 + (y[ii]-y[jj])**2)-r[ii]-r[jj]]

    def jac(z):
        a = z[:63].reshape(n, 3)
        dx = a[ii, 0]-a[jj, 0]
        dy = a[ii, 1]-a[jj, 1]
        d = np.maximum(np.sqrt(dx*dx+dy*dy), 1.e-100)
        j = base.copy()
        j[rr, 3*ii] = dx/d
        j[rr, 3*jj] = -dx/d
        j[rr, 3*ii+1] = dy/d
        j[rr, 3*jj+1] = -dy/d
        return j

    def repair(c):
        a = np.asarray(c[:63], dtype=float).reshape(n, 3).copy()
        if not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return None
        a[:, :2] -= (a[:, :2].min(axis=0)+a[:, :2].max(axis=0))*.5
        for rep in range(4):
            dx = a[ii, 0]-a[jj, 0]
            dy = a[ii, 1]-a[jj, 1]
            d = np.sqrt(dx*dx+dy*dy)
            f = min(1., float(np.min(d/(a[ii, 2]+a[jj, 2]))))
            if f <= 0:
                return None
            a[:, 2] *= np.nextafter(f, 0.)
            loid = np.argmin(a[:, :2]-a[:, 2, None], axis=0)
            hiid = np.argmax(a[:, :2]+a[:, 2, None], axis=0)
            wh = Fraction(0)
            for k in range(2):
                l, h = int(loid[k]), int(hiid[k])
                wh += Fraction(float(a[h, k])) + Fraction(float(a[h, 2]))
                wh -= Fraction(float(a[l, k])) - Fraction(float(a[l, 2]))
            if wh <= 0:
                return None
            f = np.nextafter(float(Fraction(2)/wh), 0.)
            a *= f
            dx = a[ii, 0]-a[jj, 0]
            dy = a[ii, 1]-a[jj, 1]
            if np.all(np.sqrt(dx*dx+dy*dy) >= a[ii, 2]+a[jj, 2]):
                return a.ravel()
        return a.ravel()

    def signature(c):
        a = c.reshape(n, 3)
        return np.r_[np.sort(a[:, 2]), np.sort(np.hypot(a[ii, 0]-a[jj, 0], a[ii, 1]-a[jj, 1]))[::7]]

    def consider(c, store=False):
        nonlocal best, best_value, best_sum, archive
        c = repair(c)
        if c is None:
            return None
        s = float(c[2::3].sum())
        if s > best_sum:
            val = float(score(c))
            if np.isfinite(val) and val < best_value:
                best, best_value, best_sum = c.copy(), val, s
        if store and s > best_sum-.10:
            sig = signature(c)
            if not any(np.linalg.norm(sig-t[2]) < 2.e-4 for t in archive):
                archive.append((s, c.copy(), sig))
                archive.sort(key=lambda t: -t[0])
                archive = archive[:24]
        return c

    def to_z(c):
        a = np.asarray(c[:63]).reshape(n, 3).copy()
        a[:, 2] = np.maximum(a[:, 2], 1.01e-11)
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        f = min(1., (2.-2.e-14)/sum(hi-lo))
        a[:, :2] -= lo
        a *= f
        return np.r_[a.ravel(), np.clip((hi[0]-lo[0])*f, .015, 1.985)]

    def optimize(c, nit=140, objective=None, local_bounds=None, ftol=2.e-13):
        z = to_z(c)
        obj = g if objective is None else objective
        last = [z.copy()]
        cnt = [0]
        def fun(x):
            check()
            return float(obj @ x)
        def cb(x):
            check()
            last[0] = x.copy()
            cnt[0] += 1
            if objective is None and cnt[0] % 35 == 0:
                consider(x)
        try:
            res = minimize(fun, z, jac=lambda x: obj, method='SLSQP',
                           bounds=bounds if local_bounds is None else local_bounds,
                           constraints={'type': 'ineq', 'fun': constraints, 'jac': jac},
                           callback=cb, options={'maxiter': nit, 'ftol': ftol})
            last[0] = res.x
        except Stop:
            consider(last[0])
            raise
        except (ValueError, FloatingPointError):
            pass
        out = consider(last[0], objective is None)
        return out if out is not None else np.asarray(c[:63]).copy()

    def contacts(c):
        z = to_z(c)
        ids = np.flatnonzero(constraints(z) < 3.e-7)
        j = jac(z)[ids]
        try:
            lam = nnls(j.T, -g, maxiter=2000)[0]
        except Exception:
            lam = np.maximum(lstsq(j.T, -g, lapack_driver='gelsy')[0], 0.)
        return z, ids, lam

    def snap(c):
        # Extended-precision residuals avoid paying a uniform 1e-13
        # feasibility tax at an already very good rigid packing.
        z, ids, lam = contacts(c)
        if not len(ids):
            return c
        z = z.astype(np.longdouble)
        lam = lam.copy()
        for it in range(12):
            check()
            j = jac(z)[ids].astype(float)
            cv = np.asarray(constraints(z)[ids], dtype=float)
            h = np.zeros((64, 64))
            a = z[:63].reshape(n, 3)
            for q, edge in enumerate(ids):
                if edge < 84:
                    continue
                k = edge-84
                i, jdisk = int(ii[k]), int(jj[k])
                dxy = np.asarray(a[i, :2]-a[jdisk, :2], dtype=float)
                d = np.linalg.norm(dxy)
                if d < 1.e-10:
                    continue
                u = dxy/d
                hh = -lam[q]*(np.eye(2)-np.outer(u, u))/d
                ix = [3*i, 3*i+1]
                jx = [3*jdisk, 3*jdisk+1]
                h[np.ix_(ix, ix)] += hh
                h[np.ix_(jx, jx)] += hh
                h[np.ix_(ix, jx)] -= hh
                h[np.ix_(jx, ix)] -= hh
            m = len(ids)
            mat = np.zeros((64+m, 64+m))
            mat[:64, :64] = h
            mat[:64, 64:] = -j.T
            mat[64:, :64] = j
            rhs = np.r_[-g+j.T@lam, -cv]
            try:
                step = lstsq(mat, rhs, cond=1.e-12, lapack_driver='gelsy')[0]
            except Exception:
                break
            if np.max(np.abs(step[:64])) > 2.e-3:
                break
            z += step[:64].astype(np.longdouble)
            lam += step[64:]
            consider(z)
            if np.min(constraints(z)) < -1.e-6:
                break
        out = consider(z, True)
        return out if out is not None and out[2::3].sum() >= c[2::3].sum()-1.e-10 else c

    triple_cache = {}
    def hole_move(c, count=1):
        a = c.reshape(n, 3).copy()
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        prob = 1./np.maximum(a[:, 2], .012)**.7
        removed = list(rng.choice(n, count, replace=False, p=prob/prob.sum()))
        alive = np.ones(n, dtype=bool)
        alive[removed] = False
        normals = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
        wallb = np.array([lo[0], -hi[0], lo[1], -hi[1]])
        for disk in removed:
            check()
            b = a[alive]
            m = len(b)
            if m not in triple_cache:
                triple_cache[m] = (np.array(list(itertools.combinations(range(m), 3))),
                                   np.array(list(itertools.combinations(range(m), 2))))
            tri, pairs = triple_cache[m]
            blocksA, blocksb, blocksq, anchors = [], [], [], []
            # Each triple gives p = p0 + rho*p1, followed by one quadratic.
            i, j, k = tri.T
            aa = 2*np.stack((b[j, :2]-b[i, :2], b[k, :2]-b[i, :2]), axis=1)
            norm = np.sum(b[:, :2]**2, axis=1)-b[:, 2]**2
            bb = np.stack((norm[j]-norm[i], norm[k]-norm[i]), axis=1)
            qq = -2*np.stack((b[j, 2]-b[i, 2], b[k, 2]-b[i, 2]), axis=1)
            blocksA.append(aa); blocksb.append(bb); blocksq.append(qq); anchors.append(b[i])
            i, j = pairs.T
            for w in range(4):
                aa = np.empty((len(i), 2, 2))
                aa[:, 0] = 2*(b[j, :2]-b[i, :2])
                aa[:, 1] = normals[w]
                bb = np.c_[norm[j]-norm[i], np.full(len(i), wallb[w])]
                qq = np.c_[-2*(b[j, 2]-b[i, 2]), np.ones(len(i))]
                blocksA.append(aa); blocksb.append(bb); blocksq.append(qq); anchors.append(b[i])
            for w, t in [(0, 2), (0, 3), (1, 2), (1, 3)]:
                blocksA.append(np.tile(np.stack((normals[w], normals[t])), (m, 1, 1)))
                blocksb.append(np.tile([wallb[w], wallb[t]], (m, 1)))
                blocksq.append(np.ones((m, 2))); anchors.append(b)
            aa = np.concatenate(blocksA)
            bb = np.concatenate(blocksb)
            qq = np.concatenate(blocksq)
            anc = np.concatenate(anchors)
            det = aa[:, 0, 0]*aa[:, 1, 1]-aa[:, 0, 1]*aa[:, 1, 0]
            ok = np.abs(det) > 1.e-11
            aa, bb, qq, anc, det = aa[ok], bb[ok], qq[ok], anc[ok], det[ok]
            def solve2(rhs):
                return np.c_[(aa[:, 1, 1]*rhs[:, 0]-aa[:, 0, 1]*rhs[:, 1])/det,
                             (-aa[:, 1, 0]*rhs[:, 0]+aa[:, 0, 0]*rhs[:, 1])/det]
            p0, p1 = solve2(bb), solve2(qq)
            t = p0-anc[:, :2]
            qa = np.sum(p1*p1, axis=1)-1
            qb = 2*(np.sum(t*p1, axis=1)-anc[:, 2])
            qc = np.sum(t*t, axis=1)-anc[:, 2]**2
            disc = qb*qb-4*qa*qc
            candidates = []
            for sign in [-1., 1.]:
                den = 2*qa
                rad = np.divide(-qb+sign*np.sqrt(np.maximum(disc, 0)), den,
                                out=np.full_like(den, -1.), where=np.abs(den)>1.e-12)
                linear = (np.abs(qa) <= 1.e-12) & (np.abs(qb)>1.e-12)
                rad[linear] = -qc[linear]/qb[linear]
                ok = (disc >= -1.e-13) & (rad > 1.e-7) & (rad < 1.)
                pts = p0[ok]+p1[ok]*rad[ok, None]
                candidates.append(pts)
            pts = np.concatenate(candidates)
            if len(pts) == 0:
                alive[disk] = True
                continue
            ok = ((pts > lo).all(axis=1) & (pts < hi).all(axis=1) &
                  (np.linalg.norm(pts-a[disk, :2], axis=1) > .045))
            pts = pts[ok]
            if len(pts) == 0:
                alive[disk] = True
                continue
            gap = np.minimum(pts-lo, hi-pts).min(axis=1)
            gap = np.minimum(gap, np.min(np.linalg.norm(pts[:, None]-b[None, :, :2], axis=2)-b[None, :, 2], axis=1))
            order = np.argsort(-gap)
            options = []
            for q in order:
                if gap[q] <= 1.e-6:
                    break
                if all(np.linalg.norm(pts[q]-pts[t]) > .035 for t in options):
                    options.append(int(q))
                    if len(options) == 8:
                        break
            if options:
                weights = np.exp(-np.arange(len(options))/.95)
                q = options[int(rng.choice(len(options), p=weights/weights.sum()))]
                a[disk, :2] = pts[q]
                a[disk, 2] = gap[q]*(1.-1.e-10)
            alive[disk] = True
        return a.ravel()

    try:
        c = optimize(best, 350, ftol=2.e-14)
        snap(c)
        consider(best, True)
        chain = best.copy()
        trial = 0
        while True:
            check()
            trial += 1
            u = rng.random()
            if u < .65:
                source = best.copy()
            elif u < .85:
                source = chain.copy()
            else:
                ranks = np.arange(len(archive))
                prob = np.exp(-ranks/5.)
                source = archive[int(rng.choice(len(archive), p=prob/prob.sum()))][1].copy() if archive else best.copy()
            mode = rng.random()
            if mode < .57:
                z, ids, lam = contacts(source)
                j = jac(z)
                if len(ids):
                    # A contact multiplier measures the first-order cost of
                    # opening that contact. Tilt just beyond that threshold.
                    prob = 1./(.035+lam)**.65
                    q = int(rng.choice(len(ids), p=prob/prob.sum()))
                    direction = j[ids[q]].copy()
                    strength = (lam[q]+.025)*float(rng.choice([1.15, 1.6, 2.4, 4.]))
                    if rng.random() < .20:
                        q2 = int(rng.integers(len(ids)))
                        direction += .7*j[ids[q2]]
                    if rng.random() < .20:
                        direction[2:63:3] = 0.
                    obj = g-strength*direction
                else:
                    obj = g.copy()
                    obj[:63] += rng.normal(0, .15, 63)
                bb = list(bounds)
                floor = float(rng.choice([.12, .35, .65]))
                for k in range(n):
                    bb[3*k+2] = (max(1.e-11, floor*z[3*k+2]), 1.)
                c = optimize(source, 65, obj, bb, ftol=1.e-10)
            elif mode < .82:
                c = hole_move(source, 1 if rng.random() < .82 else 2)
            else:
                a = source.reshape(n, 3).copy()
                kind = int(rng.integers(4))
                if kind == 0:
                    k = int(rng.integers(n))
                    group = np.argsort(np.linalg.norm(a[:, :2]-a[k, :2], axis=1))[:int(rng.integers(3, 8))]
                    a[group, :2] += rng.normal(0, float(rng.choice([.015, .035, .07])), (len(group), 2))
                elif kind == 1:
                    k, l = rng.choice(n, 2, replace=False)
                    a[[k, l], :2] = a[[l, k], :2]
                elif kind == 2:
                    f = np.exp(rng.normal(0, .14))
                    a[:, 0] *= f
                    a[:, 1] /= f
                else:
                    a[:, :2] += rng.normal(0, .025, (n, 2))
                    a[:, 2] *= np.exp(rng.normal(0, .12, n))
                c = a.ravel()
            c = optimize(c, 170)
            s = float(c[2::3].sum())
            if s > best_sum-2.e-7 or trial % 18 == 0:
                c = snap(c)
                s = float(c[2::3].sum())
            temperature = float(rng.choice([.003, .008, .02]))
            if s >= chain[2::3].sum() or rng.random() < np.exp(max(-700., (s-chain[2::3].sum())/temperature)):
                chain = c.copy()
            if trial % 24 == 0:
                chain = best.copy()
    except Stop:
        pass
    return best
