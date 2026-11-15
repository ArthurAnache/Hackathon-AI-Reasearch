import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"iterations": (105, 50, 160), "search_fraction": (0.72, 0.5, 0.85), "jitter": (0.018, 0.002, 0.06)}


def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + 0.70 * budget_s
    search_end = started + 0.70 * budget_s * params['search_fraction']
    rng = np.random.default_rng(seed)
    ai, aj = np.triu_indices(21, 1)
    best = init
    best_score = -float('inf')
    best_array = None

    class Stop(Exception):
        pass

    def check():
        if time.monotonic() >= deadline:
            raise Stop()

    def array_of(obj):
        if 'circles' in obj:
            return np.asarray(obj['circles'], dtype=float)
        return np.column_stack((obj['centers'], obj['radii']))

    def perimeter(a):
        x = [Fraction(float(t)) for t in a[:, 0]]
        y = [Fraction(float(t)) for t in a[:, 1]]
        r = [Fraction(float(t)) for t in a[:, 2]]
        return (max(u+v for u,v in zip(x,r)) - min(u-v for u,v in zip(x,r))
                + max(u+v for u,v in zip(y,r)) - min(u-v for u,v in zip(y,r)))

    def exact(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return -float('inf')
        if perimeter(a) > 2:
            return -float('inf')
        d = a[ai, :2] - a[aj, :2]
        if np.any(np.sqrt(d[:, 0]**2 + d[:, 1]**2) < a[ai, 2] + a[aj, 2]):
            return -float('inf')
        return float(np.sum(a[:, 2]))

    if init is not None:
        try:
            a = array_of(init)
            best_score = exact(a)
            if np.isfinite(best_score):
                best_array = a.copy()
        except (ValueError, KeyError, TypeError):
            pass

    def retain(a):
        nonlocal best, best_score, best_array
        a = np.asarray(a, dtype=float).copy()
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return
        a[:, :2] -= np.mean(a[:, :2], axis=0)
        d = a[ai, :2] - a[aj, :2]
        dist = np.sqrt(d[:, 0]**2 + d[:, 1]**2)
        factor = min(1.0, float(np.min(dist / (a[ai, 2] + a[aj, 2]))))
        if factor <= 0:
            return
        a[:, 2] *= factor * (1 - 4e-14)
        pfast = np.sum(np.max(a[:, :2] + a[:, 2, None], axis=0) - np.min(a[:, :2] - a[:, 2, None], axis=0))
        if 2 * np.sum(a[:, 2]) / pfast <= best_score:
            return
        p = perimeter(a)
        a *= float(Fraction(2) / p) * (1 - 4e-14)
        score = exact(a)
        if score > best_score:
            best_score = score
            best_array = a.copy()
            best = {'centers': a[:, :2].tolist(), 'radii': a[:, 2].tolist()}

    # A quotient contains s axis circles and m representatives above the axis.
    def quotient(counts):
        s = counts[0]
        m = sum(counts[1:])
        n = s + m
        iy = n
        ir = n + m
        iw = ir + n
        size = iw + 1
        I, J = np.triu_indices(n, 1)
        U, V = np.triu_indices(m, 1)
        U, V = U+s, V+s
        pi = np.r_[I, U, np.arange(s, n)]
        pj = np.r_[J, V, np.arange(s, n)]
        sign = np.r_[np.ones(len(I)), -np.ones(len(U)+m)]
        nr = len(pi)
        rr = np.arange(nr)
        ids = np.arange(n)
        yidx = np.r_[np.full(s, -1, dtype=int), np.arange(m)+iy]
        weights = np.r_[np.ones(s), np.full(m, 2.0)]
        grad = np.zeros(size)
        grad[ir:ir+n] = -weights
        z = np.zeros(size)
        row = 0
        for level, count in enumerate(counts):
            check()
            for col in range(count):
                check()
                z[row] = (col-(count-1)/2)*0.20
                if level == 0 and s == 3:
                    z[row] *= 1.45
                if level:
                    z[yidx[row]] = level*0.177
                z[ir+row] = 0.095
                row += 1
        y = np.r_[np.zeros(s), z[iy:ir]]
        lo = np.min(z[:n]-z[ir:ir+n])
        hi = np.max(z[:n]+z[ir:ir+n])
        hh = np.max(y+z[ir:ir+n])
        scale = 2/(hi-lo+2*hh)
        z[:iw] *= scale
        z[:n] -= lo*scale
        z[iw] = (hi-lo)*scale

        def expand(t):
            yy = np.r_[np.zeros(s), t[iy:ir]]
            a = np.column_stack((t[:n], yy, t[ir:ir+n]))
            reflected = a[s:].copy()
            reflected[:, 1] *= -1
            return np.concatenate((a, reflected))

        def constraints(t, jac=False):
            check()
            x = t[:n]
            y = np.r_[np.zeros(s), t[iy:ir]]
            r = t[ir:ir+n]
            dx = x[pi]-x[pj]
            dy = y[pi]-sign*y[pj]
            d = np.sqrt(dx*dx+dy*dy)
            if not jac:
                return np.r_[d-r[pi]-r[pj], x-r, t[iw]-x-r,
                             1-t[iw]/2-y-r, t[iy:ir], r-1e-8]
            A = np.zeros((nr+4*n+m, size))
            dx = dx/np.maximum(d, 1e-15)
            dy = dy/np.maximum(d, 1e-15)
            np.add.at(A, (rr, pi), dx)
            np.add.at(A, (rr, pj), -dx)
            mask = pi >= s
            np.add.at(A, (rr[mask], yidx[pi[mask]]), dy[mask])
            mask = pj >= s
            np.add.at(A, (rr[mask], yidx[pj[mask]]), -sign[mask]*dy[mask])
            np.add.at(A, (rr, ir+pi), -1)
            np.add.at(A, (rr, ir+pj), -1)
            off = nr
            A[off+ids, ids] = 1
            A[off+ids, ir+ids] = -1
            off += n
            A[off+ids, ids] = -1
            A[off+ids, ir+ids] = -1
            A[off+ids, iw] = 1
            off += n
            A[off+ids, ir+ids] = -1
            A[off+ids, iw] = -0.5
            A[off+np.arange(s,n), np.arange(iy,ir)] = -1
            off += n
            A[off+np.arange(m), np.arange(iy,ir)] = 1
            off += m
            A[off+ids, ir+ids] = 1
            return A

        latest = [z.copy()]
        count = [0]
        def callback(t):
            check()
            latest[0] = t.copy()
            count[0] += 1
            if count[0] % 15 == 0:
                retain(expand(t))
        def optimize(t, iterations):
            latest[0] = t.copy()
            try:
                res = minimize(lambda v: float(grad@v), t, jac=lambda v: grad,
                               constraints={'type':'ineq', 'fun':constraints,
                                            'jac':lambda v: constraints(v, True)},
                               method='SLSQP', callback=callback,
                               options={'maxiter':iterations, 'ftol':2e-11, 'disp':False})
                latest[0] = res.x
            except Stop:
                retain(expand(latest[0]))
                raise
            retain(expand(latest[0]))
            return latest[0]
        return z, expand, optimize

    def full_polish(a):
        check()
        a = a.copy()
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        a[:, 0] -= lo[0]
        a[:, 1] -= (lo[1]+hi[1])/2
        v = np.r_[a.ravel(), hi[0]-lo[0]]
        ids = np.arange(21)
        rows = np.arange(210)
        grad = np.zeros(64)
        grad[2:63:3] = -1
        latest = [v.copy()]
        def constraints(t, jac=False):
            check()
            x,y,r = t[:63].reshape(21,3).T
            dx = x[ai]-x[aj]
            dy = y[ai]-y[aj]
            d = np.sqrt(dx*dx+dy*dy)
            if not jac:
                return np.r_[d-r[ai]-r[aj], x-r, t[63]-x-r,
                             1-t[63]/2+y-r, 1-t[63]/2-y-r, r-1e-8]
            A = np.zeros((315,64))
            den = np.maximum(d,1e-15)
            A[rows,3*ai] = dx/den
            A[rows,3*aj] = -dx/den
            A[rows,3*ai+1] = dy/den
            A[rows,3*aj+1] = -dy/den
            A[rows,3*ai+2] = -1
            A[rows,3*aj+2] = -1
            A[210+ids,3*ids] = 1
            A[210+ids,3*ids+2] = -1
            A[231+ids,3*ids] = -1
            A[231+ids,3*ids+2] = -1
            A[231+ids,63] = 1
            A[252+ids,3*ids+1] = 1
            A[252+ids,3*ids+2] = -1
            A[252+ids,63] = -.5
            A[273+ids,3*ids+1] = -1
            A[273+ids,3*ids+2] = -1
            A[273+ids,63] = -.5
            A[294+ids,3*ids+2] = 1
            return A
        def callback(t):
            check()
            latest[0] = t.copy()
        try:
            result = minimize(lambda t: float(grad@t),v,jac=lambda t:grad,
                              method='SLSQP',callback=callback,
                              constraints={'type':'ineq','fun':constraints,
                                           'jac':lambda t:constraints(t,True)},
                              options={'maxiter':65,'ftol':1e-12,'disp':False})
            latest[0] = result.x
        except Stop:
            retain(latest[0][:63].reshape(21,3))
            raise
        retain(latest[0][:63].reshape(21,3))

    try:
        layouts = ((3,5,4), (3,4,5), (5,4,4), (3,3,4,2), (5,5,3))
        records = []
        for counts in layouts:
            check()
            if time.monotonic() > search_end and records:
                break
            z, expand, optimize = quotient(counts)
            retain(expand(z))
            z = optimize(z, params['iterations'])
            score = float(np.sum(expand(z)[:,2]))
            records.append((score, z, expand, optimize))
        if best_array is not None and time.monotonic() < deadline-0.12:
            full_polish(best_array)
        while records and time.monotonic() < deadline-0.08:
            check()
            _, z, expand, optimize = max(records, key=lambda q:q[0])
            t = z.copy()
            t += rng.normal(0, params['jitter'], len(t))
            t = optimize(t, 55)
            records.append((float(np.sum(expand(t)[:,2])), t, expand, optimize))
    except (Stop, ValueError, np.linalg.LinAlgError):
        pass
    if best is None or not np.isfinite(best_score):
        return {'circles': [[(i%7)*.2, (i//7)*.2, .099999999999] for i in range(21)]}
    return best
