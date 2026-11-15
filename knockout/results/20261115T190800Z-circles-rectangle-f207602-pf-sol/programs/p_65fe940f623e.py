import time
import numpy as np
from fractions import Fraction
from scipy.optimize import minimize

PARAMS = {'iterations': (125, 40, 220), 'restart_scale': (0.012, 0.001, 0.06)}


def construct(params, seed, budget_s, init=None):
    deadline = time.monotonic() + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    ai, aj = np.triu_indices(21, 1)
    best = init
    best_score = -float('inf')
    saved = []

    class Stop(Exception):
        pass

    def check():
        if time.monotonic() >= deadline:
            raise Stop()

    def array_of(obj):
        if 'circles' in obj:
            return np.asarray(obj['circles'], dtype=float)
        return np.column_stack((obj['centers'], obj['radii']))

    def exact(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return -float('inf')
        d = a[ai, :2] - a[aj, :2]
        if np.any(np.sqrt(d[:, 0]**2 + d[:, 1]**2) < a[ai, 2] + a[aj, 2]):
            return -float('inf')
        x, y, r = [[Fraction(float(t)) for t in a[:, k]] for k in range(3)]
        p = max(u+v for u, v in zip(x, r)) - min(u-v for u, v in zip(x, r))
        p += max(u+v for u, v in zip(y, r)) - min(u-v for u, v in zip(y, r))
        return float(np.sum(a[:, 2])) if p <= 2 else -float('inf')

    if init is not None:
        try:
            best_score = exact(array_of(init))
        except (ValueError, TypeError, KeyError):
            pass

    def retain(a):
        nonlocal best, best_score
        a = np.array(a, dtype=float, copy=True)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return
        a[:, :2] -= np.mean(a[:, :2], axis=0)
        d = a[ai, :2]-a[aj, :2]
        ratio = np.min(np.sqrt(np.sum(d*d, axis=1))/(a[ai, 2]+a[aj, 2]))
        if ratio <= 0:
            return
        a[:, 2] *= min(1.0, ratio)*(1-4e-14)
        x, y, r = [[Fraction(float(t)) for t in a[:, k]] for k in range(3)]
        p = max(u+v for u, v in zip(x, r))-min(u-v for u, v in zip(x, r))
        p += max(u+v for u, v in zip(y, r))-min(u-v for u, v in zip(y, r))
        a *= float(Fraction(2)/p)*(1-4e-14)
        s = exact(a)
        if s > best_score:
            best_score = s
            best = {'centers': a[:, :2].tolist(), 'radii': a[:, 2].tolist()}

    # A strictly feasible fallback, also better than either tiny cold start.
    retain(np.array([[(i % 7)*.2, (i//7)*.2, .09999999999999] for i in range(21)]))

    def family(counts):
        check()
        coords = []
        reflected = []
        reps = []
        ix, iy, ir, sy = [], [], [], []
        initial = []
        for level, count in enumerate(counts):
            check()
            for col in range(count):
                check()
                x = (col-(count-1)/2)*.20
                if level == 0 and count == 3:
                    x *= 1.45
                k = len(initial)
                if level == 0:
                    initial.extend([x, .095])
                    i = len(ix)
                    ix.append(k); iy.append(k); ir.append(k+1); sy.append(0.)
                    reflected.append(i); reps.append(i)
                else:
                    initial.extend([x, level*.177, .095])
                    i = len(ix)
                    ix.extend([k, k]); iy.extend([k+1, k+1]); ir.extend([k+2, k+2]); sy.extend([1., -1.])
                    reflected.extend([i+1, i]); reps.append(i)
        ix, iy, ir = map(np.array, (ix, iy, ir))
        sy = np.array(sy)
        reps = np.array(reps)
        m = len(initial)+1
        z = np.r_[initial, 1.0]
        def unpack(t):
            return np.column_stack((t[ix], sy*t[iy], t[ir]))
        a = unpack(z)
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        scale = 2/np.sum(hi-lo)
        z[:-1] *= scale
        for k in set(ix.tolist()):
            check()
            z[k] -= lo[0]*scale
        z[-1] = (hi[0]-lo[0])*scale
        pairs = []
        for i, j in zip(ai, aj):
            check()
            other = tuple(sorted((reflected[i], reflected[j])))
            if (int(i), int(j)) <= other:
                pairs.append((i, j))
        pi, pj = np.array(pairs).T
        nr = len(pi)
        rows = np.arange(nr)
        X = np.zeros((21, m)); Y = X.copy(); R = X.copy()
        X[np.arange(21), ix] = 1
        Y[np.arange(21), iy] = sy
        R[np.arange(21), ir] = 1
        DX = X[pi]-X[pj]
        DY = Y[pi]-Y[pj]
        RR = R[pi]+R[pj]
        wall = np.vstack((X[reps]-R[reps], -X[reps]-R[reps], -Y[reps]-R[reps]))
        q = len(reps)
        wall[q:2*q, -1] = 1
        wall[2*q:, -1] = -.5
        offset = np.r_[np.zeros(2*q), np.ones(q)]
        gradient = -np.sum(R, axis=0)
        lower = np.full(m, -np.inf)
        upper = np.full(m, np.inf)
        lower[np.unique(ir)] = 1e-7
        lower[np.unique(iy[sy != 0])] = 1e-7
        lower[-1], upper[-1] = .15, 1.85
        bounds = list(zip(lower, upper))
        cache_t = None
        cache_c = None
        cache_j = None
        def calculate(t):
            nonlocal cache_t, cache_c, cache_j
            check()
            if cache_t is not None and np.array_equal(t, cache_t):
                return
            a = unpack(t)
            dx = a[pi, 0]-a[pj, 0]
            dy = a[pi, 1]-a[pj, 1]
            dist = np.sqrt(dx*dx+dy*dy)
            den = np.maximum(dist, 1e-15)
            cache_c = np.r_[dist-a[pi, 2]-a[pj, 2], wall@t+offset]
            cache_j = np.vstack(((dx/den)[:, None]*DX+(dy/den)[:, None]*DY-RR, wall))
            cache_t = t.copy()
        def cf(t):
            calculate(t)
            return cache_c
        def cj(t):
            calculate(t)
            return cache_j
        def run(t, iterations):
            check()
            latest = [t.copy()]
            ticks = [0]
            def callback(v):
                latest[0] = v.copy()
                check()
                ticks[0] += 1
                if ticks[0] % 18 == 0:
                    retain(unpack(v))
            try:
                result = minimize(lambda v: float(gradient@v), t,
                                  jac=lambda v: gradient, bounds=bounds,
                                  constraints={'type': 'ineq', 'fun': cf, 'jac': cj},
                                  callback=callback, method='SLSQP',
                                  options={'maxiter': iterations, 'ftol': 9e-12, 'disp': False})
                latest[0] = result.x
            except Stop:
                retain(unpack(latest[0]))
                raise
            retain(unpack(latest[0]))
            return latest[0], -float(gradient@latest[0])
        return z, run, unpack, ix, iy, ir

    try:
        for counts in ((3, 5, 4), (5, 4, 4), (3, 4, 5), (5, 5, 3)):
            check()
            z, run, unpack, ix, iy, ir = family(counts)
            retain(unpack(z))
            z, score = run(z, params['iterations'])
            saved.append((score, z, run, unpack, ix, iy, ir))
        saved.sort(key=lambda item: item[0], reverse=True)
        attempt = 0
        while saved and time.monotonic() < deadline:
            check()
            score, z, run, unpack, ix, iy, ir = saved[attempt % min(2, len(saved))]
            t = z.copy()
            ids = np.unique(np.r_[ix, iy])
            t[ids] += rng.normal(0, params['restart_scale'], len(ids))
            t[np.unique(ir)] = np.maximum(t[np.unique(ir)], 1e-6)
            u, s = run(t, 65)
            if s > score:
                saved[attempt % min(2, len(saved))] = (s, u, run, unpack, ix, iy, ir)
            attempt += 1
    except (Stop, ValueError, np.linalg.LinAlgError):
        pass
    return best
