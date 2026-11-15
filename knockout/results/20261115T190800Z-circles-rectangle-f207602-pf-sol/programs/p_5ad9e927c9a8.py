import time
import math
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'time_fraction': (0.69, 0.5, 0.75),
    'iterations': (110, 40, 180),
    'perturbation': (0.018, 0.0, 0.08),
}


def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + float(params['time_fraction']) * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    best = None
    best_score = -float('inf')

    def canonical(obj):
        if obj is None:
            return None
        if 'circles' in obj:
            return obj
        return {'circles': [[float(p[0]), float(p[1]), float(r)]
                            for p, r in zip(obj['centers'], obj['radii'])]}

    def exact_score(obj):
        try:
            a = np.asarray(obj['circles'], dtype=np.float64)
            if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
                return -float('inf')
            xs = list(map(lambda v: Fraction(float(v)), a[:, 0]))
            ys = list(map(lambda v: Fraction(float(v)), a[:, 1]))
            rs = list(map(lambda v: Fraction(float(v)), a[:, 2]))
            w = max(x+r for x, r in zip(xs, rs)) - min(x-r for x, r in zip(xs, rs))
            h = max(y+r for y, r in zip(ys, rs)) - min(y-r for y, r in zip(ys, rs))
            if w+h > 2:
                return -float('inf')
            delta = a[ii, :2] - a[jj, :2]
            dist = np.sqrt(delta[:, 0]**2 + delta[:, 1]**2)
            if np.any(dist < a[ii, 2] + a[jj, 2]):
                return -float('inf')
            return float(np.sum(a[:, 2]))
        except (ValueError, TypeError, KeyError, OverflowError):
            return -float('inf')

    original = canonical(init)
    if original is not None:
        best_score = exact_score(original)
        if np.isfinite(best_score):
            best = init

    def consider(a):
        nonlocal best, best_score
        a = np.asarray(a, dtype=float).copy()
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return
        delta = a[ii, :2] - a[jj, :2]
        dist = np.sqrt(delta[:, 0]**2 + delta[:, 1]**2)
        factor = min(1.0, float(np.min(dist / (a[ii, 2] + a[jj, 2]))))
        if factor <= 0:
            return
        a[:, 2] *= factor * (1.0 - 8e-14)
        lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
        hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
        perimeter = float(np.sum(hi-lo))
        if perimeter <= 0:
            return
        a[:, :2] -= (lo+hi)/2
        a *= (2.0/perimeter) * (1.0 - 8e-14)
        if float(np.sum(a[:, 2])) <= best_score:
            return
        obj = {'circles': a.tolist()}
        score = exact_score(obj)
        if score > best_score:
            best_score = score
            best = obj

    fallback = np.array([[0.2*x, 0.2*y, 0.1] for y in range(3) for x in range(7)], dtype=float)
    consider(fallback)

    class Finished(Exception):
        pass

    def rows(lengths):
        out = []
        for row, n in enumerate(lengths):
            if time.monotonic() >= deadline:
                break
            for j in range(n):
                if time.monotonic() >= deadline:
                    break
                out.append((2*j-(n-1), math.sqrt(3)*row+1, 1.0))
        return out

    def square(extra_y=4.0, nx=4, ny=4):
        out = []
        for j in range(ny):
            if time.monotonic() >= deadline:
                break
            for i in range(nx):
                if time.monotonic() >= deadline:
                    break
                out.append((2*i-(nx-1), 2*j+1, 1.0))
        small = math.sqrt(2)-1
        if nx == 4 and ny == 4:
            out.extend([(0, 2, small), (0, 4, small), (0, 6, small),
                        (-2, extra_y, small), (2, extra_y, small)])
        elif nx == 4 and ny == 5:
            out.append((0, extra_y, small))
        elif nx == 5 and ny == 4:
            out.append((0, extra_y, small))
        return out

    def solve_layout(layout, perturb=0.0):
        if time.monotonic() >= deadline or len(layout) != 21:
            return
        raw = np.asarray(layout, dtype=float)
        axis = raw[np.abs(raw[:, 0]) < 1e-8]
        side = raw[raw[:, 0] > 1e-8]
        ns, npair = len(axis), len(side)
        k = ns + npair
        if ns + 2*npair != 21:
            return
        orb = np.concatenate((axis, side), axis=0).copy()
        if perturb:
            orb[:, 1] += rng.normal(0, perturb, k)
            orb[ns:, 0] += rng.normal(0, perturb, npair)
            orb[:, 2] *= np.exp(rng.normal(0, perturb*0.2, k))
        lower_y = float(np.min(orb[:, 1]-orb[:, 2]))
        upper_y = float(np.max(orb[:, 1]+orb[:, 2]))
        width = 2*float(np.max(orb[:, 0]+orb[:, 2]))
        height = upper_y-lower_y
        scale = 2/(width+height)
        orb[:, 0] *= scale
        orb[:, 1] = (orb[:, 1]-lower_y)*scale
        orb[:, 2] *= scale
        nv = npair+2*k+1
        y0, r0, wi = npair, npair+k, nv-1
        z = np.zeros(nv)
        z[:npair] = orb[ns:, 0]
        z[y0:y0+k] = orb[:, 1]
        z[r0:r0+k] = orb[:, 2]
        z[wi] = width*scale
        X = np.zeros((k, nv))
        Y = np.zeros((k, nv))
        R = np.zeros((k, nv))
        for j in range(k):
            if time.monotonic() >= deadline:
                return
            Y[j, y0+j] = 1
            R[j, r0+j] = 1
            if j >= ns:
                X[j, j-ns] = 1
        oi, oj = np.triu_indices(k, 1)
        DX = X[oi]-X[oj]
        DY = Y[oi]-Y[oj]
        SR = R[oi]+R[oj]
        ew = np.zeros(nv)
        ew[wi] = 1
        A = np.concatenate((0.5*ew[None, :]-X-R,
                            Y-R,
                            -ew[None, :]-Y-R,
                            X[ns:]-R[ns:]), axis=0)
        b = np.zeros(len(A))
        b[2*k:3*k] = 2
        weights = np.concatenate((np.ones(ns), np.full(npair, 2.0)))
        gradient = np.zeros(nv)
        gradient[r0:r0+k] = -weights
        cached_z = None
        cached_c = None
        cached_j = None

        def update(v):
            nonlocal cached_z, cached_c, cached_j
            if cached_z is not None and np.array_equal(v, cached_z):
                return
            if time.monotonic() >= deadline:
                raise Finished()
            dx, dy = DX@v, DY@v
            d = np.sqrt(dx*dx+dy*dy)
            safe_d = np.maximum(d, 1e-15)
            cached_c = np.concatenate((A@v+b, d-SR@v))
            cached_j = np.concatenate((A, (dx/safe_d)[:, None]*DX + (dy/safe_d)[:, None]*DY-SR), axis=0)
            cached_z = v.copy()

        def constraints(v):
            update(v)
            return cached_c

        def jacobian(v):
            update(v)
            return cached_j

        def expand(v):
            x = X@v
            y = v[y0:y0+k]
            r = v[r0:r0+k]
            base = np.column_stack((x, y, r))
            reflected = base[ns:].copy()
            reflected[:, 0] *= -1
            return np.concatenate((base, reflected), axis=0)

        latest = z.copy()
        count = 0
        def callback(v):
            nonlocal latest, count
            if time.monotonic() >= deadline:
                raise Finished()
            latest = v.copy()
            count += 1
            if count % 12 == 0:
                consider(expand(v))

        bounds = [(0.0, 2.0)]*npair + [(0.0, 2.0)]*k + [(1e-7, 0.6)]*k + [(0.1, 1.9)]
        consider(expand(z))
        try:
            result = minimize(lambda v: float(gradient@v), z, jac=lambda v: gradient,
                              method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                              callback=callback,
                              options={'maxiter': int(params['iterations']), 'ftol': 2e-12, 'disp': False})
            latest = result.x
        except Finished:
            pass
        consider(expand(latest))

    layouts = [
        lambda: square(4),
        lambda: rows([4, 3, 4, 3, 4, 3]),
        lambda: square(2),
        lambda: rows([3, 4, 5, 4, 5]),
        lambda: square(4, 4, 5),
        lambda: rows([5, 4, 3, 4, 5]),
        lambda: square(4, 5, 4),
        lambda: rows([4, 5, 4, 5, 3]),
        lambda: rows([3, 3, 3, 3, 3, 3, 3]),
    ]
    for index, make in enumerate(layouts):
        if time.monotonic() >= deadline:
            break
        perturb = float(params['perturbation']) if index != 1 else 0.0
        solve_layout(make(), perturb)

    if best is None:
        return {'circles': [[0.2*x, 0.2*y, 0.099999999999] for y in range(3) for x in range(7)]}
    return best
