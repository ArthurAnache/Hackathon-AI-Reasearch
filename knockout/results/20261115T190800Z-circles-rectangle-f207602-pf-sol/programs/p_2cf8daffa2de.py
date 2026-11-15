import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"first_share": (0.40, 0.25, 0.55), "second_share": (0.39, 0.25, 0.55), "iterations": (160, 60, 250)}


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70 * budget_s
    best = init
    best_score = -float('inf')
    ia, ib = np.triu_indices(21, 1)

    def canonical(obj):
        if 'circles' in obj:
            return np.array(obj['circles'], dtype=float)
        return np.array(obj['state'], dtype=float).reshape(3, 21).T.copy()

    def exact(c):
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return -float('inf')
        d = c[ia, :2] - c[ib, :2]
        if np.any(np.sqrt(d[:, 0]**2 + d[:, 1]**2) < c[ia, 2] + c[ib, 2]):
            return -float('inf')
        xs = [Fraction(float(v)) for v in c[:, 0]]
        ys = [Fraction(float(v)) for v in c[:, 1]]
        rs = [Fraction(float(v)) for v in c[:, 2]]
        w = max(x+r for x, r in zip(xs, rs))-min(x-r for x, r in zip(xs, rs))
        h = max(y+r for y, r in zip(ys, rs))-min(y-r for y, r in zip(ys, rs))
        if w+h > 2:
            return -float('inf')
        return float(np.sum(c[:, 2]))

    if init is not None:
        try:
            best_score = exact(canonical(init))
        except (ValueError, KeyError, TypeError):
            pass

    def consider(c):
        nonlocal best, best_score
        if time.monotonic() >= end or not np.isfinite(c).all():
            return
        c = c.copy()
        c[:, 2] = np.maximum(c[:, 2], 1e-12)
        d = c[ia, :2]-c[ib, :2]
        dist = np.sqrt(d[:, 0]**2+d[:, 1]**2)
        f = min(1.0, float(np.min(dist/(c[ia, 2]+c[ib, 2]))))
        if f <= 0:
            return
        c[:, 2] *= f*(1-2e-13)
        lo = np.min(c[:, :2]-c[:, 2, None], axis=0)
        hi = np.max(c[:, :2]+c[:, 2, None], axis=0)
        perimeter = float(np.sum(hi-lo))
        if perimeter <= 0:
            return
        c[:, :2] -= (hi+lo)/2
        c *= (2/perimeter)*(1-2e-13)
        if float(np.sum(c[:, 2])) <= best_score:
            return
        score = exact(c)
        if score > best_score:
            best_score = score
            best = {'circles': c.tolist()}

    def quotient(counts):
        groups = []
        circle = 0
        nv = 0
        mirror = []
        for col, count in enumerate(counts):
            if time.monotonic() >= end:
                break
            if count % 2:
                groups.append((col, 0., [circle], nv, False))
                mirror.append(circle)
                circle += 1
                nv += 2
            for k in range(count//2):
                if time.monotonic() >= end:
                    break
                yy = 2*k+1 if count % 2 == 0 else 2*k+2
                groups.append((col, float(yy), [circle, circle+1], nv, True))
                mirror.extend([circle+1, circle])
                circle += 2
                nv += 3
        if circle != 21:
            return None
        X = np.zeros((21, nv+1)); Y = X.copy(); R = X.copy()
        q = np.zeros(nv+1)
        lower = np.full(nv+1, -1.0); upper = np.ones(nv+1)
        raw = np.zeros((21, 3))
        for col, yy, ids, j, pair in groups:
            if time.monotonic() >= end:
                return None
            xx = (col-(len(counts)-1)/2)*np.sqrt(3.)
            for k, i in enumerate(ids):
                X[i, j] = 1
                if pair:
                    Y[i, j+1] = 1 if k == 0 else -1
                R[i, j+(2 if pair else 1)] = 1
                raw[i] = [xx, yy if k == 0 else -yy, 1.]
            q[j] = xx
            if pair:
                q[j+1] = yy
                lower[j+1] = 0.
            rid = j+(2 if pair else 1)
            q[rid] = 1.
            lower[rid] = 1e-6
            upper[rid] = 0.5
        lo = np.min(raw[:, :2]-1, axis=0)
        hi = np.max(raw[:, :2]+1, axis=0)
        scale = 2/np.sum(hi-lo)
        q[:-1] *= scale
        q[-1] = (hi[0]-lo[0])*scale
        lower[-1] = 0.15; upper[-1] = 1.85
        keep = []
        for k, (a, b) in enumerate(zip(ia, ib)):
            if time.monotonic() >= end:
                return None
            reflected = tuple(sorted((mirror[a], mirror[b])))
            if (a, b) <= reflected:
                keep.append(k)
        return X, Y, R, q, list(zip(lower, upper)), np.array(keep, dtype=int)

    class Stop(Exception):
        pass

    def solve(model, deadline):
        X, Y, R, q, bounds, keep = model
        A = X[ia[keep]]-X[ib[keep]]
        B = Y[ia[keep]]-Y[ib[keep]]
        S = R[ia[keep]]+R[ib[keep]]
        W = np.vstack((-X-R, X-R, -Y-R, Y-R))
        W[:42, -1] += .5
        W[42:, -1] -= .5
        rhs = np.concatenate((np.zeros(42), np.ones(42)))
        combined = np.column_stack((W, rhs))
        _, unique = np.unique(combined, axis=0, return_index=True)
        W = W[unique]; rhs = rhs[unique]
        objective = -np.sum(R, axis=0)
        last = [q.copy()]
        count = [0]

        def fun(v):
            if time.monotonic() >= deadline:
                raise Stop()
            return float(objective @ v)

        def con(v):
            dx = A @ v; dy = B @ v
            return np.concatenate((np.sqrt(dx*dx+dy*dy)-S@v, W@v+rhs))

        def jac(v):
            dx = A @ v; dy = B @ v
            d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-20)
            return np.vstack(((dx/d)[:, None]*A+(dy/d)[:, None]*B-S, W))

        def callback(v):
            last[0] = v.copy()
            count[0] += 1
            if time.monotonic() >= deadline:
                raise Stop()
            if count[0] % 7 == 0:
                consider(np.column_stack((X@v, Y@v, R@v)))

        consider(np.column_stack((X@q, Y@q, R@q)))
        try:
            res = minimize(fun, q, jac=lambda v: objective, bounds=bounds,
                           constraints={'type': 'ineq', 'fun': con, 'jac': jac},
                           callback=callback, method='SLSQP',
                           options={'maxiter': int(params['iterations']), 'ftol': 2e-12, 'disp': False})
            last[0] = res.x
        except Stop:
            pass
        v = last[0]
        consider(np.column_stack((X@v, Y@v, R@v)))

    duration = max(0., end-start)
    families = ([3, 4, 3, 4, 3, 4], [4, 5, 4, 5, 3])
    fractions = (float(params['first_share']), float(params['second_share']))
    cumulative = 0.
    for family, share in zip(families, fractions):
        if time.monotonic() >= end-0.04:
            break
        cumulative += share
        model = quotient(family)
        if model is not None:
            solve(model, min(end-0.04, start+cumulative*duration))

    if best is not None and time.monotonic() < end-0.045:
        c = canonical(best)
        X = np.zeros((21, 64)); Y = X.copy(); R = X.copy()
        X[:, :21] = np.eye(21)
        Y[:, 21:42] = np.eye(21)
        R[:, 42:63] = np.eye(21)
        q = np.zeros(64)
        q[:63] = c.T.reshape(-1)
        q[-1] = np.ptp(c[:, 0])+2*np.max(c[:, 2])
        q[-1] = np.clip(q[-1], .2, 1.8)
        bounds = [(-1., 1.)]*42+[(1e-6, .5)]*21+[(.15, 1.85)]
        solve((X, Y, R, q, bounds, np.arange(len(ia))), end-0.012)
    return best
