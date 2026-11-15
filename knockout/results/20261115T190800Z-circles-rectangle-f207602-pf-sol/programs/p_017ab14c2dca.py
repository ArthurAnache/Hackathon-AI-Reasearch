import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"iterations": (160, 40, 300), "jitter": (0.035, 0.0, 0.15)}


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.71 * budget_s
    best = init
    best_score = -float('inf')

    def exact(obj):
        try:
            if 'circles' in obj:
                a = np.asarray(obj['circles'], dtype=float)
            else:
                a = np.column_stack((obj['centers'], obj['radii']))
            if a.shape != (21, 3) or not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
                return -float('inf')
            fs = [[Fraction(float(v)) for v in row] for row in a]
            wh = sum(max(row[k] + row[2] for row in fs) - min(row[k] - row[2] for row in fs) for k in (0, 1))
            if wh > 2:
                return -float('inf')
            for i in range(21):
                if time.monotonic() > start + 0.94 * budget_s:
                    return -float('inf')
                for j in range(i):
                    dx = a[i, 0] - a[j, 0]
                    dy = a[i, 1] - a[j, 1]
                    if np.sqrt(dx*dx + dy*dy) < a[i, 2] + a[j, 2]:
                        return -float('inf')
            return float(np.sum(a[:, 2]))
        except Exception:
            return -float('inf')

    if init is not None:
        best_score = exact(init)

    rng = np.random.default_rng(3217)

    def offer(a):
        nonlocal best, best_score
        if time.monotonic() >= start + 0.88 * budget_s:
            return
        a = np.asarray(a, float).copy()
        if not np.isfinite(a).all():
            return
        a[:, 2] = np.maximum(a[:, 2], 1e-12)
        ii, jj = np.triu_indices(21, 1)
        dist = np.sqrt(np.sum((a[ii, :2] - a[jj, :2])**2, axis=1))
        ratio = min(1.0, float(np.min(dist / (a[ii, 2] + a[jj, 2]))))
        if ratio <= 0:
            return
        a[:, 2] *= ratio * (1 - 3e-13)
        lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
        hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
        a[:, :2] -= (lo + hi) / 2
        a *= (2.0 / float(np.sum(hi - lo))) * (1 - 3e-13)
        obj = {'centers': a[:, :2].tolist(), 'radii': a[:, 2].tolist()}
        val = exact(obj)
        if val > best_score:
            best_score, best = val, obj

    # A universally valid fallback, independent of the supplied warm start.
    fallback = np.array([[0.1 + 0.2*(i % 7), 0.1 + 0.2*(i // 7), 0.099] for i in range(21)])
    offer(fallback)

    class Stop(Exception):
        pass

    layouts = [(4,5,3,5,4), (5,4,3,4,5), (4,4,5,4,4),
               (3,5,5,5,3), (3,4,7,4,3), (6,3,3,3,6),
               (5,5,1,5,5), (3,3,3,3,3,3,3)]

    for trial, rows in enumerate(layouts):
        if time.monotonic() >= deadline:
            break
        mid = len(rows)//2
        singles = rows[mid]
        pairs = (21-singles)//2
        groups = singles + pairs
        dim = 2*singles + 3*pairs + 1
        X = np.zeros((21, dim))
        Y = np.zeros_like(X)
        R = np.zeros_like(X)
        z = np.zeros(dim)
        ref = np.arange(21)
        rcols = []
        ycols = []
        k = 0
        idx = 0
        # Build axis circles first.
        for j in range(singles):
            X[idx,k] = 1
            R[idx,k+1] = 1
            z[k] = 2*(j-(singles-1)/2)
            z[k+1] = 0.97
            rcols.append(k+1)
            k += 2
            idx += 1
        for row in range(mid+1, len(rows)):
            count = rows[row]
            offset = (0.18 if (row-mid)%2 else -0.12)
            for j in range(count):
                X[idx,k] = X[idx+1,k] = 1
                Y[idx,k+1] = 1
                Y[idx+1,k+1] = -1
                R[idx,k+2] = R[idx+1,k+2] = 1
                z[k] = 2*(j-(count-1)/2) + offset + rng.normal(0, params['jitter'])
                z[k+1] = (row-mid)*np.sqrt(3)
                z[k+2] = 0.97
                rcols.append(k+2)
                ycols.append(k+1)
                ref[idx], ref[idx+1] = idx+1, idx
                idx += 2
                k += 3
        xx, yy, rr = X@z, Y@z, R@z
        width = np.max(xx+rr)-np.min(xx-rr)
        height = np.max(yy+rr)-np.min(yy-rr)
        scale = 2/(width+height)
        z[:-1] *= scale
        # Center the rectangle at the origin; width is its full width.
        z[-1] = width*scale
        representatives = list(range(singles)) + list(range(singles,21,2))
        A = []
        b = []
        for i in representatives:
            for sign in (-1,1):
                v = -R[i] + sign*X[i]
                v = v.copy(); v[-1] += 0.5
                A.append(v); b.append(0.0)
            v = -R[i]-Y[i]
            v = v.copy(); v[-1] -= 0.5
            A.append(v); b.append(1.0)
        A = np.asarray(A)
        b = np.asarray(b)
        edges = []
        for i in range(21):
            if time.monotonic() >= deadline:
                break
            for j in range(i):
                edge = (j,i)
                mirror = tuple(sorted((int(ref[j]),int(ref[i]))))
                if edge <= mirror:
                    edges.append(edge)
        if time.monotonic() >= deadline:
            break
        ei = np.array([e[0] for e in edges])
        ej = np.array([e[1] for e in edges])
        DX = X[ei]-X[ej]
        DY = Y[ei]-Y[ej]
        SR = R[ei]+R[ej]
        objective = -np.sum(R, axis=0)
        latest = [z.copy()]
        cache = [None, None, None]

        def compute(v):
            if time.monotonic() >= deadline:
                raise Stop()
            if cache[0] is not None and np.array_equal(v,cache[0]):
                return cache[1],cache[2]
            dx = DX@v
            dy = DY@v
            ds = np.maximum(np.sqrt(dx*dx+dy*dy),1e-20)
            cv = np.concatenate((A@v+b, ds-SR@v))
            jac = np.vstack((A, (dx/ds)[:,None]*DX + (dy/ds)[:,None]*DY-SR))
            cache[:] = [v.copy(),cv,jac]
            return cv,jac

        def callback(v):
            latest[0] = v.copy()
            if time.monotonic() >= deadline:
                raise Stop()

        bounds = [(None,None)]*dim
        for col in rcols:
            bounds[col] = (1e-6,0.65)
        for col in ycols:
            bounds[col] = (0,1)
        bounds[-1] = (0.25,1.75)
        try:
            result = minimize(lambda v: float(objective@v), z, jac=lambda v: objective,
                              method='SLSQP', bounds=bounds,
                              constraints={'type':'ineq','fun':lambda v:compute(v)[0],
                                           'jac':lambda v:compute(v)[1]},
                              callback=callback,
                              options={'maxiter':int(params['iterations']), 'ftol':2e-12, 'disp':False})
            latest[0] = result.x
        except Stop:
            pass
        except Exception:
            pass
        v = latest[0]
        offer(np.column_stack((X@v,Y@v,R@v)))
    return best
