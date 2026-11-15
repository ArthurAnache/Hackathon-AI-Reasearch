import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"iterations": (95, 40, 180), "polish": (38, 10, 80), "perturbation": (0.018, 0.001, 0.06)}


def construct(params, seed, budget_s, init=None):
    deadline = time.monotonic() + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    best = init
    score = -float('inf')
    best_a = None

    class Stop(Exception):
        pass

    def check():
        if time.monotonic() >= deadline:
            raise Stop()

    def array(obj):
        if 'circles' in obj:
            return np.asarray(obj['circles'], dtype=float)
        return np.column_stack((obj['centers'], obj['radii']))

    def perimeter(a):
        x = [Fraction(float(t)) for t in a[:, 0]]
        y = [Fraction(float(t)) for t in a[:, 1]]
        r = [Fraction(float(t)) for t in a[:, 2]]
        return (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
                +max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)))

    def exact(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return -float('inf')
        d = a[ii, :2] - a[jj, :2]
        if np.any(np.sqrt(d[:, 0]**2+d[:, 1]**2) < a[ii, 2]+a[jj, 2]):
            return -float('inf')
        if perimeter(a) > 2:
            return -float('inf')
        return float(np.sum(a[:, 2]))

    if init is not None:
        try:
            score = exact(array(init))
        except (ValueError, KeyError, TypeError):
            pass

    def retain(a):
        nonlocal best, score, best_a
        a = np.asarray(a, dtype=float).copy()
        if not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return
        a[:, :2] -= np.mean(a[:, :2], axis=0)
        d = a[ii, :2]-a[jj, :2]
        distances = np.sqrt(d[:, 0]**2+d[:, 1]**2)
        contraction = min(1., float(np.min(distances/(a[ii, 2]+a[jj, 2]))))
        if contraction <= 0:
            return
        a[:, 2] *= contraction*(1-4e-14)
        a *= float(Fraction(2)/perimeter(a))*(1-4e-14)
        s = exact(a)
        if s > score:
            score = s
            best_a = a.copy()
            best = {'centers': a[:, :2].tolist(), 'radii': a[:, 2].tolist()}

    def model(counts=None, initial=None):
        check()
        if counts is None:
            B = np.eye(64)
            a = initial.copy()
            lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
            hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
            a[:, 0] -= lo[0]
            a[:, 1] -= (lo[1]+hi[1])/2
            z = np.r_[a.ravel(), hi[0]-lo[0]]
            representatives = list(range(21))
            pairs = list(zip(ii, jj))
            upper_only = False
        else:
            pts, singles, mirrored = [], [], []
            for level, count in enumerate(counts):
                check()
                for col in range(count):
                    check()
                    x = (col-(count-1)/2)*.20
                    if level == 0 and count == 3:
                        x *= 1.45
                    if level == 0:
                        singles.append(len(pts))
                        pts.append([x, 0., .095])
                    else:
                        k = len(pts)
                        mirrored.append((k, k+1))
                        pts.extend([[x, level*.177, .095], [x, -level*.177, .095]])
            a = np.array(pts)
            lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
            hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
            scale = 2/np.sum(hi-lo)
            a *= scale
            a[:, 0] -= lo[0]*scale
            dim = 2*len(singles)+3*len(mirrored)+1
            B = np.zeros((64, dim))
            c = 0
            for k in singles:
                check()
                B[3*k, c] = 1
                B[3*k+2, c+1] = 1
                c += 2
            reflection = np.arange(21)
            for k,l in mirrored:
                check()
                B[3*k, c] = B[3*l, c] = 1
                B[3*k+1, c+1] = 1
                B[3*l+1, c+1] = -1
                B[3*k+2, c+2] = B[3*l+2, c+2] = 1
                reflection[k], reflection[l] = l, k
                c += 3
            B[63, c] = 1
            full = np.r_[a.ravel(), (hi[0]-lo[0])*scale]
            z = (B.T@full)/np.sum(B*B, axis=0)
            pairs = []
            for i,j in zip(ii,jj):
                check()
                reflected = tuple(sorted((int(reflection[i]), int(reflection[j]))))
                if (i,j) <= reflected:
                    pairs.append((i,j))
            representatives = singles+[k for k,l in mirrored]
            upper_only = True
        X, Y, R = B[0:63:3], B[1:63:3], B[2:63:3]
        p = np.array([i for i,j in pairs])
        q = np.array([j for i,j in pairs])
        DX, DY, RS = X[p]-X[q], Y[p]-Y[q], R[p]+R[q]
        C, constants = [], []
        for k in representatives:
            check()
            C.extend((X[k]-R[k], B[63]-X[k]-R[k], -.5*B[63]-Y[k]-R[k], R[k]))
            constants.extend((0., 0., 1., -1e-7))
            if not upper_only:
                C.append(-.5*B[63]+Y[k]-R[k])
                constants.append(1.)
        C = np.array(C)
        constants = np.array(constants)
        g = -np.sum(R, axis=0)
        return z, B, DX, DY, RS, C, constants, g

    def optimize(m, iterations):
        z, B, DX, DY, RS, C, constants, g = m
        latest = [z.copy()]
        cache = [None, None, None, None]
        def calculate(t):
            check()
            if cache[0] is None or not np.array_equal(t, cache[0]):
                dx, dy = DX@t, DY@t
                dist = np.sqrt(dx*dx+dy*dy)
                cache[:] = [t.copy(), dx, dy, dist]
            return cache[1:]
        def constraints(t):
            dx, dy, dist = calculate(t)
            return np.r_[dist-RS@t, C@t+constants]
        def jac(t):
            dx, dy, dist = calculate(t)
            den = np.maximum(dist, 1e-15)
            return np.vstack((DX*(dx/den)[:, None]+DY*(dy/den)[:, None]-RS, C))
        def fun(t):
            check()
            return float(g@t)
        count = [0]
        def callback(t):
            check()
            latest[0] = t.copy()
            count[0] += 1
            if count[0] % 20 == 0:
                retain((B@t)[:63].reshape(21,3))
        try:
            res = minimize(fun, z, jac=lambda t: g, method='SLSQP',
                           constraints={'type': 'ineq', 'fun': constraints, 'jac': jac},
                           callback=callback,
                           options={'maxiter': iterations, 'ftol': 2e-12, 'disp': False})
            latest[0] = res.x
        except Stop:
            retain((B@latest[0])[:63].reshape(21,3))
            raise
        retain((B@latest[0])[:63].reshape(21,3))
        return (B@latest[0])[:63].reshape(21,3)

    try:
        fallback = np.array([[(k%7)*.2, (k//7)*.2, .0999999999999] for k in range(21)])
        retain(fallback)
        if score > 2.30 and init is not None:
            optimize(model(initial=array(init)), params['polish'])
        for counts in ((3,5,4), (3,4,5), (5,4,4), (5,5,3)):
            check()
            m = model(counts)
            retain((m[1]@m[0])[:63].reshape(21,3))
            optimize(m, params['iterations'])
        if best_a is not None:
            optimize(model(initial=best_a), params['polish'])
        while time.monotonic() < deadline and best_a is not None:
            check()
            a = best_a.copy()
            ids = rng.choice(21, 5, replace=False)
            a[ids, :2] += rng.normal(0, params['perturbation'], (5,2))
            optimize(model(initial=a), params['polish'])
    except (Stop, ValueError, np.linalg.LinAlgError):
        pass
    if best is None:
        return {'circles': [[(k%7)*.2, (k//7)*.2, .099999999999] for k in range(21)]}
    return best
