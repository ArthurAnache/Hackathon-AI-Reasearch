import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {'iterations': (100, 40, 180), 'perturbation': (0.018, 0.001, 0.06), 'portfolio': (4, 1, 6)}


def construct(params, seed, budget_s, init=None):
    deadline = time.monotonic() + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    best = init
    best_score = -float('inf')
    best_v = None

    class Stop(Exception):
        pass

    def check():
        if time.monotonic() >= deadline:
            raise Stop()

    def array(obj):
        if 'circles' in obj:
            return np.asarray(obj['circles'], dtype=float)
        return np.column_stack((obj['centers'], obj['radii']))

    def exact(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        x = [Fraction(float(t)) for t in a[:, 0]]
        y = [Fraction(float(t)) for t in a[:, 1]]
        r = [Fraction(float(t)) for t in a[:, 2]]
        p = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        p += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        if p > 2:
            return -float('inf')
        d = a[ii, :2]-a[jj, :2]
        if np.any(np.sqrt(d[:, 0]**2+d[:, 1]**2) < a[ii, 2]+a[jj, 2]):
            return -float('inf')
        return float(np.sum(a[:, 2]))

    if init is not None:
        try:
            best_score = exact(array(init))
        except (ValueError, TypeError, KeyError):
            pass

    def retain(v):
        nonlocal best, best_score, best_v
        a = np.array(v[:63]).reshape(21, 3).copy()
        if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return
        a[:, :2] -= np.mean(a[:, :2], axis=0)
        d = a[ii, :2]-a[jj, :2]
        f = min(1., float(np.min(np.sqrt(d[:, 0]**2+d[:, 1]**2)/(a[ii, 2]+a[jj, 2]))))
        if f <= 0:
            return
        a[:, 2] *= f*(1-5e-14)
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        a *= 2/float(np.sum(hi-lo))*(1-8e-14)
        score = exact(a)
        if score > best_score:
            best_score = score
            best = {'centers': a[:, :2].tolist(), 'radii': a[:, 2].tolist()}
            lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
            hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
            a[:, 0] -= lo[0]
            a[:, 1] -= (lo[1]+hi[1])/2
            best_v = np.r_[a.ravel(), hi[0]-lo[0]]

    def layered(counts):
        singles, pairs, pts = [], [], []
        for level, count in enumerate(counts):
            check()
            for col in range(count):
                check()
                x = (col-(count-1)/2)*0.20
                if level == 0 and count == 3:
                    x *= 1.45
                if level == 0:
                    singles.append(len(pts))
                    pts.append([x, 0., .095])
                else:
                    i = len(pts)
                    pairs.append((i, i+1))
                    pts.extend([[x, level*.177, .095], [x, -level*.177, .095]])
        a = np.array(pts)
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        scale = 2/np.sum(hi-lo)
        a *= scale
        a[:, 0] -= lo[0]*scale
        B = np.zeros((64, 2*len(singles)+3*len(pairs)+1))
        c = 0
        for i in singles:
            check()
            B[3*i, c] = 1
            B[3*i+2, c+1] = 1
            c += 2
        for i,j in pairs:
            check()
            B[3*i, c] = B[3*j, c] = 1
            B[3*i+1, c+1] = 1
            B[3*j+1, c+1] = -1
            B[3*i+2, c+2] = B[3*j+2, c+2] = 1
            c += 3
        B[63, c] = 1
        return np.r_[a.ravel(), (hi[0]-lo[0])*scale], B

    def optimize(v, B, iterations):
        check()
        X, Y, R = B[:63:3], B[1:63:3], B[2:63:3]
        W = B[63]
        A, D, S = X[ii]-X[jj], Y[ii]-Y[jj], R[ii]+R[jj]
        # A distance is unchanged by either coordinate reflection.
        keys = np.concatenate((np.abs(A), np.abs(D), S), axis=1)
        _, keep = np.unique(keys, axis=0, return_index=True)
        A, D, S = A[keep], D[keep], S[keep]
        L = np.vstack((X-R, W-X-R, Y-R-W/2, -Y-R-W/2, R))
        offsets = np.r_[np.zeros(42), np.ones(42), np.full(21, -1e-8)]
        _, keep = np.unique(np.column_stack((L, offsets)), axis=0, return_index=True)
        L, offsets = L[keep], offsets[keep]
        g = -np.sum(R, axis=0)
        z = (B.T@v)/np.sum(B*B, axis=0)
        cache_z = None
        cache_c = None
        cache_j = None
        latest = [z.copy()]
        count = [0]

        def prepare(t):
            nonlocal cache_z, cache_c, cache_j
            check()
            if cache_z is not None and np.array_equal(t, cache_z):
                return
            dx, dy = A@t, D@t
            dist = np.sqrt(dx*dx+dy*dy)
            den = np.maximum(dist, 1e-15)
            cache_c = np.r_[dist-S@t, L@t+offsets]
            cache_j = np.vstack(((dx/den)[:, None]*A+(dy/den)[:, None]*D-S, L))
            cache_z = t.copy()

        def constraints(t):
            prepare(t)
            return cache_c

        def jac(t):
            prepare(t)
            return cache_j

        def fun(t):
            check()
            return float(g@t)

        def callback(t):
            check()
            latest[0] = t.copy()
            count[0] += 1
            if count[0] % 20 == 0:
                retain(B@t)

        try:
            result = minimize(fun, z, jac=lambda t: g, method='SLSQP',
                              constraints={'type': 'ineq', 'fun': constraints, 'jac': jac},
                              callback=callback,
                              options={'maxiter': iterations, 'ftol': 2e-11, 'disp': False})
            latest[0] = result.x
        except Stop:
            retain(B@latest[0])
            raise
        out = B@latest[0]
        retain(out)
        return out

    fallback = np.array([[(i % 7)*.2, (i//7)*.2, .0999999999999] for i in range(21)])
    if best_score == -float('inf'):
        best = {'circles': fallback.tolist()}
        best_score = exact(fallback)
    families = ((3,5,4), (5,4,4), (5,5,3), (3,4,5), (1,4,4,2), (3,3,4,2))
    solutions = []
    try:
        if best_score > 2.1:
            a = array(best).copy()
            lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
            hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
            a[:, 0] -= lo[0]
            a[:, 1] -= (lo[1]+hi[1])/2
            optimize(np.r_[a.ravel(), hi[0]-lo[0]], np.eye(64), 65)
        for counts in families[:params['portfolio']]:
            check()
            v, B = layered(counts)
            u = optimize(v, B, params['iterations'])
            solutions.append((float(np.sum(u[2:63:3])), u, B))
        solutions.sort(key=lambda t: t[0], reverse=True)
        if best_v is not None:
            optimize(best_v.copy(), np.eye(64), 65)
        attempt = 0
        while time.monotonic() < deadline and solutions:
            check()
            _, v, B = solutions[attempt % min(2, len(solutions))]
            z = (B.T@v)/np.sum(B*B, axis=0)
            noise = rng.normal(0, params['perturbation'], len(z))
            noise[-1] *= .25
            u = optimize(B@(z+noise), B, 65)
            if float(np.sum(u[2:63:3])) > solutions[0][0]:
                solutions.insert(0, (float(np.sum(u[2:63:3])), u, B))
                if best_v is not None:
                    optimize(best_v.copy(), np.eye(64), 40)
            attempt += 1
    except (Stop, ValueError, np.linalg.LinAlgError):
        pass
    return best
