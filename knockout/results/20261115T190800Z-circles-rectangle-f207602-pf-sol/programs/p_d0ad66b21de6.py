import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {'exploration': (0.51, 0.35, 0.58), 'perturbation': (0.025, 0.005, 0.07), 'tolerance': (2e-11, 1e-13, 1e-8, 'log')}


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    best, best_score, best_c = init, -float('inf'), None
    archive = []

    class Stop(Exception):
        pass

    def exact(c):
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return -float('inf')
        dx = c[ii, 0] - c[jj, 0]
        dy = c[ii, 1] - c[jj, 1]
        if np.any(np.sqrt(dx * dx + dy * dy) < c[ii, 2] + c[jj, 2]):
            return -float('inf')
        x = [Fraction(float(v)) for v in c[:, 0]]
        y = [Fraction(float(v)) for v in c[:, 1]]
        r = [Fraction(float(v)) for v in c[:, 2]]
        w = max(a+b for a,b in zip(x,r)) - min(a-b for a,b in zip(x,r))
        h = max(a+b for a,b in zip(y,r)) - min(a-b for a,b in zip(y,r))
        if w+h > 2:
            return -float('inf')
        return float(np.sum(c[:, 2]))

    if init is not None:
        try:
            if 'circles' in init:
                c = np.asarray(init['circles'], dtype=float)
            else:
                z = np.asarray(init['state'], dtype=float)
                c = np.column_stack((z[:21], z[21:42], z[42:63]))
            best_score = exact(c)
            if np.isfinite(best_score):
                best_c = c.copy()
        except Exception:
            pass

    def consider(c):
        nonlocal best, best_score, best_c
        if time.monotonic() >= deadline or not np.isfinite(c).all():
            return
        c = c.copy()
        c[:, 2] = np.maximum(c[:, 2], 1e-10)
        dx = c[ii, 0]-c[jj, 0]
        dy = c[ii, 1]-c[jj, 1]
        rat = np.min(np.sqrt(dx*dx+dy*dy)/(c[ii, 2]+c[jj, 2]))
        c[:, 2] *= min(1.0, float(rat))*(1-5e-14)
        lo = np.min(c[:, :2]-c[:, 2, None], axis=0)
        hi = np.max(c[:, :2]+c[:, 2, None], axis=0)
        total = float(np.sum(hi-lo))
        if total <= 0:
            return
        c[:, :2] -= (hi+lo)*0.5
        c *= (2/total)*(1-5e-14)
        if float(np.sum(c[:, 2])) <= best_score:
            return
        value = exact(c)
        if value > best_score:
            best_score = value
            best_c = c.copy()
            best = {'circles': c.tolist()}

    grid = np.array([[(i-2)*2/9, (j-1.5)*2/9, 1/9] for j in range(4) for i in range(5)] + [[-1/9, -2/9, (np.sqrt(2)-1)/9]])
    consider(grid)

    def family(axis, near, far):
        m = 2*axis + 3*(near+far) + 1
        A = np.zeros((64, m))
        q = np.zeros(m)
        bounds = []
        k = 0
        n = 0
        for x in np.linspace(-0.11*(axis-1), 0.11*(axis-1), axis):
            if time.monotonic() >= deadline:
                raise Stop()
            A[n, k] = 1
            q[k] = x
            bounds.append((-0.95, 0.95))
            A[n+42, k+1] = 1
            q[k+1] = 0.105
            bounds.append((1e-7, 0.49))
            n += 1
            k += 2
        for count, height in ((near, 0.195), (far, 0.39)):
            if time.monotonic() >= deadline:
                raise Stop()
            for x in np.linspace(-0.11*(count-1), 0.11*(count-1), count):
                if time.monotonic() >= deadline:
                    raise Stop()
                A[n, k] = A[n+1, k] = 1
                A[n+21, k+1] = 1
                A[n+22, k+1] = -1
                A[n+42, k+2] = A[n+43, k+2] = 1
                q[k:k+3] = (x, height, 0.105)
                bounds.extend([(-0.95, 0.95), (0.001, 0.95), (1e-7, 0.49)])
                n += 2
                k += 3
        A[63, -1] = 1
        q[-1] = 1.0
        bounds.append((0.15, 1.85))
        return prepare(A, bounds), q

    def prepare(A, bounds):
        X, Y, R = A[:21], A[21:42], A[42:63]
        dx, dy, sr = X[ii]-X[jj], Y[ii]-Y[jj], R[ii]+R[jj]
        # Distance is unchanged by independently reversing either component.
        keys = set()
        keep = []
        for k in range(len(ii)):
            if time.monotonic() >= deadline:
                raise Stop()
            a, b = dx[k].copy(), dy[k].copy()
            ia, ib = np.flatnonzero(a), np.flatnonzero(b)
            if len(ia) and a[ia[0]] < 0:
                a = -a
            if len(ib) and b[ib[0]] < 0:
                b = -b
            key = (a.tobytes(), b.tobytes(), sr[k].tobytes())
            if key not in keys:
                keys.add(key)
                keep.append(k)
        dx, dy, sr = dx[keep], dy[keep], sr[keep]
        W = A[63]
        wall = np.concatenate((W[None, :]/2-X-R, W[None, :]/2+X-R,
                               -W[None, :]/2-Y-R, -W[None, :]/2+Y-R))
        off = np.r_[np.zeros(42), np.ones(42)]
        _, unique = np.unique(np.column_stack((wall, off)), axis=0, return_index=True)
        wall, off = wall[unique], off[unique]
        return (A, bounds, dx, dy, sr, wall, off, -R.sum(axis=0))

    def solve(problem, initial, until):
        A, bounds, DX, DY, SR, wall, off, objective = problem
        until = min(until, deadline)
        last = initial.copy()
        local_score = -float('inf')
        local_q = initial.copy()
        def check():
            if time.monotonic() >= until:
                raise Stop()
        def fun(q):
            check()
            return float(objective @ q)
        def con(q):
            check()
            dx, dy = DX@q, DY@q
            return np.r_[np.sqrt(dx*dx+dy*dy)-SR@q, wall@q+off]
        def jac(q):
            check()
            dx, dy = DX@q, DY@q
            dist = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-16)
            return np.vstack(((dx/dist)[:, None]*DX+(dy/dist)[:, None]*DY-SR, wall))
        def callback(q):
            nonlocal last, local_score, local_q
            check()
            last = q.copy()
            z = A@q
            c = np.column_stack((z[:21], z[21:42], z[42:63]))
            # Rank feasible quotient iterates without Fraction overhead.
            violation = float(np.min(con(q)))
            if violation >= -1e-7:
                value = -float(objective@q)
                if value > local_score:
                    local_score, local_q = value, q.copy()
            consider(c)
        try:
            result = minimize(fun, initial, jac=lambda q: objective, bounds=bounds,
                              constraints={'type': 'ineq', 'fun': con, 'jac': jac},
                              method='SLSQP', callback=callback,
                              options={'maxiter': 160, 'ftol': params['tolerance'], 'disp': False})
            callback(result.x)
        except (Stop, ValueError, FloatingPointError):
            pass
        return local_score, local_q if np.isfinite(local_score) else last

    try:
        families = [(3, 5, 4), (3, 4, 5), (5, 4, 4)]
        explore_end = start + params['exploration']*budget_s
        for k, counts in enumerate(families):
            if time.monotonic() >= explore_end:
                break
            p, q = family(*counts)
            remaining = len(families)-k
            until = time.monotonic() + max(0, explore_end-time.monotonic())/remaining
            score, q = solve(p, q, until)
            archive.append((score, p, q))
        perturb_end = start + 0.61*budget_s
        for attempt in range(2):
            if time.monotonic() >= perturb_end or not archive:
                break
            _, p, q0 = max(archive, key=lambda item: item[0])
            q = q0.copy()
            noise = rng.normal(0, params['perturbation'], len(q))
            # Keep radii initially unchanged; rearrange centers and aspect ratio.
            radius_columns = np.any(p[0][42:63] != 0, axis=0)
            noise[radius_columns] = 0
            q += noise
            for j, (low, high) in enumerate(p[1]):
                if time.monotonic() >= deadline:
                    raise Stop()
                q[j] = np.clip(q[j], low+1e-8, high-1e-8)
            until = time.monotonic() + max(0, perturb_end-time.monotonic())/(2-attempt)
            score, q = solve(p, q, until)
            archive.append((score, p, q))
        if best_c is not None and time.monotonic() < deadline:
            c = best_c.copy()
            lo = np.min(c[:, :2]-c[:, 2, None], axis=0)
            hi = np.max(c[:, :2]+c[:, 2, None], axis=0)
            c[:, :2] -= (hi+lo)/2
            q = np.r_[c[:, 0], c[:, 1], c[:, 2], hi[0]-lo[0]]
            p = prepare(np.eye(64), [(-1, 1)]*42+[(1e-8, 0.5)]*21+[(0.1, 1.9)])
            solve(p, q, deadline)
    except Stop:
        pass
    return best
