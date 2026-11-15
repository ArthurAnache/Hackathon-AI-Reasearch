import time
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize

PARAMS = {
    'time_fraction': (0.70, 0.55, 0.75),
    'screen_fraction': (0.48, 0.25, 0.65),
    'screen_steps': (28, 10, 70),
    'polish_steps': (100, 20, 180),
    'minimum_radius': (0.000001, 0.0000001, 0.001, 'log')
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
P = np.arange(M)
K = np.arange(N)

class Deadline(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    begin = time.monotonic()
    end = begin + float(params['time_fraction']) * budget_s
    rng = np.random.default_rng(seed)
    best = None
    best_score = -np.inf
    best_array = None

    def guard(limit=None):
        if time.monotonic() >= (end if limit is None else min(end, limit)):
            raise Deadline()

    def exact_score(a):
        if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -np.inf
        d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
        if np.any(d < a[I, 2]+a[J, 2]):
            return -np.inf
        xl, xh, yl, yh = [], [], [], []
        for x, y, r in a:
            guard()
            x, y, r = Fraction(float(x)), Fraction(float(y)), Fraction(float(r))
            xl.append(x-r)
            xh.append(x+r)
            yl.append(y-r)
            yh.append(y+r)
        if max(xh)-min(xl)+max(yh)-min(yl) > 2:
            return -np.inf
        return float(np.sum(a[:, 2]))

    def consider(a):
        nonlocal best, best_score, best_array
        guard()
        if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return
        c = a.copy()
        c[:, :2] -= np.mean(c[:, :2], axis=0)
        d = np.sqrt((c[I, 0]-c[J, 0])**2 + (c[I, 1]-c[J, 1])**2)
        factor = min(1.0, float(np.min(d/(c[I, 2]+c[J, 2]))))
        if factor <= 0:
            return
        c[:, 2] *= factor * (1.0-2e-12)
        lo = np.min(c[:, :2]-c[:, 2, None], axis=0)
        hi = np.max(c[:, :2]+c[:, 2, None], axis=0)
        c *= 2.0 / float(np.sum(hi-lo)) * (1.0-2e-12)
        if float(np.sum(c[:, 2])) <= best_score:
            return
        score = exact_score(c)
        if score > best_score:
            best_score, best_array = score, c.copy()
            best = {'coordinate_rows': c.T.tolist()}

    def layout(counts):
        rows = []
        for row, count in enumerate(counts):
            guard()
            for col in range(count):
                guard()
                rows.append((2.0*col-count+1.0, row*np.sqrt(3.0), 1.0))
        a = np.asarray(rows)
        a *= 2.0/float(np.sum(np.ptp(a[:, :2], axis=0)+2.0))
        return a

    def pack(a):
        a = a.copy()
        a[:, 2] = np.maximum(np.abs(a[:, 2]), 1e-7)
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        scale = 2.0/max(float(np.sum(hi-lo)), 1e-12)
        a[:, :2] = (a[:, :2]-lo)*scale
        a[:, 2] *= scale
        return np.r_[a[:, 0], a[:, 1], a[:, 2], (hi[0]-lo[0])*scale]

    def unpack(z):
        return np.column_stack((z[:N], z[N:2*N], z[2*N:3*N]))

    raw = None
    try:
        if init is not None:
            raw = np.asarray(init['circles'] if 'circles' in init else np.asarray(init['coordinate_rows']).T, dtype=float)
            score = exact_score(raw)
            if score > best_score:
                best, best_score, best_array = init, score, raw.copy()
    except (ValueError, KeyError, TypeError, OverflowError):
        raw = None

    try:
        first = layout([4, 3, 4, 3, 4, 3])
        consider(first)
        second = layout([5, 4, 3, 4, 5])
        consider(second)
        starts = [first, second]
        if raw is not None and raw.shape == (N, 3) and np.isfinite(raw).all():
            starts.append(raw)

        B = np.zeros((4*N, 3*N+1))
        B[K, K] = -1
        B[K, 2*N+K] = 1
        B[N+K, K] = 1
        B[N+K, 2*N+K] = 1
        B[N+K, -1] = -1
        B[2*N+K, N+K] = -1
        B[2*N+K, 2*N+K] = 1
        B[3*N+K, N+K] = 1
        B[3*N+K, 2*N+K] = 1
        B[3*N+K, -1] = 1
        bb = np.r_[np.zeros(3*N), np.full(N, 2.0)]
        A = np.zeros((M+4*N, 3*N+1))
        A[M:] = B
        A[P, 2*N+I] = 1
        A[P, 2*N+J] = 1
        rhs = np.r_[np.zeros(M), bb]
        objective = np.r_[np.zeros(2*N), -np.ones(N), 0.0]
        bounds = [(0.0, 2.0)]*(2*N)+[(float(params['minimum_radius']), 0.7)]*N+[(0.02, 1.98)]
        screen_end = begin + (end-begin)*float(params['screen_fraction'])
        finalists = []

        for s, initial in enumerate(starts):
            guard()
            z = pack(initial)
            until = time.monotonic() + max(0.0, screen_end-time.monotonic())/(len(starts)-s)
            previous = -np.inf
            stale = 0
            for iteration in range(int(params['screen_steps'])):
                guard()
                remaining = until-time.monotonic()
                if remaining < 0.008:
                    break
                dx, dy = z[I]-z[J], z[N+I]-z[N+J]
                dist = np.hypot(dx, dy)
                zero = dist < 1e-14
                if np.any(zero):
                    angles = rng.uniform(0.0, 2*np.pi, int(np.sum(zero)))
                    dx[zero], dy[zero], dist[zero] = np.cos(angles), np.sin(angles), 1.0
                A[P, I], A[P, J] = -dx/dist, dx/dist
                A[P, N+I], A[P, N+J] = -dy/dist, dy/dist
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    result = linprog(objective, A_ub=A, b_ub=rhs, bounds=bounds, method='highs-ds', options={'threads': 1, 'time_limit': max(0.001, remaining-0.004), 'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9})
                guard()
                if not result.success:
                    break
                z = result.x
                consider(unpack(z))
                value = float(np.sum(z[2*N:3*N]))
                stale = stale+1 if value-previous < 1e-8 else 0
                previous = value
                if stale >= 2:
                    break
            finalists.append(z.copy())

        if best_array is not None:
            finalists.append(pack(best_array))
        finalists.sort(key=lambda v: -float(np.sum(v[2*N:3*N])))
        polish_end = end

        def constraints(v):
            guard(polish_end)
            return np.r_[np.hypot(v[I]-v[J], v[N+I]-v[N+J])-v[2*N+I]-v[2*N+J], bb-B@v]

        def jacobian(v):
            guard(polish_end)
            dx, dy = v[I]-v[J], v[N+I]-v[N+J]
            d = np.maximum(np.hypot(dx, dy), 1e-15)
            out = np.zeros_like(A)
            out[P, I], out[P, J] = dx/d, -dx/d
            out[P, N+I], out[P, N+J] = dy/d, -dy/d
            out[P, 2*N+I], out[P, 2*N+J] = -1, -1
            out[M:] = -B
            return out

        def fun(v):
            guard(polish_end)
            return float(objective@v)

        def callback(v):
            consider(unpack(v))
            guard(polish_end)

        used = []
        for index, z in enumerate(finalists):
            guard()
            if end-time.monotonic() < 0.025:
                break
            if used and min(float(np.linalg.norm(z-u)) for u in used) < 1e-5:
                continue
            used.append(z.copy())
            polish_end = time.monotonic() + (end-time.monotonic())*(0.75 if index == 0 else 1.0)
            try:
                result = minimize(fun, z, jac=lambda v: objective, method='SLSQP', bounds=bounds, constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian}, callback=callback, options={'maxiter': int(params['polish_steps']), 'ftol': 2e-12, 'disp': False})
                consider(unpack(result.x))
            except Deadline:
                if time.monotonic() >= end:
                    break
    except Deadline:
        pass
    return best if best is not None else init
