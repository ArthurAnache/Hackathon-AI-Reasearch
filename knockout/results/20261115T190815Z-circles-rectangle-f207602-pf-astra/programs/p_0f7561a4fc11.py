import time
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize, OptimizeWarning

PARAMS = {
    'time_fraction': (0.70, 0.55, 0.72),
    'lp_fraction': (0.46, 0.30, 0.55),
    'stage_iterations': (40, 10, 100),
    'small_jitter': (0.008, 0.001, 0.04),
    'large_jitter': (0.035, 0.01, 0.10),
    'polish_iterations': (100, 20, 200)
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
    lp_end = begin + float(params['lp_fraction']) * budget_s
    rng = np.random.default_rng(seed)
    best_obj = None
    best_a = None
    best_score = -float('inf')

    def guard():
        if time.monotonic() >= end:
            raise Deadline()

    def score(a):
        if a is None or a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
        if np.any(d < a[I, 2]+a[J, 2]):
            return -float('inf')
        xmin = ymin = xmax = ymax = None
        for x, y, r in a:
            guard()
            x, y, r = Fraction(float(x)), Fraction(float(y)), Fraction(float(r))
            xmin = x-r if xmin is None else min(xmin, x-r)
            xmax = x+r if xmax is None else max(xmax, x+r)
            ymin = y-r if ymin is None else min(ymin, y-r)
            ymax = y+r if ymax is None else max(ymax, y+r)
        if xmax-xmin+ymax-ymin > 2:
            return -float('inf')
        return float(np.sum(a[:, 2]))

    def consider(a):
        nonlocal best_obj, best_a, best_score
        guard()
        if a is None or a.shape != (N, 3) or not np.isfinite(a).all():
            return
        c = a.copy()
        if np.any(c[:, 2] <= 0):
            return
        c[:, :2] -= np.mean(c[:, :2], axis=0)
        d = np.sqrt((c[I, 0]-c[J, 0])**2 + (c[I, 1]-c[J, 1])**2)
        factor = min(1.0, float(np.min(d/(c[I, 2]+c[J, 2]))))
        if factor <= 0:
            return
        c[:, 2] *= factor*(1-3e-12)
        lo = np.min(c[:, :2]-c[:, 2, None], axis=0)
        hi = np.max(c[:, :2]+c[:, 2, None], axis=0)
        c[:, :2] -= (lo+hi)/2
        c *= (2.0/float(np.sum(hi-lo)))*(1-3e-12)
        if float(np.sum(c[:, 2])) <= best_score:
            return
        value = score(c)
        if value > best_score:
            best_score = value
            best_a = c.copy()
            best_obj = {'coordinate_rows': c.T.tolist()}

    def layout(counts):
        rows = []
        for k, count in enumerate(counts):
            guard()
            x = 2.0*(np.arange(count)-(count-1)/2)
            rows.append(np.column_stack((x, np.full(count, k*np.sqrt(3.0)), np.ones(count))))
        a = np.concatenate(rows)
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        a *= 2.0/float(np.sum(hi-lo))
        return a

    def pack(a):
        a = a.copy()
        a[:, 2] = np.maximum(np.abs(a[:, 2]), 1e-7)
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        scale = 2.0/max(float(np.sum(hi-lo)), 1e-15)
        a[:, :2] = (a[:, :2]-lo)*scale
        a[:, 2] *= scale
        return np.r_[a[:, 0], a[:, 1], a[:, 2], (hi[0]-lo[0])*scale]

    def unpack(z):
        return np.column_stack((z[:N], z[N:2*N], z[2*N:3*N]))

    original = None
    try:
        if init is not None:
            try:
                original = np.asarray(init['circles'] if 'circles' in init else np.asarray(init['coordinate_rows']).T, dtype=float)
            except (KeyError, ValueError, TypeError):
                original = None
            initial_score = score(original)
            if np.isfinite(initial_score):
                best_obj, best_score, best_a = init, initial_score, original.copy()
        fallback = np.column_stack(((K % 5 + 0.5)*0.2, (K // 5 + 0.5)*0.2, np.full(N, 0.1*(1-1e-12))))
        consider(fallback)
        first = layout([4, 3, 4, 3, 4, 3])
        consider(first)

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
        bounds = [(0.0, 2.0)]*(2*N)+[(1e-7, 0.5)]*N+[(0.04, 1.96)]

        def improve(a, until, steps):
            if a is None:
                return
            z = pack(a)
            previous = -1.0
            stale = 0
            for iteration in range(steps):
                guard()
                remaining = min(until, lp_end)-time.monotonic()
                if remaining < 0.012:
                    break
                dx = z[I]-z[J]
                dy = z[N+I]-z[N+J]
                dist = np.hypot(dx, dy)
                zero = dist < 1e-14
                if np.any(zero):
                    angles = rng.uniform(0, 2*np.pi, int(np.sum(zero)))
                    dx[zero], dy[zero] = np.cos(angles), np.sin(angles)
                    dist[zero] = 1.0
                nx, ny = dx/dist, dy/dist
                A[P, I], A[P, J] = -nx, nx
                A[P, N+I], A[P, N+J] = -ny, ny
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', OptimizeWarning)
                    result = linprog(objective, A_ub=A, b_ub=rhs, bounds=bounds, method='highs-ds', options={'threads': 1, 'time_limit': max(0.001, remaining-0.005), 'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9})
                guard()
                if not result.success:
                    break
                z = result.x
                consider(unpack(z))
                value = float(np.sum(z[2*N:3*N]))
                stale = stale+1 if abs(value-previous) < 2e-9 else 0
                previous = value
                if stale >= 2:
                    break

        steps = int(params['stage_iterations'])
        for stage in range(5):
            guard()
            if time.monotonic() >= lp_end-0.025:
                break
            if stage == 0:
                a, cap = first, 0.105
            elif stage == 1:
                a, cap = layout([5, 4, 3, 4, 5]), 0.105
            elif stage == 2:
                if original is not None and original.shape == (N, 3) and np.isfinite(original).all() and float(np.sum(original[:, 2])) > 2.0:
                    a = original.copy()
                else:
                    a = layout([4, 5, 3, 5, 4])
                cap = 0.075
            else:
                a = best_a.copy()
                amplitude = float(params['small_jitter'] if stage == 3 else params['large_jitter'])
                a[:, :2] += rng.normal(0.0, amplitude, (N, 2))
                cap = 0.055
            improve(a, min(lp_end, time.monotonic()+cap*budget_s), steps)
        improve(best_a, lp_end, 100)

        def constraints(z):
            guard()
            return np.r_[np.hypot(z[I]-z[J], z[N+I]-z[N+J])-z[2*N+I]-z[2*N+J], bb-B@z]

        def jacobian(z):
            guard()
            dx, dy = z[I]-z[J], z[N+I]-z[N+J]
            d = np.maximum(np.hypot(dx, dy), 1e-15)
            out = np.zeros_like(A)
            out[P, I], out[P, J] = dx/d, -dx/d
            out[P, N+I], out[P, N+J] = dy/d, -dy/d
            out[P, 2*N+I] = -1
            out[P, 2*N+J] = -1
            out[M:] = -B
            return out

        def fun(z):
            guard()
            return float(objective@z)

        if best_a is not None and time.monotonic() < end-0.025:
            result = minimize(fun, pack(best_a), jac=lambda z: objective, method='SLSQP', bounds=bounds, constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian}, callback=lambda z: consider(unpack(z)), options={'maxiter': int(params['polish_iterations']), 'ftol': 1e-12, 'disp': False})
            consider(unpack(result.x))
    except Deadline:
        pass
    return best_obj if best_obj is not None else init
