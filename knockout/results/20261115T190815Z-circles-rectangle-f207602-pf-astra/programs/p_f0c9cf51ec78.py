import time
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize

PARAMS = {
    'trust': (0.20, 0.05, 0.40),
    'screen_steps': (12, 5, 25),
    'screen_fraction': (0.34, 0.20, 0.45),
    'polish_steps': (90, 20, 160)
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
K = np.arange(N)
P = np.arange(M)

class Deadline(Exception):
    pass

def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return -np.inf
    x = list(map(Fraction, map(float, a[:, 0])))
    y = list(map(Fraction, map(float, a[:, 1])))
    r = list(map(Fraction, map(float, a[:, 2])))
    width = max(map(sum, zip(x, r))) - min(map(lambda q: q[0]-q[1], zip(x, r)))
    height = max(map(sum, zip(y, r))) - min(map(lambda q: q[0]-q[1], zip(y, r)))
    return float(np.sum(a[:, 2])) if width+height <= 2 else -np.inf

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70*budget_s
    best, best_score = None, -np.inf
    raw = None
    if init is not None:
        try:
            raw = np.asarray(init['circles'] if 'circles' in init else np.asarray(init['coordinate_rows']).T, dtype=float)
            sc = exact_score(raw)
            if sc > best_score:
                best, best_score = init, sc
        except (ValueError, TypeError, KeyError, OverflowError):
            raw = None
    kk = np.arange(N)
    fallback = np.column_stack(((kk % 5 + 0.5)*0.2, (kk // 5 + 0.5)*0.2, np.full(N, 0.1*(1-1e-12))))
    fs = exact_score(fallback)
    if fs > best_score:
        best, best_score = {'coordinate_rows': fallback.T.tolist()}, fs
    if time.monotonic() >= end:
        return best

    def normalize(a):
        a = a.copy()
        a[:, 2] = np.maximum(np.abs(a[:, 2]), 1e-7)
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        scale = 2/max(float(np.sum(hi-lo)), 1e-12)
        a[:, :2] = (a[:, :2]-lo)*scale
        a[:, 2] *= scale
        return np.r_[a[:, 0], a[:, 1], a[:, 2], (hi[0]-lo[0])*scale]

    def consider(v):
        nonlocal best, best_score
        if time.monotonic() >= end:
            raise Deadline()
        c = np.column_stack((v[:N], v[N:2*N], np.maximum(v[2*N:3*N], 1e-12)))
        d = np.sqrt((c[I, 0]-c[J, 0])**2 + (c[I, 1]-c[J, 1])**2)
        factor = min(1.0, float(np.min(d/(c[I, 2]+c[J, 2]))))
        if factor <= 0 or not np.isfinite(factor):
            return
        c[:, 2] *= factor*(1-3e-12)
        lo = np.min(c[:, :2]-c[:, 2, None], axis=0)
        hi = np.max(c[:, :2]+c[:, 2, None], axis=0)
        c[:, :2] -= (lo+hi)*0.5
        c *= (2/float(np.sum(hi-lo)))*(1-3e-12)
        if float(np.sum(c[:, 2])) > best_score:
            sc = exact_score(c)
            if sc > best_score:
                best_score = sc
                best = {'coordinate_rows': c.T.tolist()}

    objective = np.r_[np.zeros(2*N), -np.ones(N), 0.0]
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

    def lp_run(z, steps, deadline):
        stale = 0
        for iteration in range(steps):
            now = time.monotonic()
            if now >= min(end, deadline)-0.004:
                break
            dx, dy = z[I]-z[J], z[N+I]-z[N+J]
            dist = np.maximum(np.hypot(dx, dy), 1e-15)
            nx, ny = dx/dist, dy/dist
            A[P, I], A[P, J] = -nx, nx
            A[P, N+I], A[P, N+J] = -ny, ny
            trust = float(params['trust'])
            bounds = np.empty((3*N+1, 2))
            bounds[:2*N, 0] = z[:2*N]-trust
            bounds[:2*N, 1] = z[:2*N]+trust
            bounds[2*N:3*N] = [1e-7, 0.5]
            bounds[-1] = [0.05, 1.95]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result = linprog(objective, A_ub=A, b_ub=rhs, bounds=bounds, method='highs', options={'threads': 1, 'time_limit': max(0.001, min(end, deadline)-time.monotonic()), 'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9})
            if not result.success:
                break
            gain = float(np.sum(result.x[2*N:3*N])-np.sum(z[2*N:3*N]))
            z = result.x
            consider(z)
            stale = stale+1 if gain < 2e-8 else 0
            if stale >= 2:
                break
        return z

    local_end = end
    def guard():
        if time.monotonic() >= min(local_end, end):
            raise Deadline()

    def constraints(v):
        guard()
        return np.r_[np.hypot(v[I]-v[J], v[N+I]-v[N+J])-v[2*N+I]-v[2*N+J], bb-B@v]

    def jacobian(v):
        guard()
        dx, dy = v[I]-v[J], v[N+I]-v[N+J]
        dist = np.maximum(np.hypot(dx, dy), 1e-15)
        out = np.zeros_like(A)
        out[P, I], out[P, J] = dx/dist, -dx/dist
        out[P, N+I], out[P, N+J] = dy/dist, -dy/dist
        out[P, 2*N+I] = -1
        out[P, 2*N+J] = -1
        out[M:] = -B
        return out

    def fun(v):
        guard()
        return float(objective@v)

    starts = []
    if raw is not None and raw.shape == (N, 3) and np.isfinite(raw).all():
        starts.append(normalize(raw))
    for rows in ((4, 3, 4, 3, 4, 3), (5, 4, 5, 4, 3), (4, 5, 4, 5, 3)):
        if time.monotonic() >= end:
            return best
        chunks = []
        for row, count in enumerate(rows):
            if time.monotonic() >= end:
                return best
            chunks.append(np.column_stack((2*np.arange(count)-(count-1), np.full(count, row*np.sqrt(3.0)), np.ones(count))))
        starts.append(normalize(np.concatenate(chunks)))

    trajectories = []
    screen_end = start+float(params['screen_fraction'])*budget_s
    try:
        for index, z in enumerate(starts):
            if time.monotonic() >= screen_end:
                break
            remaining = len(starts)-index
            slot_end = time.monotonic()+max(0.0, screen_end-time.monotonic())/remaining
            consider(z)
            z = lp_run(z, int(params['screen_steps']), slot_end)
            trajectories.append(z)
        trajectories.sort(key=lambda v: float(np.sum(v[2*N:3*N])), reverse=True)
        if not trajectories:
            return best
        selected = trajectories[:2]
        for rank, z in enumerate(selected):
            now = time.monotonic()
            if now >= end-0.02:
                break
            local_end = end if rank == len(selected)-1 else now+0.60*(end-now)
            last = [z]
            def callback(v):
                guard()
                last[0] = v.copy()
                consider(v)
            try:
                z = lp_run(z, 10, min(local_end, now+0.10*budget_s))
                last[0] = z
                guard()
                result = minimize(fun, z, jac=lambda v: objective, method='SLSQP', bounds=[(None, None)]*(2*N)+[(1e-7, 0.5)]*N+[(0.05, 1.95)], constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian}, callback=callback, options={'maxiter': int(params['polish_steps']), 'ftol': 1e-12, 'disp': False})
                consider(result.x)
            except Deadline:
                if time.monotonic() >= end:
                    break
    except Deadline:
        pass
    return best
