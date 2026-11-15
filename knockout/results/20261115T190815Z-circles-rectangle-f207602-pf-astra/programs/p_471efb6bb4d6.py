import time
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize

PARAMS = {
    'screen_iterations': (18, 8, 40),
    'refine_iterations': (48, 15, 100),
    'screen_fraction': (0.35, 0.20, 0.45),
    'lp_fraction': (0.49, 0.40, 0.58),
    'radius_floor': (0.0001, 0.000001, 0.005, 'log'),
    'polish_iterations': (90, 25, 180),
    'probe_iterations': (16, 6, 30),
    'probe_fraction': (0.55, 0.25, 0.75)
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
P = np.arange(M)
K = np.arange(N)
D = 65

class Deadline(Exception):
    pass

class ProbeDeadline(Exception):
    pass

def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    r = a[:, 2]
    dist = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(dist < r[I]+r[J]):
        return -np.inf
    x = list(map(Fraction, map(float, a[:, 0])))
    y = list(map(Fraction, map(float, a[:, 1])))
    q = list(map(Fraction, map(float, r)))
    width = max(a+b for a, b in zip(x, q))-min(a-b for a, b in zip(x, q))
    height = max(a+b for a, b in zip(y, q))-min(a-b for a, b in zip(y, q))
    return float(np.sum(r)) if width+height <= 2 else -np.inf

def encode(a):
    return {'packing_rows': a.T.tolist()}

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    end = started + 0.70*budget_s
    screen_end = started + float(params['screen_fraction'])*budget_s
    lp_end = started + float(params['lp_fraction'])*budget_s
    rng = np.random.default_rng(seed)
    best = None
    best_score = -np.inf
    raw = None

    def guard():
        if time.monotonic() >= end:
            raise Deadline()

    if init is not None:
        try:
            if 'circles' in init:
                raw = np.asarray(init['circles'], dtype=float)
            elif 'packing_rows' in init:
                raw = np.asarray(init['packing_rows'], dtype=float).T
            else:
                raw = np.asarray(init['coordinate_rows'], dtype=float).T
            sc = exact_score(raw)
            if sc > best_score:
                best, best_score = init, sc
        except (KeyError, ValueError, TypeError, OverflowError):
            raw = None
    fallback = np.column_stack(((np.arange(N)%5+0.5)*0.2,
                                (np.arange(N)//5+0.5)*0.2,
                                np.full(N, 0.1*(1-1e-12))))
    sc = exact_score(fallback)
    if sc > best_score:
        best, best_score = encode(fallback), sc

    objective = np.zeros(D)
    objective[-2:] = 1.0
    equality = np.zeros((1, D))
    equality[0, 2*N:3*N] = 1.0
    B = np.zeros((4*N, D))
    B[K, K] = -1.0
    B[K, 2*N+K] = 1.0
    B[N+K, K] = 1.0
    B[N+K, 2*N+K] = 1.0
    B[N+K, -2] = -1.0
    B[2*N+K, N+K] = -1.0
    B[2*N+K, 2*N+K] = 1.0
    B[3*N+K, N+K] = 1.0
    B[3*N+K, 2*N+K] = 1.0
    B[3*N+K, -1] = -1.0
    A = np.zeros((M+4*N, D))
    A[M:] = B
    A[P, 2*N+I] = 1.0
    A[P, 2*N+J] = 1.0
    rhs = np.zeros(M+4*N)
    floor = float(params['radius_floor'])
    bounds = [(0.0, 2.0)]*(2*N)+[(floor, 0.5)]*N+[(0.01, 2.0)]*2

    def state(a):
        a = np.array(a, dtype=float, copy=True)
        a[:, 2] = np.maximum(np.abs(a[:, 2]), 1e-8)
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        total = float(a[:, 2].sum())
        a[:, :2] -= lo
        a /= total
        return np.r_[a[:, 0], a[:, 1], a[:, 2], (hi-lo)/total]

    def consider(v, certify=False):
        nonlocal best, best_score
        guard()
        c = np.column_stack((v[:N], v[N:2*N], np.maximum(v[2*N:3*N], 1e-14)))
        dist = np.sqrt((c[I, 0]-c[J, 0])**2+(c[I, 1]-c[J, 1])**2)
        factor = min(1.0, float(np.min(dist/(c[I, 2]+c[J, 2]))))
        if not np.isfinite(factor) or factor <= 0:
            return -np.inf
        c[:, 2] *= factor*(1-3e-12)
        lo = np.min(c[:, :2]-c[:, 2, None], axis=0)
        hi = np.max(c[:, :2]+c[:, 2, None], axis=0)
        perimeter = float((hi-lo).sum())
        if not np.isfinite(perimeter) or perimeter <= 0:
            return -np.inf
        c[:, :2] -= (hi+lo)*0.5
        c *= (2.0/perimeter)*(1-3e-12)
        score = float(c[:, 2].sum())
        if certify or score > best_score:
            score = exact_score(c)
            if score > best_score:
                best, best_score = encode(c), score
        return score

    def run_lp(v, count, until):
        stale = 0
        for iteration in range(count):
            guard()
            now = time.monotonic()
            if now >= until-0.004:
                break
            dx = v[I]-v[J]
            dy = v[N+I]-v[N+J]
            distances = np.maximum(np.hypot(dx, dy), 1e-15)
            nx, ny = dx/distances, dy/distances
            A[P, I] = -nx
            A[P, J] = nx
            A[P, N+I] = -ny
            A[P, N+J] = ny
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result = linprog(objective, A_ub=A, b_ub=rhs,
                    A_eq=equality, b_eq=[1.0], bounds=bounds, method='highs',
                    options={'threads': 1, 'time_limit': max(0.001, until-now),
                             'primal_feasibility_tolerance': 1e-9,
                             'dual_feasibility_tolerance': 1e-9})
            guard()
            if not result.success:
                break
            gain = float(v[-2:].sum()-result.x[-2:].sum())
            v = result.x
            consider(v)
            stale = stale+1 if abs(gain) < 2e-10 else 0
            if stale >= 3:
                break
        return v

    seeds = []
    try:
        guard()
        if raw is not None and raw.shape == (N, 3) and np.isfinite(raw).all():
            a = raw.copy()
            extent = float(np.ptp(a[:, :2], axis=0).sum())
            if extent > 0:
                a[:, 2] = max(extent/40.0, 1e-6)
                seeds.append(state(a))
        patterns = [(4, 5, 3, 5, 4), (5, 4, 3, 4, 5),
                    (4, 5, 4, 5, 3), (4, 4, 5, 4, 4)]
        for index, rows in enumerate(patterns):
            guard()
            circles = []
            for row, count in enumerate(rows):
                guard()
                xs = 2*np.arange(count, dtype=float)-(count-1)
                if index == 3:
                    xs += 0.22*(row%2)
                ys = np.full(count, np.sqrt(3.0)*row)
                circles.append(np.column_stack((xs, ys, np.ones(count))))
            a = np.vstack(circles)
            a[:, :2] += rng.normal(0.0, 0.018, (N, 2))
            seeds.append(state(a))
        survivors = []
        for index, v in enumerate(seeds):
            guard()
            now = time.monotonic()
            if now >= screen_end-0.008:
                break
            remaining = len(seeds)-index
            allowance = (screen_end-now)/remaining
            v = run_lp(v, int(params['screen_iterations']), min(screen_end, now+allowance))
            survivors.append(v)
        if not survivors:
            survivors = [state(fallback)]
        survivors.sort(key=lambda v: float(v[-2:].sum()))
        v = run_lp(survivors[0], int(params['refine_iterations']), lp_end)
        consider(v)
        survivors[0] = v

        local_end = end

        def polish_guard():
            guard()
            if time.monotonic() >= local_end:
                raise ProbeDeadline()

        def constraints(z):
            polish_guard()
            dx = z[I]-z[J]
            dy = z[N+I]-z[N+J]
            return np.r_[np.hypot(dx, dy)-z[2*N+I]-z[2*N+J], -B@z]

        def jacobian(z):
            polish_guard()
            dx = z[I]-z[J]
            dy = z[N+I]-z[N+J]
            distance = np.maximum(np.hypot(dx, dy), 1e-15)
            out = np.zeros((M+4*N, D))
            out[P, I] = dx/distance
            out[P, J] = -dx/distance
            out[P, N+I] = dy/distance
            out[P, N+J] = -dy/distance
            out[P, 2*N+I] = -1
            out[P, 2*N+J] = -1
            out[M:] = -B
            return out

        def fun(z):
            polish_guard()
            return float(z[-2]+z[-1])

        def eqfun(z):
            polish_guard()
            return np.array([z[2*N:3*N].sum()-1.0])

        def polish(start, count, until):
            nonlocal local_end
            local_end = until
            champion = start.copy()
            champion_score = consider(champion, certify=True)

            def callback(z):
                nonlocal champion, champion_score
                polish_guard()
                sc = consider(z, certify=True)
                if sc > champion_score:
                    champion, champion_score = z.copy(), sc

            try:
                result = minimize(fun, start, jac=lambda z: objective, method='SLSQP',
                    bounds=bounds,
                    constraints=[{'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                                 {'type': 'eq', 'fun': eqfun, 'jac': lambda z: equality}],
                    callback=callback,
                    options={'maxiter': count, 'ftol': 3e-13, 'disp': False})
                callback(result.x)
            except ProbeDeadline:
                pass
            return champion_score, champion

        # Rank nonlinear basins by certified output, not the unfinished LP bound.
        finalists = survivors[:2]
        now = time.monotonic()
        probe_end = now + float(params['probe_fraction'])*max(0.0, end-now)
        polished = []
        for index, candidate in enumerate(finalists):
            guard()
            now = time.monotonic()
            if now >= probe_end-0.006:
                polished.append((consider(candidate, certify=True), candidate))
                continue
            until = now+(probe_end-now)/(len(finalists)-index)
            polished.append(polish(candidate, int(params['probe_iterations']), until))
        guard()
        polished.sort(key=lambda item: item[0], reverse=True)
        if polished and time.monotonic() < end-0.012:
            polish(polished[0][1], int(params['polish_iterations']), end)
    except Deadline:
        pass
    return best
