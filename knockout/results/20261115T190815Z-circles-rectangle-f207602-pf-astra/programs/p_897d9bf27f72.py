import time
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize

PARAMS = {
    'screen_fraction': (0.26, 0.15, 0.40),
    'screen_iterations': (14, 5, 30),
    'polish_iterations': (110, 30, 180),
    'polish_candidates': (3, 1, 6),
    'radius_floor': (0.00003, 0.000001, 0.002, 'log'),
    'jitter': (0.035, 0.001, 0.15, 'log')
}

N = 21
D = 35
II, JJ = np.triu_indices(N, 1)

class Deadline(Exception):
    pass

class PhaseEnd(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    r = a[:, 2]
    dist = np.sqrt((a[II, 0]-a[JJ, 0])**2 + (a[II, 1]-a[JJ, 1])**2)
    if np.any(dist < r[II]+r[JJ]):
        return -np.inf
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    q = [Fraction(float(v)) for v in r]
    w = max(u+v for u, v in zip(x, q))-min(u-v for u, v in zip(x, q))
    h = max(u+v for u, v in zip(y, q))-min(u-v for u, v in zip(y, q))
    return float(np.sum(r)) if w+h <= 2 else -np.inf

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    finish = start + 0.70*budget_s
    screen_end = start + float(params['screen_fraction'])*budget_s
    rng = np.random.default_rng(seed)
    best = None
    best_score = -np.inf
    best_z = None

    def guard():
        if time.monotonic() >= finish:
            raise Deadline()

    if init is not None:
        try:
            if 'circles' in init:
                a = np.asarray(init['circles'], dtype=float)
            else:
                a = np.asarray(init.get('packing_rows', init.get('coordinate_rows')), dtype=float).T
            s = exact_score(a)
            if s > best_score:
                best, best_score = init, s
        except (ValueError, TypeError, OverflowError):
            pass
    fallback = np.column_stack(((np.arange(21)%5+0.5)*0.2,
                                (np.arange(21)//5+0.5)*0.2,
                                np.full(21, 0.1*(1-1e-12))))
    s = exact_score(fallback)
    if s > best_score:
        best, best_score = {'packing_rows': fallback.T.tolist()}, s

    try:
        guard()
        # Coordinates: twelve representative x's, nine positive y's,
        # twelve representative radii, width, and full height.
        rep = np.r_[np.arange(12), np.arange(3, 12)]
        EX = np.zeros((N, D))
        EY = np.zeros((N, D))
        ER = np.zeros((N, D))
        EX[np.arange(N), rep] = 1
        ER[np.arange(N), 21+rep] = 1
        EY[np.arange(3, 12), np.arange(12, 21)] = 1
        EY[np.arange(12, 21), np.arange(12, 21)] = -1
        reflection = np.r_[np.arange(3), np.arange(12, 21), np.arange(3, 12)]
        pairs = []
        for i, j in zip(II, JJ):
            guard()
            reflected = tuple(sorted((int(reflection[i]), int(reflection[j]))))
            if (int(i), int(j)) <= reflected:
                pairs.append((i, j))
        pi, pj = np.asarray(pairs).T
        DX = EX[pi]-EX[pj]
        DY = EY[pi]-EY[pj]
        RR = ER[pi]+ER[pj]
        H = np.vstack((EX[:12]-ER[:12], -EX[:12]-ER[:12], -EY[:12]-ER[:12]))
        H[12:24, 33] = 1
        H[24:, 34] = 0.5
        objective = np.zeros(D)
        objective[33:] = 1
        equality = np.sum(ER, axis=0, keepdims=True)
        floor = float(params['radius_floor'])
        bounds = [(0., 2.)]*21 + [(floor, 0.5)]*12 + [(0.01, 2.)]*2
        rhs = np.zeros(len(pi)+len(H))

        def consider(z):
            nonlocal best, best_score, best_z
            guard()
            c = np.column_stack((EX@z, EY@z, ER@z))
            if np.any(c[:, 2] <= 0) or not np.isfinite(c).all():
                return
            distances = np.sqrt((c[II, 0]-c[JJ, 0])**2+(c[II, 1]-c[JJ, 1])**2)
            shrink = min(1., float(np.min(distances/(c[II, 2]+c[JJ, 2]))))
            if shrink <= 0:
                return
            c[:, 2] *= shrink*(1-3e-12)
            lo = np.min(c[:, :2]-c[:, 2, None], axis=0)
            hi = np.max(c[:, :2]+c[:, 2, None], axis=0)
            perimeter = float(np.sum(hi-lo))
            if perimeter <= 0:
                return
            c[:, :2] -= (hi+lo)/2
            c *= 2*(1-3e-12)/perimeter
            if float(np.sum(c[:, 2])) > best_score:
                score = exact_score(c)
                if score > best_score:
                    best = {'packing_rows': c.T.tolist()}
                    best_score = score
                    best_z = z.copy()

        def seed_layout(inner, variant):
            guard()
            outer = 9-inner
            x = np.r_[2*np.arange(3)-2,
                      2*np.arange(inner)-(inner-1),
                      2*np.arange(outer)-(outer-1)].astype(float)
            y = np.r_[np.full(inner, np.sqrt(3.)), np.full(outer, 2*np.sqrt(3.))]
            jitter = float(params['jitter'])
            x += rng.normal(0, jitter, 12)
            y += rng.normal(0, jitter*0.4, 9)
            if variant == 1:
                x[3:3+inner] += 0.45
                x[3+inner:] -= 0.25
            x -= np.min(x)-1
            return np.r_[x, y, np.ones(12), np.max(x)+1, 2*(np.max(y)+1)]/21

        def run_lp(z, until):
            stale = 0
            for iteration in range(int(params['screen_iterations'])):
                guard()
                now = time.monotonic()
                if now >= until-0.003:
                    break
                dx, dy = DX@z, DY@z
                distance = np.maximum(np.hypot(dx, dy), 1e-15)
                A = np.vstack((RR-(dx/distance)[:, None]*DX-(dy/distance)[:, None]*DY, -H))
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    result = linprog(objective, A_ub=A, b_ub=rhs,
                                     A_eq=equality, b_eq=[1.], bounds=bounds,
                                     method='highs', options={
                                         'threads': 1,
                                         'time_limit': max(0.001, until-now),
                                         'primal_feasibility_tolerance': 1e-9,
                                         'dual_feasibility_tolerance': 1e-9})
                guard()
                if not result.success:
                    break
                change = abs(float(objective@(z-result.x)))
                z = result.x
                consider(z)
                stale = stale+1 if change < 1e-10 else 0
                if stale >= 2:
                    break
            return z

        layouts = [(5, 0), (4, 0), (5, 1), (4, 1), (6, 0), (3, 0)]
        survivors = []
        for index, (inner, variant) in enumerate(layouts):
            guard()
            now = time.monotonic()
            if now >= screen_end-0.005:
                break
            z = seed_layout(inner, variant)
            until = now+(screen_end-now)/(len(layouts)-index)
            z = run_lp(z, until)
            survivors.append(z)
        if not survivors:
            survivors.append(seed_layout(5, 0))
        survivors.sort(key=lambda z: float(objective@z))

        def polish(z, until):
            latest = z.copy()
            def check():
                guard()
                if time.monotonic() >= until:
                    raise PhaseEnd()
            def fun(v):
                check()
                return float(objective@v)
            def constraints(v):
                check()
                return np.r_[np.hypot(DX@v, DY@v)-RR@v, H@v]
            def jacobian(v):
                check()
                dx, dy = DX@v, DY@v
                d = np.maximum(np.hypot(dx, dy), 1e-15)
                return np.vstack(((dx/d)[:, None]*DX+(dy/d)[:, None]*DY-RR, H))
            def callback(v):
                nonlocal latest
                check()
                latest = v.copy()
                consider(v)
            try:
                result = minimize(fun, z, jac=lambda v: objective, method='SLSQP',
                    bounds=bounds,
                    constraints=[{'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                                 {'type': 'eq', 'fun': lambda v: equality@v-1.,
                                  'jac': lambda v: equality}],
                    callback=callback,
                    options={'maxiter': int(params['polish_iterations']),
                             'ftol': 2e-13, 'disp': False})
                latest = result.x
                consider(latest)
            except PhaseEnd:
                pass
            return latest

        count = min(int(params['polish_candidates']), len(survivors))
        for index in range(count):
            guard()
            now = time.monotonic()
            remaining = finish-now
            if remaining < 0.015:
                break
            allowance = remaining/(count-index+0.35)
            survivors[index] = polish(survivors[index], now+allowance)
        guard()
        if best_z is not None and time.monotonic() < finish-0.015:
            polish(best_z, finish)
    except Deadline:
        pass
    return best
