import time
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize

PARAMS = {
    'screen_fraction': (0.30, 0.15, 0.45),
    'screen_iterations': (16, 6, 35),
    'polish_iterations': (110, 30, 200),
    'finalists': (3, 1, 4),
    'perturbation': (0.025, 0.0, 0.15),
    'radius_floor': (0.00002, 0.000001, 0.001, 'log')
}
N = 21
I, J = np.triu_indices(N, 1)

class Deadline(Exception):
    pass


def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    r = a[:, 2]
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < r[I]+r[J]):
        return -np.inf
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    q = [Fraction(float(v)) for v in r]
    w = max(a+b for a, b in zip(x, q))-min(a-b for a, b in zip(x, q))
    h = max(a+b for a, b in zip(y, q))-min(a-b for a, b in zip(y, q))
    return float(np.sum(r)) if w+h <= 2 else -np.inf


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70*budget_s
    screen_end = start + float(params['screen_fraction'])*budget_s
    rng = np.random.default_rng(seed)
    best = None
    best_score = -np.inf
    raw = None

    def guard(until=None):
        if time.monotonic() >= (end if until is None else min(end, until)):
            raise Deadline()

    if init is not None:
        try:
            raw = np.asarray(init['circles'], dtype=float) if 'circles' in init else np.asarray(init.get('packing_rows', init.get('coordinate_rows')), dtype=float).T
            score = exact_score(raw)
            if score > best_score:
                best, best_score = init, score
        except (ValueError, TypeError, KeyError, OverflowError):
            raw = None
    fallback = np.column_stack(((np.arange(N)%5+0.5)*0.2, (np.arange(N)//5+0.5)*0.2, np.full(N, 0.1*(1-1e-12))))
    score = exact_score(fallback)
    if score > best_score:
        best_score = score
        best = {'packing_rows': fallback.T.tolist()}

    def consider(v):
        nonlocal best, best_score
        guard()
        a = np.column_stack((v[:N], v[N:2*N], np.maximum(v[2*N:3*N], 1e-15)))
        d = np.sqrt((a[I, 0]-a[J, 0])**2+(a[I, 1]-a[J, 1])**2)
        shrink = min(1.0, float(np.min(d/(a[I, 2]+a[J, 2]))))
        if not np.isfinite(shrink) or shrink <= 0:
            return
        a[:, 2] *= shrink*(1-2e-12)
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        wh = float(np.sum(hi-lo))
        if wh <= 0:
            return
        a[:, :2] -= (lo+hi)/2
        a *= (2/wh)*(1-2e-12)
        if float(np.sum(a[:, 2])) > best_score:
            score = exact_score(a)
            if score > best_score:
                best_score = score
                best = {'packing_rows': a.T.tolist()}

    # Full-space box inequalities; vertical coordinates are centered.
    k = np.arange(N)
    B = np.zeros((4*N, 65))
    B[k, k] = -1
    B[k, 2*N+k] = 1
    B[N+k, k] = 1
    B[N+k, 2*N+k] = 1
    B[N+k, -2] = -1
    B[2*N+k, N+k] = 1
    B[2*N+k, 2*N+k] = 1
    B[2*N+k, -1] = -0.5
    B[3*N+k, N+k] = -1
    B[3*N+k, 2*N+k] = 1
    B[3*N+k, -1] = -0.5
    floor = float(params['radius_floor'])

    def model(axis, rows):
        guard()
        p = sum(rows)
        u = axis+p
        dim = 2*u+p+2
        T = np.zeros((65, dim))
        ids = np.r_[np.arange(u), np.arange(axis, u)]
        T[k, ids] = 1
        T[2*N+k, u+p+ids] = 1
        T[N+axis+np.arange(p), u+np.arange(p)] = 1
        T[N+u+np.arange(p), u+np.arange(p)] = -1
        T[-2, -2] = 1
        T[-1, -1] = 1
        reflection = np.r_[np.arange(axis), np.arange(u, N), np.arange(axis, u)]
        ri = np.minimum(reflection[I], reflection[J])
        rj = np.maximum(reflection[I], reflection[J])
        keep = (I*N+J <= ri*N+rj)
        ii, jj = I[keep], J[keep]
        DX = T[ii]-T[jj]
        DY = T[N+ii]-T[N+jj]
        R = T[2*N+ii]+T[2*N+jj]
        box = np.unique(B@T, axis=0)
        eq = np.sum(T[2*N:3*N], axis=0)
        obj = T[-2]+T[-1]
        bounds = [(0.0, 2.0)]*(u+p)+[(floor, 0.5)]*u+[(0.01, 2.0)]*2
        xs = [2*np.arange(axis, dtype=float)-(axis-1)]
        ys = []
        for level, count in enumerate(rows):
            guard()
            xs.append(2*np.arange(count, dtype=float)-(count-1))
            ys.append(np.full(count, np.sqrt(3.0)*(level+1)))
        x = np.concatenate(xs)
        x += rng.normal(0, float(params['perturbation']), u)
        y = np.concatenate(ys)
        x -= np.min(x)-1
        z = np.r_[x, y, np.ones(u), np.max(x)+1, 2*(np.max(y)+1)]/N
        return {'T': T, 'DX': DX, 'DY': DY, 'R': R, 'B': box, 'eq': eq, 'obj': obj, 'bounds': bounds, 'z': z}

    def lp(m, until):
        z = m['z']
        DX, DY, R, box = m['DX'], m['DY'], m['R'], m['B']
        for iteration in range(int(params['screen_iterations'])):
            guard()
            now = time.monotonic()
            if now >= until-0.005:
                break
            dx, dy = DX@z, DY@z
            d = np.maximum(np.hypot(dx, dy), 1e-15)
            A = np.vstack((R-(dx/d)[:, None]*DX-(dy/d)[:, None]*DY, box))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res = linprog(m['obj'], A_ub=A, b_ub=np.zeros(len(A)), A_eq=m['eq'][None, :], b_eq=[1.0], bounds=m['bounds'], method='highs', options={'threads': 1, 'time_limit': max(0.001, until-now), 'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9})
            guard()
            if not res.success:
                break
            old = float(m['obj']@z)
            z = res.x
            m['z'] = z
            consider(m['T']@z)
            if abs(old-float(m['obj']@z)) < 1e-11:
                break
        return m

    def polish(m, until):
        DX, DY, R, box = m['DX'], m['DY'], m['R'], m['B']
        last = m['z'].copy()
        def fun(z):
            guard(until)
            return float(m['obj']@z)
        def con(z):
            guard(until)
            return np.r_[np.hypot(DX@z, DY@z)-R@z, -box@z]
        def jac(z):
            guard(until)
            dx, dy = DX@z, DY@z
            d = np.maximum(np.hypot(dx, dy), 1e-15)
            return np.vstack(((dx/d)[:, None]*DX+(dy/d)[:, None]*DY-R, -box))
        def callback(z):
            nonlocal last
            guard(until)
            last = z.copy()
            m['z'] = last
            consider(m['T']@last)
        try:
            res = minimize(fun, last, jac=lambda z: m['obj'], method='SLSQP', bounds=m['bounds'], constraints=[{'type': 'ineq', 'fun': con, 'jac': jac}, {'type': 'eq', 'fun': lambda z: float(m['eq']@z-1), 'jac': lambda z: m['eq']}], callback=callback, options={'maxiter': int(params['polish_iterations']), 'ftol': 2e-13, 'disp': False})
            m['z'] = res.x
            consider(m['T']@res.x)
        except Deadline:
            pass

    try:
        families = [(3, (5, 4)), (3, (4, 5)), (5, (4, 4)), (1, (5, 5))]
        pool = []
        for index, (axis, rows) in enumerate(families):
            guard()
            now = time.monotonic()
            if now >= screen_end-0.015:
                break
            m = model(axis, rows)
            until = now+(screen_end-now)/(len(families)-index)
            lp(m, until)
            pool.append(m)
        if not pool:
            pool.append(model(3, (5, 4)))
        pool.sort(key=lambda m: float(m['obj']@m['z']))
        count = min(len(pool), int(params['finalists']))
        for index in range(count):
            guard()
            now = time.monotonic()
            if now >= end-0.025:
                break
            # Reserve enough time to finish each shortlisted topology.
            until = now+(end-now-0.008)/(count-index)
            polish(pool[index], until)
        guard()
        # A final unconstrained-space polish can escape a symmetry saddle.
        if time.monotonic() < end-0.035 and best is not None:
            a = np.asarray(best['circles'], dtype=float) if 'circles' in best else np.asarray(best.get('packing_rows', best.get('coordinate_rows')), dtype=float).T
            total = float(np.sum(a[:, 2]))
            lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
            hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
            a = a.copy()
            a[:, 0] -= lo[0]
            a[:, 1] -= (lo[1]+hi[1])/2
            a /= total
            T = np.eye(65)
            eq = np.zeros(65)
            eq[42:63] = 1
            obj = np.zeros(65)
            obj[-2:] = 1
            m = {'T': T, 'DX': T[I]-T[J], 'DY': T[N+I]-T[N+J], 'R': T[2*N+I]+T[2*N+J], 'B': B, 'eq': eq, 'obj': obj, 'bounds': [(0, 2)]*N+[(-2, 2)]*N+[(floor, 0.5)]*N+[(0.01, 2)]*2, 'z': np.r_[a[:, 0], a[:, 1], a[:, 2], (hi-lo)/total]}
            polish(m, end-0.005)
    except Deadline:
        pass
    return best
