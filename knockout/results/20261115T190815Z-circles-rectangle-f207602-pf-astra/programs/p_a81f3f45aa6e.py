import time
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize

PARAMS = {
    'screen_iterations': (16, 8, 35),
    'screen_fraction': (0.25, 0.15, 0.35),
    'symmetric_fraction': (0.56, 0.42, 0.62),
    'polish_iterations': (85, 30, 150),
    'radius_floor': (0.00002, 0.000001, 0.001, 'log')
}
N = 21
I, J = np.triu_indices(N, 1)
K = np.arange(N)
D = 65

class Deadline(Exception):
    pass


def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    r = a[:, 2]
    distance = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(distance < r[I]+r[J]):
        return -np.inf
    x = list(map(Fraction, map(float, a[:, 0])))
    y = list(map(Fraction, map(float, a[:, 1])))
    q = list(map(Fraction, map(float, r)))
    w = max(a+b for a, b in zip(x, q))-min(a-b for a, b in zip(x, q))
    h = max(a+b for a, b in zip(y, q))-min(a-b for a, b in zip(y, q))
    return float(np.sum(r)) if w+h <= 2 else -np.inf


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70*budget_s
    screen_end = start + params['screen_fraction']*budget_s
    sym_end = start + params['symmetric_fraction']*budget_s
    best = init
    best_score = -np.inf
    best_array = None
    raw = None

    def guard(until=None):
        if time.monotonic() >= (end if until is None else min(end, until)):
            raise Deadline()

    if init is not None:
        try:
            raw = np.asarray(init['circles'], dtype=float) if 'circles' in init else np.asarray(init.get('packing_rows', init.get('coordinate_rows')), dtype=float).T
            best_score = exact_score(raw)
            if np.isfinite(best_score):
                best_array = raw.copy()
        except (ValueError, TypeError, KeyError, OverflowError):
            raw = None
    fallback = np.column_stack(((np.arange(N)%5+0.5)*0.2, (np.arange(N)//5+0.5)*0.2, np.full(N, 0.1*(1-1e-12))))
    fs = exact_score(fallback)
    if fs > best_score:
        best_score = fs
        best_array = fallback.copy()
        best = {'packing_rows': fallback.T.tolist()}

    B = np.zeros((84, D))
    B[K, K] = -1
    B[K, 42+K] = 1
    B[21+K, K] = 1
    B[21+K, 42+K] = 1
    B[21+K, -2] = -1
    B[42+K, 21+K] = -1
    B[42+K, 42+K] = 1
    B[63+K, 21+K] = 1
    B[63+K, 42+K] = 1
    B[63+K, -1] = -1

    def consider(v):
        nonlocal best, best_score, best_array
        guard()
        c = np.column_stack((v[:21], v[21:42], np.maximum(v[42:63], 1e-15)))
        d = np.sqrt((c[I, 0]-c[J, 0])**2+(c[I, 1]-c[J, 1])**2)
        f = min(1.0, float(np.min(d/(c[I, 2]+c[J, 2]))))
        if not np.isfinite(f) or f <= 0:
            return
        c[:, 2] *= f*(1-2e-12)
        lo = np.min(c[:, :2]-c[:, 2, None], axis=0)
        hi = np.max(c[:, :2]+c[:, 2, None], axis=0)
        perimeter = float(np.sum(hi-lo))
        if perimeter <= 0:
            return
        c[:, :2] -= (hi+lo)*0.5
        c *= (2/perimeter)*(1-2e-12)
        if float(c[:, 2].sum()) > best_score:
            sc = exact_score(c)
            if sc > best_score:
                best_score = sc
                best_array = c.copy()
                best = {'packing_rows': c.T.tolist()}

    def state(a):
        a = a.copy()
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        total = a[:, 2].sum()
        a[:, :2] -= lo
        a /= total
        return np.r_[a[:, 0], a[:, 1], a[:, 2], (hi-lo)/total]

    def model(T, bounds):
        X, Y, R = T[:21], T[21:42], T[42:63]
        dx, dy, rr = X[I]-X[J], Y[I]-Y[J], R[I]+R[J]
        # Remove constraints identical under reflection, including reversed pairs.
        def canonical_rows(a):
            idx = np.argmax(np.abs(a)>1e-14, axis=1)
            sign = np.sign(a[np.arange(len(a)), idx])
            sign[sign == 0] = 1
            return a*sign[:, None]
        signatures = np.hstack((canonical_rows(dx), canonical_rows(dy), rr))
        _, keep = np.unique(signatures, axis=0, return_index=True)
        dx, dy, rr = dx[keep], dy[keep], rr[keep]
        bb = np.unique(B@T, axis=0)
        eq = R.sum(axis=0)[None, :]
        obj = T[-2]+T[-1]
        return T, bounds, dx, dy, rr, bb, eq, obj

    def lp(m, z, until):
        T, bounds, dx, dy, rr, bb, eq, obj = m
        previous = np.inf
        for iteration in range(int(params['screen_iterations'])):
            guard(until)
            u, v = dx@z, dy@z
            d = np.maximum(np.hypot(u, v), 1e-15)
            A = np.vstack((rr-(u/d)[:, None]*dx-(v/d)[:, None]*dy, bb))
            now = time.monotonic()
            if now >= until-0.005:
                break
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res = linprog(obj, A_ub=A, b_ub=np.zeros(len(A)), A_eq=eq, b_eq=[1.0], bounds=bounds, method='highs', options={'threads': 1, 'time_limit': max(0.001, until-now), 'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9})
            guard()
            if not res.success:
                break
            z = res.x
            consider(T@z)
            value = float(obj@z)
            if abs(previous-value) < 1e-10:
                break
            previous = value
        return z

    def polish(m, z, until):
        T, bounds, dx, dy, rr, bb, eq, obj = m
        last = z.copy()
        def fun(v):
            guard(until)
            return float(obj@v)
        def cons(v):
            guard(until)
            return np.r_[np.hypot(dx@v, dy@v)-rr@v, -bb@v]
        def jac(v):
            guard(until)
            x, y = dx@v, dy@v
            d = np.maximum(np.hypot(x, y), 1e-15)
            return np.vstack(((x/d)[:, None]*dx+(y/d)[:, None]*dy-rr, -bb))
        def callback(v):
            nonlocal last
            guard(until)
            last = v.copy()
            consider(T@v)
        try:
            guard(until)
            result = minimize(fun, z, jac=lambda v: obj, method='SLSQP', bounds=bounds,
                constraints=[{'type': 'ineq', 'fun': cons, 'jac': jac}, {'type': 'eq', 'fun': lambda v: eq@v-1, 'jac': lambda v: eq}],
                callback=callback, options={'maxiter': int(params['polish_iterations']), 'ftol': 2e-13, 'disp': False})
            last = result.x
            consider(T@last)
        except Deadline:
            pass
        return last

    floor = float(params['radius_floor'])
    survivors = []
    try:
        patterns = [(3, 5, 4), (3, 4, 5), (5, 4, 4)]
        for index, (axis, near, far) in enumerate(patterns):
            guard()
            if time.monotonic() >= screen_end-0.012:
                break
            pairs = near+far
            unique = axis+pairs
            dim = unique+pairs+unique+2
            T = np.zeros((D, dim))
            rr0 = unique+pairs
            for k in range(unique):
                guard()
                T[k, k] = 1
                T[42+k, rr0+k] = 1
                T[21+k, -1] = 0.5
            for k in range(pairs):
                guard()
                upper, lower = axis+k, unique+k
                T[lower, upper] = 1
                T[42+lower, rr0+upper] = 1
                T[21+upper, unique+k] = 1
                T[21+lower, unique+k] = -1
                T[21+lower, -1] = 0.5
            T[-2, -2] = T[-1, -1] = 1
            xs = np.r_[2*np.arange(axis)-(axis-1), 2*np.arange(near)-(near-1), 2*np.arange(far)-(far-1)]
            ys = np.r_[np.full(near, np.sqrt(3.0)), np.full(far, 2*np.sqrt(3.0))]
            width = float(xs.max()-xs.min()+2)
            height = float(2*(ys.max()+1))
            z = np.r_[xs-xs.min()+1, ys, np.ones(unique), width, height]/21
            bounds = [(0, 2)]*(unique+pairs)+[(floor, 0.5)]*unique+[(0.01, 2)]*2
            m = model(T, bounds)
            now = time.monotonic()
            until = now+(screen_end-now)/(len(patterns)-index)
            try:
                z = lp(m, z, until)
            except Deadline:
                pass
            survivors.append((float(m[-1]@z), m, z))
        survivors.sort(key=lambda item: item[0])
        for rank, (_, m, z) in enumerate(survivors[:2]):
            guard()
            now = time.monotonic()
            if now >= sym_end-0.01:
                break
            remaining = min(2, len(survivors))-rank
            until = now+(sym_end-now)/remaining
            polish(m, z, until)
        guard()
        if best_array is not None and time.monotonic() < end-0.025:
            bounds = [(0, 2)]*42+[(floor, 0.5)]*21+[(0.01, 2)]*2
            full = model(np.eye(D), bounds)
            polish(full, state(best_array), end-0.006)
    except Deadline:
        pass
    return best
