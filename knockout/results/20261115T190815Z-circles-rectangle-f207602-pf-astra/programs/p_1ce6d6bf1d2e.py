import time
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize

PARAMS = {
    'screen_fraction': (0.25, 0.12, 0.40),
    'screen_steps': (10, 4, 24),
    'finalists': (4, 2, 6),
    'polish_steps': (100, 30, 180),
    'radius_floor': (0.00002, 0.000001, 0.001, 'log')
}
N = 21
II, JJ = np.triu_indices(N, 1)

class Deadline(Exception):
    pass


def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    start = time.monotonic()
    for i, j in zip(II, JJ):
        if time.monotonic() - start > 0.04:
            return -np.inf
        d = np.sqrt((a[i, 0]-a[j, 0])**2 + (a[i, 1]-a[j, 1])**2)
        if d < a[i, 2]+a[j, 2]:
            return -np.inf
    x = list(map(Fraction, map(float, a[:, 0])))
    y = list(map(Fraction, map(float, a[:, 1])))
    r = list(map(Fraction, map(float, a[:, 2])))
    w = max(u+v for u, v in zip(x, r))-min(u-v for u, v in zip(x, r))
    h = max(u+v for u, v in zip(y, r))-min(u-v for u, v in zip(y, r))
    return float(np.sum(a[:, 2])) if w+h <= 2 else -np.inf


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70*budget_s
    screen_end = start + float(params['screen_fraction'])*budget_s
    best = None
    score = -np.inf

    def guard(until=None):
        if time.monotonic() >= (end if until is None else min(end, until)):
            raise Deadline()

    if init is not None:
        try:
            if 'circles' in init:
                a = np.asarray(init['circles'], dtype=float)
            else:
                a = np.asarray(init.get('packing_rows', init.get('coordinate_rows')), dtype=float).T
            s = exact_score(a)
            if s > score:
                best, score = init, s
        except (TypeError, ValueError, OverflowError):
            pass
    fallback = np.column_stack(((np.arange(N)%5+0.5)*0.2,
                                (np.arange(N)//5+0.5)*0.2,
                                np.full(N, 0.1*(1-1e-12))))
    s = exact_score(fallback)
    if s > score:
        best, score = {'packing_rows': fallback.T.tolist()}, s

    def consider(model, z):
        nonlocal best, score
        guard()
        X, Y, R = model['X'], model['Y'], model['R']
        a = np.column_stack((X@z, Y@z, np.maximum(R@z, 1e-14)))
        d = np.hypot(a[II, 0]-a[JJ, 0], a[II, 1]-a[JJ, 1])
        factor = min(1.0, float(np.min(d/(a[II, 2]+a[JJ, 2]))))
        if not np.isfinite(factor) or factor <= 0:
            return
        a[:, 2] *= factor*(1-2e-12)
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        perimeter = float(np.sum(hi-lo))
        if perimeter <= 0:
            return
        a[:, :2] -= (hi+lo)/2
        a *= (2/perimeter)*(1-2e-12)
        estimated = float(np.sum(a[:, 2]))
        if estimated > score:
            s = exact_score(a)
            if s > score:
                score, best = s, {'packing_rows': a.T.tolist()}

    def build(rows):
        guard()
        orbits = []
        for row, count in enumerate(rows):
            guard()
            if count%2:
                orbits.append((0.0, np.sqrt(3.0)*(row-(len(rows)-1)/2), False))
            for k in range(count//2):
                guard()
                orbits.append((float(2*k+1+(count%2)), np.sqrt(3.0)*(row-(len(rows)-1)/2), True))
        pairs = sum(int(o[2]) for o in orbits)
        q = len(orbits)
        dim = pairs+2*q+2
        X = np.zeros((N, dim))
        Y = np.zeros_like(X)
        R = np.zeros_like(X)
        z = np.zeros(dim)
        mirror = np.empty(N, dtype=int)
        i = 0
        xp = 0
        for k, (x, y, paired) in enumerate(orbits):
            guard()
            z[pairs+k] = y/21
            z[pairs+q+k] = 1/21
            Y[i, pairs+k] = 1
            R[i, pairs+q+k] = 1
            if paired:
                X[i, xp] = 1
                X[i+1, xp] = -1
                Y[i+1, pairs+k] = 1
                R[i+1, pairs+q+k] = 1
                z[xp] = x/21
                mirror[i], mirror[i+1] = i+1, i
                i += 2
                xp += 1
            else:
                mirror[i] = i
                i += 1
        z[-2] = 2*np.max(np.abs(X@z)+R@z)
        z[-1] = 2*np.max(np.abs(Y@z)+R@z)
        selected = []
        for i, j in zip(II, JJ):
            guard()
            reflected = tuple(sorted((int(mirror[i]), int(mirror[j]))))
            if (i, j) <= reflected:
                selected.append((i, j))
        ij = np.asarray(selected)
        dx = X[ij[:, 0]]-X[ij[:, 1]]
        dy = Y[ij[:, 0]]-Y[ij[:, 1]]
        rr = R[ij[:, 0]]+R[ij[:, 1]]
        bx1, bx2 = X+R, -X+R
        by1, by2 = Y+R, -Y+R
        bx1[:, -2] -= 0.5
        bx2[:, -2] -= 0.5
        by1[:, -1] -= 0.5
        by2[:, -1] -= 0.5
        B = np.unique(np.vstack((bx1, bx2, by1, by2)), axis=0)
        objective = np.zeros(dim)
        objective[-2:] = 1
        equality = np.sum(R, axis=0, keepdims=True)
        floor = float(params['radius_floor'])
        bounds = [(0, 2)]*pairs + [(-2, 2)]*q + [(floor, 0.5)]*q + [(0.01, 2)]*2
        return {'X': X, 'Y': Y, 'R': R, 'dx': dx, 'dy': dy,
                'rr': rr, 'B': B, 'obj': objective, 'eq': equality,
                'bounds': bounds, 'z': z}

    def screen(model, until):
        z = model['z']
        dx, dy, rr, B = model['dx'], model['dy'], model['rr'], model['B']
        for iteration in range(int(params['screen_steps'])):
            guard()
            if time.monotonic() >= until-0.004:
                break
            x, y = dx@z, dy@z
            d = np.maximum(np.hypot(x, y), 1e-15)
            A = np.vstack((rr-(x/d)[:, None]*dx-(y/d)[:, None]*dy, B))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res = linprog(model['obj'], A_ub=A, b_ub=np.zeros(len(A)),
                              A_eq=model['eq'], b_eq=[1.0], bounds=model['bounds'],
                              method='highs', options={'threads': 1,
                              'time_limit': max(0.001, until-time.monotonic()),
                              'primal_feasibility_tolerance': 1e-9,
                              'dual_feasibility_tolerance': 1e-9})
            guard()
            if not res.success:
                break
            change = abs(float(np.sum(z[-2:])-res.fun))
            z = res.x
            model['z'] = z
            consider(model, z)
            if change < 1e-10:
                break
        return model

    def polish(model, until):
        dx, dy, rr, B = model['dx'], model['dy'], model['rr'], model['B']
        def fun(z):
            guard(until)
            return float(z[-2]+z[-1])
        def con(z):
            guard(until)
            return np.r_[np.hypot(dx@z, dy@z)-rr@z, -B@z]
        def jac(z):
            guard(until)
            x, y = dx@z, dy@z
            d = np.maximum(np.hypot(x, y), 1e-15)
            return np.vstack(((x/d)[:, None]*dx+(y/d)[:, None]*dy-rr, -B))
        def callback(z):
            guard(until)
            model['z'] = z.copy()
            consider(model, z)
        try:
            res = minimize(fun, model['z'], jac=lambda z: model['obj'],
                method='SLSQP', bounds=model['bounds'],
                constraints=[{'type': 'ineq', 'fun': con, 'jac': jac},
                    {'type': 'eq', 'fun': lambda z: model['eq']@z-1,
                     'jac': lambda z: model['eq']}],
                callback=callback,
                options={'maxiter': int(params['polish_steps']), 'ftol': 2e-13, 'disp': False})
            model['z'] = res.x
            consider(model, res.x)
        except Deadline:
            pass

    try:
        patterns = [(4, 5, 3, 5, 4), (5, 4, 3, 4, 5),
                    (4, 4, 5, 4, 4), (3, 5, 5, 5, 3),
                    (4, 5, 4, 5, 3), (5, 6, 5, 5)]
        models = []
        for index, rows in enumerate(patterns):
            guard()
            now = time.monotonic()
            if now >= screen_end-0.008:
                break
            model = build(rows)
            until = now+(screen_end-now)/(len(patterns)-index)
            models.append(screen(model, until))
        models.sort(key=lambda m: float(np.sum(m['z'][-2:])))
        finalists = models[:int(params['finalists'])]
        for k, model in enumerate(finalists):
            guard()
            now = time.monotonic()
            left = len(finalists)-k
            share = 1.0 if left == 1 else min(0.52, 1.5/left)
            until = min(end-0.004, now+share*(end-now))
            if until-now > 0.015:
                polish(model, until)
    except Deadline:
        pass
    return best
