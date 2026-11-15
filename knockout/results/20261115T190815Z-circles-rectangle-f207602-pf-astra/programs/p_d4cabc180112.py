import time
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize

PARAMS = {
    'screen_steps': (8, 3, 20),
    'screen_fraction': (0.25, 0.12, 0.4),
    'polish_iterations': (100, 30, 180),
    'radius_floor': (0.00002, 0.000001, 0.001, 'log'),
    'jitter': (0.008, 0.0, 0.06)
}
N = 21
II, JJ = np.triu_indices(N, 1)

class Deadline(Exception):
    pass

class SliceEnd(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -float('inf')
    d = np.sqrt((a[II, 0]-a[JJ, 0])**2 + (a[II, 1]-a[JJ, 1])**2)
    if np.any(d < a[II, 2]+a[JJ, 2]):
        return -float('inf')
    x = list(map(Fraction, map(float, a[:, 0])))
    y = list(map(Fraction, map(float, a[:, 1])))
    r = list(map(Fraction, map(float, a[:, 2])))
    w = max(u+v for u, v in zip(x, r))-min(u-v for u, v in zip(x, r))
    h = max(u+v for u, v in zip(y, r))-min(u-v for u, v in zip(y, r))
    return float(np.sum(a[:, 2])) if w+h <= 2 else -float('inf')

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    best = None
    best_score = -float('inf')

    def guard():
        if time.monotonic() >= end:
            raise Deadline()

    if init is not None:
        try:
            raw = np.asarray(init['circles'], dtype=float) if 'circles' in init else np.asarray(init.get('packing_rows', init.get('coordinate_rows')), dtype=float).T
            score = exact_score(raw)
            if score > best_score:
                best, best_score = init, score
        except (ValueError, TypeError, KeyError, OverflowError):
            pass
    fallback = np.column_stack(((np.arange(N)%5+0.5)*0.2, (np.arange(N)//5+0.5)*0.2, np.full(N, 0.1*(1-1e-12))))
    fs = exact_score(fallback)
    if fs > best_score:
        best, best_score = {'packing_rows': fallback.T.tolist()}, fs

    def certify(model, z):
        nonlocal best, best_score
        guard()
        X, Y, R = model['X'], model['Y'], model['R']
        a = np.column_stack((X@z, Y@z, R@z))
        if np.any(a[:, 2] <= 0) or not np.isfinite(a).all():
            return -float('inf')
        d = np.sqrt((a[II, 0]-a[JJ, 0])**2+(a[II, 1]-a[JJ, 1])**2)
        shrink = min(1.0, float(np.min(d/(a[II, 2]+a[JJ, 2]))))
        if shrink <= 0:
            return -float('inf')
        a[:, 2] *= shrink*(1-2e-12)
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        a[:, :2] -= (lo+hi)*0.5
        a *= (2/float(np.sum(hi-lo)))*(1-2e-12)
        score = float(np.sum(a[:, 2]))
        if score > best_score:
            score = exact_score(a)
            if score > best_score:
                best, best_score = {'packing_rows': a.T.tolist()}, score
        return score

    def model_for(rows):
        groups = []
        pair_count = sum(c//2 for c in rows)
        for row, count in enumerate(rows):
            guard()
            if count % 2:
                groups.append((row, 0.0, False))
            for k in range(count//2):
                guard()
                groups.append((row, float(2*k+1+(count%2)), True))
        g = len(groups)
        dim = pair_count+2*g+2
        X = np.zeros((N, dim)); Y = X.copy(); R = X.copy()
        z = np.zeros(dim)
        ci = 0
        pi = 0
        for gi, (row, dx, paired) in enumerate(groups):
            guard()
            z[pair_count+gi] = (1+np.sqrt(3.0)*row+rng.normal(0, params['jitter']))/N
            z[pair_count+g+gi] = 1/N
            if paired:
                z[pi] = (dx+rng.normal(0, params['jitter']))/N
                X[ci, pi] = -1
                X[ci+1, pi] = 1
                Y[ci:ci+2, pair_count+gi] = 1
                R[ci:ci+2, pair_count+g+gi] = 1
                ci += 2
                pi += 1
            else:
                Y[ci, pair_count+gi] = 1
                R[ci, pair_count+g+gi] = 1
                ci += 1
        z[-2] = 2*max(rows)/N
        z[-1] = (2+np.sqrt(3.0)*(len(rows)-1))/N
        dx = X[II]-X[JJ]
        dy = Y[II]-Y[JJ]
        rr = R[II]+R[JJ]
        seen = set(); keep = []
        for k in range(len(II)):
            guard()
            a = dx[k].copy(); b = dy[k].copy()
            nz = np.flatnonzero(a)
            if len(nz) and a[nz[0]] < 0:
                a = -a
            nz = np.flatnonzero(b)
            if len(nz) and b[nz[0]] < 0:
                b = -b
            key = tuple(np.r_[a, b, rr[k]].astype(int))
            if key not in seen:
                seen.add(key); keep.append(k)
        dx, dy, rr = dx[keep], dy[keep], rr[keep]
        B = np.vstack((R-X, R+X, R-Y, R+Y))
        B[:2*N, -2] -= 0.5
        B[3*N:, -1] -= 1
        B = np.unique(B, axis=0)
        objective = np.zeros(dim); objective[-2:] = 1
        eq = R.sum(axis=0)[None, :]
        bounds = [(1e-7, None)]*pair_count + [(0.0, None)]*g + [(float(params['radius_floor']), 0.5)]*g + [(0.001, 2.0)]*2
        return dict(X=X, Y=Y, R=R, dx=dx, dy=dy, rr=rr, B=B, objective=objective, eq=eq, bounds=bounds, z=z, score=-float('inf'))

    def screen(m, until):
        z = m['z']
        for iteration in range(int(params['screen_steps'])):
            guard()
            now = time.monotonic()
            if now >= until-0.004:
                break
            dx, dy = m['dx']@z, m['dy']@z
            dist = np.maximum(np.hypot(dx, dy), 1e-15)
            A = np.vstack((m['rr']-(dx/dist)[:, None]*m['dx']-(dy/dist)[:, None]*m['dy'], m['B']))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res = linprog(m['objective'], A_ub=A, b_ub=np.zeros(len(A)), A_eq=m['eq'], b_eq=[1.0], bounds=m['bounds'], method='highs', options={'threads': 1, 'time_limit': max(0.001, until-now), 'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9})
            guard()
            if not res.success:
                break
            z = res.x
            m['score'] = certify(m, z)
        m['z'] = z

    def polish(m, until):
        def check():
            guard()
            if time.monotonic() >= until:
                raise SliceEnd()
        dxm, dym, rr, B = m['dx'], m['dy'], m['rr'], m['B']
        def fun(z):
            check()
            return float(z[-2]+z[-1])
        def cons(z):
            check()
            return np.r_[np.hypot(dxm@z, dym@z)-rr@z, -B@z]
        def jac(z):
            check()
            dx, dy = dxm@z, dym@z
            d = np.maximum(np.hypot(dx, dy), 1e-15)
            return np.vstack(((dx/d)[:, None]*dxm+(dy/d)[:, None]*dym-rr, -B))
        def callback(z):
            check()
            score = certify(m, z)
            if score >= m['score']:
                m['score'] = score
                m['z'] = z.copy()
        def eqfun(z):
            check()
            return m['eq']@z-1
        try:
            result = minimize(fun, m['z'], jac=lambda z: m['objective'], method='SLSQP', bounds=m['bounds'], constraints=[{'type': 'ineq', 'fun': cons, 'jac': jac}, {'type': 'eq', 'fun': eqfun, 'jac': lambda z: m['eq']}], callback=callback, options={'maxiter': int(params['polish_iterations']), 'ftol': 2e-13, 'disp': False})
            callback(result.x)
        except SliceEnd:
            pass

    try:
        patterns = [(4,5,3,5,4), (5,4,3,4,5), (4,5,4,5,3), (4,4,5,4,4)]
        models = []
        for rows in patterns:
            guard()
            models.append(model_for(rows))
        screen_end = start+float(params['screen_fraction'])*budget_s
        for k, m in enumerate(models):
            guard()
            now = time.monotonic()
            if now >= screen_end:
                break
            screen(m, now+(screen_end-now)/(len(models)-k))
        models.sort(key=lambda m: -m['score'])
        for k, m in enumerate(models):
            guard()
            now = time.monotonic()
            remaining = end-now
            if remaining < 0.025:
                break
            share = remaining/(len(models)-k+0.45)
            polish(m, min(end-0.006, now+share))
        models.sort(key=lambda m: -m['score'])
        if time.monotonic() < end-0.015:
            polish(models[0], end-0.004)
    except Deadline:
        pass
    return best
