import time
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize

PARAMS = {
    'screen_iterations': (14, 6, 30),
    'refine_iterations': (28, 10, 60),
    'screen_fraction': (0.27, 0.15, 0.35),
    'refine_fraction': (0.46, 0.36, 0.55),
    'polish_iterations': (100, 30, 180),
    'jitter': (0.025, 0.001, 0.1, 'log'),
    'radius_floor': (0.00001, 0.000001, 0.001, 'log')
}
N = 21
AI, AJ = np.triu_indices(N, 1)

class Deadline(Exception):
    pass

class StageEnd(Exception):
    pass


def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    r = a[:, 2]
    d = np.sqrt((a[AI, 0]-a[AJ, 0])**2 + (a[AI, 1]-a[AJ, 1])**2)
    if np.any(d < r[AI]+r[AJ]):
        return -np.inf
    x = list(map(Fraction, map(float, a[:, 0])))
    y = list(map(Fraction, map(float, a[:, 1])))
    q = list(map(Fraction, map(float, r)))
    w = max(u+v for u, v in zip(x, q))-min(u-v for u, v in zip(x, q))
    h = max(u+v for u, v in zip(y, q))-min(u-v for u, v in zip(y, q))
    if w+h > 2:
        return -np.inf
    return float(np.sum(r))


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70*budget_s
    screen_end = start + float(params['screen_fraction'])*budget_s
    refine_end = start + float(params['refine_fraction'])*budget_s
    rng = np.random.default_rng(seed)
    best = None
    best_score = -np.inf

    def guard():
        if time.monotonic() >= end:
            raise Deadline()

    def save(a):
        nonlocal best, best_score
        if float(np.sum(a[:, 2])) <= best_score:
            return
        score = exact_score(a)
        if score > best_score:
            best_score = score
            best = {'packing_rows': a.T.tolist()}

    if init is not None:
        try:
            if 'circles' in init:
                a = np.asarray(init['circles'], dtype=float)
            else:
                a = np.asarray(init.get('packing_rows', init.get('coordinate_rows')), dtype=float).T
            score = exact_score(a)
            if score > best_score:
                best_score, best = score, init
        except (TypeError, ValueError, KeyError, OverflowError):
            pass
    fallback = np.column_stack(((np.arange(21)%5+0.5)*0.2,
                                (np.arange(21)//5+0.5)*0.2,
                                np.full(21, 0.1*(1-1e-12))))
    save(fallback)

    def consider(model, z):
        guard()
        full = model['T'] @ z
        c = np.column_stack((full[:21], full[21:42], full[42:63]))
        if not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return
        dist = np.sqrt((c[AI, 0]-c[AJ, 0])**2+(c[AI, 1]-c[AJ, 1])**2)
        shrink = min(1.0, float(np.min(dist/(c[AI, 2]+c[AJ, 2]))))
        if shrink <= 0:
            return
        c[:, 2] *= shrink*(1-2e-12)
        lo = np.min(c[:, :2]-c[:, 2, None], axis=0)
        hi = np.max(c[:, :2]+c[:, 2, None], axis=0)
        perimeter = float(np.sum(hi-lo))
        if perimeter <= 0:
            return
        c[:, :2] -= (lo+hi)*0.5
        c *= (2.0/perimeter)*(1-2e-12)
        save(c)

    def make_model(ns, inner, outer):
        guard()
        npair = inner+outer
        nr = ns+npair
        d = 2*nr+npair+2
        ry = nr
        rr = nr+npair
        T = np.zeros((65, d))
        reps = np.r_[np.arange(nr), np.arange(ns, nr)]
        ids = np.arange(21)
        T[ids, reps] = 1
        T[21+ids, -1] = 1
        T[21+np.arange(ns, nr), ry+np.arange(npair)] = 1
        T[21+np.arange(nr, 21), ry+np.arange(npair)] = -1
        T[42+ids, rr+reps] = 1
        T[63, -2] = 1
        T[64, -1] = 2
        reflect = np.r_[np.arange(ns), np.arange(nr, 21), np.arange(ns, nr)]
        mirrored_i = np.minimum(reflect[AI], reflect[AJ])
        mirrored_j = np.maximum(reflect[AI], reflect[AJ])
        keep = (AI < mirrored_i) | ((AI == mirrored_i) & (AJ <= mirrored_j))
        ii, jj = AI[keep], AJ[keep]
        dx = T[ii]-T[jj]
        dy = T[21+ii]-T[21+jj]
        radii = T[42+ii]+T[42+jj]
        walls = np.vstack((T[42:63]-T[:21],
                           T[42:63]+T[:21]-T[63],
                           T[42:63]-T[21:42],
                           T[42:63]+T[21:42]-T[64]))
        walls = np.unique(walls, axis=0)
        objective = T[63]+T[64]
        eq = np.sum(T[42:63], axis=0, keepdims=True)
        bounds = [(0.0, 2.0)]*nr + [(0.0, 1.0)]*npair
        bounds += [(float(params['radius_floor']), 0.5)]*nr
        bounds += [(0.01, 2.0), (0.005, 1.0)]
        xs = np.r_[2*np.arange(ns)-(ns-1),
                   2*np.arange(inner)-(inner-1),
                   2*np.arange(outer)-(outer-1)].astype(float)
        yp = np.r_[np.full(inner, np.sqrt(3.0)), np.full(outer, 2*np.sqrt(3.0))]
        xs += rng.normal(0.0, float(params['jitter']), nr)
        yp += rng.normal(0.0, float(params['jitter'])*0.3, npair)
        w = float(np.ptp(xs)+2)
        xs -= float(np.min(xs)-1)
        z = np.r_[xs, yp, np.ones(nr), w, np.max(yp)+1]/21.0
        model = {'T': T, 'dx': dx, 'dy': dy, 'rr': radii,
                 'B': walls, 'obj': objective, 'eq': eq,
                 'bounds': bounds, 'z': z}
        model['A'] = np.vstack((radii.copy(), walls))
        model['rhs'] = np.zeros(len(radii)+len(walls))
        return model

    def run_lp(model, count, until):
        z = model['z']
        old_obj = float(model['obj'] @ z)
        stale = 0
        for iteration in range(count):
            guard()
            now = time.monotonic()
            if now >= until-0.003:
                break
            dx = model['dx'] @ z
            dy = model['dy'] @ z
            distance = np.maximum(np.hypot(dx, dy), 1e-15)
            model['A'][:len(dx)] = (model['rr']
                - (dx/distance)[:, None]*model['dx']
                - (dy/distance)[:, None]*model['dy'])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result = linprog(model['obj'], A_ub=model['A'], b_ub=model['rhs'],
                    A_eq=model['eq'], b_eq=[1.0], bounds=model['bounds'],
                    method='highs', options={'threads': 1,
                        'time_limit': max(0.001, until-now),
                        'primal_feasibility_tolerance': 1e-9,
                        'dual_feasibility_tolerance': 1e-9})
            guard()
            if not result.success:
                break
            z = result.x
            model['z'] = z
            consider(model, z)
            obj = float(model['obj'] @ z)
            stale = stale+1 if abs(old_obj-obj) < 1e-11 else 0
            old_obj = obj
            if stale >= 3:
                break
        return model

    def polish(model, until):
        def check():
            guard()
            if time.monotonic() >= until:
                raise StageEnd()

        def fun(z):
            check()
            return float(model['obj'] @ z)

        def con(z):
            check()
            dx = model['dx'] @ z
            dy = model['dy'] @ z
            return np.r_[np.hypot(dx, dy)-model['rr'] @ z, -model['B'] @ z]

        def jac(z):
            check()
            dx = model['dx'] @ z
            dy = model['dy'] @ z
            distance = np.maximum(np.hypot(dx, dy), 1e-15)
            return np.vstack(((dx/distance)[:, None]*model['dx']
                              +(dy/distance)[:, None]*model['dy']-model['rr'],
                              -model['B']))

        def callback(z):
            check()
            model['z'] = z.copy()
            consider(model, z)

        def eqfun(z):
            check()
            return model['eq'] @ z-1.0

        try:
            result = minimize(fun, model['z'], method='SLSQP',
                jac=lambda z: model['obj'], bounds=model['bounds'],
                constraints=[{'type': 'ineq', 'fun': con, 'jac': jac},
                             {'type': 'eq', 'fun': eqfun, 'jac': lambda z: model['eq']}],
                callback=callback,
                options={'maxiter': int(params['polish_iterations']), 'ftol': 8e-14, 'disp': False})
            model['z'] = result.x
            consider(model, result.x)
        except StageEnd:
            pass

    try:
        patterns = [(3, 5, 4), (3, 4, 5), (5, 4, 4), (1, 5, 5), (7, 3, 4)]
        models = []
        for index, pattern in enumerate(patterns):
            guard()
            now = time.monotonic()
            if now >= screen_end-0.008:
                break
            model = make_model(*pattern)
            until = now+(screen_end-now)/(len(patterns)-index)
            run_lp(model, int(params['screen_iterations']), until)
            models.append(model)
        if not models:
            return best
        models.sort(key=lambda m: float(m['obj'] @ m['z']))
        models = models[:3]
        for index, model in enumerate(models):
            guard()
            now = time.monotonic()
            if now >= refine_end-0.006:
                break
            until = now+(refine_end-now)/(len(models)-index)
            run_lp(model, int(params['refine_iterations']), until)
        models.sort(key=lambda m: float(m['obj'] @ m['z']))
        first_end = time.monotonic()+0.64*max(0.0, end-time.monotonic())
        if time.monotonic() < first_end-0.012:
            polish(models[0], first_end)
        if len(models) > 1 and time.monotonic() < end-0.018:
            polish(models[1], end-0.004)
        if time.monotonic() < end-0.018:
            polish(models[0], end-0.003)
    except Deadline:
        pass
    return best
