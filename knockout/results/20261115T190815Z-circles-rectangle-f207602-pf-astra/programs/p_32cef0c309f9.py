import time
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize

PARAMS = {
    'screen_steps': (7, 3, 15),
    'refine_steps': (18, 5, 40),
    'radius_floor': (0.003, 0.0001, 0.015, 'log'),
    'time_fraction': (0.69, 0.5, 0.75)
}

N = 21
II, JJ = np.triu_indices(N, 1)


def exact_score(obj):
    try:
        if 'circles' in obj:
            a = np.asarray(obj['circles'], dtype=np.float64)
        else:
            a = np.asarray(obj.get('packing_rows', obj.get('coordinate_rows')), dtype=np.float64).T
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        xs = [Fraction(float(v)) for v in a[:, 0]]
        ys = [Fraction(float(v)) for v in a[:, 1]]
        rs = [Fraction(float(v)) for v in a[:, 2]]
        w = max(x+r for x, r in zip(xs, rs)) - min(x-r for x, r in zip(xs, rs))
        h = max(y+r for y, r in zip(ys, rs)) - min(y-r for y, r in zip(ys, rs))
        if w+h > 2:
            return -float('inf')
        dx = a[II, 0]-a[JJ, 0]
        dy = a[II, 1]-a[JJ, 1]
        if np.any(np.sqrt(dx**2+dy**2) < a[II, 2]+a[JJ, 2]):
            return -float('inf')
        return float(np.sum(a[:, 2]))
    except Exception:
        return -float('inf')


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + float(params['time_fraction'])*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = exact_score(init) if init is not None else -float('inf')
    floor = float(params['radius_floor'])
    ids = np.arange(N)
    pair_ids = np.arange(len(II))
    A = np.zeros((84+len(II), 65))
    A[ids, ids] = -1
    A[ids, 42+ids] = 1
    A[21+ids, ids] = 1
    A[21+ids, 42+ids] = 1
    A[21+ids, 63] = -1
    A[42+ids, 21+ids] = -1
    A[42+ids, 42+ids] = 1
    A[63+ids, 21+ids] = 1
    A[63+ids, 42+ids] = 1
    A[63+ids, 64] = -1
    A[84+pair_ids, 42+II] = 1
    A[84+pair_ids, 42+JJ] = 1
    rhs = np.zeros(len(A))
    eq = np.zeros((1, 65))
    eq[0, 42:63] = 1
    objective = np.zeros(65)
    objective[63:] = 1
    bounds = [(None, None)]*42 + [(floor, None)]*21 + [(0.0, None)]*2

    def certify(z):
        nonlocal best, best_score
        if z is None or time.monotonic() >= deadline:
            return
        a = np.column_stack((z[:21], z[21:42], z[42:63])).copy()
        if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return
        dx = a[II, 0]-a[JJ, 0]
        dy = a[II, 1]-a[JJ, 1]
        ratios = np.sqrt(dx*dx+dy*dy)/(a[II, 2]+a[JJ, 2])
        shrink = min(1.0, float(np.min(ratios)))*(1.0-2e-12)
        if shrink <= 0:
            return
        a[:, 2] *= shrink
        a[:, :2] -= np.min(a[:, :2]-a[:, 2, None], axis=0)
        xs = [Fraction(float(v)) for v in a[:, 0]]
        ys = [Fraction(float(v)) for v in a[:, 1]]
        rs = [Fraction(float(v)) for v in a[:, 2]]
        wh = max(x+r for x, r in zip(xs, rs))-min(x-r for x, r in zip(xs, rs))
        wh += max(y+r for y, r in zip(ys, rs))-min(y-r for y, r in zip(ys, rs))
        a *= (2.0/float(wh))*(1.0-2e-14)
        obj = {'packing_rows': a.T.tolist()}
        score = exact_score(obj)
        if score > best_score:
            best_score, best = score, obj

    def lp_steps(z, count):
        old_value = float('inf')
        for k in range(count):
            if time.monotonic() >= deadline-0.015:
                break
            dx = z[II]-z[JJ]
            dy = z[21+II]-z[21+JJ]
            d = np.maximum(np.hypot(dx, dy), 1e-15)
            ux, uy = dx/d, dy/d
            A[84+pair_ids, II] = -ux
            A[84+pair_ids, JJ] = ux
            A[84+pair_ids, 21+II] = -uy
            A[84+pair_ids, 21+JJ] = uy
            remaining = max(0.002, deadline-time.monotonic()-0.01)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result = linprog(objective, A_ub=A, b_ub=rhs, A_eq=eq, b_eq=[1.0], bounds=bounds, method='highs', options={'threads': 1, 'time_limit': remaining, 'dual_feasibility_tolerance': 1e-8, 'primal_feasibility_tolerance': 1e-8})
            if not result.success:
                break
            z = result.x
            value = float(z[63]+z[64])
            if old_value-value < 2e-10:
                break
            old_value = value
        return z

    families = [
        [4, 3, 4, 3, 4, 3],
        [4, 4, 5, 4, 4],
        [4, 5, 3, 5, 4],
        [5, 4, 3, 4, 5],
        [3, 4, 4, 4, 3, 3],
        [4, 5, 4, 5, 3],
        [5, 4, 4, 4, 4]
    ]
    pool = []
    for fi, counts in enumerate(families):
        if time.monotonic() >= start+0.38*budget_s or time.monotonic() >= deadline-0.05:
            break
        xparts, yparts = [], []
        for row, count in enumerate(counts):
            if time.monotonic() >= deadline-0.025:
                break
            xx = 2.0*np.arange(count)-(count-1)
            if row and count == counts[row-1]:
                xx = xx + (0.45 if row % 2 else -0.45)
            xparts.append(xx)
            yparts.append(np.full(count, row*np.sqrt(3.0)))
        if len(xparts) != len(counts):
            break
        xx = np.concatenate(xparts)
        yy = np.concatenate(yparts)
        xx += rng.normal(0, 0.012, N)
        yy += rng.normal(0, 0.012, N)
        xx = (xx-xx.min()+1)/21
        yy = (yy-yy.min()+1)/21
        z = np.r_[xx, yy, np.full(N, 1/21), xx.max()+1/21, yy.max()+1/21]
        z = lp_steps(z, int(params['screen_steps']))
        certify(z)
        pool.append(z)
    if not pool:
        xx, yy = np.meshgrid(np.arange(7), np.arange(3))
        z = np.r_[(2*xx.ravel()+1)/21, (2*yy.ravel()+1)/21, np.full(21, 1/21), 14/21, 6/21]
        certify(z)
        pool.append(z)
    pool.sort(key=lambda z: z[63]+z[64])
    finalists = []
    for z in pool[:3]:
        if time.monotonic() >= deadline-0.08:
            break
        z = lp_steps(z, int(params['refine_steps']))
        certify(z)
        finalists.append(z)
    if not finalists:
        finalists = pool[:1]
    finalists.sort(key=lambda z: z[63]+z[64])

    # Exact Euclidean constraints, with analytic derivatives.
    wall_A = A[:84].copy()
    def constraints(z):
        dx = z[II]-z[JJ]
        dy = z[21+II]-z[21+JJ]
        return np.r_[-wall_A@z, np.hypot(dx, dy)-z[42+II]-z[42+JJ]]

    def jacobian(z):
        dx = z[II]-z[JJ]
        dy = z[21+II]-z[21+JJ]
        d = np.maximum(np.hypot(dx, dy), 1e-15)
        out = np.zeros((294, 65))
        out[:84] = -wall_A
        out[84+pair_ids, II] = dx/d
        out[84+pair_ids, JJ] = -dx/d
        out[84+pair_ids, 21+II] = dy/d
        out[84+pair_ids, 21+JJ] = -dy/d
        out[84+pair_ids, 42+II] = -1
        out[84+pair_ids, 42+JJ] = -1
        return out

    class Timeout(Exception):
        pass

    latest = [finalists[0]]
    def callback(z):
        if time.monotonic() >= deadline-0.025:
            raise Timeout()
        latest[0] = z.copy()

    for z in finalists[:2]:
        if time.monotonic() >= deadline-0.1:
            break
        latest[0] = z
        try:
            result = minimize(lambda v: float(v[63]+v[64]), z, jac=lambda v: objective,
                method='SLSQP', bounds=bounds,
                constraints=[{'type': 'eq', 'fun': lambda v: np.sum(v[42:63])-1.0, 'jac': lambda v: eq[0]},
                             {'type': 'ineq', 'fun': constraints, 'jac': jacobian}],
                callback=callback, options={'maxiter': 75, 'ftol': 2e-12, 'disp': False})
            certify(result.x)
        except Timeout:
            pass
        certify(latest[0])
    if time.monotonic() < deadline-0.04:
        z = lp_steps(latest[0], 30)
        certify(z)
    return best
