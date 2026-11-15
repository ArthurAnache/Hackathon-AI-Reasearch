import time
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize

PARAMS = {
    'screen_rounds': (2, 1, 5),
    'minimum_radius': (0.003, 0.0001, 0.02, 'log'),
    'lp_refinements': (12, 3, 30),
    'polish_iterations': (45, 10, 100),
    'screen_fraction': (0.38, 0.2, 0.55),
}


def exact_score(data):
    try:
        if 'circles' in data:
            a = np.asarray(data['circles'], dtype=np.float64)
        else:
            a = np.asarray(data.get('packing_rows', data.get('coordinate_rows')), dtype=np.float64).T
        if a.shape != (21, 3) or not np.isfinite(a).all() or not (a[:, 2] > 0).all():
            return -float('inf')
        xs = [Fraction(float(v)) for v in a[:, 0]]
        ys = [Fraction(float(v)) for v in a[:, 1]]
        rs = [Fraction(float(v)) for v in a[:, 2]]
        w = max(x+r for x,r in zip(xs,rs))-min(x-r for x,r in zip(xs,rs))
        h = max(y+r for y,r in zip(ys,rs))-min(y-r for y,r in zip(ys,rs))
        if w+h > 2:
            return -float('inf')
        for i in range(21):
            for j in range(i):
                d = np.sqrt((a[i,0]-a[j,0])**2+(a[i,1]-a[j,1])**2)
                if d < a[i,2]+a[j,2]:
                    return -float('inf')
        return float(np.sum(a[:,2]))
    except (ValueError, TypeError, KeyError, OverflowError):
        return -float('inf')


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    screen_end = start + float(params['screen_fraction']) * 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = exact_score(init) if init is not None else -float('inf')
    ii, jj = np.triu_indices(21, 1)
    nr = len(ii)
    ridx = np.arange(nr)
    ids = np.arange(21)

    def offer(z):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        a = np.column_stack((z[:21], z[21:42], z[42:63])).copy()
        if not np.isfinite(a).all() or np.min(a[:,2]) <= 0:
            return
        dx = a[ii,0]-a[jj,0]
        dy = a[ii,1]-a[jj,1]
        dist = np.sqrt(dx*dx+dy*dy)
        ratio = float(np.min(dist/(a[ii,2]+a[jj,2])))
        if ratio <= 0:
            return
        a[:,2] *= min(1.0, ratio) * (1.0-3e-13)
        a[:,:2] -= np.min(a[:,:2]-a[:,2,None], axis=0)
        extent = np.max(a[:,:2]+a[:,2,None], axis=0)-np.min(a[:,:2]-a[:,2,None], axis=0)
        a *= (2.0/float(np.sum(extent))) * (1.0-3e-14)
        estimate = float(np.sum(a[:,2]))
        if estimate <= best_score:
            return
        candidate = {'packing_rows': a.T.tolist()}
        value = exact_score(candidate)
        if value > best_score:
            best_score, best = value, candidate

    def from_points(p):
        p = np.asarray(p, dtype=float)
        p = p - np.min(p, axis=0)
        p *= 1.0/42.0
        r = np.full(21, 1.0/21.0)
        z = np.empty(65)
        z[:21] = p[:,0]+r
        z[21:42] = p[:,1]+r
        z[42:63] = r
        z[63] = np.max(z[:21]+r)
        z[64] = np.max(z[21:42]+r)
        return z

    fallback = from_points([(2*(k%5),2*(k//5)) for k in range(21)])
    offer(fallback)
    if time.monotonic() >= deadline:
        return best

    objective = np.zeros(65)
    objective[63:] = 1.0
    eq = np.zeros((1,65))
    eq[0,42:63] = 1.0
    wall = np.zeros((84,65))
    wall[ids,ids] = -1
    wall[ids,42+ids] = 1
    wall[21+ids,ids] = 1
    wall[21+ids,42+ids] = 1
    wall[21+ids,63] = -1
    wall[42+ids,21+ids] = -1
    wall[42+ids,42+ids] = 1
    wall[63+ids,21+ids] = 1
    wall[63+ids,42+ids] = 1
    wall[63+ids,64] = -1
    matrix = np.zeros((nr+84,65))
    matrix[nr:] = wall
    matrix[ridx,42+ii] = 1
    matrix[ridx,42+jj] = 1
    bounds = [(0,None)]*42 + [(float(params['minimum_radius']),None)]*21 + [(0,None)]*2
    zeros = np.zeros(nr+84)

    def lp_step(z, limit):
        if time.monotonic() >= limit-0.006:
            return None
        dx = z[ii]-z[jj]
        dy = z[21+ii]-z[21+jj]
        d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-15)
        ux, uy = dx/d, dy/d
        matrix[ridx,ii] = -ux
        matrix[ridx,jj] = ux
        matrix[ridx,21+ii] = -uy
        matrix[ridx,21+jj] = uy
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = linprog(objective, A_ub=matrix, b_ub=zeros,
                             A_eq=eq, b_eq=[1.0], bounds=bounds,
                             method='highs', options={'threads':1,
                             'time_limit':max(0.001,limit-time.monotonic()-0.004),
                             'primal_feasibility_tolerance':1e-9,
                             'dual_feasibility_tolerance':1e-9})
        if result.success:
            return result.x
        return None

    families = [
        [4,5,3,5,4], [4,4,5,4,4], [3,5,5,5,3],
        [5,4,3,4,5], [4,5,4,5,3], [3,4,3,4,3,4],
        [2,3,4,3,4,3,2], [5,5,6,5], [5,4,4,4,4],
    ]
    seeds = []
    if init is not None and best_score > 2.25:
        if 'circles' in init:
            a = np.asarray(init['circles'],dtype=float)
        else:
            a = np.asarray(init.get('packing_rows',init.get('coordinate_rows')),dtype=float).T
        a = a / np.sum(a[:,2])
        a[:,:2] -= np.min(a[:,:2]-a[:,2,None],axis=0)
        z = np.r_[a[:,0],a[:,1],a[:,2],np.max(a[:,0]+a[:,2]),np.max(a[:,1]+a[:,2])]
        seeds.append(z)
    for lengths in families:
        if time.monotonic() >= screen_end:
            break
        pts = []
        for row, count in enumerate(lengths):
            if time.monotonic() >= screen_end:
                break
            offset = -(count-1)
            if row and count == lengths[row-1]:
                offset += 0.35*(1 if row%2 else -1)
            pts.extend((2*k+offset,np.sqrt(3.0)*row) for k in range(count))
        if len(pts) == 21:
            seeds.append(from_points(pts))

    pool = []
    for z in seeds:
        if time.monotonic() >= screen_end:
            break
        for step in range(int(params['screen_rounds'])):
            if time.monotonic() >= screen_end:
                break
            zz = lp_step(z, screen_end)
            if zz is None:
                break
            z = zz
        offer(z)
        pool.append(z)
    if not pool:
        pool = [fallback]
    pool.sort(key=lambda z: z[63]+z[64])
    refine_end = start + 0.69*(deadline-start)
    finalists = pool[:3]
    for iteration in range(int(params['lp_refinements'])):
        if time.monotonic() >= refine_end:
            break
        for index in range(len(finalists)):
            if time.monotonic() >= refine_end:
                break
            z = finalists[index]
            zz = lp_step(z, refine_end)
            if zz is not None:
                finalists[index] = zz
                offer(zz)
    finalists.sort(key=lambda z:z[63]+z[64])

    class Finished(Exception):
        pass

    def inequalities(z):
        dx = z[ii]-z[jj]
        dy = z[21+ii]-z[21+jj]
        return np.r_[np.sqrt(dx*dx+dy*dy)-z[42+ii]-z[42+jj], -wall.dot(z)]

    def jacobian(z):
        dx = z[ii]-z[jj]
        dy = z[21+ii]-z[21+jj]
        dist = np.maximum(np.sqrt(dx*dx+dy*dy),1e-15)
        jac = np.zeros((nr+84,65))
        jac[ridx,ii] = dx/dist
        jac[ridx,jj] = -dx/dist
        jac[ridx,21+ii] = dy/dist
        jac[ridx,21+jj] = -dy/dist
        jac[ridx,42+ii] = -1
        jac[ridx,42+jj] = -1
        jac[nr:] = -wall
        return jac

    def callback(z):
        if time.monotonic() >= deadline-0.025:
            raise Finished()
        offer(z)

    constraints = [
        {'type':'eq','fun':lambda z:np.sum(z[42:63])-1.0,'jac':lambda z:eq[0]},
        {'type':'ineq','fun':inequalities,'jac':jacobian}
    ]
    for z in finalists[:2]:
        if time.monotonic() >= deadline-0.04:
            break
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result = minimize(lambda v:v[63]+v[64],z,jac=lambda v:objective,
                    method='SLSQP',bounds=bounds,constraints=constraints,callback=callback,
                    options={'maxiter':int(params['polish_iterations']),'ftol':2e-12,'disp':False})
            offer(result.x)
        except Finished:
            break
    return best
