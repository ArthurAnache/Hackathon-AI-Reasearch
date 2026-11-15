import time
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize

PARAMS = {
    'screen_steps': (13, 6, 25),
    'refine_steps': (24, 8, 60),
    'radius_floor': (0.00003, 0.000001, 0.001, 'log'),
}


def construct(params: dict, seed: int, budget_s: float, init=None):
    started = time.monotonic()
    deadline = started + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    ids = np.arange(n)
    best = init
    best_score = -float('inf')

    def exact_score(obj):
        try:
            a = np.asarray(obj['circles'], dtype=float)
            if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
                return -float('inf')
            dx = a[ii, 0] - a[jj, 0]
            dy = a[ii, 1] - a[jj, 1]
            if np.any(np.sqrt(dx**2 + dy**2) < a[ii, 2] + a[jj, 2]):
                return -float('inf')
            x = list(map(Fraction, map(float, a[:, 0])))
            y = list(map(Fraction, map(float, a[:, 1])))
            r = list(map(Fraction, map(float, a[:, 2])))
            w = max(a+b for a,b in zip(x,r)) - min(a-b for a,b in zip(x,r))
            h = max(a+b for a,b in zip(y,r)) - min(a-b for a,b in zip(y,r))
            if w+h > 2:
                return -float('inf')
            return float(np.sum(np.asarray(obj['circles'], dtype=float)[:, 2]))
        except (ValueError, TypeError, KeyError, OverflowError):
            return -float('inf')

    if init is not None:
        best_score = exact_score(init)

    def save(v):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        a = np.column_stack((v[:n], v[n:2*n], v[2*n:3*n])).copy()
        if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return
        d = np.sqrt((a[ii,0]-a[jj,0])**2 + (a[ii,1]-a[jj,1])**2)
        ratio = float(np.min(d/(a[ii,2]+a[jj,2])))
        a[:,2] *= min(1.0, ratio) * (1.0-2e-12)
        if np.any(a[:,2] <= 0):
            return
        a[:,:2] -= np.min(a[:,:2]-a[:,2,None], axis=0)
        extent = np.max(a[:,:2]+a[:,2,None],axis=0)-np.min(a[:,:2]-a[:,2,None],axis=0)
        a *= (2.0 / float(np.sum(extent))) * (1.0-2e-12)
        if float(np.sum(a[:,2])) <= best_score:
            return
        obj = {'circles': a.tolist()}
        score = exact_score(obj)
        if score > best_score:
            best, best_score = obj, score

    # A guaranteed fallback also supplies a useful equal-circle family.
    fallback = np.zeros(65)
    fallback[:21] = np.tile(np.arange(7),3)*2+1
    fallback[21:42] = np.repeat(np.arange(3),7)*2+1
    fallback[42:63] = 1
    fallback[63:] = (14,6)
    save(fallback)

    A = np.zeros((84+m,65))
    A[ids,ids] = -1
    A[ids,42+ids] = 1
    A[21+ids,ids] = 1
    A[21+ids,42+ids] = 1
    A[21+ids,63] = -1
    A[42+ids,21+ids] = -1
    A[42+ids,42+ids] = 1
    A[63+ids,21+ids] = 1
    A[63+ids,42+ids] = 1
    A[63+ids,64] = -1
    rows = 84+np.arange(m)
    A[rows,42+ii] = 1
    A[rows,42+jj] = 1
    box = A[:84].copy()
    eq = np.zeros((1,65))
    eq[0,42:63] = 1
    obj = np.zeros(65)
    obj[63:] = 1
    bounds = [(None,None)]*42 + [(float(params['radius_floor']),None)]*21 + [(0,None)]*2
    zero = np.zeros(84+m)

    def directions(v):
        dx = v[ii]-v[jj]
        dy = v[21+ii]-v[21+jj]
        d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-15)
        return dx/d, dy/d, d

    def lp_refine(v, steps, stop):
        old = float('inf')
        for k in range(steps):
            now = time.monotonic()
            if now >= stop:
                break
            ux,uy,_ = directions(v)
            A[rows,ii] = -ux
            A[rows,jj] = ux
            A[rows,21+ii] = -uy
            A[rows,21+jj] = uy
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result = linprog(obj, A_ub=A, b_ub=zero, A_eq=eq, b_eq=[1.0],
                    bounds=bounds, method='highs', options={
                        'threads':1, 'primal_feasibility_tolerance':1e-9,
                        'dual_feasibility_tolerance':1e-8,
                        'time_limit':max(0.001,stop-now)})
            if not result.success:
                break
            v = result.x
            val = float(v[63]+v[64])
            if old-val < 2e-10:
                break
            old = val
        save(v)
        return v

    patterns = [
        [4,5,4,5,3],
        [5,4,5,4,3],
        [4,3,4,3,4,3],
        [4,5,3,5,4],
        [5,4,3,4,5],
        [4,4,5,4,4],
        [3,4,5,4,5],
        [3,5,5,5,3],
    ]
    pool = []
    screen_end = started + 0.37*budget_s
    for counts in patterns:
        if time.monotonic() >= screen_end:
            break
        xy = []
        for row,count in enumerate(counts):
            if time.monotonic() >= screen_end:
                break
            offset = 0.0
            if row and counts[row-1] == count:
                offset = 0.35*(-1 if row%2 else 1)
            xy.extend((j-(count-1)/2+offset,row*np.sqrt(3)/2) for j in range(count))
        if len(xy) != 21:
            break
        xy = np.asarray(xy,dtype=float)
        xy += rng.normal(0,0.006,xy.shape)
        xy -= xy.min(axis=0)
        v = np.zeros(65)
        v[:21] = (xy[:,0]+0.5)/10.5
        v[21:42] = (xy[:,1]+0.5)/10.5
        v[42:63] = 1/21
        v[63:65] = (xy.max(axis=0)+1)/10.5
        v = lp_refine(v,int(params['screen_steps']),screen_end)
        pool.append(v)

    if init is not None and best_score > 2.2 and time.monotonic() < screen_end:
        a = np.asarray(best['circles'],dtype=float)
        v = np.zeros(65)
        scale = a[:,2].sum()
        v[:21] = (a[:,0]-np.min(a[:,0]-a[:,2]))/scale
        v[21:42] = (a[:,1]-np.min(a[:,1]-a[:,2]))/scale
        v[42:63] = a[:,2]/scale
        v[63] = np.max(v[:21]+v[42:63])
        v[64] = np.max(v[21:42]+v[42:63])
        pool.append(v)

    pool.sort(key=lambda v: v[63]+v[64])
    refine_end = started + 0.51*budget_s
    for k in range(min(3,len(pool))):
        if time.monotonic() >= refine_end:
            break
        pool[k] = lp_refine(pool[k],int(params['refine_steps']),refine_end)
    pool.sort(key=lambda v: v[63]+v[64])

    class Stop(Exception):
        pass

    def check():
        if time.monotonic() >= deadline-0.012:
            raise Stop()

    def fun(v):
        check()
        return float(v[63]+v[64])

    def con(v):
        check()
        _,_,d = directions(v)
        return np.concatenate((-box@v,d-v[42+ii]-v[42+jj]))

    def jac(v):
        check()
        ux,uy,_ = directions(v)
        J = -A.copy()
        J[rows,ii] = ux
        J[rows,jj] = -ux
        J[rows,21+ii] = uy
        J[rows,21+jj] = -uy
        return J

    for v in pool[:2]:
        if time.monotonic() >= deadline-0.025:
            break
        latest = [v]
        def callback(z):
            check()
            latest[0] = z.copy()
        try:
            res = minimize(fun,v,jac=lambda z:obj,method='SLSQP',bounds=bounds,
                constraints=[{'type':'eq','fun':lambda z:np.sum(z[42:63])-1,
                              'jac':lambda z:eq[0]},
                             {'type':'ineq','fun':con,'jac':jac}],
                callback=callback,options={'maxiter':100,'ftol':2e-12,'disp':False})
            save(res.x)
        except Stop:
            save(latest[0])
            break
        save(latest[0])
    return best
