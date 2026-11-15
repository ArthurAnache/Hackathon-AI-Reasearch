import time
import math
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"maxiter": (180, 60, 400), "perturb": (0.055, 0.001, 0.2)}


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.69 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    bestscore = -float('inf')

    def exact_score(obj):
        try:
            a = np.asarray(obj['circles'] if 'circles' in obj else [[p[0], p[1], r] for p, r in zip(obj['centers'], obj['radii'])], dtype=float)
            if a.shape != (21, 3) or not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
                return -float('inf')
            for i in range(21):
                if time.monotonic() > start + .94 * budget_s:
                    return -float('inf')
                for j in range(i):
                    if np.sqrt((a[i, 0]-a[j, 0])**2 + (a[i, 1]-a[j, 1])**2) < a[i, 2]+a[j, 2]:
                        return -float('inf')
            x = [Fraction(float(t)) for t in a[:, 0]]
            y = [Fraction(float(t)) for t in a[:, 1]]
            r = [Fraction(float(t)) for t in a[:, 2]]
            if max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))+max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)) > 2:
                return -float('inf')
            return float(np.sum(a[:, 2]))
        except (ValueError, KeyError, TypeError):
            return -float('inf')

    if init is not None:
        bestscore = exact_score(init)

    # Three circles on y=0, and nine representatives in y>0.
    owners = np.array(list(range(12)) + list(range(3, 12)))
    signs = np.array([0]*3 + [1]*9 + [-1]*9)
    ex = np.zeros((21, 34)); ey = np.zeros((21, 34)); er = np.zeros((21, 34))
    for k in range(21):
        if time.monotonic() >= deadline:
            break
        o = owners[k]
        ex[k, o] = 1
        er[k, 21+o] = 1
        if o >= 3:
            ey[k, 12+o-3] = signs[k]
    pairs = []
    for i in range(21):
        if time.monotonic() >= deadline:
            break
        for j in range(i):
            # Keep only one of a pair of reflected constraints.
            ri = i if i < 3 else (i+9 if i < 12 else i-9)
            rj = j if j < 3 else (j+9 if j < 12 else j-9)
            if (i,j) <= (max(ri,rj), min(ri,rj)):
                pairs.append((i,j))
    ii = np.array([p[0] for p in pairs], dtype=int)
    jj = np.array([p[1] for p in pairs], dtype=int)
    dxm = ex[ii]-ex[jj]; dym = ey[ii]-ey[jj]; rsm = er[ii]+er[jj]
    # Linear rectangle constraints. Width is z[33], height is 2-width.
    rows = []; const = []
    for k in range(12):
        if time.monotonic() >= deadline:
            break
        v = np.zeros(34); v[33] = .5; v[k] = -1; v[21+k] = -1
        rows.append(v); const.append(0.)
        v = np.zeros(34); v[33] = .5; v[k] = 1; v[21+k] = -1
        rows.append(v); const.append(0.)
        v = np.zeros(34); v[33] = -.5; v[21+k] = -1
        if k >= 3:
            v[12+k-3] = -1
        rows.append(v); const.append(1.)
    lin = np.asarray(rows); offset = np.asarray(const)
    objective = np.zeros(34); objective[21:24] = -1; objective[24:33] = -2
    bounds = [(-1.,1.)]*12 + [(1e-7,1.)]*9 + [(1e-7,.65)]*12 + [(.12,1.88)]

    def fun(z):
        return float(objective @ z)

    def con(z):
        dx = dxm @ z; dy = dym @ z; rr = rsm @ z
        return np.concatenate((dx*dx+dy*dy-rr*rr, lin @ z+offset))

    def jac(z):
        dx = dxm @ z; dy = dym @ z; rr = rsm @ z
        return np.vstack((2*(dx[:,None]*dxm+dy[:,None]*dym-rr[:,None]*rsm),lin))

    class Stop(Exception):
        pass

    last = [None]
    def callback(z):
        last[0] = z.copy()
        if time.monotonic() >= deadline:
            raise Stop()

    def emit(z):
        nonlocal best, bestscore
        if z is None or time.monotonic() > start + .87*budget_s:
            return
        a = np.column_stack((ex @ z, ey @ z, er @ z))
        if not np.isfinite(a).all() or np.min(a[:,2]) <= 0:
            return
        ratio = 1.
        for i in range(21):
            if time.monotonic() > start + .89*budget_s:
                return
            for j in range(i):
                d = math.hypot(a[i,0]-a[j,0],a[i,1]-a[j,1])
                ratio = min(ratio,d/(a[i,2]+a[j,2]))
        a[:,2] *= max(0.,ratio)*(1.-3e-13)
        if np.min(a[:,2]) <= 0:
            return
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        a[:,:2] -= (lo+hi)/2
        a *= (2./np.sum(hi-lo))*(1.-3e-13)
        obj = {'centers': a[:,:2].tolist(), 'radii': a[:,2].tolist()}
        score = exact_score(obj)
        if score > bestscore:
            bestscore = score; best = obj

    def make_start(kind):
        single = [(-2.,0.,1.),(0.,0.,1.),(2.,0.,1.)]
        upper = []
        if kind == 0 or kind == 3:
            upper = [(x,2.,1.) for x in (-2.,0.,2.)] + [(x,4.,1.) for x in (-2.,0.,2.)]
            small = math.sqrt(2)-1
            upper += [(-1.,1.,small),(1.,1.,small),((-1. if kind == 0 else 1.),3.,small)]
        elif kind == 1:
            upper = [(x,math.sqrt(3),1.) for x in (-3.,-1.,1.,3.)]
            upper += [(x,2*math.sqrt(3),1.) for x in (-4.,-2.,0.,2.,4.)]
        else:
            upper = [(x,math.sqrt(3),1.) for x in (-3.,-1.,1.,3.)]
            upper += [(x,2*math.sqrt(3),1.) for x in (-2.,0.,2.)]
            upper += [(x,3*math.sqrt(3),1.) for x in (-1.,1.)]
        a = np.asarray(single+upper)
        w = np.max(a[:,0]+a[:,2])-np.min(a[:,0]-a[:,2])
        h = 2*np.max(a[:,1]+a[:,2])
        s = 2/(w+h)
        z = np.zeros(34); z[:12] = a[:,0]*s; z[12:21] = a[3:,1]*s
        z[21:33] = a[:,2]*s; z[33] = w*s
        return z

    pool = []
    attempt = 0
    while time.monotonic() < deadline:
        if attempt < 4:
            z = make_start(attempt)
        elif pool:
            z = max(pool,key=lambda q:q[0])[1].copy()
            scale = float(params['perturb']) * (1. if attempt % 3 else 2.)
            z[:21] += rng.normal(0.,scale,21)
            z[12:21] = np.maximum(z[12:21],.001)
            z[21:33] *= rng.uniform(.8,1.,12)
        else:
            z = make_start(0)
        last[0] = z
        try:
            result = minimize(fun,z,jac=lambda z:objective,method='SLSQP',bounds=bounds,
                              constraints={'type':'ineq','fun':con,'jac':jac},callback=callback,
                              options={'maxiter':int(params['maxiter']),'ftol':2e-12,'disp':False})
            z = result.x
        except Stop:
            z = last[0]
        emit(z)
        if z is not None:
            violation = max(0.,-float(np.min(con(z))))
            pool.append((-fun(z)-100*violation,z.copy()))
        attempt += 1
    if best is not None and bestscore > -float('inf'):
        return best
    # A strictly feasible emergency construction.
    a = [[(i%7)*.12,(i//7)*.12,.055] for i in range(21)]
    return {'circles':a}
