import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'iterations': (120, 40, 250),
    'first_share': (0.64, 0.4, 0.85),
    'perturb': (0.003, 0.0001, 0.025, 'log'),
    'penalty': (1200.0, 100.0, 10000.0, 'log'),
    'safety': (8e-14, 1e-14, 1e-10, 'log')
}

N = 21
I, J = np.triu_indices(N, 1)
ROWS = np.arange(len(I))
EYE = np.eye(N)
GRAD = np.r_[np.zeros(42), -np.ones(21), 0.0]

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    safety = float(params['safety'])
    best = None
    best_score = -float('inf')
    best_a = None

    def exact(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        perimeter = (max(u+v for u,v in zip(x,r)) - min(u-v for u,v in zip(x,r))
                     + max(u+v for u,v in zip(y,r)) - min(u-v for u,v in zip(y,r)))
        if perimeter > 2:
            return -float('inf')
        d = np.sqrt((a[I,0]-a[J,0])**2 + (a[I,1]-a[J,1])**2)
        if np.any(d < a[I,2]+a[J,2]):
            return -float('inf')
        return float(np.sum(a[:,2]))

    def retain(a):
        nonlocal best, best_score, best_a
        if time.monotonic() >= deadline:
            return
        a = np.array(a, dtype=float, copy=True)
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        d = np.sqrt((a[I,0]-a[J,0])**2 + (a[I,1]-a[J,1])**2)
        factor = min(1.0, float(np.min(d/(a[I,2]+a[J,2]))))
        if factor <= 0:
            return
        a[:,2] *= factor*(1-safety)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        a[:,:2] -= (hi+lo)/2
        a *= (2.0/float(np.sum(hi-lo)))*(1-safety)
        if float(np.sum(a[:,2])) <= best_score:
            return
        value = exact(a)
        if value > best_score:
            best_score = value
            best_a = a.copy()
            best = {'circles': a.tolist()}

    initial_a = None
    if init is not None:
        try:
            a = np.asarray(init['circles'], dtype=float)
            value = exact(a)
            if value > best_score:
                best, best_score, best_a = init, value, a.copy()
            if a.shape == (21,3) and np.isfinite(a).all() and np.all(a[:,2] > 0):
                initial_a = a.copy()
                retain(a)
        except (ValueError, KeyError, TypeError):
            pass

    base = np.array([[2*c+1.,2*r+1.,1.] for r in range(4) for c in range(5)]
                    + [[4.,4.,np.sqrt(2.)-1.]])/9.
    retain(base)
    if initial_a is None:
        initial_a = base.copy()

    def encode(a):
        a = a.copy()
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        a[:,:2] -= lo
        wh = np.max(a[:,:2]+a[:,2,None],axis=0)
        scale = 2.0/float(wh.sum())
        a *= scale
        return np.r_[a[:,0], a[:,1], a[:,2], wh[0]*scale]

    def unpack(v):
        return np.column_stack((v[:21], v[21:42], v[42:63]))

    def constraints(v):
        x,y,r,w = v[:21],v[21:42],v[42:63],v[63]
        dx,dy = x[I]-x[J],y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        return np.r_[d-r[I]-r[J],x-r,w-x-r,y-r,2-w-y-r]

    def jacobian(v):
        dx,dy = v[I]-v[J],v[21+I]-v[21+J]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        a = np.zeros((294,64))
        a[ROWS,I],a[ROWS,J] = dx/d,-dx/d
        a[ROWS,21+I],a[ROWS,21+J] = dy/d,-dy/d
        a[ROWS,42+I],a[ROWS,42+J] = -1,-1
        a[210:231,:21] = EYE
        a[231:252,:21] = -EYE
        a[252:273,21:42] = EYE
        a[273:294,21:42] = -EYE
        a[210:231,42:63] = -EYE
        a[231:252,42:63] = -EYE
        a[252:273,42:63] = -EYE
        a[273:294,42:63] = -EYE
        a[231:252,63] = 1
        a[273:294,63] = -1
        return a

    bounds = [(None,None)]*42+[(1e-9,0.7)]*21+[(0.05,1.95)]

    def polish(z, end):
        last = z.copy()
        def check():
            if time.monotonic() >= end:
                raise StopSearch()
        def fun(v):
            check()
            return -float(v[42:63].sum())
        def con(v):
            check()
            return constraints(v)
        def jac(v):
            check()
            return jacobian(v)
        def callback(v):
            nonlocal last
            check()
            last = v.copy()
            retain(unpack(v))
        try:
            res = minimize(fun,z,jac=lambda v:GRAD,method='SLSQP',bounds=bounds,
                           constraints={'type':'ineq','fun':con,'jac':jac},callback=callback,
                           options={'maxiter':int(params['iterations']),'ftol':2e-13,'disp':False})
            last = res.x
        except StopSearch:
            pass
        retain(unpack(last))
        return last

    first_end = started + 0.70*budget_s*float(params['first_share'])
    z = polish(encode(initial_a), first_end)
    attempt = 0
    while time.monotonic() < deadline - 0.08:
        attempt += 1
        a = best_a.copy() if best_a is not None else initial_a.copy()
        a[:,:2] += rng.normal(0,float(params['perturb']),size=(21,2))
        z = encode(a)
        penalty_end = min(deadline-0.055,time.monotonic()+0.09*budget_s)
        mu = float(params['penalty'])
        last = z.copy()
        def penalty(v):
            nonlocal last
            if time.monotonic() >= penalty_end:
                raise StopSearch()
            last = v.copy()
            c = constraints(v)
            violation = np.minimum(c,0)
            return (-float(v[42:63].sum()) + mu*float(violation@violation),
                    GRAD + 2*mu*(violation@jacobian(v)))
        try:
            res = minimize(penalty,z,jac=True,method='L-BFGS-B',bounds=bounds,
                           options={'maxiter':24,'maxls':15,'ftol':1e-11})
            z = res.x
        except StopSearch:
            z = last
        if time.monotonic() < deadline-0.025:
            z = polish(z,deadline)
        else:
            retain(unpack(z))
        if attempt >= 3:
            break

    if best is not None:
        return best
    fallback = base.copy()
    fallback[:,2] *= 1-1e-10
    fallback *= 1-1e-10
    return {'circles':fallback.tolist()}
