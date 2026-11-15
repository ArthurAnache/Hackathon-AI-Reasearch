import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'iterations': (160, 50, 300),
    'first_share': (0.66, 0.4, 0.9),
    'perturb': (0.004, 0.0001, 0.04, 'log'),
    'penalty': (900.0, 100.0, 10000.0, 'log'),
    'penalty_seconds': (0.075, 0.02, 0.2),
    'safety': (6e-14, 1e-14, 1e-10, 'log')
}
N = 21
I, J = np.triu_indices(N, 1)
K = np.arange(len(I))
Q = np.arange(N)
GRAD = np.r_[np.zeros(42), -np.ones(21), 0.0]

class StopSearch(Exception):
    pass

def exact(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -float('inf')
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return -float('inf')
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    perimeter = (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
                 +max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)))
    return float(a[:, 2].sum()) if perimeter <= 2 else -float('inf')

def unpack(z):
    return np.column_stack((z[:21], z[21:42], z[42:63]))

def encode(a):
    a = a.copy()
    lo = np.min(a[:, :2]-a[:, 2,None], axis=0)
    a[:, :2] -= lo
    wh = np.max(a[:, :2]+a[:, 2,None], axis=0)
    s = 2.0/float(wh.sum())
    a *= s
    return np.r_[a[:,0], a[:,1], a[:,2], wh[0]*s]

def constraints(z):
    x,y,r,w = z[:21],z[21:42],z[42:63],z[63]
    d = np.sqrt((x[I]-x[J])**2+(y[I]-y[J])**2+1e-30)
    return np.r_[d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r]

def jacobian(z):
    dx,dy = z[I]-z[J], z[21+I]-z[21+J]
    d = np.sqrt(dx*dx+dy*dy+1e-30)
    a = np.zeros((294,64))
    a[K,I],a[K,J] = dx/d,-dx/d
    a[K,21+I],a[K,21+J] = dy/d,-dy/d
    a[K,42+I],a[K,42+J] = -1,-1
    a[210+Q,Q] = 1
    a[231+Q,Q] = -1
    a[252+Q,21+Q] = 1
    a[273+Q,21+Q] = -1
    a[210+Q,42+Q] = -1
    a[231+Q,42+Q] = -1
    a[252+Q,42+Q] = -1
    a[273+Q,42+Q] = -1
    a[231:252,63] = 1
    a[273:294,63] = -1
    return a

def penalty(z, mu):
    x,y,r,w = z[:21],z[21:42],z[42:63],z[63]
    dx,dy = x[I]-x[J],y[I]-y[J]
    d = np.sqrt(dx*dx+dy*dy+1e-30)
    v = np.maximum(r[I]+r[J]-d,0)
    b0,b1 = np.maximum(r-x,0),np.maximum(r+x-w,0)
    b2,b3 = np.maximum(r-y,0),np.maximum(r+y-(2-w),0)
    c = 2*mu*v
    gx = np.bincount(I,-c*dx/d,minlength=21)+np.bincount(J,c*dx/d,minlength=21)
    gy = np.bincount(I,-c*dy/d,minlength=21)+np.bincount(J,c*dy/d,minlength=21)
    gr = -np.ones(21)+np.bincount(I,c,minlength=21)+np.bincount(J,c,minlength=21)
    gx += 2*mu*(-b0+b1)
    gy += 2*mu*(-b2+b3)
    gr += 2*mu*(b0+b1+b2+b3)
    value = -r.sum()+mu*(v@v+b0@b0+b1@b1+b2@b2+b3@b3)
    return value,np.r_[gx,gy,gr,2*mu*(-b1.sum()+b3.sum())]

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started+0.70*budget_s
    rng = np.random.default_rng(seed)
    safety = float(params['safety'])
    best,best_a,best_score = None,None,-float('inf')

    def retain(a, raw=False):
        nonlocal best,best_a,best_score
        if time.monotonic() >= deadline:
            return
        a = np.asarray(a,dtype=float).copy()
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        if not raw:
            d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
            f = min(1.,float(np.min(d/(a[I,2]+a[J,2]))))
            if f <= 0:
                return
            a[:,2] *= f*(1-safety)
            lo = np.min(a[:,:2]-a[:,2,None],axis=0)
            hi = np.max(a[:,:2]+a[:,2,None],axis=0)
            a[:,:2] -= (hi+lo)/2
            a *= 2/float((hi-lo).sum())*(1-safety)
        if float(a[:,2].sum()) <= best_score:
            return
        value = exact(a)
        if value > best_score:
            best_score,best_a = value,a.copy()
            best = {'circles':a.tolist()}

    initial = None
    if init is not None:
        try:
            a = np.asarray(init['circles'],dtype=float)
            value = exact(a)
            if value > best_score:
                best,best_score,best_a = init,value,a.copy()
            if a.shape == (21,3) and np.isfinite(a).all() and np.all(a[:,2]>0):
                initial = a.copy()
                retain(a)
        except (ValueError,TypeError,KeyError,OverflowError):
            pass
    base = np.array([[2*c+1.,2*r+1.,1.] for r in range(4) for c in range(5)]
                    + [[4.,4.,np.sqrt(2.)-1.]])/9.
    retain(base)
    if initial is None:
        initial = base.copy()
    bounds = [(None,None)]*42+[(1e-9,0.8)]*21+[(0.05,1.95)]

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
                           options={'maxiter':int(params['iterations']),'ftol':8e-14})
            last = res.x
        except StopSearch:
            pass
        retain(unpack(last))
        return last

    z = polish(encode(initial), started+0.70*budget_s*float(params['first_share']))
    attempt = 0
    while time.monotonic() < deadline-0.07:
        a = best_a.copy() if best_a is not None else initial.copy()
        amplitude = float(params['perturb'])*(1.8**min(attempt,3))
        a[:,:2] += rng.normal(0,amplitude,(21,2))
        z = encode(a)
        penalty_end = min(deadline-0.045,time.monotonic()+float(params['penalty_seconds']))
        last = z.copy()
        mu = float(params['penalty'])
        def soft(v):
            nonlocal last
            if time.monotonic() >= penalty_end:
                raise StopSearch()
            last = v.copy()
            return penalty(v,mu)
        try:
            z = minimize(soft,z,jac=True,method='L-BFGS-B',bounds=bounds,
                         options={'maxiter':55,'maxls':15,'ftol':1e-11}).x
        except StopSearch:
            z = last
        retain(unpack(z))
        if time.monotonic() < deadline-0.025:
            remaining = deadline-time.monotonic()
            local_end = deadline if attempt >= 1 else time.monotonic()+0.72*remaining
            polish(z,local_end)
        attempt += 1
        if attempt >= 4:
            break
    if best is not None:
        return best
    base[:,2] *= 1-1e-10
    base *= 1-1e-10
    return {'circles':base.tolist()}
