import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import lstsq

PARAMS = {
    'aspect_shift': (0.16, 0.04, 0.35),
    'penalty': (350.0, 30.0, 3000.0, 'log'),
    'aspect_force': (110.0, 10.0, 1000.0, 'log'),
    'penalty_share': (0.24, 0.1, 0.4),
    'maxiter': (100, 30, 250)
}
N = 21
I, J = np.triu_indices(N, 1)
K = np.arange(N)
M = len(I)
Q = np.arange(M)
GRAD = np.r_[np.zeros(42), -np.ones(21), 0.0]
BASE = np.zeros((M + 84, 64))
BASE[Q,42+I] = -1
BASE[Q,42+J] = -1
BASE[M+K,K] = 1
BASE[M+K,42+K] = -1
BASE[M+21+K,K] = -1
BASE[M+21+K,42+K] = -1
BASE[M+21+K,63] = 1
BASE[M+42+K,21+K] = 1
BASE[M+42+K,42+K] = -1
BASE[M+63+K,21+K] = -1
BASE[M+63+K,42+K] = -1
BASE[M+63+K,63] = -1

class Deadline(Exception):
    pass

def score(a):
    if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
        return -np.inf
    d = np.sqrt((a[I,0]-a[J,0])**2 + (a[I,1]-a[J,1])**2)
    if np.any(d < a[I,2]+a[J,2]):
        return -np.inf
    x = list(map(lambda v: Fraction(float(v)), a[:,0]))
    y = list(map(lambda v: Fraction(float(v)), a[:,1]))
    r = list(map(lambda v: Fraction(float(v)), a[:,2]))
    span = (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
            + max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)))
    if span > 2:
        return -np.inf
    return float(np.sum(a[:,2]))

def geometry(z, jac=False):
    x,y,r,w = z[:21],z[21:42],z[42:63],z[63]
    dx,dy = x[I]-x[J],y[I]-y[J]
    d = np.sqrt(dx*dx+dy*dy)
    c = np.r_[d-r[I]-r[J],x-r,w-x-r,y-r,2-w-y-r]
    if not jac:
        return c
    safe = np.maximum(d,1e-15)
    a = BASE.copy()
    a[Q,I],a[Q,J] = dx/safe,-dx/safe
    a[Q,21+I],a[Q,21+J] = dy/safe,-dy/safe
    return c,a

def repaired(z):
    if not np.isfinite(z).all() or np.min(z[42:63]) <= 0:
        return None
    p = np.column_stack((z[:21],z[21:42]))
    r = z[42:63].copy()
    d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
    factor = min(1.0,float(np.min(d/(r[I]+r[J]))))
    if factor <= 0:
        return None
    r *= factor*(1-1e-14)
    lo = np.min(p-r[:,None],axis=0)
    hi = np.max(p+r[:,None],axis=0)
    scale = (2-4e-14)/float(np.sum(hi-lo))
    return np.column_stack(((p-lo)*scale,r*scale))

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    grid = np.column_stack(((2*(K%7)+1)/10,(2*(K//7)+1)/10,np.full(21,0.1-1e-14)))
    best = {'circles':grid.tolist()}
    best_score = score(grid)
    original = grid
    if init is not None:
        try:
            aa = np.asarray(init['circles'],dtype=float)
            ss = score(aa)
            if ss >= best_score:
                original,best,best_score = aa,init,ss
        except (ValueError,TypeError,KeyError,OverflowError):
            pass
    p,r = original[:,:2].copy(),original[:,2].copy()
    lo = np.min(p-r[:,None],axis=0)
    span = np.max(p+r[:,None],axis=0)-lo
    scale = 2/float(span.sum())
    p,r = (p-lo)*scale,r*scale
    z0 = np.r_[p[:,0],p[:,1],r,span[0]*scale]

    def consider(z):
        nonlocal best,best_score
        if time.monotonic() >= end:
            return
        a = repaired(z)
        if a is not None and float(a[:,2].sum()) > best_score:
            ss = score(a)
            if ss > best_score:
                best,best_score = {'circles':a.tolist()},ss

    def newton(z, stop):
        z = z.copy()
        c = geometry(z)
        if np.min(c) < -2e-6:
            return z
        active = np.flatnonzero(np.abs(c)<3e-6)
        if len(active) < 64:
            return z
        for iteration in range(5):
            if time.monotonic() >= stop:
                break
            c,a = geometry(z,True)
            delta,_,rank,_ = lstsq(a[active],-c[active],cond=1e-12,lapack_driver='gelsy',check_finite=False)
            if rank < 64 or not np.isfinite(delta).all() or np.max(np.abs(delta)) > 0.01:
                break
            trial = z+delta
            if np.min(geometry(trial)) < -max(1e-10,2*float(np.max(np.abs(c[active])))):
                break
            z = trial
            consider(z)
            if np.max(np.abs(delta)) < 1e-15:
                break
        return z

    def polish(z, stop):
        latest = z.copy()
        cache_z = None
        cache_c = None
        cache_a = None
        def check():
            if time.monotonic() >= stop:
                raise Deadline()
        def obj(v):
            check()
            return -float(v[42:63].sum()),GRAD
        def calc(v):
            nonlocal cache_z,cache_c,cache_a
            check()
            if cache_z is None or not np.array_equal(v,cache_z):
                cache_c,cache_a = geometry(v,True)
                cache_z = v.copy()
        def con(v):
            calc(v)
            return cache_c
        def jac(v):
            calc(v)
            return cache_a
        def cb(v):
            nonlocal latest
            check()
            latest = v.copy()
            consider(v)
        try:
            res = minimize(obj,z,jac=True,method='SLSQP',
                bounds=[(None,None)]*42+[(1e-8,0.5)]*21+[(0.25,1.75)],
                constraints={'type':'ineq','fun':con,'jac':jac},callback=cb,
                options={'maxiter':int(params['maxiter']),'ftol':3e-13,'disp':False})
            latest = res.x
        except Deadline:
            pass
        consider(latest)
        return latest

    z0 = newton(z0,min(end,start+0.06*budget_s))
    consider(z0)
    directions = [1,-1] if rng.integers(2) else [-1,1]
    for attempt,direction in enumerate(directions):
        now = time.monotonic()
        if now >= end-0.03:
            break
        local_end = now+(end-now)/(2-attempt)
        pen_end = now+float(params['penalty_share'])*(local_end-now)
        target = float(np.clip(z0[63]+direction*float(params['aspect_shift']),0.45,1.55))
        z = z0.copy()
        # Start halfway along an anisotropic deformation, not with local jitter.
        halfway = 0.5*(target+z0[63])
        z[:21] *= halfway/z0[63]
        z[21:42] *= (2-halfway)/(2-z0[63])
        z[63] = halfway
        latest = z.copy()
        mu = float(params['penalty'])
        force = float(params['aspect_force'])
        def pen(v):
            if time.monotonic() >= pen_end:
                raise Deadline()
            x,y,r,w = v[:21],v[21:42],v[42:63],v[63]
            dx,dy = x[I]-x[J],y[I]-y[J]
            d = np.sqrt(dx*dx+dy*dy+1e-24)
            overlap = np.maximum(r[I]+r[J]-d,0)
            b0,b1 = np.maximum(r-x,0),np.maximum(r+x-w,0)
            b2,b3 = np.maximum(r-y,0),np.maximum(r+y-(2-w),0)
            f = -float(r.sum())+mu*(overlap@overlap+b0@b0+b1@b1+b2@b2+b3@b3)+force*(w-target)**2
            t = 2*mu*overlap
            gx = np.bincount(I,weights=-t*dx/d,minlength=21)+np.bincount(J,weights=t*dx/d,minlength=21)+2*mu*(-b0+b1)
            gy = np.bincount(I,weights=-t*dy/d,minlength=21)+np.bincount(J,weights=t*dy/d,minlength=21)+2*mu*(-b2+b3)
            gr = -np.ones(21)+np.bincount(I,weights=t,minlength=21)+np.bincount(J,weights=t,minlength=21)+2*mu*(b0+b1+b2+b3)
            gw = 2*mu*(-b1.sum()+b3.sum())+2*force*(w-target)
            return f,np.r_[gx,gy,gr,gw]
        def pcb(v):
            nonlocal latest
            if time.monotonic() >= pen_end:
                raise Deadline()
            latest = v.copy()
        try:
            result = minimize(pen,z,jac=True,method='L-BFGS-B',callback=pcb,
                bounds=[(None,None)]*42+[(1e-6,0.5)]*21+[(0.3,1.7)],
                options={'maxiter':220,'ftol':1e-11,'gtol':1e-7,'maxls':20})
            latest = result.x
        except Deadline:
            pass
        consider(latest)
        if time.monotonic() < local_end-0.015:
            latest = polish(latest,local_end-0.012)
            newton(latest,local_end)
    return best
