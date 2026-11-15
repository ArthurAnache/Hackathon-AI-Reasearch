import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'excursion': (0.22, 0.04, 0.5),
    'paths': (3, 1, 8),
    'penalty_iterations': (65, 15, 150),
    'release_iterations': (110, 20, 250),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class Deadline(Exception):
    pass

def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return -np.inf
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    per = (max(v+q for v,q in zip(x,r))-min(v-q for v,q in zip(x,r))
           +max(v+q for v,q in zip(y,r))-min(v-q for v,q in zip(y,r)))
    return float(a[:, 2].sum()) if per <= 2 else -np.inf

def unpack(z):
    return z[:N], z[N:2*N], z[2*N:3*N], z[-1]

def repair(z, deadline):
    x,y,r,w = unpack(z)
    a = np.column_stack((x,y,np.maximum(r,1e-12)))
    lo = (a[:,:2]-a[:,2,None]).min(axis=0)
    a[:,:2] -= lo
    side = (a[:,:2]+a[:,2,None]).max(axis=0)
    scale = min(1.0, 2.0/side.sum())*(1-3e-14)
    a *= scale
    for _ in range(6):
        if time.monotonic() >= deadline:
            break
        d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
        ratio = float(np.min(d/(a[I,2]+a[J,2])))
        if ratio < 1:
            a[:,2] *= max(0.0,ratio)*(1-3e-14)
        if exact_score(a) > -np.inf:
            return a
        a *= 1-2e-13
    return a

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.69*budget_s
    rng = np.random.default_rng(seed)
    if init is not None:
        a0 = np.asarray(init['circles'], dtype=float)
        best = init
        bestscore = exact_score(a0)
    else:
        a0 = np.array([[(k%5+0.5)/5, (k//5+0.5)/5, 0.075] for k in range(N)])
        best = {'circles': a0.tolist()}
        bestscore = exact_score(a0)
    lo = (a0[:,:2]-a0[:,2,None]).min(axis=0)
    sides = (a0[:,:2]+a0[:,2,None]).max(axis=0)-lo
    scale = 2/sides.sum()
    p0 = (a0[:,:2]-lo)*scale
    r0 = a0[:,2]*scale
    w0 = sides[0]*scale
    base = np.r_[p0[:,0],p0[:,1],r0,w0]
    objective_gradient = np.r_[np.zeros(2*N),-np.ones(N),0.0]
    bounds = [(-1.,3.)]*(2*N)+[(1e-9,1.)]*N+[(0.1,1.9)]
    paths = int(params['paths'])

    for path in range(paths):
        now = time.monotonic()
        if now >= end-0.025:
            break
        local_end = now+(end-now-0.018)/(paths-path)
        direction = -1 if path%2 == 0 else 1
        amplitude = float(params['excursion'])*(1+0.40*(path//2))
        target = float(np.clip(w0+direction*amplitude,0.25,1.75))
        z = base.copy()
        z[:N] *= target/w0
        z[N:2*N] *= (2-target)/(2-w0)
        z[2*N:3*N] *= min(target/w0,(2-target)/(2-w0))
        z[-1] = target
        last = [z.copy()]
        fixed_bounds = bounds[:-1]+[(target,target)]
        penalty_end = now+0.48*(local_end-now)

        def penalty(v, mu):
            if time.monotonic() >= penalty_end:
                raise Deadline()
            x,y,r,w = unpack(v)
            dx,dy = x[I]-x[J],y[I]-y[J]
            d = np.sqrt(dx*dx+dy*dy+1e-28)
            ov = np.maximum(r[I]+r[J]-d,0)
            b0,b1 = np.maximum(r-x,0),np.maximum(r+x-w,0)
            b2,b3 = np.maximum(r-y,0),np.maximum(r+y-(2-w),0)
            f = -r.sum()+mu*(ov@ov+b0@b0+b1@b1+b2@b2+b3@b3)
            c = 2*mu*ov
            gx = np.bincount(I,weights=-c*dx/d,minlength=N)+np.bincount(J,weights=c*dx/d,minlength=N)
            gy = np.bincount(I,weights=-c*dy/d,minlength=N)+np.bincount(J,weights=c*dy/d,minlength=N)
            gr = -np.ones(N)+np.bincount(I,weights=c,minlength=N)+np.bincount(J,weights=c,minlength=N)
            gx += 2*mu*(-b0+b1)
            gy += 2*mu*(-b2+b3)
            gr += 2*mu*(b0+b1+b2+b3)
            return f,np.r_[gx,gy,gr,2*mu*(-b1.sum()+b3.sum())]

        def remember(v):
            last[0] = v.copy()
            if time.monotonic() >= local_end:
                raise Deadline()

        for mu in (30.,300.,3000.):
            if time.monotonic() >= penalty_end:
                break
            try:
                z = minimize(penalty,z,args=(mu,),jac=True,method='L-BFGS-B',bounds=fixed_bounds,
                             callback=remember,options={'maxiter':int(params['penalty_iterations']),'ftol':1e-12,'maxls':15}).x
                last[0] = z.copy()
            except Deadline:
                z = last[0].copy()
                break

        def constraints(v):
            if time.monotonic() >= local_end:
                raise Deadline()
            x,y,r,w = unpack(v)
            d = np.sqrt((x[I]-x[J])**2+(y[I]-y[J])**2+1e-30)
            return np.r_[d-r[I]-r[J], x-r,w-x-r,y-r,2-w-y-r]

        def jacobian(v):
            if time.monotonic() >= local_end:
                raise Deadline()
            x,y,r,w = unpack(v)
            dx,dy = x[I]-x[J],y[I]-y[J]
            d = np.sqrt(dx*dx+dy*dy+1e-30)
            jac = np.zeros((M+4*N,3*N+1))
            row = np.arange(M)
            jac[row,I] = dx/d
            jac[row,J] = -dx/d
            jac[row,N+I] = dy/d
            jac[row,N+J] = -dy/d
            jac[row,2*N+I] = -1
            jac[row,2*N+J] = -1
            k = np.arange(N)
            for b in range(4):
                if time.monotonic() >= local_end:
                    raise Deadline()
                rr = M+b*N+k
                jac[rr,2*N+k] = -1
                jac[rr,(0 if b<2 else N)+k] = 1 if b%2==0 else -1
            jac[M+N:M+2*N,-1] = 1
            jac[M+3*N:,-1] = -1
            return jac

        if time.monotonic() < local_end-0.01:
            last[0] = z.copy()
            try:
                z = minimize(lambda v: (-v[2*N:3*N].sum(),objective_gradient),z,jac=True,
                             method='SLSQP',bounds=bounds,
                             constraints={'type':'ineq','fun':constraints,'jac':jacobian},callback=remember,
                             options={'maxiter':int(params['release_iterations']),'ftol':2e-12}).x
            except Deadline:
                z = last[0]
        if time.monotonic() < end-0.008:
            a = repair(z,end)
            score = exact_score(a)
            if score > bestscore:
                bestscore = score
                best = {'circles':a.tolist()}
    return best
