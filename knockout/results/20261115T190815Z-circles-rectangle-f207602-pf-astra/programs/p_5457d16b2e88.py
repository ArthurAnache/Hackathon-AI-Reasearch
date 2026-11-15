import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'aspect_excursion': (0.17, 0.06, 0.35),
    'penalty_share': (0.34, 0.2, 0.5),
    'mu_start': (100.0, 20.0, 1000.0, 'log'),
    'mu_growth': (10.0, 3.0, 30.0),
    'trials': (2, 1, 4),
    'finish_fraction': (0.70, 0.60, 0.74)
}
N = 21
I, J = np.triu_indices(N, 1)
K = np.arange(N)
M = len(I)

class Timeout(Exception):
    pass


def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    x, y, r = ([Fraction(float(v)) for v in a[:, k]] for k in range(3))
    perimeter = (max(u+v for u,v in zip(x,r)) - min(u-v for u,v in zip(x,r))
                 + max(u+v for u,v in zip(y,r)) - min(u-v for u,v in zip(y,r)))
    if perimeter > 2:
        return -np.inf
    d = np.sqrt((a[I,0]-a[J,0])**2 + (a[I,1]-a[J,1])**2)
    if np.any(d < a[I,2]+a[J,2]):
        return -np.inf
    return float(np.sum(a[:,2]))


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + float(params['finish_fraction'])*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf
    if init is not None:
        a = np.asarray(init['circles'], dtype=float)
        best_score = exact_score(a)
    else:
        a = np.column_stack(((np.arange(N)%3)*0.2,
                             (np.arange(N)//3)*0.2,
                             np.full(N,0.1*(1-1e-12))))
        best = {'circles': a.tolist()}
        best_score = exact_score(a)

    def consider(v):
        nonlocal best, best_score
        if time.monotonic() >= end or not np.isfinite(v).all():
            return
        p = np.column_stack((v[:N],v[N:2*N]))
        r = v[2*N:3*N].copy()
        if np.any(r <= 0):
            return
        d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
        ratio = min(1.0,float(np.min(d/(r[I]+r[J]))))
        if ratio <= 0:
            return
        r *= ratio*(1-3e-14)
        lo = np.min(p-r[:,None],axis=0)
        hi = np.max(p+r[:,None],axis=0)
        scale = (2.0/float(np.sum(hi-lo)))*(1-3e-14)
        candidate = np.column_stack(((p-(lo+hi)*0.5)*scale,r*scale))
        if float(np.sum(candidate[:,2])) <= best_score:
            return
        score = exact_score(candidate)
        if score > best_score:
            best_score = score
            best = {'circles': candidate.tolist()}

    lo = np.min(a[:,:2]-a[:,2,None],axis=0)
    hi = np.max(a[:,:2]+a[:,2,None],axis=0)
    scale = 2.0/float(np.sum(hi-lo))
    p = (a[:,:2]-lo)*scale
    z0 = np.r_[p[:,0],p[:,1],a[:,2]*scale,(hi[0]-lo[0])*scale]
    rows = np.arange(M)
    base = np.zeros((M+4*N,3*N+1))
    base[rows,2*N+I] = -1
    base[rows,2*N+J] = -1
    base[M+K,K] = 1
    base[M+K,2*N+K] = -1
    base[M+N+K,K] = -1
    base[M+N+K,2*N+K] = -1
    base[M+N+K,-1] = 1
    base[M+2*N+K,N+K] = 1
    base[M+2*N+K,2*N+K] = -1
    base[M+3*N+K,N+K] = -1
    base[M+3*N+K,2*N+K] = -1
    base[M+3*N+K,-1] = -1
    grad = np.r_[np.zeros(2*N),-np.ones(N),0.0]
    bounds = [(None,None)]*(2*N)+[(1e-7,0.5)]*N+[(0.2,1.8)]

    def polish(z, deadline):
        accepted = z.copy()
        cached = None
        value = None
        jac = None
        def check():
            if time.monotonic() >= deadline:
                raise Timeout
        def constraints(v):
            nonlocal cached, value, jac
            check()
            if cached is None or not np.array_equal(v,cached):
                x,y,r,w = v[:N],v[N:2*N],v[2*N:3*N],v[-1]
                dx,dy = x[I]-x[J],y[I]-y[J]
                d = np.sqrt(dx*dx+dy*dy)
                value = np.r_[d-r[I]-r[J],x-r,w-x-r,y-r,2-w-y-r]
                dd = np.maximum(d,1e-30)
                jac = base.copy()
                jac[rows,I] = dx/dd
                jac[rows,J] = -dx/dd
                jac[rows,N+I] = dy/dd
                jac[rows,N+J] = -dy/dd
                cached = v.copy()
            return value,jac
        def objective(v):
            check()
            return -float(np.sum(v[2*N:3*N])),grad
        def callback(v):
            nonlocal accepted
            check()
            accepted = v.copy()
            consider(v)
        try:
            result = minimize(objective,z,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda v:constraints(v)[0],
                             'jac':lambda v:constraints(v)[1]},callback=callback,
                options={'maxiter':180,'ftol':1e-13,'disp':False})
            accepted = result.x
        except Timeout:
            pass
        consider(accepted)
        return accepted

    def penalty(z,mu,deadline):
        accepted = z.copy()
        target = float(z[-1])
        fixed_bounds = bounds[:-1]+[(target,target)]
        def objective(v):
            if time.monotonic() >= deadline:
                raise Timeout
            x,y,r,w = v[:N],v[N:2*N],v[2*N:3*N],v[-1]
            dx,dy = x[I]-x[J],y[I]-y[J]
            d = np.sqrt(dx*dx+dy*dy+1e-24)
            overlap = np.maximum(r[I]+r[J]-d,0)
            b0 = np.maximum(r-x,0)
            b1 = np.maximum(r+x-w,0)
            b2 = np.maximum(r-y,0)
            b3 = np.maximum(r+y-(2-w),0)
            c = 2*mu*overlap
            cx,cy = c*dx/d,c*dy/d
            gx = -np.bincount(I,cx,minlength=N)+np.bincount(J,cx,minlength=N)
            gy = -np.bincount(I,cy,minlength=N)+np.bincount(J,cy,minlength=N)
            gr = -np.ones(N)+np.bincount(I,c,minlength=N)+np.bincount(J,c,minlength=N)
            gx += 2*mu*(-b0+b1)
            gy += 2*mu*(-b2+b3)
            gr += 2*mu*(b0+b1+b2+b3)
            f = -r.sum()+mu*(overlap@overlap+b0@b0+b1@b1+b2@b2+b3@b3)
            return float(f),np.r_[gx,gy,gr,2*mu*(-b1.sum()+b3.sum())]
        def callback(v):
            nonlocal accepted
            if time.monotonic() >= deadline:
                raise Timeout
            accepted = v.copy()
        try:
            result = minimize(objective,z,jac=True,method='L-BFGS-B',bounds=fixed_bounds,
                callback=callback,options={'maxiter':100,'ftol':1e-12,'gtol':1e-7,'maxls':25})
            accepted = result.x
        except Timeout:
            pass
        return accepted

    # Preserve a short direct polish, then spend almost all time outside its basin.
    polish(z0,min(end,start+0.06*budget_s))
    exploration_start = time.monotonic()
    trials = int(params['trials'])
    first_sign = 1 if rng.integers(2) else -1
    for trial in range(trials):
        now = time.monotonic()
        if now >= end:
            break
        trial_end = exploration_start+(end-exploration_start)*(trial+1)/trials
        if trial_end <= now:
            continue
        penalty_end = now+float(params['penalty_share'])*(trial_end-now)
        z = z0.copy()
        sign = first_sign if trial%2 == 0 else -first_sign
        target = float(np.clip(z0[-1]+sign*float(params['aspect_excursion'])*(1+0.25*(trial//2)),0.35,1.65))
        z[:N] *= target/z0[-1]
        z[N:2*N] *= (2-target)/(2-z0[-1])
        z[-1] = target
        stage_start = time.monotonic()
        for stage in range(3):
            if time.monotonic() >= penalty_end:
                break
            stage_end = stage_start+(penalty_end-stage_start)*(stage+1)/3
            mu = float(params['mu_start'])*float(params['mu_growth'])**stage
            z = penalty(z,mu,stage_end)
        if time.monotonic() < trial_end:
            polish(z,trial_end)
    return best
