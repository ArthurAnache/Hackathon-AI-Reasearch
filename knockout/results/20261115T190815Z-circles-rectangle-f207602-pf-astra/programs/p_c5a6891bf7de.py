import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linprog

PARAMS = {
    'aspect': (1.0, 0.5, 2.0, 'log'),
    'rows': (5, 3, 7),
    'jitter': (0.03, 0.0, 0.15),
    'init_jitter': (0.0, 0.0, 0.05),
    'mu0': (10.0, 1.0, 1000.0, 'log'),
    'mu_growth': (8.0, 2.0, 30.0),
    'stages': (6, 2, 12),
    'maxiter': (150, 30, 1000),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
K = np.arange(N)

class Deadline(Exception):
    pass

def valid(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or not (a[:, 2] > 0).all():
        return False
    p, r = a[:, :2], a[:, 2]
    d = np.sqrt((p[I, 0]-p[J, 0])**2 + (p[I, 1]-p[J, 1])**2)
    if np.any(d < r[I]+r[J]):
        return False
    x = [Fraction(float(v)) for v in p[:, 0]]
    y = [Fraction(float(v)) for v in p[:, 1]]
    q = [Fraction(float(v)) for v in r]
    perimeter = (max(u+v for u,v in zip(x,q))-min(u-v for u,v in zip(x,q))
                 +max(u+v for u,v in zip(y,q))-min(u-v for u,v in zip(y,q)))
    return perimeter <= 2

def repair(p, r, deadline):
    p = np.asarray(p).copy()
    r = np.maximum(np.asarray(r).copy(), 1e-12)
    if not np.isfinite(p).all() or not np.isfinite(r).all():
        return None
    d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
    if np.min(d) <= 0:
        return None
    factor = min(1.0, float(np.min(d/(r[I]+r[J]))))
    r *= factor*(1-2e-12)
    lo = (p-r[:,None]).min(axis=0)
    span = (p+r[:,None]).max(axis=0)-lo
    scale = min(1.0, 2.0/float(span.sum()))*(1-2e-12)
    p = (p-lo)*scale
    r *= scale
    a = np.column_stack((p,r))
    for _ in range(4):
        if time.monotonic() >= deadline:
            break
        if valid(a):
            return a
        a[:,2] *= 1-1e-10
    return a if valid(a) else None

def penalty(z, mu):
    x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
    dx,dy = x[I]-x[J],y[I]-y[J]
    d = np.sqrt(dx*dx+dy*dy+1e-18)
    v = np.maximum(r[I]+r[J]-d,0)
    b0,b1,b2,b3 = np.maximum(r-x,0),np.maximum(r+x-w,0),np.maximum(r-y,0),np.maximum(r+y-(2-w),0)
    neg = np.maximum(-r,0)
    f = -r.sum()+mu*(v@v+b0@b0+b1@b1+b2@b2+b3@b3+neg@neg)
    c = 2*mu*v
    gx = np.bincount(I,weights=-c*dx/d,minlength=N)+np.bincount(J,weights=c*dx/d,minlength=N)
    gy = np.bincount(I,weights=-c*dy/d,minlength=N)+np.bincount(J,weights=c*dy/d,minlength=N)
    gr = -np.ones(N)+np.bincount(I,weights=c,minlength=N)+np.bincount(J,weights=c,minlength=N)
    gx += 2*mu*(-b0+b1)
    gy += 2*mu*(-b2+b3)
    gr += 2*mu*(b0+b1+b2+b3-neg)
    return f,np.r_[gx,gy,gr,2*mu*(-b1.sum()+b3.sum())]

def constraints(z):
    x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
    dx,dy = x[I]-x[J],y[I]-y[J]
    d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-15)
    return np.r_[d-r[I]-r[J],x-r,w-x-r,y-r,2-w-y-r]

def constraint_jac(z):
    x,y = z[:N],z[N:2*N]
    dx,dy = x[I]-x[J],y[I]-y[J]
    d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-15)
    a = np.zeros((M+4*N,3*N+1))
    t = np.arange(M)
    a[t,I],a[t,J] = dx/d,-dx/d
    a[t,N+I],a[t,N+J] = dy/d,-dy/d
    a[t,2*N+I],a[t,2*N+J] = -1,-1
    a[M+K,K],a[M+K,2*N+K] = 1,-1
    a[M+N+K,K],a[M+N+K,2*N+K],a[M+N+K,-1] = -1,-1,1
    a[M+2*N+K,N+K],a[M+2*N+K,2*N+K] = 1,-1
    a[M+3*N+K,N+K],a[M+3*N+K,2*N+K],a[M+3*N+K,-1] = -1,-1,-1
    return a

def lp_radii(p, remaining):
    a = np.zeros((M+4*N+1,N+4))
    b = np.zeros(M+4*N+1)
    a[np.arange(M),I] = 1
    a[np.arange(M),J] = 1
    b[:M] = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
    a[M+K,K],a[M+K,N],b[M+K] = 1,1,p[:,0]
    a[M+N+K,K],a[M+N+K,N+1],b[M+N+K] = 1,-1,-p[:,0]
    a[M+2*N+K,K],a[M+2*N+K,N+2],b[M+2*N+K] = 1,1,p[:,1]
    a[M+3*N+K,K],a[M+3*N+K,N+3],b[M+3*N+K] = 1,-1,-p[:,1]
    a[-1,N:] = [-1,1,-1,1]
    b[-1] = 2
    res = linprog(-np.r_[np.ones(N),np.zeros(4)],A_ub=a,b_ub=b,
                  bounds=[(1e-12,None)]*N+[(None,None)]*4,method='highs',
                  options={'time_limit': max(0.001,remaining)})
    return res.x[:N] if res.success else None

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start+0.70*budget_s
    penalty_end = start+0.38*budget_s
    polish_end = start+0.61*budget_s
    rng = np.random.default_rng(seed)
    # Always have an independently valid fallback.
    fallback = np.array([[(i%7+0.5)/7,(i//7+0.5)/3,0.06] for i in range(N)])
    best = {'circles': fallback.tolist()}
    best_score = float(fallback[:,2].sum())
    a0 = None
    if init is not None:
        try:
            a0 = np.asarray(init['circles'],dtype=float)
            if valid(a0):
                score = float(a0[:,2].sum())
                if score >= best_score:
                    best,best_score = init,score
        except (ValueError,KeyError,TypeError):
            a0 = None
    if a0 is not None and a0.shape == (N,3) and np.isfinite(a0).all():
        p,r = a0[:,:2].copy(),np.maximum(np.abs(a0[:,2]),1e-12)
        lo = (p-r[:,None]).min(0)
        side = (p+r[:,None]).max(0)-lo
        s = 2/side.sum()
        p,r,w = (p-lo)*s,r*s,side[0]*s
        p += rng.normal(0,float(params['init_jitter']),p.shape)
    else:
        rows = int(params['rows'])
        cols = int(np.ceil(N/rows))
        w = 2*float(params['aspect'])/(1+float(params['aspect']))
        p = np.array([[(i%cols+0.5+0.25*((i//cols)%2))/cols,(i//cols+0.5)/rows] for i in range(N)])
        p = np.clip(p+rng.normal(0,float(params['jitter']),p.shape),0.02,0.98)*[w,2-w]
        r = np.full(N,0.05)
    z = np.r_[p[:,0],p[:,1],r,w]
    def consider(v):
        nonlocal best,best_score
        if time.monotonic() >= end:
            return
        a = repair(np.column_stack((v[:N],v[N:2*N])),v[2*N:3*N],end)
        if a is not None and valid(a):
            score = float(np.sum(a[:,2]))
            if score > best_score:
                best,best_score = {'circles':a.tolist()},score
    consider(z)
    mu = float(params['mu0'])
    latest = z.copy()
    def penfun(v):
        if time.monotonic() >= penalty_end:
            raise Deadline()
        return penalty(v,mu)
    def pencb(v):
        nonlocal latest
        latest = v.copy()
        if time.monotonic() >= penalty_end:
            raise Deadline()
    for _ in range(max(1,int(params['stages'])-2)):
        if time.monotonic() >= penalty_end:
            break
        try:
            z = minimize(penfun,z,jac=True,method='L-BFGS-B',callback=pencb,
                         options={'maxiter':int(params['maxiter']),'ftol':1e-11}).x
        except Deadline:
            z = latest.copy()
        consider(z)
        mu *= float(params['mu_growth'])
    # Joint constrained polishing replaces the ill-conditioned final penalty stages.
    objective_grad = np.r_[np.zeros(2*N),-np.ones(N),0.0]
    def objective(v):
        if time.monotonic() >= polish_end:
            raise Deadline()
        return -float(v[2*N:3*N].sum()),objective_grad
    def callback(v):
        nonlocal latest
        latest = v.copy()
        if time.monotonic() >= polish_end:
            raise Deadline()
        consider(v)
    latest = z.copy()
    if time.monotonic() < polish_end:
        try:
            z = minimize(objective,z,jac=True,method='SLSQP',callback=callback,
                         bounds=[(None,None)]*(2*N)+[(1e-10,0.5)]*N+[(0.05,1.95)],
                         constraints={'type':'ineq','fun':constraints,'jac':constraint_jac},
                         options={'maxiter':100,'ftol':1e-11}).x
        except Deadline:
            z = latest
    consider(z)
    if end-time.monotonic() > 0.08:
        p = np.column_stack((z[:N],z[N:2*N]))
        rr = lp_radii(p,end-time.monotonic()-0.025)
        if rr is not None:
            z[2*N:3*N] = rr
            consider(z)
    return best
