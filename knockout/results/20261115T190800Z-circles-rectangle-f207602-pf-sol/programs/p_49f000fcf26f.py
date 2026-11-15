import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
PARAMS = {
    'aspect': (1.0, 0.5, 2.0, 'log'),
    'rows': (5, 3, 7),
    'jitter': (0.03, 0.0, 0.15),
    'mu0': (10.0, 1.0, 1000.0, 'log'),
    'mu_growth': (8.0, 2.0, 30.0),
    'stages': (6, 2, 12),
    'maxiter': (150, 30, 1000)
}

class StopSearch(Exception):
    pass

def perimeter(p, r):
    x = [Fraction(float(v)) for v in p[:, 0]]
    y = [Fraction(float(v)) for v in p[:, 1]]
    q = [Fraction(float(v)) for v in r]
    return (max(a+b for a,b in zip(x,q))-min(a-b for a,b in zip(x,q))
            +max(a+b for a,b in zip(y,q))-min(a-b for a,b in zip(y,q)))

def score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
        return -np.inf
    p, r = a[:,:2], a[:,2]
    d = np.sqrt((p[I,0]-p[J,0])**2 + (p[I,1]-p[J,1])**2)
    if np.any(d < r[I]+r[J]) or perimeter(p,r) > 2:
        return -np.inf
    return float(np.sum(r))

def repaired(z):
    p = np.column_stack((z[:N],z[N:2*N])).copy()
    r = np.maximum(z[2*N:3*N], 1e-10).copy()
    if not np.isfinite(p).all() or not np.isfinite(r).all():
        return None
    d = np.sqrt((p[I,0]-p[J,0])**2 + (p[I,1]-p[J,1])**2)
    factor = min(1.0, float(np.min(d/(r[I]+r[J]))))
    if factor <= 0:
        return None
    r *= factor*(1-2e-13)
    p -= np.min(p-r[:,None],axis=0)
    h = float(perimeter(p,r))
    if h <= 0:
        return None
    scale = 2.0/h*(1-2e-13)
    return np.column_stack((p*scale,r*scale))

def penalty(z, mu):
    x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
    dx,dy = x[I]-x[J],y[I]-y[J]
    d = np.sqrt(dx*dx+dy*dy+1e-24)
    v = np.maximum(r[I]+r[J]-d,0)
    b = [np.maximum(r-x,0),np.maximum(r+x-w,0),
         np.maximum(r-y,0),np.maximum(r+y-(2-w),0)]
    neg = np.maximum(-r,0)
    f = -r.sum()+mu*(v@v+sum(t@t for t in b)+neg@neg)
    gx,gy,gr = np.zeros(N),np.zeros(N),-np.ones(N)
    c = 2*mu*v
    np.add.at(gr,I,c); np.add.at(gr,J,c)
    np.add.at(gx,I,-c*dx/d); np.add.at(gx,J,c*dx/d)
    np.add.at(gy,I,-c*dy/d); np.add.at(gy,J,c*dy/d)
    gr += 2*mu*(b[0]+b[1]+b[2]+b[3]-neg)
    gx += 2*mu*(-b[0]+b[1])
    gy += 2*mu*(-b[2]+b[3])
    return f,np.r_[gx,gy,gr,2*mu*(-b[1].sum()+b[3].sum())]

def constraints(z):
    x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
    dx,dy = x[I]-x[J],y[I]-y[J]
    d = np.sqrt(dx*dx+dy*dy+1e-30)
    return np.r_[d-r[I]-r[J],x-r,w-x-r,y-r,2-w-y-r]

def constraint_jac(z):
    dx,dy = z[I]-z[J],z[N+I]-z[N+J]
    d = np.sqrt(dx*dx+dy*dy+1e-30)
    a = np.zeros((M+4*N,3*N+1))
    k = np.arange(M)
    a[k,I] = dx/d; a[k,J] = -dx/d
    a[k,N+I] = dy/d; a[k,N+J] = -dy/d
    a[k,2*N+I] = -1; a[k,2*N+J] = -1
    k = np.arange(N)
    for t in range(4):
        rows = M+t*N+k
        a[rows,2*N+k] = -1
        a[rows,(0 if t<2 else N)+k] = 1 if t%2==0 else -1
    a[M+N:M+2*N,-1] = 1
    a[M+3*N:M+4*N,-1] = -1
    return a

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start+0.69*budget_s
    penalty_end = start+0.43*budget_s
    rng = np.random.default_rng(seed)
    best = None
    best_score = -np.inf
    if init is not None:
        try:
            a = np.asarray(init['circles'],dtype=float)
            s = score(a)
            if s > best_score:
                best,best_score = init,s
        except (ValueError,KeyError,TypeError):
            a = None
    else:
        a = None
    if a is not None and a.shape == (N,3) and np.isfinite(a).all():
        p,r = a[:,:2].copy(),np.maximum(np.abs(a[:,2]),1e-8)
        lo = np.min(p-r[:,None],axis=0)
        sides = np.max(p+r[:,None],axis=0)-lo
        scale = 2/sides.sum()
        p,r,w = (p-lo)*scale,r*scale,sides[0]*scale
    else:
        rows = int(params['rows'])
        cols = int(np.ceil(N/rows))
        w = 2*params['aspect']/(1+params['aspect'])
        p = np.array([((k%cols+0.5+(0.25 if (k//cols)%2 else 0))/cols,
                       (k//cols+0.5)/rows) for k in range(N)])
        p = np.clip(p+rng.normal(0,params['jitter'],p.shape),.02,.98)*[w,2-w]
        r = np.full(N,.05)
    z = np.r_[p[:,0],p[:,1],r,w]
    def consider(v):
        nonlocal best,best_score
        candidate = repaired(v)
        if candidate is not None:
            s = score(candidate)
            if s > best_score:
                best_score = s
                best = {'circles':candidate.tolist()}
    consider(z)
    latest = z.copy()
    def pf(v,mu):
        nonlocal latest
        if time.monotonic() >= penalty_end:
            raise StopSearch()
        latest = v.copy()
        return penalty(v,mu)
    mu = float(params['mu0'])
    for stage in range(int(params['stages'])):
        if time.monotonic() >= penalty_end:
            break
        try:
            z = minimize(pf,z,args=(mu,),jac=True,method='L-BFGS-B',
                         options={'maxiter':int(params['maxiter']),'ftol':1e-13}).x
        except StopSearch:
            z = latest.copy()
            break
        consider(z)
        mu *= float(params['mu_growth'])
    consider(z)
    objective_jac = np.r_[np.zeros(2*N),-np.ones(N),0.0]
    def objective(v):
        if time.monotonic() >= end:
            raise StopSearch()
        return -float(v[2*N:3*N].sum())
    def callback(v):
        if time.monotonic() >= end:
            raise StopSearch()
        consider(v)
    if time.monotonic() < end:
        try:
            result = minimize(objective,z,jac=lambda v:objective_jac,method='SLSQP',
                bounds=[(0,2)]*(2*N)+[(1e-9,1)]*N+[(.01,1.99)],
                constraints={'type':'ineq','fun':constraints,'jac':constraint_jac},
                callback=callback,options={'maxiter':180,'ftol':2e-12})
            consider(result.x)
        except StopSearch:
            pass
    if best is None:
        p = np.array([[(k%7)*.1,(k//7)*.1] for k in range(N)])
        best = {'circles':np.column_stack((p,np.full(N,.001))).tolist()}
    return best
