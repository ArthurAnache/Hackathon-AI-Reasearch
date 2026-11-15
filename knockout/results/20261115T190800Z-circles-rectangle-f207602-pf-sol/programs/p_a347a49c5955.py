import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linprog

N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
PARAMS = {
    'aspect': (1.0, 0.5, 2.0, 'log'),
    'rows': (5, 3, 7),
    'jitter': (0.03, 0.0, 0.15),
    'mu0': (10.0, 1.0, 1000.0, 'log'),
    'mu_growth': (8.0, 2.0, 30.0),
    'stages': (4, 2, 8),
    'maxiter': (120, 30, 500),
    'polish_iterations': (100, 10, 250)
}

class StopSearch(Exception):
    pass


def valid(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or not (a[:, 2] > 0).all():
        return False
    r = a[:, 2]
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < r[I]+r[J]):
        return False
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    q = [Fraction(float(v)) for v in r]
    perimeter = (max(u+v for u,v in zip(x,q))-min(u-v for u,v in zip(x,q))
                 +max(u+v for u,v in zip(y,q))-min(u-v for u,v in zip(y,q)))
    return perimeter <= 2


def repair(p, r):
    p = np.asarray(p, dtype=float).copy()
    r = np.maximum(np.asarray(r, dtype=float), 1e-10).copy()
    if not np.isfinite(p).all() or not np.isfinite(r).all():
        return None
    d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
    if np.min(d) <= 0:
        return None
    factor = min(1.0, float(np.min(d/(r[I]+r[J]))))
    r *= factor*(1-2e-12)
    lo = np.min(p-r[:,None], axis=0)
    hi = np.max(p+r[:,None], axis=0)
    scale = 2.0/float(np.sum(hi-lo))*(1-2e-12)
    p = (p-lo)*scale
    r *= scale
    return np.column_stack((p,r))


def penalty(z, mu):
    x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
    dx,dy=x[I]-x[J],y[I]-y[J]
    d=np.sqrt(dx*dx+dy*dy+1e-24)
    v=np.maximum(r[I]+r[J]-d,0)
    b0=np.maximum(r-x,0)
    b1=np.maximum(r+x-w,0)
    b2=np.maximum(r-y,0)
    b3=np.maximum(r+y-(2-w),0)
    neg=np.maximum(-r,0)
    f=-r.sum()+mu*(v@v+b0@b0+b1@b1+b2@b2+b3@b3+neg@neg)
    c=2*mu*v
    gx=np.bincount(I,-c*dx/d,minlength=N)+np.bincount(J,c*dx/d,minlength=N)
    gy=np.bincount(I,-c*dy/d,minlength=N)+np.bincount(J,c*dy/d,minlength=N)
    gr=-np.ones(N)+np.bincount(I,c,minlength=N)+np.bincount(J,c,minlength=N)
    gx+=2*mu*(-b0+b1)
    gy+=2*mu*(-b2+b3)
    gr+=2*mu*(b0+b1+b2+b3-neg)
    gw=2*mu*(-b1.sum()+b3.sum())
    return f,np.r_[gx,gy,gr,gw]


def constraints(z):
    x,y,r,w=z[:N],z[N:2*N],z[2*N:3*N],z[-1]
    d=np.sqrt((x[I]-x[J])**2+(y[I]-y[J])**2+1e-30)
    return np.r_[d-r[I]-r[J],x-r,w-x-r,y-r,2-w-y-r]


def constraint_jac(z):
    x,y=z[:N],z[N:2*N]
    dx,dy=x[I]-x[J],y[I]-y[J]
    d=np.sqrt(dx*dx+dy*dy+1e-30)
    a=np.zeros((M+4*N,3*N+1))
    k=np.arange(M)
    a[k,I]=dx/d
    a[k,J]=-dx/d
    a[k,N+I]=dy/d
    a[k,N+J]=-dy/d
    a[k,2*N+I]=-1
    a[k,2*N+J]=-1
    q=np.arange(N)
    a[M+q,q]=1
    a[M+N+q,q]=-1
    a[M+N+q,-1]=1
    a[M+2*N+q,N+q]=1
    a[M+3*N+q,N+q]=-1
    a[M+3*N+q,-1]=-1
    for k in range(4):
        a[M+k*N+q,2*N+q]=-1
    return a


def lp_radii(p, limit):
    a=np.zeros((M+4*N+1,N+4))
    b=np.zeros(M+4*N+1)
    a[np.arange(M),I]=1
    a[np.arange(M),J]=1
    b[:M]=np.sqrt(np.sum((p[I]-p[J])**2,axis=1))
    q=np.arange(N)
    for k,(col,sign,coord) in enumerate(((N,1,0),(N+1,-1,0),(N+2,1,1),(N+3,-1,1))):
        rows=M+k*N+q
        a[rows,q]=1
        a[rows,col]=sign
        b[rows]=sign*p[:,coord]
    a[-1,N:]=[-1,1,-1,1]
    b[-1]=2
    res=linprog(-np.r_[np.ones(N),np.zeros(4)],A_ub=a,b_ub=b,
                bounds=[(1e-10,None)]*N+[(None,None)]*4,
                method='highs',options={'time_limit': max(0.001,limit)})
    return res.x[:N] if res.success else None


def construct(params, seed, budget_s, init=None):
    start=time.monotonic()
    end=start+0.70*budget_s
    penalty_end=start+0.38*budget_s
    polish_end=start+0.63*budget_s
    rng=np.random.default_rng(seed)
    best=None
    best_score=-float('inf')

    def consider(a, original=None):
        nonlocal best,best_score
        if a is not None and valid(a):
            score=float(np.sum(a[:,2]))
            if score>best_score:
                best_score=score
                best=original if original is not None else {'circles':a.tolist()}

    if init is not None:
        try:
            initial=np.array(init['circles'],dtype=float)
            consider(initial,init)
        except (ValueError,KeyError,TypeError):
            initial=None
    else:
        initial=None
    # A deterministic feasible baseline also protects against unusable starts.
    grid=np.array([[(i%7+0.5)*0.2,(i//7+0.5)*0.2,0.1*(1-1e-12)] for i in range(N)])
    consider(grid)

    if initial is not None and initial.shape==(N,3) and np.isfinite(initial).all():
        p=initial[:,:2].copy()
        r=np.maximum(np.abs(initial[:,2]),1e-5)
        lo=np.min(p-r[:,None],axis=0)
        sides=np.max(p+r[:,None],axis=0)-lo
        s=2/max(float(sides.sum()),1e-10)
        p=(p-lo)*s
        r=r*s
        w=float(sides[0]*s)
    else:
        aspect=float(params['aspect'])
        w=2*aspect/(1+aspect)
        rows=int(params['rows'])
        cols=int(np.ceil(N/rows))
        p=np.array([[(i%cols+0.5+0.25*((i//cols)%2))/cols,
                     (i//cols+0.5)/rows] for i in range(N)])
        p=np.clip(p+rng.normal(0,float(params['jitter']),p.shape),0.02,0.98)
        p*=np.array([w,2-w])
        r=np.full(N,0.05)
    z=np.r_[p[:,0],p[:,1],r,w]
    last=z.copy()
    mu=float(params['mu0'])
    for stage in range(int(params['stages'])):
        if time.monotonic()>=penalty_end:
            break
        def fun(v):
            nonlocal last
            if time.monotonic()>=penalty_end:
                raise StopSearch()
            last=v.copy()
            return penalty(v,mu)
        try:
            z=minimize(fun,z,jac=True,method='L-BFGS-B',
                       options={'maxiter':int(params['maxiter']),'ftol':1e-12}).x
        except StopSearch:
            z=last.copy()
            break
        mu*=float(params['mu_growth'])
    consider(repair(np.column_stack((z[:N],z[N:2*N])),z[2*N:3*N]))

    # Replace the ill-conditioned terminal penalty stages by joint constrained refinement.
    last=z.copy()
    objective_grad=np.r_[np.zeros(2*N),-np.ones(N),0.0]
    def obj(v):
        if time.monotonic()>=polish_end:
            raise StopSearch()
        return -float(np.sum(v[2*N:3*N]))
    def callback(v):
        nonlocal last
        last=v.copy()
        if time.monotonic()>=polish_end:
            raise StopSearch()
    if time.monotonic()<polish_end:
        try:
            result=minimize(obj,z,jac=lambda v:objective_grad,method='SLSQP',
                bounds=[(None,None)]*(2*N)+[(1e-9,1.0)]*N+[(0.05,1.95)],
                constraints=[{'type':'ineq','fun':constraints,'jac':constraint_jac}],
                callback=callback,options={'maxiter':int(params['polish_iterations']),'ftol':1e-11})
            z=result.x
        except StopSearch:
            z=last
        consider(repair(np.column_stack((z[:N],z[N:2*N])),z[2*N:3*N]))
    if time.monotonic()<end-0.12:
        p=np.column_stack((z[:N],z[N:2*N]))
        try:
            r=lp_radii(p,min(0.08,end-time.monotonic()-0.04))
            if r is not None:
                consider(repair(p,r))
        except (ValueError,RuntimeError):
            pass
    return best
