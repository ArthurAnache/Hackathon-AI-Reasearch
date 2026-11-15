import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linprog

PARAMS = {
    'mu0': (10.0, 1.0, 1000.0, 'log'),
    'mu_growth': (8.0, 2.0, 30.0),
    'penalty_stages': (4, 2, 7),
    'maxiter': (110, 30, 300),
    'polish_iter': (100, 20, 200),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class StopSearch(Exception):
    pass

def perimeter(p, r):
    x = [Fraction(float(v)) for v in p[:, 0]]
    y = [Fraction(float(v)) for v in p[:, 1]]
    q = [Fraction(float(v)) for v in r]
    return (max(a+b for a,b in zip(x,q))-min(a-b for a,b in zip(x,q))
            +max(a+b for a,b in zip(y,q))-min(a-b for a,b in zip(y,q)))

def valid(p, r):
    if not np.isfinite(p).all() or not np.isfinite(r).all() or not (r > 0).all():
        return False
    d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
    return not np.any(d < r[I]+r[J]) and perimeter(p,r) <= 2

def output(p, r):
    return {'circles': np.column_stack((p,r)).tolist()}

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    descent_end = start + 0.40*budget_s
    rng = np.random.default_rng(seed)
    best = None
    best_score = -np.inf

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def remember(p, r, repair=True):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        p = np.asarray(p, dtype=float).copy()
        r = np.asarray(r, dtype=float).copy()
        if not np.isfinite(p).all() or not np.isfinite(r).all():
            return
        if repair:
            r = np.maximum(r, 1e-12)
            d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
            factor = min(1.0, float(np.min(d/(r[I]+r[J]))))
            r *= max(0.0, factor)*(1-2e-12)
            if not (r > 0).all():
                return
            lo = (p-r[:,None]).min(axis=0)
            p -= lo
            side = (p+r[:,None]).max(axis=0)-(p-r[:,None]).min(axis=0)
            scale = min(1.0, 2.0/float(side.sum()))*(1-2e-12)
            p *= scale
            r *= scale
        score = float(np.sum(r))
        if score > best_score and valid(p,r):
            best = output(p,r)
            best_score = score

    if init is not None:
        a = np.asarray(init['circles'], dtype=float)
        if a.shape == (N,3) and np.isfinite(a).all():
            p, r = a[:,:2].copy(), a[:,2].copy()
            if valid(p,r):
                best, best_score = init, float(np.sum(r))
            lo = (p-np.abs(r)[:,None]).min(axis=0)
            side = (p+np.abs(r)[:,None]).max(axis=0)-lo
            s = 2.0/max(float(side.sum()),1e-15)
            p = (p-lo)*s
            r = np.maximum(np.abs(r)*s,1e-8)
            w = float(side[0]*s)
        else:
            init = None
    if init is None:
        p = np.array([((i%5+0.5)/5,(i//5+0.5)/5) for i in range(N)])
        p += rng.normal(0,0.02,p.shape)
        r = np.full(N,0.06)
        w = 1.0
    remember(p,r)
    z = np.r_[p[:,0],p[:,1],r,w]
    last = z.copy()

    def penalty(v, mu):
        nonlocal last
        check()
        if time.monotonic() > descent_end:
            raise StopSearch()
        last = v.copy()
        x,y,r,w = v[:N],v[N:2*N],v[2*N:3*N],v[-1]
        dx,dy = x[I]-x[J],y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-24)
        ov = np.maximum(r[I]+r[J]-d,0)
        b0,b1 = np.maximum(r-x,0),np.maximum(r+x-w,0)
        b2,b3 = np.maximum(r-y,0),np.maximum(r+y-(2-w),0)
        neg = np.maximum(-r,0)
        f = -r.sum()+mu*(ov@ov+b0@b0+b1@b1+b2@b2+b3@b3+neg@neg)
        c = 2*mu*ov
        gr = -np.ones(N)+np.bincount(I,c,minlength=N)+np.bincount(J,c,minlength=N)
        gx = np.bincount(I,-c*dx/d,minlength=N)+np.bincount(J,c*dx/d,minlength=N)
        gy = np.bincount(I,-c*dy/d,minlength=N)+np.bincount(J,c*dy/d,minlength=N)
        gr += 2*mu*(b0+b1+b2+b3-neg)
        gx += 2*mu*(-b0+b1)
        gy += 2*mu*(-b2+b3)
        return f,np.r_[gx,gy,gr,2*mu*(-b1.sum()+b3.sum())]

    mu = float(params['mu0'])
    for stage in range(int(params['penalty_stages'])):
        if time.monotonic() >= min(deadline,descent_end):
            break
        try:
            res = minimize(penalty,z,args=(mu,),jac=True,method='L-BFGS-B',
                           options={'maxiter':int(params['maxiter']),'ftol':1e-12,'maxls':25})
            z = res.x
        except StopSearch:
            z = last.copy()
            break
        remember(np.column_stack((z[:N],z[N:2*N])),z[2*N:3*N])
        mu *= float(params['mu_growth'])

    remember(np.column_stack((z[:N],z[N:2*N])),z[2*N:3*N])
    objective_grad = np.r_[np.zeros(2*N),-np.ones(N),0.0]
    eye = np.eye(N)
    def objective(v):
        check()
        return -float(v[2*N:3*N].sum())
    def constraints(v):
        check()
        x,y,r,w = v[:N],v[N:2*N],v[2*N:3*N],v[-1]
        dx,dy = x[I]-x[J],y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        return np.r_[d-r[I]-r[J],x-r,w-x-r,y-r,2-w-y-r]
    def jacobian(v):
        check()
        x,y = v[:N],v[N:2*N]
        dx,dy = x[I]-x[J],y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        a = np.zeros((M+4*N,3*N+1))
        rows = np.arange(M)
        a[rows,I],a[rows,J] = dx/d,-dx/d
        a[rows,N+I],a[rows,N+J] = dy/d,-dy/d
        a[rows,2*N+I] = -1
        a[rows,2*N+J] = -1
        for k in range(4):
            if time.monotonic() >= deadline:
                raise StopSearch()
            sl = slice(M+k*N,M+(k+1)*N)
            a[sl,2*N:3*N] = -eye
            col = 0 if k<2 else N
            a[sl,col:col+N] = eye if k%2==0 else -eye
            if k==1:
                a[sl,-1] = 1
            elif k==3:
                a[sl,-1] = -1
        return a
    def callback(v):
        remember(np.column_stack((v[:N],v[N:2*N])),v[2*N:3*N])
        check()
    if time.monotonic() < deadline-0.08:
        try:
            res = minimize(objective,z,jac=lambda v:objective_grad,method='SLSQP',
                bounds=[(None,None)]*(2*N)+[(1e-10,0.5)]*N+[(0.05,1.95)],
                constraints={'type':'ineq','fun':constraints,'jac':jacobian},
                callback=callback,options={'maxiter':int(params['polish_iter']),'ftol':1e-12})
            z = res.x
            remember(np.column_stack((z[:N],z[N:2*N])),z[2*N:3*N])
        except StopSearch:
            pass

    if time.monotonic() < deadline-0.20 and best is not None:
        pc = np.asarray(best['circles'])[:,:2]
        a = np.zeros((M+4*N+1,N+4))
        b = np.zeros(M+4*N+1)
        a[np.arange(M),I] = 1
        a[np.arange(M),J] = 1
        b[:M] = np.sqrt(((pc[I]-pc[J])**2).sum(axis=1))
        for k,(col,sign,coord) in enumerate([(N,1,0),(N+1,-1,0),(N+2,1,1),(N+3,-1,1)]):
            if time.monotonic() >= deadline-0.15:
                break
            rows = M+k*N+np.arange(N)
            a[rows,np.arange(N)] = 1
            a[rows,col] = sign
            b[rows] = sign*pc[:,coord]
        else:
            a[-1,N:] = [-1,1,-1,1]
            b[-1] = 2
            res = linprog(-np.r_[np.ones(N),np.zeros(4)],A_ub=a,b_ub=b,
                bounds=[(1e-12,None)]*N+[(None,None)]*4,method='highs',
                options={'time_limit':max(0.01,deadline-time.monotonic()-0.05)})
            if res.success:
                remember(pc,res.x[:N])
    if best is None:
        p = np.array([[0.03+0.045*k,0.03] for k in range(N)])
        best = output(p,np.full(N,0.01))
    return best
