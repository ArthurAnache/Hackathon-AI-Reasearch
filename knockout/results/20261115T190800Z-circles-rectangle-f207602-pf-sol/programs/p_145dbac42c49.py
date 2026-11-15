import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.linalg import lstsq

PARAMS = {
    'aspect': (1.0, 0.5, 2.0, 'log'),
    'rows': (5, 3, 7),
    'jitter': (0.03, 0.0, 0.15),
    'mu0': (10.0, 1.0, 1000.0, 'log'),
    'mu_growth': (8.0, 2.0, 30.0),
    'stages': (6, 2, 12),
    'maxiter': (150, 30, 1000),
    'contact_tol': (2e-7, 1e-10, 1e-4, 'log')
}
N = 21
I, J = np.triu_indices(N, 1)

class Stop(Exception):
    pass

def perimeter(p, r):
    x = [Fraction(float(v)) for v in p[:, 0]]
    y = [Fraction(float(v)) for v in p[:, 1]]
    q = [Fraction(float(v)) for v in r]
    return (max(a+b for a,b in zip(x,q)) - min(a-b for a,b in zip(x,q))
            + max(a+b for a,b in zip(y,q)) - min(a-b for a,b in zip(y,q)))

def valid(p, r):
    if not np.isfinite(p).all() or not np.isfinite(r).all() or not (r > 0).all():
        return False
    d = np.sqrt((p[I,0]-p[J,0])**2 + (p[I,1]-p[J,1])**2)
    return not (d < r[I]+r[J]).any() and perimeter(p,r) <= 2

def repair(p, r, deadline):
    r = np.maximum(r, 1e-12).copy() * (1-1e-12)
    d = np.sqrt((p[I,0]-p[J,0])**2 + (p[I,1]-p[J,1])**2)
    for _ in range(5):
        if time.monotonic() >= deadline:
            break
        if not (d < r[I]+r[J]).any():
            break
        for i,j,dd in zip(I,J,d):
            if time.monotonic() >= deadline:
                return r
            if r[i]+r[j] > dd*(1-1e-12):
                s = dd*(1-1e-12)/(r[i]+r[j])
                r[i] *= s
                r[j] *= s
    for _ in range(60):
        if time.monotonic() >= deadline:
            break
        e = perimeter(p,r)-2
        if e <= 0:
            break
        r *= 1-max(2*float(e)/float(r.sum()),1e-12)
    return r

def unpack(z):
    return np.column_stack((z[:N],z[N:2*N])),z[2*N:3*N]

def constraints(z, jac=False):
    x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
    dx,dy = x[I]-x[J],y[I]-y[J]
    d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-30)
    g = np.r_[d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r]
    if not jac:
        return g
    a = np.zeros((len(g),64))
    k = np.arange(len(I))
    a[k,I] = dx/d
    a[k,J] = -dx/d
    a[k,N+I] = dy/d
    a[k,N+J] = -dy/d
    a[k,2*N+I] = -1
    a[k,2*N+J] = -1
    k = len(I)+np.arange(N)
    a[k,np.arange(N)] = 1
    a[k,2*N+np.arange(N)] = -1
    k = k+N
    a[k,np.arange(N)] = -1
    a[k,2*N+np.arange(N)] = -1
    a[k,-1] = 1
    k = k+N
    a[k,N+np.arange(N)] = 1
    a[k,2*N+np.arange(N)] = -1
    k = k+N
    a[k,N+np.arange(N)] = -1
    a[k,2*N+np.arange(N)] = -1
    a[k,-1] = -1
    return g,a

def polish(z, tol, deadline):
    active = np.flatnonzero(constraints(z) < tol)
    if len(active) < 64:
        return z
    z = z.copy()
    for _ in range(10):
        if time.monotonic() >= deadline:
            break
        g,a = constraints(z,True)
        step = lstsq(a[active],-g[active],lapack_driver='gelsy',check_finite=False)[0]
        if not np.isfinite(step).all() or np.max(np.abs(step)) > .02:
            break
        zz = z+step
        if constraints(zz).min() < -max(1e-9,2*np.max(np.abs(g[active]))):
            break
        z = zz
        if np.max(np.abs(step)) < 2e-15:
            break
    return z

def penalty(z, mu, deadline):
    if time.monotonic() >= deadline:
        raise Stop()
    g,a = constraints(z,True)
    v = np.minimum(g,0)
    r = z[2*N:3*N]
    neg = np.minimum(r,0)
    grad = 2*mu*(a.T@v)
    grad[2*N:3*N] += -1+2*mu*neg
    return -r.sum()+mu*(v@v+neg@neg),grad

def lp(p):
    m = len(I)
    a = np.zeros((m+4*N+1,N+4))
    b = np.zeros(len(a))
    a[np.arange(m),I] = 1
    a[np.arange(m),J] = 1
    b[:m] = np.sqrt(((p[I]-p[J])**2).sum(1))
    for k,(col,sign,coord) in enumerate([(N,1,0),(N+1,-1,0),(N+2,1,1),(N+3,-1,1)]):
        rows = m+k*N+np.arange(N)
        a[rows,np.arange(N)] = 1
        a[rows,col] = sign
        b[rows] = sign*p[:,coord]
    a[-1,N:] = [-1,1,-1,1]
    b[-1] = 2
    res = linprog(-np.r_[np.ones(N),np.zeros(4)],A_ub=a,b_ub=b,
                  bounds=[(0,None)]*N+[(None,None)]*4,method='highs')
    return res.x[:N] if res.success else np.full(N,1e-12)

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start+.70*budget_s
    work_end = start+.58*budget_s
    rng = np.random.default_rng(seed)
    best = init
    score = -np.inf
    if init is not None:
        c = np.asarray(init['circles'],float)
        p,r = c[:,:2].copy(),c[:,2].copy()
        if valid(p,r):
            score = float(r.sum())
        lo = (p-r[:,None]).min(0)
        side = (p+r[:,None]).max(0)-lo
        s = 2/side.sum()
        p,r,w = (p-lo)*s,r*s,side[0]*s
    else:
        w = 2*params['aspect']/(1+params['aspect'])
        rows = int(params['rows'])
        cols = int(np.ceil(N/rows))
        idx = np.arange(N)
        p = np.column_stack(((idx%cols+.5)/cols,(idx//cols+.5)/rows))
        p += rng.normal(0,params['jitter'],p.shape)
        p = np.clip(p,.02,.98)*[w,2-w]
        r = np.full(N,.05)
    z = np.r_[p[:,0],p[:,1],r,w]

    def consider(zz, use_lp=False):
        nonlocal best,score
        if time.monotonic() >= deadline:
            return
        pp,rr = unpack(zz)
        if use_lp:
            rr = lp(pp)
        rr = repair(pp,rr,deadline)
        ss = float(np.sum(rr))
        if ss > score and valid(pp,rr):
            best = {'circles':np.column_stack((pp,rr)).tolist()}
            score = ss

    if time.monotonic() < work_end:
        zp = polish(z,float(params['contact_tol']),work_end)
        consider(zp)
    mu = float(params['mu0'])
    for _ in range(int(params['stages'])):
        if time.monotonic() >= work_end:
            break
        try:
            z = minimize(penalty,z,args=(mu,work_end),jac=True,method='L-BFGS-B',
                         options={'maxiter':int(params['maxiter'])}).x
        except Stop:
            break
        mu *= float(params['mu_growth'])
    if time.monotonic() < work_end:
        consider(polish(z,float(params['contact_tol']),work_end))
    if time.monotonic() < work_end:
        consider(z,True)
    if best is None:
        # Deterministic cold-start fallback with an ample geometric margin.
        q = np.arange(N,dtype=float)
        best = {'circles':np.column_stack((q/22,np.zeros(N),np.full(N,.001))).tolist()}
    return best
