import time
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize

N = 21
PARAMS = {
    'aspect': (1.0, 0.5, 2.0, 'log'),
    'rows': (5, 3, 7),
    'jitter': (0.03, 0.0, 0.15),
    'init_jitter': (0.0, 0.0, 0.05),
    'init_shear': (0.12, -0.3, 0.3),
    'mu0': (10.0, 1.0, 1000.0, 'log'),
    'mu_growth': (8.0, 2.0, 30.0),
    'stages': (6, 2, 12),
    'maxiter': (150, 30, 1000),
}
I, J = np.triu_indices(N, 1)

class StopSearch(Exception):
    pass

def perimeter(p, r):
    x = [Fraction(float(v)) for v in p[:, 0]]
    y = [Fraction(float(v)) for v in p[:, 1]]
    q = [Fraction(float(v)) for v in r]
    return (max(a+b for a,b in zip(x,q)) - min(a-b for a,b in zip(x,q))
            + max(a+b for a,b in zip(y,q)) - min(a-b for a,b in zip(y,q)))

def valid(p, r):
    if p.shape != (N,2) or r.shape != (N,):
        return False
    if not np.isfinite(p).all() or not np.isfinite(r).all() or not (r > 0).all():
        return False
    d = np.sqrt((p[I,0]-p[J,0])**2 + (p[I,1]-p[J,1])**2)
    return not (d < r[I]+r[J]).any() and perimeter(p,r) <= 2

def output(p,r):
    return {'circles': np.column_stack((p,r)).tolist()}

def repair(p, r, deadline):
    r = np.maximum(r, 1e-12) * (1-1e-12)
    d = np.sqrt((p[I,0]-p[J,0])**2 + (p[I,1]-p[J,1])**2)
    for k in range(5):
        if time.monotonic() >= deadline:
            return None
        if not (d < r[I]+r[J]).any():
            break
        scale = min(1.0, float(np.min(d / (r[I]+r[J]))))
        r *= scale * (1-1e-12)
    for k in range(60):
        if time.monotonic() >= deadline:
            return None
        excess = perimeter(p,r)-2
        if excess <= 0:
            return r
        r *= 1-max(2*float(excess)/float(r.sum()),1e-12)
    return r

def lp_radii(p, deadline):
    remaining = deadline-time.monotonic()
    if remaining < 0.02:
        return None
    m = len(I)
    a = np.zeros((m+4*N+1,N+4))
    b = np.zeros(m+4*N+1)
    a[np.arange(m),I] = 1
    a[np.arange(m),J] = 1
    b[:m] = np.sqrt(((p[I]-p[J])**2).sum(1))
    for k,(col,sign,coord) in enumerate(((N,1,0),(N+1,-1,0),(N+2,1,1),(N+3,-1,1))):
        if time.monotonic() >= deadline:
            return None
        rows = m+k*N+np.arange(N)
        a[rows,np.arange(N)] = 1
        a[rows,col] = sign
        b[rows] = sign*p[:,coord]
    a[-1,N:] = [-1,1,-1,1]
    b[-1] = 2
    res = linprog(-np.r_[np.ones(N),np.zeros(4)],A_ub=a,b_ub=b,
                  bounds=[(0,None)]*N+[(None,None)]*4,method='highs',
                  options={'time_limit': max(0.001,deadline-time.monotonic())})
    return res.x[:N] if res.success else None

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start+0.70*budget_s
    optimize_end = start+0.57*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf
    if init is not None:
        arr = np.asarray(init['circles'],dtype=float)
        p0,r0 = arr[:,:2].copy(),arr[:,2].copy()
        if valid(p0,r0):
            best_score = float(np.sum(r0))
        lo = (p0-np.abs(r0)[:,None]).min(0)
        side = (p0+np.abs(r0)[:,None]).max(0)-lo
        scale = 2/side.sum()
        p = (p0-lo)*scale
        r = np.abs(r0)*scale
        w = side[0]*scale
        p += rng.normal(0,float(params['init_jitter']),p.shape)
        p[:,0] += float(params['init_shear'])*(p[:,1]-(2-w)/2)
    else:
        aspect = float(params['aspect'])
        w = 2*aspect/(1+aspect)
        rows = int(params['rows'])
        cols = int(np.ceil(N/rows))
        idx = np.arange(N)
        p = np.column_stack(((idx%cols+0.5)/cols,(idx//cols+0.5)/rows))
        p += rng.normal(0,float(params['jitter']),p.shape)
        p = np.clip(p,0.02,0.98)*[w,2-w]
        r = np.full(N,0.05)
        fallback = np.column_stack((0.03+0.09*idx,np.zeros(N)))
        best = output(fallback,np.full(N,0.001))
        best_score = 0.021
    z = np.r_[p[:,0],p[:,1],r,w]
    last = z.copy()
    mu = float(params['mu0'])
    def penalty(v):
        nonlocal last
        if time.monotonic() >= optimize_end:
            raise StopSearch
        last = v.copy()
        x,y,r,w = v[:N],v[N:2*N],v[2*N:3*N],v[-1]
        dx,dy = x[I]-x[J],y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-18)
        ov = np.maximum(r[I]+r[J]-d,0)
        b = np.maximum(np.array([r-x,r+x-w,r-y,r+y-(2-w)]),0)
        neg = np.maximum(-r,0)
        f = -r.sum()+mu*((ov*ov).sum()+(b*b).sum()+(neg*neg).sum())
        gx,gy,gr = np.zeros(N),np.zeros(N),-np.ones(N)
        c = 2*mu*ov
        np.add.at(gr,I,c)
        np.add.at(gr,J,c)
        np.add.at(gx,I,-c*dx/d)
        np.add.at(gx,J,c*dx/d)
        np.add.at(gy,I,-c*dy/d)
        np.add.at(gy,J,c*dy/d)
        gr += 2*mu*(b.sum(axis=0)-neg)
        gx += 2*mu*(-b[0]+b[1])
        gy += 2*mu*(-b[2]+b[3])
        gw = 2*mu*(-b[1].sum()+b[3].sum())
        return f,np.r_[gx,gy,gr,gw]
    for stage in range(int(params['stages'])):
        if time.monotonic() >= optimize_end:
            break
        try:
            z = minimize(penalty,z,jac=True,method='L-BFGS-B',
                         options={'maxiter':int(params['maxiter'])}).x
            last = z.copy()
        except StopSearch:
            z = last
            break
        mu *= float(params['mu_growth'])
    if time.monotonic() < deadline:
        pc = np.column_stack((z[:N],z[N:2*N]))
        rc = lp_radii(pc,deadline)
        if rc is not None:
            rc = repair(pc,rc,deadline)
            if rc is not None and valid(pc,rc):
                score = float(np.sum(rc))
                if score > best_score:
                    best = output(pc,rc)
    return best
