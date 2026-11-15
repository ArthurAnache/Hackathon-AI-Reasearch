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
    'stages': (8, 2, 12),
    'maxiter': (150, 30, 1000),
}
N = 21
I, J = np.triu_indices(N, 1)

class Deadline(Exception):
    pass

def perimeter(p, r):
    x = [Fraction(float(v)) for v in p[:, 0]]
    y = [Fraction(float(v)) for v in p[:, 1]]
    q = [Fraction(float(v)) for v in r]
    return (max(a+b for a,b in zip(x,q))-min(a-b for a,b in zip(x,q))
            +max(a+b for a,b in zip(y,q))-min(a-b for a,b in zip(y,q)))

def valid(p, r):
    if p.shape != (21,2) or r.shape != (21,):
        return False
    if not np.isfinite(p).all() or not np.isfinite(r).all() or not (r > 0).all():
        return False
    d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
    return not (d < r[I]+r[J]).any() and perimeter(p,r) <= 2

def penalty(z, mu):
    x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
    dx,dy = x[I]-x[J],y[I]-y[J]
    d = np.sqrt(dx*dx+dy*dy+1e-18)
    v = np.maximum(r[I]+r[J]-d,0)
    b = [np.maximum(r-x,0),np.maximum(r+x-w,0),
         np.maximum(r-y,0),np.maximum(r+y-(2-w),0)]
    neg = np.maximum(-r,0)
    f = -r.sum()+mu*((v*v).sum()+sum((a*a).sum() for a in b)+(neg*neg).sum())
    gx,gy,gr = np.zeros(N),np.zeros(N),-np.ones(N)
    c = 2*mu*v
    np.add.at(gr,I,c); np.add.at(gr,J,c)
    np.add.at(gx,I,-c*dx/d); np.add.at(gx,J,c*dx/d)
    np.add.at(gy,I,-c*dy/d); np.add.at(gy,J,c*dy/d)
    gr += 2*mu*(b[0]+b[1]+b[2]+b[3]-neg)
    gx += 2*mu*(-b[0]+b[1])
    gy += 2*mu*(-b[2]+b[3])
    gw = 2*mu*(-b[1].sum()+b[3].sum())
    return f,np.r_[gx,gy,gr,gw]

def radii_lp(p, deadline):
    remaining = deadline-time.monotonic()
    if remaining <= 0.005:
        return None
    m = len(I)
    a = np.zeros((m+4*N+1,N+4)); b = np.zeros(m+4*N+1)
    a[np.arange(m),I] = 1; a[np.arange(m),J] = 1
    b[:m] = np.sqrt(((p[I]-p[J])**2).sum(1))
    for k,(col,sign,coord) in enumerate([(N,1,0),(N+1,-1,0),(N+2,1,1),(N+3,-1,1)]):
        if time.monotonic() >= deadline:
            return None
        rows = m+k*N+np.arange(N)
        a[rows,np.arange(N)] = 1
        a[rows,col] = sign
        b[rows] = sign*p[:,coord]
    a[-1,N:] = [-1,1,-1,1]; b[-1] = 2
    result = linprog(-np.r_[np.ones(N),np.zeros(4)], A_ub=a,b_ub=b,
                     bounds=[(0,None)]*N+[(None,None)]*4,method='highs',
                     options={'time_limit': max(0.001,deadline-time.monotonic())})
    return result.x[:N] if result.success else None

def repair(p,r,deadline):
    r = np.maximum(r,1e-12)*(1-1e-12)
    d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
    for _ in range(5):
        if time.monotonic() >= deadline:
            return r
        if not (d < r[I]+r[J]).any():
            break
        for i,j,dd in zip(I,J,d):
            if time.monotonic() >= deadline:
                return r
            if r[i]+r[j] > dd*(1-1e-12):
                s = dd*(1-1e-12)/(r[i]+r[j])
                r[i] *= s; r[j] *= s
    for _ in range(60):
        if time.monotonic() >= deadline:
            break
        excess = perimeter(p,r)-2
        if excess <= 0:
            break
        r *= 1-max(2*float(excess)/float(r.sum()),1e-12)
    return r

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start+0.70*budget_s
    search_end = start+0.56*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf
    if init is not None:
        ar = np.asarray(init['circles'],dtype=float)
        p0,r0 = ar[:,:2].copy(),ar[:,2].copy()
        if valid(p0,r0):
            best_score = float(np.sum(r0))
        lo = (p0-np.abs(r0)[:,None]).min(0)
        side = (p0+np.abs(r0)[:,None]).max(0)-lo
        scale = 2/side.sum()
        p,r,w = (p0-lo)*scale,np.abs(r0)*scale,side[0]*scale
        p += rng.normal(0,float(params['init_jitter']),p.shape)
        initial_p = p.copy()
    else:
        aspect = float(params['aspect']); w = 2*aspect/(1+aspect)
        rows = int(params['rows']); cols = int(np.ceil(N/rows))
        pts = []
        for k in range(rows):
            if time.monotonic() >= deadline:
                break
            count = min(cols,N-len(pts))
            pts.extend([((i+0.5)/cols+(0.25/cols if k%2 else 0),(k+0.5)/rows) for i in range(count)])
        if len(pts) != N:
            p = np.column_stack((np.arange(N)*0.04,np.zeros(N)))
            return {'circles':np.column_stack((p,np.full(N,0.001))).tolist()}
        p = np.clip(np.asarray(pts)+rng.normal(0,float(params['jitter']),(N,2)),0.02,0.98)*[w,2-w]
        r = np.full(N,0.05)
        initial_p = p.copy()
    z = np.r_[p[:,0],p[:,1],r,w]
    mu = float(params['mu0'])
    latest = z.copy()
    def fun(q, strength):
        nonlocal latest
        if time.monotonic() >= search_end:
            raise Deadline
        latest = q.copy()
        return penalty(q,strength)
    for _ in range(int(params['stages'])):
        if time.monotonic() >= search_end:
            break
        try:
            z = minimize(fun,z,args=(mu,),jac=True,method='L-BFGS-B',
                         options={'maxiter':int(params['maxiter'])}).x
        except Deadline:
            z = latest
            break
        mu *= float(params['mu_growth'])
    for pc in [np.column_stack((z[:N],z[N:2*N])),initial_p]:
        if time.monotonic() >= deadline:
            break
        rc = radii_lp(pc,deadline)
        if rc is None:
            continue
        rc = repair(pc,rc,deadline)
        if valid(pc,rc):
            score = float(np.sum(rc))
            if score > best_score:
                best_score = score
                best = {'circles':np.column_stack((pc,rc)).tolist()}
    if best is None:
        p = np.column_stack((np.arange(N)*0.04,np.zeros(N)))
        best = {'circles':np.column_stack((p,np.full(N,0.001))).tolist()}
    return best
