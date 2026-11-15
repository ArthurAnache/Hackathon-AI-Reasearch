import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linprog

PARAMS = {
    'aspect': (1.0, 0.5, 2.0, 'log'),
    'rows': (5, 3, 7),
    'jitter': (0.03, 0.0, 0.15),
    'init_jitter': (0.018, 0.0, 0.05),
    'mu0': (10.0, 1.0, 1000.0, 'log'),
    'mu_growth': (8.0, 2.0, 30.0),
    'stages': (6, 2, 12),
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
    if not np.isfinite(p).all() or not np.isfinite(r).all() or not (r > 0).all():
        return False
    d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
    return not (d < r[I]+r[J]).any() and perimeter(p,r) <= 2

def repair(p, r, deadline):
    r = np.maximum(r, 1e-12)*(1-1e-12)
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
                r[i] *= s
                r[j] *= s
    for _ in range(60):
        if time.monotonic() >= deadline:
            break
        excess = perimeter(p,r)-2
        if excess <= 0:
            break
        r *= 1-max(2*float(excess)/float(r.sum()),1e-12)
    return r

def lp_radii(p, remaining):
    m = len(I)
    a = np.zeros((m+4*N+1,N+4))
    b = np.zeros(m+4*N+1)
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
    res = linprog(-np.r_[np.ones(N),np.zeros(4)], A_ub=a,b_ub=b,
                  bounds=[(0,None)]*N+[(None,None)]*4, method='highs',
                  options={'time_limit':max(0.001,remaining),'threads':1})
    return res.x[:N] if res.success else np.zeros(N)

def penalty(z, mu):
    x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
    dx,dy = x[I]-x[J],y[I]-y[J]
    d = np.sqrt(dx*dx+dy*dy+1e-18)
    v = np.maximum(r[I]+r[J]-d,0)
    b = [np.maximum(r-x,0),np.maximum(r+x-w,0),
         np.maximum(r-y,0),np.maximum(r+y-(2-w),0)]
    neg = np.maximum(-r,0)
    f = -r.sum()+mu*((v*v).sum()+sum((t*t).sum() for t in b)+(neg*neg).sum())
    gx,gy,gr = np.zeros(N),np.zeros(N),-np.ones(N)
    c = 2*mu*v
    np.add.at(gr,I,c)
    np.add.at(gr,J,c)
    np.add.at(gx,I,-c*dx/d)
    np.add.at(gx,J,c*dx/d)
    np.add.at(gy,I,-c*dy/d)
    np.add.at(gy,J,c*dy/d)
    gr += 2*mu*(b[0]+b[1]+b[2]+b[3])-2*mu*neg
    gx += 2*mu*(-b[0]+b[1])
    gy += 2*mu*(-b[2]+b[3])
    gw = 2*mu*(-b[1].sum()+b[3].sum())
    return f,np.r_[gx,gy,gr,gw]

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start+0.70*budget_s
    descent_end = start+0.56*budget_s
    rng = np.random.default_rng(seed)
    best,best_sum = None,-np.inf
    candidates = []
    if init is not None:
        a = np.asarray(init['circles'],dtype=float)
        p0,r0 = a[:,:2].copy(),a[:,2].copy()
        if valid(p0,r0):
            best,best_sum = init,float(np.sum(r0))
        lo = (p0-np.abs(r0)[:,None]).min(0)
        side = (p0+np.abs(r0)[:,None]).max(0)-lo
        s = 2/side.sum()
        p,r,w = (p0-lo)*s,np.abs(r0)*s,side[0]*s
        p += rng.normal(0,float(params['init_jitter']),p.shape)
        candidates.append(p.copy())
    else:
        aspect = float(params['aspect'])
        w = 2*aspect/(1+aspect)
        rows = int(params['rows'])
        cols = int(np.ceil(N/rows))
        pts = []
        for k in range(rows):
            if time.monotonic() >= deadline:
                break
            count = min(cols,N-len(pts))
            off = 0.25/cols if k%2 else 0
            pts.extend([((i+0.5)/cols+off,(k+0.5)/rows) for i in range(max(0,count))])
        if len(pts) != N:
            pts = [((i%5+0.5)/5,(i//5+0.5)/5) for i in range(N)]
        p = np.clip(np.array(pts)+rng.normal(0,float(params['jitter']),(N,2)),0.02,0.98)*[w,2-w]
        r = np.full(N,0.05)
    z = np.r_[p[:,0],p[:,1],r,w]
    last = z.copy()
    def callback(zk):
        nonlocal last
        last = zk.copy()
        if time.monotonic() >= descent_end:
            raise Deadline()
    mu = float(params['mu0'])
    for _ in range(int(params['stages'])):
        if time.monotonic() >= descent_end:
            break
        def fun(zk):
            if time.monotonic() >= descent_end:
                raise Deadline()
            return penalty(zk,mu)
        try:
            z = minimize(fun,z,jac=True,method='L-BFGS-B',callback=callback,
                         options={'maxiter':int(params['maxiter'])}).x
            last = z.copy()
        except Deadline:
            z = last
            break
        mu *= float(params['mu_growth'])
    candidates.append(np.stack([z[:N],z[N:2*N]],axis=1))
    for pc in candidates:
        if time.monotonic() >= deadline-0.01:
            break
        rc = repair(pc,lp_radii(pc,deadline-time.monotonic()-0.01),deadline)
        if valid(pc,rc):
            score = float(np.sum(rc))
            if score > best_sum:
                best,best_sum = {'circles':np.column_stack([pc,rc]).tolist()},score
    if best is None:
        # Deterministic strictly feasible cold-start fallback.
        p = np.array([[(i%5)*0.19,(i//5)*0.19] for i in range(N)])
        best = {'circles':np.column_stack([p,np.full(N,0.01)]).tolist()}
    return best
