import time
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize

N = 21
PARAMS = {
    'aspect': (1.0, 0.5, 2.0, 'log'),
    'rows': (5, 3, 7),
    'jitter': (0.03, 0.0, 0.15),
    'init_jitter': (0.065, 0.0, 0.15),
    'mu0': (10.0, 1.0, 1000.0, 'log'),
    'mu_growth': (8.0, 2.0, 30.0),
    'stages': (6, 2, 12),
    'maxiter': (150, 30, 1000),
}
I, J = np.triu_indices(N, 1)

class Deadline(Exception):
    pass

def valid(p, r):
    if p.shape != (N, 2) or r.shape != (N,):
        return False
    if not np.isfinite(p).all() or not np.isfinite(r).all() or not (r > 0).all():
        return False
    d = np.sqrt((p[I, 0]-p[J, 0])**2 + (p[I, 1]-p[J, 1])**2)
    if (d < r[I]+r[J]).any():
        return False
    xs = list(map(lambda x: Fraction(float(x)), p[:, 0]))
    ys = list(map(lambda x: Fraction(float(x)), p[:, 1]))
    rs = list(map(lambda x: Fraction(float(x)), r))
    perimeter = max(x+q for x,q in zip(xs,rs))-min(x-q for x,q in zip(xs,rs))
    perimeter += max(y+q for y,q in zip(ys,rs))-min(y-q for y,q in zip(ys,rs))
    return perimeter <= 2

def penalty(z, mu):
    x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
    dx,dy = x[I]-x[J],y[I]-y[J]
    d = np.sqrt(dx*dx+dy*dy+1e-18)
    v = np.maximum(r[I]+r[J]-d,0)
    b = [np.maximum(r-x,0),np.maximum(r+x-w,0),np.maximum(r-y,0),np.maximum(r+y-(2-w),0)]
    neg = np.maximum(-r,0)
    f = -r.sum()+mu*((v*v).sum()+sum((a*a).sum() for a in b)+(neg*neg).sum())
    gx,gy,gr = np.zeros(N),np.zeros(N),-np.ones(N)
    c = 2*mu*v
    np.add.at(gr,I,c); np.add.at(gr,J,c)
    np.add.at(gx,I,-c*dx/d); np.add.at(gx,J,c*dx/d)
    np.add.at(gy,I,-c*dy/d); np.add.at(gy,J,c*dy/d)
    gr += 2*mu*(b[0]+b[1]+b[2]+b[3])-2*mu*neg
    gx += 2*mu*(-b[0]+b[1])
    gy += 2*mu*(-b[2]+b[3])
    gw = 2*mu*(-b[1].sum()+b[3].sum())
    return f,np.r_[gx,gy,gr,gw]

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
    res = linprog(-np.r_[np.ones(N),np.zeros(4)], A_ub=a, b_ub=b,
                  bounds=[(0,None)]*N+[(None,None)]*4, method='highs',
                  options={'time_limit': max(0.001,remaining)})
    return res.x[:N] if res.success else None

def repair(p, r, deadline):
    r = np.maximum(np.asarray(r,dtype=float),1e-12)*(1-1e-12)
    for attempt in range(65):
        if time.monotonic() >= deadline:
            return None
        if valid(p,r):
            return r
        d = np.sqrt(((p[I]-p[J])**2).sum(1))
        ratios = d/(r[I]+r[J])
        factor = min(1-1e-12,float(ratios.min())*(1-1e-12))
        if attempt >= 5:
            factor = min(factor,1-1e-9*(2**min(attempt-5,20)))
        r *= max(factor,0.5)
    return None

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start+0.70*budget_s
    search_end = start+0.58*budget_s
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
        p,r,w = (p0-lo)*scale,np.abs(r0)*scale,side[0]*scale
        p += rng.normal(0,float(params['init_jitter']),p.shape)
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
            pts.extend([((j+0.5)/cols+(0.25/cols if k%2 else 0),(k+0.5)/rows) for j in range(count)])
        p = np.asarray(pts[:N])
        if len(p) != N:
            p = rng.uniform(0.1,0.9,(N,2))
        p = np.clip(p+rng.normal(0,float(params['jitter']),p.shape),0.02,0.98)*[w,2-w]
        r = np.full(N,0.05)
        fallback_p = np.column_stack((np.linspace(0.01,0.99,N),np.full(N,0.5)))
        best = {'circles':np.column_stack((fallback_p,np.full(N,0.001))).tolist()}
        best_score = 0.021
    z = np.r_[p[:,0],p[:,1],r,w]
    last = z.copy()
    mu = float(params['mu0'])
    def objective(v, strength):
        nonlocal last
        if time.monotonic() >= search_end:
            raise Deadline()
        last = v.copy()
        return penalty(v,strength)
    for stage in range(int(params['stages'])):
        if time.monotonic() >= search_end:
            break
        try:
            z = minimize(objective,z,args=(mu,),jac=True,method='L-BFGS-B',
                         options={'maxiter':int(params['maxiter'])}).x
        except Deadline:
            z = last
            break
        mu *= float(params['mu_growth'])
    pc = np.column_stack((z[:N],z[N:2*N]))
    if time.monotonic() < deadline-0.015:
        rc = lp_radii(pc,deadline-time.monotonic()-0.01)
        if rc is not None:
            rc = repair(pc,rc,deadline)
            if rc is not None and valid(pc,rc):
                score = float(np.sum(rc))
                if score > best_score:
                    best = {'circles':np.column_stack((pc,rc)).tolist()}
    return best
