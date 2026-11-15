import time
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize

N = 21
PARAMS = {
    'aspect': (1.0, 0.5, 2.0, 'log'),
    'rows': (5, 3, 7),
    'jitter': (0.03, 0.0, 0.15),
    'init_jitter': (0.11, 0.0, 0.15),
    'mu0': (10.0, 1.0, 1000.0, 'log'),
    'mu_growth': (8.0, 2.0, 30.0),
    'stages': (6, 2, 12),
    'maxiter': (150, 30, 1000),
}
I, J = np.triu_indices(N, 1)

class StopSearch(Exception):
    pass

def perimeter(p, r):
    xs = list(map(Fraction, map(float, p[:, 0])))
    ys = list(map(Fraction, map(float, p[:, 1])))
    rs = list(map(Fraction, map(float, r)))
    return (max(x+t for x,t in zip(xs,rs))-min(x-t for x,t in zip(xs,rs))
            +max(y+t for y,t in zip(ys,rs))-min(y-t for y,t in zip(ys,rs)))

def valid(p, r):
    if p.shape != (N,2) or r.shape != (N,):
        return False
    if not np.isfinite(p).all() or not np.isfinite(r).all() or not (r > 0).all():
        return False
    d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
    return not np.any(d < r[I]+r[J]) and perimeter(p,r) <= 2

def output(p,r):
    return {'circles': np.column_stack((p,r)).tolist()}

def penalty(z, mu, deadline):
    if time.monotonic() >= deadline:
        raise StopSearch
    x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
    dx,dy = x[I]-x[J],y[I]-y[J]
    d = np.sqrt(dx*dx+dy*dy+1e-18)
    v = np.maximum(r[I]+r[J]-d,0)
    a = np.maximum(r-x,0)
    b = np.maximum(r+x-w,0)
    c = np.maximum(r-y,0)
    e = np.maximum(r+y-(2-w),0)
    neg = np.maximum(-r,0)
    f = -r.sum()+mu*(v@v+a@a+b@b+c@c+e@e+neg@neg)
    gx,gy,gr = np.zeros(N),np.zeros(N),-np.ones(N)
    q = 2*mu*v
    np.add.at(gr,I,q)
    np.add.at(gr,J,q)
    np.add.at(gx,I,-q*dx/d)
    np.add.at(gx,J,q*dx/d)
    np.add.at(gy,I,-q*dy/d)
    np.add.at(gy,J,q*dy/d)
    gr += 2*mu*(a+b+c+e-neg)
    gx += 2*mu*(-a+b)
    gy += 2*mu*(-c+e)
    return f,np.r_[gx,gy,gr,2*mu*(-b.sum()+e.sum())]

def lp_radii(p, deadline):
    if time.monotonic() >= deadline:
        return None
    m = len(I)
    a = np.zeros((m+4*N+1,N+4))
    b = np.zeros(m+4*N+1)
    a[np.arange(m),I] = 1
    a[np.arange(m),J] = 1
    b[:m] = np.sqrt(((p[I]-p[J])**2).sum(axis=1))
    for k,(col,sign,coord) in enumerate(((N,1,0),(N+1,-1,0),(N+2,1,1),(N+3,-1,1))):
        if time.monotonic() >= deadline:
            return None
        rows = m+k*N+np.arange(N)
        a[rows,np.arange(N)] = 1
        a[rows,col] = sign
        b[rows] = sign*p[:,coord]
    a[-1,N:] = [-1,1,-1,1]
    b[-1] = 2
    remaining = deadline-time.monotonic()
    if remaining <= 0:
        return None
    res = linprog(-np.r_[np.ones(N),np.zeros(4)],A_ub=a,b_ub=b,
                  bounds=[(0,None)]*N+[(None,None)]*4,method='highs',
                  options={'time_limit': max(0.001,remaining)})
    return res.x[:N] if res.success else None

def repair(p,r,deadline):
    r = np.maximum(np.asarray(r,dtype=float),1e-12)*(1-1e-12)
    d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
    for _ in range(5):
        if time.monotonic() >= deadline:
            return r
        if not np.any(d < r[I]+r[J]):
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

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start+0.70*budget_s
    descent_end = start+0.58*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -float('inf')
    candidates = []
    if init is not None:
        ar = np.asarray(init['circles'],dtype=float)
        p0,r0 = ar[:,:2].copy(),ar[:,2].copy()
        if valid(p0,r0):
            best_score = float(np.sum(r0))
        lo = (p0-np.abs(r0)[:,None]).min(axis=0)
        side = (p0+np.abs(r0)[:,None]).max(axis=0)-lo
        scale = 2/side.sum()
        p = (p0-lo)*scale
        r = np.abs(r0)*scale
        w = side[0]*scale
        p += rng.normal(0,float(params['init_jitter']),p.shape)
        candidates.append(p.copy())
    else:
        aspect = float(params['aspect'])
        w = 2*aspect/(1+aspect)
        rows = int(params['rows'])
        per = int(np.ceil(N/rows))
        pts = []
        for k in range(rows):
            if time.monotonic() >= deadline:
                break
            count = min(per,N-len(pts))
            if count <= 0:
                break
            off = 0.25/per if k%2 else 0
            pts.extend(((i+0.5)/per+off,(k+0.5)/rows) for i in range(count))
        if len(pts) != N:
            pts = [((i%5+0.5)/5,(i//5+0.5)/5) for i in range(N)]
        p = np.clip(np.asarray(pts)+rng.normal(0,float(params['jitter']),(N,2)),0.02,0.98)
        p *= np.array([w,2-w])
        r = np.full(N,0.05)
        fallback_p = np.column_stack((np.arange(N)*0.04,np.zeros(N)))
        best = output(fallback_p,np.full(N,0.001))
        best_score = 0.021
    z = np.r_[p[:,0],p[:,1],r,w]
    mu = float(params['mu0'])
    latest = [z.copy()]
    def objective(v,m):
        result = penalty(v,m,descent_end)
        latest[0] = v.copy()
        return result
    for _ in range(int(params['stages'])):
        if time.monotonic() >= descent_end:
            break
        try:
            z = minimize(objective,z,args=(mu,),jac=True,method='L-BFGS-B',
                         options={'maxiter':int(params['maxiter'])}).x
        except StopSearch:
            z = latest[0]
            break
        mu *= float(params['mu_growth'])
    candidates.append(np.column_stack((z[:N],z[N:2*N])))
    for pc in reversed(candidates):
        if time.monotonic() >= deadline:
            break
        rc = lp_radii(pc,deadline)
        if rc is None:
            continue
        rc = repair(pc,rc,deadline)
        score = float(np.sum(rc))
        if score > best_score and valid(pc,rc):
            best,best_score = output(pc,rc),score
    return best
