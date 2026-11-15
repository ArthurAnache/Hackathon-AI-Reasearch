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
    'mu_growth': (3.0, 2.0, 30.0),
    'stages': (6, 2, 12),
    'maxiter': (150, 30, 1000),
}
N = 21
I, J = np.triu_indices(N, 1)

class Deadline(Exception):
    pass

def perimeter(p, r):
    xs = list(map(lambda v: Fraction(float(v)), p[:, 0]))
    ys = list(map(lambda v: Fraction(float(v)), p[:, 1]))
    rs = list(map(lambda v: Fraction(float(v)), r))
    return (max(x+q for x,q in zip(xs,rs))-min(x-q for x,q in zip(xs,rs))
            +max(y+q for y,q in zip(ys,rs))-min(y-q for y,q in zip(ys,rs)))

def valid(p, r):
    if not (np.isfinite(p).all() and np.isfinite(r).all() and (r > 0).all()):
        return False
    d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
    return not (d < r[I]+r[J]).any() and perimeter(p,r) <= 2

def penalty(z, mu):
    x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
    dx,dy = x[I]-x[J],y[I]-y[J]
    d = np.sqrt(dx*dx+dy*dy+1e-18)
    v = np.maximum(r[I]+r[J]-d,0)
    b0,b1 = np.maximum(r-x,0),np.maximum(r+x-w,0)
    b2,b3 = np.maximum(r-y,0),np.maximum(r+y-(2-w),0)
    neg = np.maximum(-r,0)
    f = -r.sum()+mu*(v@v+b0@b0+b1@b1+b2@b2+b3@b3+neg@neg)
    c = 2*mu*v
    gx = np.bincount(I,-c*dx/d,N)+np.bincount(J,c*dx/d,N)+2*mu*(-b0+b1)
    gy = np.bincount(I,-c*dy/d,N)+np.bincount(J,c*dy/d,N)+2*mu*(-b2+b3)
    gr = -np.ones(N)+np.bincount(I,c,N)+np.bincount(J,c,N)+2*mu*(b0+b1+b2+b3-neg)
    return f,np.r_[gx,gy,gr,2*mu*(-b1.sum()+b3.sum())]

def lp_radii(p, deadline):
    m = len(I)
    a = np.zeros((m+4*N+1,N+4))
    b = np.zeros(m+4*N+1)
    a[np.arange(m),I] = 1
    a[np.arange(m),J] = 1
    b[:m] = np.sqrt(((p[I]-p[J])**2).sum(1))
    for k,(col,sign,coord) in enumerate([(N,1,0),(N+1,-1,0),(N+2,1,1),(N+3,-1,1)]):
        if time.monotonic() >= deadline:
            return None
        rows = m+k*N+np.arange(N)
        a[rows,np.arange(N)] = 1
        a[rows,col] = sign
        b[rows] = sign*p[:,coord]
    a[-1,N:] = [-1,1,-1,1]
    b[-1] = 2
    remaining = deadline-time.monotonic()
    if remaining <= 0.01:
        return None
    res = linprog(-np.r_[np.ones(N),np.zeros(4)],A_ub=a,b_ub=b,
                  bounds=[(0,None)]*N+[(None,None)]*4,method='highs',
                  options={'time_limit': max(0.001,remaining)})
    return res.x[:N] if res.success else None

def repair(p,r,deadline):
    r = np.maximum(r,1e-12)*(1-1e-12)
    d = np.sqrt(((p[I]-p[J])**2).sum(1))
    for _ in range(5):
        if time.monotonic() >= deadline:
            return None
        bad = np.flatnonzero(d < r[I]+r[J])
        if not len(bad):
            break
        for k in bad:
            if time.monotonic() >= deadline:
                return None
            i,j = I[k],J[k]
            s = min(1.0,d[k]*(1-1e-12)/(r[i]+r[j]))
            r[i] *= s
            r[j] *= s
    for _ in range(60):
        if time.monotonic() >= deadline:
            return None
        excess = perimeter(p,r)-2
        if excess <= 0:
            return r
        r *= 1-max(2*float(excess)/float(r.sum()),1e-12)
    return r

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start+0.70*budget_s
    opt_end = start+0.53*budget_s
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
        candidates = [p.copy()]
    else:
        aspect = float(params['aspect'])
        w = 2*aspect/(1+aspect)
        rows = int(params['rows'])
        count = int(np.ceil(N/rows))
        idx = np.arange(N)
        p = np.column_stack(((idx%count+0.5)/count+(idx//count%2)*0.25/count,
                             (idx//count+0.5)/rows))
        p = np.clip(p+rng.normal(0,float(params['jitter']),p.shape),0.02,0.98)*[w,2-w]
        r = np.full(N,0.05)
        candidates = []
        fallback_p = np.column_stack((np.linspace(0.01,1.99,N),np.zeros(N)))
        best = {'circles': np.column_stack((fallback_p,np.full(N,0.001))).tolist()}
        best_score = 0.021
    z = np.r_[p[:,0],p[:,1],r,w]
    latest = z.copy()
    def callback(x):
        nonlocal latest
        latest = x.copy()
        if time.monotonic() >= opt_end:
            raise Deadline()
    mu = float(params['mu0'])
    for _ in range(int(params['stages'])):
        if time.monotonic() >= opt_end:
            break
        try:
            z = minimize(penalty,z,args=(mu,),jac=True,method='L-BFGS-B',callback=callback,
                         options={'maxiter':int(params['maxiter'])}).x
            latest = z.copy()
        except Deadline:
            z = latest
            break
        mu *= float(params['mu_growth'])
    candidates.append(np.column_stack((z[:N],z[N:2*N])))
    for pc in candidates:
        if time.monotonic() >= deadline:
            break
        rc = lp_radii(pc,deadline)
        if rc is None:
            continue
        rc = repair(pc,rc,deadline)
        if rc is not None and valid(pc,rc):
            score = float(np.sum(rc))
            if score > best_score:
                best_score = score
                best = {'circles':np.column_stack((pc,rc)).tolist()}
    return best
