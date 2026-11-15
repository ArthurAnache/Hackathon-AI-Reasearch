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

class StopSearch(Exception):
    pass

def half_perimeter(p, r):
    x = list(map(lambda v: Fraction(float(v)), p[:, 0]))
    y = list(map(lambda v: Fraction(float(v)), p[:, 1]))
    q = list(map(lambda v: Fraction(float(v)), r))
    return (max(a+b for a,b in zip(x,q))-min(a-b for a,b in zip(x,q))
            +max(a+b for a,b in zip(y,q))-min(a-b for a,b in zip(y,q)))

def distances(p):
    return np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)

def valid(p, r):
    return (p.shape == (N,2) and r.shape == (N,) and np.isfinite(p).all()
            and np.isfinite(r).all() and (r>0).all()
            and not (distances(p)<r[I]+r[J]).any() and half_perimeter(p,r)<=2)

def penalty(z, mu, deadline):
    if time.monotonic() >= deadline:
        raise StopSearch
    x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
    dx,dy = x[I]-x[J],y[I]-y[J]
    d = np.sqrt(dx*dx+dy*dy+1e-18)
    v = np.maximum(r[I]+r[J]-d,0)
    b = [np.maximum(r-x,0),np.maximum(r+x-w,0),
         np.maximum(r-y,0),np.maximum(r+y-(2-w),0)]
    neg = np.maximum(-r,0)
    f = -r.sum()+mu*((v*v).sum()+sum((bb*bb).sum() for bb in b)+(neg*neg).sum())
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

def lp_radii(p, deadline):
    if time.monotonic() >= deadline:
        return None
    m = len(I)
    a = np.zeros((m+4*N+1,N+4))
    b = np.zeros(m+4*N+1)
    a[np.arange(m),I] = 1
    a[np.arange(m),J] = 1
    b[:m] = distances(p)
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
    if remaining <= 0:
        return None
    result = linprog(-np.r_[np.ones(N),np.zeros(4)], A_ub=a,b_ub=b,
                     bounds=[(0,None)]*N+[(None,None)]*4,method='highs',
                     options={'time_limit':max(0.001,remaining)})
    return result.x[:N] if result.success else None

def repair(p,r,deadline):
    r = np.maximum(np.asarray(r,dtype=float),1e-12)*(1-1e-12)
    d = distances(p)
    for _ in range(5):
        if time.monotonic() >= deadline:
            return r
        if not (d<r[I]+r[J]).any():
            break
        for i,j,dd in zip(I,J,d):
            if time.monotonic() >= deadline:
                return r
            if r[i]+r[j]>dd*(1-1e-12):
                s = dd*(1-1e-12)/(r[i]+r[j])
                r[i] *= s
                r[j] *= s
    for _ in range(60):
        if time.monotonic() >= deadline:
            break
        excess = half_perimeter(p,r)-2
        if excess<=0:
            break
        r *= 1-max(2*float(excess)/float(r.sum()),1e-12)
    return np.maximum(r,1e-12)

def construct(params,seed,budget_s,init=None):
    start = time.monotonic()
    deadline = start+0.70*budget_s
    optimize_deadline = start+0.53*budget_s
    rng = np.random.default_rng(seed)
    best,best_score = None,-np.inf
    if init is not None:
        arr = np.asarray(init['circles'],dtype=float)
        p0,r0 = arr[:,:2].copy(),arr[:,2].copy()
        if valid(p0,r0):
            best,best_score = init,float(np.sum(r0))
        lo = (p0-np.abs(r0)[:,None]).min(0)
        side = (p0+np.abs(r0)[:,None]).max(0)-lo
        s = 2/side.sum()
        p,r,w = (p0-lo)*s,np.abs(r0)*s,side[0]*s
        p += rng.normal(0,float(params['init_jitter']),p.shape)
        candidates = [p.copy()]
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
            if count<=0:
                break
            offset = 0.25/cols if k%2 else 0
            pts.extend([((i+0.5)/cols+offset,(k+0.5)/rows) for i in range(count)])
        if len(pts)!=N:
            p = np.column_stack((np.linspace(0.01,1.99,N),np.zeros(N)))
            return {'circles':np.column_stack((p,np.full(N,1e-5))).tolist()}
        p = np.clip(np.asarray(pts)+rng.normal(0,float(params['jitter']),(N,2)),0.02,0.98)*[w,2-w]
        r = np.full(N,0.05)
        candidates = []
    z = np.r_[p[:,0],p[:,1],r,w]
    mu = float(params['mu0'])
    for _ in range(int(params['stages'])):
        if time.monotonic() >= optimize_deadline:
            break
        try:
            result = minimize(penalty,z,args=(mu,optimize_deadline),jac=True,method='L-BFGS-B',
                              options={'maxiter':int(params['maxiter'])})
            z = result.x
        except StopSearch:
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
        if valid(pc,rc):
            score = float(np.sum(rc))
            if score>best_score:
                best,best_score = {'circles':np.column_stack((pc,rc)).tolist()},score
    if best is None:
        p = np.column_stack((np.linspace(0.01,1.99,N),np.zeros(N)))
        best = {'circles':np.column_stack((p,np.full(N,1e-5))).tolist()}
    return best
