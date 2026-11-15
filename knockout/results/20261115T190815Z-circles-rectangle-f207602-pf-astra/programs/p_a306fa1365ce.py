import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linprog

PARAMS = {
    'aspect': (1.0, 0.5, 2.0, 'log'),
    'rows': (5, 3, 7),
    'jitter': (0.03, 0.0, 0.15),
    'init_jitter': (0.0, 0.0, 0.05),
    'soft_mu': (0.25, 0.05, 5.0, 'log'),
    'mu0': (10.0, 1.0, 1000.0, 'log'),
    'mu_growth': (8.0, 2.0, 30.0),
    'stages': (6, 2, 12),
    'maxiter': (150, 30, 1000),
}
N = 21
I, J = np.triu_indices(N, 1)

class StopSearch(Exception):
    pass

def perimeter(p, r):
    x = list(map(lambda v: Fraction(float(v)), p[:, 0]))
    y = list(map(lambda v: Fraction(float(v)), p[:, 1]))
    q = list(map(lambda v: Fraction(float(v)), r))
    return (max(a+b for a,b in zip(x,q))-min(a-b for a,b in zip(x,q))
            +max(a+b for a,b in zip(y,q))-min(a-b for a,b in zip(y,q)))

def score(p, r):
    if p.shape != (N,2) or r.shape != (N,):
        return -np.inf
    if not np.isfinite(p).all() or not np.isfinite(r).all() or not (r > 0).all():
        return -np.inf
    d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
    if np.any(d < r[I]+r[J]) or perimeter(p,r) > 2:
        return -np.inf
    return float(np.sum(r))

def penalty(z, mu):
    x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
    dx,dy = x[I]-x[J],y[I]-y[J]
    d = np.sqrt(dx*dx+dy*dy+1e-18)
    v = np.maximum(r[I]+r[J]-d,0)
    b = np.maximum(np.array([r-x,r+x-w,r-y,r+y-(2-w)]),0)
    neg = np.maximum(-r,0)
    f = -r.sum()+mu*(np.sum(v*v)+np.sum(b*b)+np.sum(neg*neg))
    c = 2*mu*v
    gx,gy,gr = np.zeros(N),np.zeros(N),-np.ones(N)
    np.add.at(gr,I,c); np.add.at(gr,J,c)
    np.add.at(gx,I,-c*dx/d); np.add.at(gx,J,c*dx/d)
    np.add.at(gy,I,-c*dy/d); np.add.at(gy,J,c*dy/d)
    gr += 2*mu*b.sum(axis=0)-2*mu*neg
    gx += 2*mu*(-b[0]+b[1])
    gy += 2*mu*(-b[2]+b[3])
    gw = 2*mu*(-b[1].sum()+b[3].sum())
    return f,np.concatenate([gx,gy,gr,[gw]])

def lp_radii(p, deadline):
    if time.monotonic() >= deadline:
        return None
    m = len(I)
    a = np.zeros((m+4*N+1,N+4))
    b = np.zeros(m+4*N+1)
    a[np.arange(m),I] = 1
    a[np.arange(m),J] = 1
    b[:m] = np.sqrt(np.sum((p[I]-p[J])**2,axis=1))
    for k,(col,sgn,coord) in enumerate([(N,1,0),(N+1,-1,0),(N+2,1,1),(N+3,-1,1)]):
        if time.monotonic() >= deadline:
            return None
        rows = m+k*N+np.arange(N)
        a[rows,np.arange(N)] = 1
        a[rows,col] = sgn
        b[rows] = sgn*p[:,coord]
    a[-1,N:] = [-1,1,-1,1]
    b[-1] = 2
    remaining = deadline-time.monotonic()
    if remaining <= 0:
        return None
    res = linprog(-np.r_[np.ones(N),np.zeros(4)],A_ub=a,b_ub=b,
                  bounds=[(0,None)]*N+[(None,None)]*4,method='highs',
                  options={'time_limit':remaining})
    return res.x[:N] if res.success else None

def repair(p,r,deadline):
    r = np.maximum(np.asarray(r,dtype=float),1e-12)*(1-1e-12)
    d = np.sqrt(np.sum((p[I]-p[J])**2,axis=1))
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
            return r
        excess = perimeter(p,r)-2
        if excess <= 0:
            break
        r *= 1-max(2*float(excess)/float(r.sum()),1e-12)
    return r

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start+0.70*budget_s
    opt_end = start+0.53*budget_s
    rng = np.random.default_rng(seed)
    best = {'circles': [[(i+0.5)/21,0.5,0.01] for i in range(N)]}
    ba = np.asarray(best['circles'])
    best_sum = score(ba[:,:2],ba[:,2])
    candidates = []
    if init is not None:
        arr = np.asarray(init['circles'],dtype=float)
        p0,r0 = arr[:,:2].copy(),arr[:,2].copy()
        initial_score = score(p0,r0)
        if initial_score >= best_sum:
            best,best_sum = init,initial_score
        lo = (p0-np.abs(r0)[:,None]).min(axis=0)
        side = (p0+np.abs(r0)[:,None]).max(axis=0)-lo
        s = 2/side.sum()
        p,r,w = (p0-lo)*s,np.abs(r0)*s,side[0]*s
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
                return best
            count = min(per,N-len(pts))
            off = 0.25/per if k%2 else 0
            pts.extend([((i+0.5)/per+off,(k+0.5)/rows) for i in range(count)])
        p = np.clip(np.asarray(pts)+rng.normal(0,float(params['jitter']),(N,2)),0.02,0.98)*[w,2-w]
        r = np.full(N,0.05)
    z = np.r_[p[:,0],p[:,1],r,w]
    last = z.copy()
    def callback(v):
        nonlocal last
        last = v.copy()
        if time.monotonic() >= opt_end:
            raise StopSearch()
    def objective(v,mu):
        if time.monotonic() >= opt_end:
            raise StopSearch()
        return penalty(v,mu)
    # The sole search change: prepend a soft stage to the original schedule.
    schedule = [float(params['soft_mu'])]
    mu = float(params['mu0'])
    for _ in range(int(params['stages'])):
        if time.monotonic() >= opt_end:
            break
        schedule.append(mu)
        mu *= float(params['mu_growth'])
    for mu in schedule:
        if time.monotonic() >= opt_end:
            break
        try:
            z = minimize(objective,z,args=(mu,),jac=True,method='L-BFGS-B',
                         callback=callback,options={'maxiter':int(params['maxiter'])}).x
            last = z.copy()
        except StopSearch:
            z = last
            break
    candidates.insert(0,np.column_stack([z[:N],z[N:2*N]]))
    for pc in candidates:
        if time.monotonic() >= deadline-0.01:
            break
        rc = lp_radii(pc,deadline-0.008)
        if rc is None:
            continue
        rc = repair(pc,rc,deadline)
        value = score(pc,rc)
        if value > best_sum:
            best_sum = value
            best = {'circles':np.column_stack([pc,rc]).tolist()}
    return best
