import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linprog

PARAMS = {
    'aspect': (1.0, 0.5, 2.0, 'log'),
    'rows': (5, 3, 7),
    'jitter': (0.03, 0.0, 0.15),
    'init_jitter': (0.035, 0.0, 0.08),
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
    x = list(map(Fraction, map(float, p[:, 0])))
    y = list(map(Fraction, map(float, p[:, 1])))
    q = list(map(Fraction, map(float, r)))
    return (max(a+b for a,b in zip(x,q))-min(a-b for a,b in zip(x,q))
            +max(a+b for a,b in zip(y,q))-min(a-b for a,b in zip(y,q)))


def valid(p, r):
    if not np.isfinite(p).all() or not np.isfinite(r).all() or not (r > 0).all():
        return False
    d = np.sqrt((p[I,0]-p[J,0])**2 + (p[I,1]-p[J,1])**2)
    return not (d < r[I]+r[J]).any() and perimeter(p,r) <= 2


def penalty(z, mu):
    x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
    dx,dy = x[I]-x[J],y[I]-y[J]
    d = np.sqrt(dx*dx+dy*dy+1e-18)
    v = np.maximum(r[I]+r[J]-d,0)
    b = np.maximum(np.array([r-x,r+x-w,r-y,r+y-(2-w)]),0)
    neg = np.maximum(-r,0)
    f = -r.sum()+mu*((v*v).sum()+(b*b).sum()+(neg*neg).sum())
    gx,gy,gr = np.zeros(N),np.zeros(N),-np.ones(N)
    c = 2*mu*v
    np.add.at(gr,I,c)
    np.add.at(gr,J,c)
    np.add.at(gx,I,-c*dx/d)
    np.add.at(gx,J,c*dx/d)
    np.add.at(gy,I,-c*dy/d)
    np.add.at(gy,J,c*dy/d)
    gx += 2*mu*(-b[0]+b[1])
    gy += 2*mu*(-b[2]+b[3])
    gr += 2*mu*(b.sum(axis=0)-neg)
    gw = 2*mu*(-b[1].sum()+b[3].sum())
    return f,np.r_[gx,gy,gr,gw]


def lp_radii(p, remaining):
    m = len(I)
    a = np.zeros((m+4*N+1,N+4))
    b = np.zeros(m+4*N+1)
    a[np.arange(m),I] = 1
    a[np.arange(m),J] = 1
    b[:m] = np.sqrt(((p[I]-p[J])**2).sum(axis=1))
    for k,(col,sign,coord) in enumerate(((N,1,0),(N+1,-1,0),(N+2,1,1),(N+3,-1,1))):
        if time.monotonic() >= remaining[0]:
            return None
        rows = m+k*N+np.arange(N)
        a[rows,np.arange(N)] = 1
        a[rows,col] = sign
        b[rows] = sign*p[:,coord]
    a[-1,N:] = [-1,1,-1,1]
    b[-1] = 2
    left = remaining[0]-time.monotonic()
    if left <= 0.01:
        return None
    res = linprog(-np.r_[np.ones(N),np.zeros(4)], A_ub=a,b_ub=b,
                  bounds=[(0,None)]*N+[(None,None)]*4,method='highs',
                  options={'time_limit': max(0.001,left)})
    return res.x[:N] if res.success else None


def repair(p, r, deadline):
    r = np.maximum(r,1e-12)*(1-1e-12)
    d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
    for unused in range(5):
        if time.monotonic() >= deadline:
            return None
        bad = np.flatnonzero(r[I]+r[J] > d*(1-1e-12))
        if len(bad) == 0:
            break
        for k in bad:
            if time.monotonic() >= deadline:
                return None
            i,j = I[k],J[k]
            factor = min(1.0,d[k]*(1-1e-12)/(r[i]+r[j]))
            r[i] *= factor
            r[j] *= factor
    for unused in range(60):
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
    opt_end = start+0.56*budget_s
    rng = np.random.default_rng(seed)
    best,score = init,-np.inf
    if init is not None:
        a = np.asarray(init['circles'],dtype=float)
        p0,r0 = a[:,:2].copy(),a[:,2].copy()
        if valid(p0,r0):
            score = float(r0.sum())
        lo = (p0-np.abs(r0)[:,None]).min(axis=0)
        side = (p0+np.abs(r0)[:,None]).max(axis=0)-lo
        scale = 2/side.sum()
        p,r,w = (p0-lo)*scale,np.abs(r0)*scale,side[0]*scale
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
            if count <= 0:
                break
            off = 0.25/cols if k%2 else 0
            pts.extend([((i+0.5)/cols+off,(k+0.5)/rows) for i in range(count)])
        if len(pts) != N:
            pts = [(i/22+1/44,0.5) for i in range(N)]
        p = np.clip(np.array(pts)+rng.normal(0,float(params['jitter']),(N,2)),0.02,0.98)
        p *= [w,2-w]
        r = np.full(N,0.05)
        candidates = []
        fallback_p = np.column_stack((np.arange(N)/11,np.full(N,0.05)))
        best = {'circles': np.column_stack((fallback_p,np.full(N,0.01))).tolist()}
        score = 0.21
    z = np.r_[p[:,0],p[:,1],r,w]
    latest = z.copy()
    def objective(v,mu):
        nonlocal latest
        if time.monotonic() >= opt_end:
            raise Deadline()
        latest = v.copy()
        return penalty(v,mu)
    mu = float(params['mu0'])
    for unused in range(int(params['stages'])):
        if time.monotonic() >= opt_end:
            break
        try:
            result = minimize(objective,z,args=(mu,),jac=True,method='L-BFGS-B',
                              options={'maxiter':int(params['maxiter'])})
            z = result.x
        except Deadline:
            z = latest
            break
        mu *= float(params['mu_growth'])
    candidates.append(np.column_stack((z[:N],z[N:2*N])))
    for pc in candidates[::-1]:
        if time.monotonic() >= deadline-0.015:
            break
        rc = lp_radii(pc,(deadline-0.008,))
        if rc is None:
            continue
        rc = repair(pc,rc,deadline)
        if rc is not None and float(rc.sum()) > score and valid(pc,rc):
            best = {'circles':np.column_stack((pc,rc)).tolist()}
            score = float(rc.sum())
    return best
