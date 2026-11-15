import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linprog

PARAMS = {
    'aspect': (1.0, 0.5, 2.0, 'log'),
    'rows': (5, 3, 7),
    'jitter': (0.03, 0.0, 0.15),
    'init_jitter': (0.03, 0.0, 0.05),
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
    x = [Fraction(float(v)) for v in p[:, 0]]
    y = [Fraction(float(v)) for v in p[:, 1]]
    q = [Fraction(float(v)) for v in r]
    return (max(a+b for a,b in zip(x,q)) - min(a-b for a,b in zip(x,q))
            + max(a+b for a,b in zip(y,q)) - min(a-b for a,b in zip(y,q)))

def distances(p):
    return np.sqrt((p[I,0]-p[J,0])**2 + (p[I,1]-p[J,1])**2)

def valid(p, r):
    return (p.shape == (N,2) and r.shape == (N,) and np.isfinite(p).all()
            and np.isfinite(r).all() and bool((r > 0).all())
            and not bool((distances(p) < r[I]+r[J]).any())
            and perimeter(p,r) <= 2)

def output(p,r):
    return {'circles': np.column_stack((p,r)).tolist()}

def penalty(z, mu):
    x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
    h = 2-w
    dx,dy = x[I]-x[J],y[I]-y[J]
    d = np.sqrt(dx*dx+dy*dy+1e-18)
    v = np.maximum(r[I]+r[J]-d,0)
    b = [np.maximum(r-x,0),np.maximum(r+x-w,0),
         np.maximum(r-y,0),np.maximum(r+y-h,0)]
    neg = np.maximum(-r,0)
    f = -r.sum()+mu*((v*v).sum()+sum((a*a).sum() for a in b)+(neg*neg).sum())
    gx,gy,gr = np.zeros(N),np.zeros(N),-np.ones(N)
    c = 2*mu*v
    np.add.at(gr,I,c)
    np.add.at(gr,J,c)
    np.add.at(gx,I,-c*dx/d)
    np.add.at(gx,J,c*dx/d)
    np.add.at(gy,I,-c*dy/d)
    np.add.at(gy,J,c*dy/d)
    gr += 2*mu*(b[0]+b[1]+b[2]+b[3]-neg)
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
    b[:m] = distances(p)
    for k,(col,sign,coord) in enumerate(((N,1,0),(N+1,-1,0),(N+2,1,1),(N+3,-1,1))):
        rows = m+k*N+np.arange(N)
        a[rows,np.arange(N)] = 1
        a[rows,col] = sign
        b[rows] = sign*p[:,coord]
    a[-1,N:] = [-1,1,-1,1]
    b[-1] = 2
    res = linprog(-np.r_[np.ones(N),np.zeros(4)], A_ub=a,b_ub=b,
                  bounds=[(0,None)]*N+[(None,None)]*4,method='highs',
                  options={'time_limit': max(0.001,remaining)})
    return res.x[:N] if res.success else None

def repair(p,r,deadline):
    r = np.maximum(np.asarray(r,dtype=float),1e-12)*(1-1e-12)
    d = distances(p)
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

def layout(rows,jitter,w,h,rng,deadline):
    cols = int(np.ceil(N/rows))
    pts = []
    for k in range(rows):
        if time.monotonic() >= deadline:
            break
        count = min(cols,N-len(pts))
        if count <= 0:
            break
        off = 0.25/cols if k%2 else 0.0
        pts.extend(((i+0.5)/cols+off,(k+0.5)/rows) for i in range(count))
    if len(pts) < N:
        return np.column_stack(((np.arange(N)%7+0.5)/7,(np.arange(N)//7+0.5)/3))*[w,h]
    p = np.clip(np.array(pts)+rng.normal(0,jitter,(N,2)),0.02,0.98)
    return p*[w,h]

def construct(params,seed,budget_s,init=None):
    start = time.monotonic()
    deadline = start+0.70*budget_s
    opt_deadline = start+0.53*budget_s
    rng = np.random.default_rng(seed)
    # A guaranteed valid fallback also avoids relying on an interrupted optimizer.
    fallback_p = np.column_stack(((np.arange(N)%7+0.5)/7,(np.arange(N)//7+0.5)/3))
    best = output(fallback_p,np.full(N,0.01))
    best_score = float(np.full(N,0.01).sum())
    if init is not None:
        a = np.asarray(init['circles'],dtype=float)
        p0,r0 = a[:,:2].copy(),a[:,2].copy()
        if valid(p0,r0):
            best,best_score = init,float(r0.sum())
        lo = (p0-np.abs(r0)[:,None]).min(0)
        side = (p0+np.abs(r0)[:,None]).max(0)-lo
        s = 2/side.sum()
        p,r,w = (p0-lo)*s,np.abs(r0)*s,side[0]*s
        p += rng.normal(0,float(params['init_jitter']),p.shape)
        candidates = [(p.copy(),None)]
    else:
        aspect = float(params['aspect'])
        w = 2*aspect/(1+aspect)
        p = layout(int(params['rows']),float(params['jitter']),w,2-w,rng,deadline)
        r = np.full(N,0.05)
        candidates = []
    z = np.r_[p[:,0],p[:,1],r,w]
    last = z.copy()
    def objective(v,mu):
        if time.monotonic() >= opt_deadline:
            raise StopSearch
        return penalty(v,mu)
    def callback(v):
        nonlocal last
        last = v.copy()
        if time.monotonic() >= opt_deadline:
            raise StopSearch
    mu = float(params['mu0'])
    for _ in range(int(params['stages'])):
        if time.monotonic() >= opt_deadline:
            break
        try:
            z = minimize(objective,z,args=(mu,),jac=True,method='L-BFGS-B',callback=callback,
                         options={'maxiter':int(params['maxiter'])}).x
            last = z.copy()
        except StopSearch:
            z = last
            break
        mu *= float(params['mu_growth'])
    pc = np.column_stack((z[:N],z[N:2*N]))
    # Prefer the completed descent when the remaining time permits only one LP.
    candidates.insert(0,(pc,z[2*N:3*N]))
    for pc,raw in candidates:
        if time.monotonic() >= deadline-0.01:
            break
        if raw is not None:
            rr = repair(pc,raw,deadline)
            if rr is not None and valid(pc,rr) and float(rr.sum()) > best_score:
                best,best_score = output(pc,rr),float(rr.sum())
        remaining = deadline-time.monotonic()-0.015
        if remaining <= 0.01:
            break
        rc = lp_radii(pc,remaining)
        if rc is None or time.monotonic() >= deadline:
            continue
        rc = repair(pc,rc,deadline)
        if rc is not None and valid(pc,rc) and float(rc.sum()) > best_score:
            best,best_score = output(pc,rc),float(rc.sum())
    return best
