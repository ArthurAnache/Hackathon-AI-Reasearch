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
    'mu_growth': (5.0, 2.0, 30.0),
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

def valid(p, r):
    if p.shape != (N,2) or r.shape != (N,):
        return False
    if not np.isfinite(p).all() or not np.isfinite(r).all() or not (r > 0).all():
        return False
    d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
    return not (d < r[I]+r[J]).any() and perimeter(p,r) <= 2

def penalty(z, mu):
    x,y,r,w = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
    dx,dy = x[I]-x[J], y[I]-y[J]
    d = np.sqrt(dx*dx+dy*dy+1e-18)
    v = np.maximum(r[I]+r[J]-d,0)
    b = [np.maximum(r-x,0),np.maximum(r+x-w,0),
         np.maximum(r-y,0),np.maximum(r+y-(2-w),0)]
    neg = np.maximum(-r,0)
    f = -r.sum()+mu*((v*v).sum()+sum((t*t).sum() for t in b)+(neg*neg).sum())
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

def lp_radii(p, remaining):
    m = len(I)
    a = np.zeros((m+4*N+1,N+4))
    b = np.zeros(m+4*N+1)
    a[np.arange(m),I]=1
    a[np.arange(m),J]=1
    b[:m]=np.sqrt(((p[I]-p[J])**2).sum(1))
    for k,(col,sign,coord) in enumerate([(N,1,0),(N+1,-1,0),(N+2,1,1),(N+3,-1,1)]):
        if remaining() <= 0:
            return None
        rows=m+k*N+np.arange(N)
        a[rows,np.arange(N)]=1
        a[rows,col]=sign
        b[rows]=sign*p[:,coord]
    a[-1,N:]=[-1,1,-1,1]
    b[-1]=2
    rem=remaining()
    if rem <= 0:
        return None
    res=linprog(-np.r_[np.ones(N),np.zeros(4)],A_ub=a,b_ub=b,
                bounds=[(0,None)]*N+[(None,None)]*4,method='highs',
                options={'time_limit':max(0.001,rem)})
    return res.x[:N] if res.success else None

def repair(p,r,deadline):
    r=np.maximum(np.asarray(r,dtype=float),1e-12)*(1-1e-12)
    d=np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
    for _ in range(5):
        if time.monotonic() >= deadline:
            break
        if not (d < r[I]+r[J]).any():
            break
        for i,j,dd in zip(I,J,d):
            if time.monotonic() >= deadline:
                return r
            if r[i]+r[j] > dd*(1-1e-12):
                s=dd*(1-1e-12)/(r[i]+r[j])
                r[i]*=s
                r[j]*=s
    for _ in range(60):
        if time.monotonic() >= deadline:
            break
        excess=perimeter(p,r)-2
        if excess <= 0:
            break
        r*=1-max(2*float(excess)/float(r.sum()),1e-12)
    return r

def construct(params,seed,budget_s,init=None):
    start=time.monotonic()
    deadline=start+0.70*budget_s
    search_end=start+0.53*budget_s
    rng=np.random.default_rng(seed)
    best=None
    score=-np.inf
    if init is not None:
        ar=np.asarray(init['circles'],dtype=float)
        p0,r0=ar[:,:2].copy(),ar[:,2].copy()
        if valid(p0,r0):
            best,score=init,float(np.sum(r0))
        lo=(p0-np.abs(r0)[:,None]).min(0)
        side=(p0+np.abs(r0)[:,None]).max(0)-lo
        s=2/side.sum()
        p=(p0-lo)*s
        r=np.abs(r0)*s
        w=side[0]*s
        p+=rng.normal(0,float(params['init_jitter']),p.shape)
        candidates=[p.copy()]
    else:
        aspect=float(params['aspect'])
        w=2*aspect/(1+aspect)
        rows=int(params['rows'])
        per_row=int(np.ceil(N/rows))
        pts=[]
        for k in range(rows):
            if time.monotonic() >= deadline:
                break
            count=min(per_row,N-len(pts))
            off=0.25/per_row if k%2 else 0
            pts.extend([((i+0.5)/per_row+off,(k+0.5)/rows) for i in range(count)])
        if len(pts)<N:
            pts.extend(rng.uniform(0.1,0.9,(N-len(pts),2)).tolist())
        p=np.clip(np.array(pts[:N])+rng.normal(0,float(params['jitter']),(N,2)),0.02,0.98)*[w,2-w]
        r=np.full(N,0.05)
        candidates=[]
    z=np.r_[p[:,0],p[:,1],r,w]
    mu=float(params['mu0'])
    for _ in range(int(params['stages'])):
        if time.monotonic() >= search_end:
            break
        def fun(zz):
            if time.monotonic() >= search_end:
                raise StopSearch()
            return penalty(zz,mu)
        try:
            z=minimize(fun,z,jac=True,method='L-BFGS-B',
                       options={'maxiter':int(params['maxiter'])}).x
        except StopSearch:
            break
        mu*=float(params['mu_growth'])
    candidates.append(np.column_stack((z[:N],z[N:2*N])))
    for pc in reversed(candidates):
        if time.monotonic() >= deadline:
            break
        rc=lp_radii(pc,lambda:deadline-time.monotonic())
        if rc is None:
            continue
        rc=repair(pc,rc,deadline)
        if valid(pc,rc) and float(np.sum(rc))>score:
            score=float(np.sum(rc))
            best={'circles':np.column_stack((pc,rc)).tolist()}
    if best is None:
        # Guaranteed valid cold fallback, independent of any unfinished solve.
        best={'circles':[[float(i)/11.0,0.0,0.001] for i in range(N)]}
    return best
