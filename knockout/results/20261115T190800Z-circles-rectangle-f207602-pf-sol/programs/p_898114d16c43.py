import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linprog

PARAMS = {
    'aspect': (1.0, 0.5, 2.0, 'log'),
    'rows': (5, 3, 7),
    'jitter': (0.03, 0.0, 0.15),
    'init_jitter': (0.025, 0.0, 0.15),
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
    if p.shape != (N,2) or r.shape != (N,):
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
    return f,np.r_[gx,gy,gr,2*mu*(-b[1].sum()+b[3].sum())]

def lp_radii(p, remaining):
    m = len(I)
    a = np.zeros((m+4*N+1,N+4))
    b = np.zeros(m+4*N+1)
    a[np.arange(m),I]=1
    a[np.arange(m),J]=1
    b[:m]=np.sqrt(((p[I]-p[J])**2).sum(1))
    for k,(col,sign,coord) in enumerate([(N,1,0),(N+1,-1,0),(N+2,1,1),(N+3,-1,1)]):
        if time.monotonic() >= remaining[0]:
            return None
        rows=m+k*N+np.arange(N)
        a[rows,np.arange(N)]=1
        a[rows,col]=sign
        b[rows]=sign*p[:,coord]
    a[-1,N:]=[-1,1,-1,1]
    b[-1]=2
    left=remaining[0]-time.monotonic()
    if left <= 0.015:
        return None
    res=linprog(-np.r_[np.ones(N),np.zeros(4)],A_ub=a,b_ub=b,
                bounds=[(0,None)]*N+[(None,None)]*4,method='highs',
                options={'time_limit':left})
    return res.x[:N] if res.success else None

def repair(p,r,end):
    if r is None or not (r > 0).all():
        return None
    r=r.copy()*(1-1e-12)
    d=np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
    factor=min(1.0,float(np.min(d/(r[I]+r[J]))))
    r*=factor*(1-1e-14)
    for _ in range(30):
        if time.monotonic() >= end:
            return None
        excess=perimeter(p,r)-2
        if excess <= 0:
            return r
        r*=1-max(2*float(excess)/float(r.sum()),1e-12)
    return None

def symmetric_noise(p, side, rng, amplitude):
    # Infer the reflection axis, including any circles fixed by reflection.
    normalized=p-side/2
    errors=[]
    matches=[]
    for axis in range(2):
        reflected=normalized.copy()
        reflected[:,axis]*=-1
        ds=((reflected[:,None,:]-normalized[None,:,:])**2).sum(2)
        match=np.argmin(ds,axis=1)
        errors.append(float(ds[np.arange(N),match].sum()))
        matches.append(match)
    axis=int(np.argmin(errors))
    match=matches[axis]
    noise=rng.normal(0,amplitude,p.shape)
    mirrored=noise[match].copy()
    mirrored[:,axis]*=-1
    return p+(noise+mirrored)/np.sqrt(2.0)

def construct(params, seed, budget_s, init=None):
    start=time.monotonic()
    end=start+0.70*budget_s
    optimize_end=start+0.55*budget_s
    rng=np.random.default_rng(seed)
    best=init
    best_sum=-np.inf
    if init is not None:
        arr=np.asarray(init['circles'],dtype=float)
        p0,r0=arr[:,:2].copy(),arr[:,2].copy()
        if valid(p0,r0):
            best_sum=float(np.sum(r0))
        lo=(p0-np.abs(r0)[:,None]).min(0)
        side=(p0+np.abs(r0)[:,None]).max(0)-lo
        scale=2/side.sum()
        p=(p0-lo)*scale
        r=np.abs(r0)*scale
        side*=scale
        w=side[0]
        original=p.copy()
        p=symmetric_noise(p,side,rng,float(params['init_jitter']))
        candidates=[original]
    else:
        w=2*float(params['aspect'])/(1+float(params['aspect']))
        rows=int(params['rows'])
        cols=int(np.ceil(N/rows))
        k=np.arange(N)
        p=np.column_stack(((k%cols+0.5)/cols,(k//cols+0.5)/rows))
        p=np.clip(p+rng.normal(0,float(params['jitter']),p.shape),0.02,0.98)
        p*=np.array([w,2-w])
        r=np.full(N,0.05)
        candidates=[]
    z=np.r_[p[:,0],p[:,1],r,w]
    mu=float(params['mu0'])
    latest=z.copy()
    def objective(v, penalty_mu):
        if time.monotonic() >= optimize_end:
            raise Deadline()
        return penalty(v,penalty_mu)
    def callback(v):
        nonlocal latest
        latest=v.copy()
        if time.monotonic() >= optimize_end:
            raise Deadline()
    for _ in range(int(params['stages'])):
        if time.monotonic() >= optimize_end:
            break
        try:
            z=minimize(objective,z,args=(mu,),jac=True,method='L-BFGS-B',
                       callback=callback,options={'maxiter':int(params['maxiter'])}).x
            latest=z.copy()
        except Deadline:
            z=latest
            break
        mu*=float(params['mu_growth'])
    candidates.insert(0,np.column_stack((z[:N],z[N:2*N])))
    for pc in candidates:
        if time.monotonic() >= end:
            break
        rc=repair(pc,lp_radii(pc,(end,)),end)
        if rc is not None and valid(pc,rc):
            score=float(np.sum(rc))
            if score > best_sum:
                best_sum=score
                best={'circles':np.column_stack((pc,rc)).tolist()}
    if best is None:
        # Deterministic valid fallback for a cold start.
        k=np.arange(N,dtype=float)
        best={'circles':np.column_stack((0.04+0.09*(k%7),0.04+0.09*(k//7),np.full(N,0.02))).tolist()}
    return best
