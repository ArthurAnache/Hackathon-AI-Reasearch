import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linprog

N = 21
PARAMS = {
    'aspect': (1.0, 0.5, 2.0, 'log'),
    'rows': (5, 3, 7),
    'jitter': (0.03, 0.0, 0.15),
    'init_jitter': (0.0, 0.0, 0.05),
    'mu0': (1.0, 1.0, 1000.0, 'log'),
    'mu_growth': (8.0, 2.0, 30.0),
    'stages': (6, 2, 12),
    'maxiter': (150, 30, 1000),
}
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
    if p.shape != (N, 2) or r.shape != (N,):
        return False
    if not np.isfinite(p).all() or not np.isfinite(r).all() or not (r > 0).all():
        return False
    d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
    return not (d < r[I]+r[J]).any() and perimeter(p,r) <= 2


def output(p,r):
    return {'circles': np.column_stack((p,r)).tolist()}


def penalty(z, mu):
    x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
    dx,dy=x[I]-x[J],y[I]-y[J]
    d=np.sqrt(dx*dx+dy*dy+1e-18)
    v=np.maximum(r[I]+r[J]-d,0)
    b=np.array([np.maximum(r-x,0),np.maximum(r+x-w,0),
                np.maximum(r-y,0),np.maximum(r+y-(2-w),0)])
    neg=np.maximum(-r,0)
    f=-r.sum()+mu*((v*v).sum()+(b*b).sum()+(neg*neg).sum())
    gx=np.zeros(N); gy=np.zeros(N); gr=-np.ones(N)
    c=2*mu*v
    np.add.at(gr,I,c); np.add.at(gr,J,c)
    np.add.at(gx,I,-c*dx/d); np.add.at(gx,J,c*dx/d)
    np.add.at(gy,I,-c*dy/d); np.add.at(gy,J,c*dy/d)
    gr+=2*mu*(b.sum(axis=0)-neg)
    gx+=2*mu*(-b[0]+b[1]); gy+=2*mu*(-b[2]+b[3])
    gw=2*mu*(-b[1].sum()+b[3].sum())
    return f,np.r_[gx,gy,gr,gw]


def lp_radii(p, remaining):
    m=len(I)
    a=np.zeros((m+4*N+1,N+4)); b=np.zeros(len(a))
    a[np.arange(m),I]=1; a[np.arange(m),J]=1
    b[:m]=np.sqrt(((p[I]-p[J])**2).sum(axis=1))
    for k,(col,sign,coord) in enumerate([(N,1,0),(N+1,-1,0),(N+2,1,1),(N+3,-1,1)]):
        if time.monotonic() >= remaining[1]:
            return None
        rows=m+k*N+np.arange(N)
        a[rows,np.arange(N)]=1
        a[rows,col]=sign
        b[rows]=sign*p[:,coord]
    a[-1,N:]=[-1,1,-1,1]; b[-1]=2
    res=linprog(-np.r_[np.ones(N),np.zeros(4)],A_ub=a,b_ub=b,
                bounds=[(0,None)]*N+[(None,None)]*4,method='highs',
                options={'time_limit':max(0.001,remaining[1]-time.monotonic())})
    return res.x[:N] if res.success else None


def repair(p,r,deadline):
    r=np.maximum(np.asarray(r,dtype=float),1e-12)*(1-1e-12)
    d=np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
    for _ in range(5):
        if time.monotonic() >= deadline:
            return None
        if not (d < r[I]+r[J]).any():
            break
        for i,j,dd in zip(I,J,d):
            if time.monotonic() >= deadline:
                return None
            if r[i]+r[j] > dd*(1-1e-12):
                s=dd*(1-1e-12)/(r[i]+r[j])
                r[i]*=s; r[j]*=s
    for _ in range(60):
        if time.monotonic() >= deadline:
            return None
        excess=perimeter(p,r)-2
        if excess <= 0:
            break
        r*=1-max(2*float(excess)/float(r.sum()),1e-12)
    return r


def construct(params,seed,budget_s,init=None):
    start=time.monotonic()
    stop=start+0.70*budget_s
    descent_stop=start+0.55*budget_s
    rng=np.random.default_rng(seed)
    best=None; best_sum=-np.inf
    if init is not None:
        a=np.asarray(init['circles'],dtype=float)
        p0=a[:,:2].copy(); r0=a[:,2].copy()
        if valid(p0,r0):
            best=init; best_sum=float(np.sum(r0))
        lo=(p0-np.abs(r0)[:,None]).min(axis=0)
        side=(p0+np.abs(r0)[:,None]).max(axis=0)-lo
        scale=2/side.sum()
        p=(p0-lo)*scale
        r=np.abs(r0)*scale
        w=side[0]*scale
        p+=rng.normal(0,float(params['init_jitter']),p.shape)
        candidates=[p.copy()]
    else:
        aspect=float(params['aspect']); w=2*aspect/(1+aspect)
        rows=int(params['rows']); per=int(np.ceil(N/rows))
        pts=[]
        for k in range(rows):
            if time.monotonic() >= stop:
                break
            count=min(per,N-len(pts))
            if count <= 0:
                break
            off=0.25/per if k%2 else 0
            pts.extend([((i+0.5)/per+off,(k+0.5)/rows) for i in range(count)])
        if len(pts) != N:
            pts=[((i%5+0.5)/5,(i//5+0.5)/5) for i in range(N)]
        p=np.clip(np.array(pts)+rng.normal(0,float(params['jitter']),(N,2)),0.02,0.98)*[w,2-w]
        r=np.full(N,0.05)
        candidates=[]
        fallback=np.array([[(i%5+0.5)/5,(i//5+0.5)/5,0.01] for i in range(N)])
        best={'circles':fallback.tolist()}; best_sum=float(fallback[:,2].sum())
    z=np.r_[p[:,0],p[:,1],r,w]
    last=[z.copy()]
    mu=float(params['mu0'])
    def callback(zz):
        last[0]=zz.copy()
        if time.monotonic() >= descent_stop:
            raise Deadline
    for _ in range(int(params['stages'])):
        if time.monotonic() >= descent_stop:
            break
        def fun(zz):
            if time.monotonic() >= descent_stop:
                raise Deadline
            return penalty(zz,mu)
        try:
            z=minimize(fun,z,jac=True,method='L-BFGS-B',callback=callback,
                       options={'maxiter':int(params['maxiter'])}).x
            last[0]=z.copy()
        except Deadline:
            z=last[0]
            break
        mu*=float(params['mu_growth'])
    candidates.append(np.column_stack((z[:N],z[N:2*N])))
    candidates.reverse()
    for pc in candidates:
        if time.monotonic() >= stop-0.025:
            break
        rc=lp_radii(pc,(start,stop-0.012))
        if rc is None:
            continue
        rc=repair(pc,rc,stop)
        if rc is not None and valid(pc,rc):
            score=float(np.sum(rc))
            if score > best_sum:
                best=output(pc,rc); best_sum=score
    return best if best is not None else init
