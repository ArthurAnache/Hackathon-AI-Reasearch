import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linprog

PARAMS = {
    'aspect': (1.0, 0.5, 2.0, 'log'),
    'rows': (5, 3, 7),
    'jitter': (0.03, 0.0, 0.15),
    'init_jitter': (0.0, 0.0, 0.05),
    'mu0': (0.5, 0.1, 1000.0, 'log'),
    'mu_growth': (8.0, 2.0, 30.0),
    'stages': (6, 2, 12),
    'maxiter': (150, 30, 1000),
}
N = 21
I, J = np.triu_indices(N, 1)

class Deadline(Exception):
    pass

def perimeter(p, r):
    xs = [Fraction(float(v)) for v in p[:, 0]]
    ys = [Fraction(float(v)) for v in p[:, 1]]
    rs = [Fraction(float(v)) for v in r]
    return (max(x+q for x,q in zip(xs,rs))-min(x-q for x,q in zip(xs,rs))
            +max(y+q for y,q in zip(ys,rs))-min(y-q for y,q in zip(ys,rs)))

def distances(p):
    return np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)

def valid(p, r):
    return (p.shape == (21,2) and r.shape == (21,) and np.isfinite(p).all()
            and np.isfinite(r).all() and (r>0).all()
            and not (distances(p)<r[I]+r[J]).any() and perimeter(p,r)<=2)

def penalty(z, mu):
    x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
    dx,dy=x[I]-x[J],y[I]-y[J]
    d=np.sqrt(dx*dx+dy*dy+1e-18)
    v=np.maximum(r[I]+r[J]-d,0)
    b=np.array([np.maximum(r-x,0),np.maximum(r+x-w,0),
                np.maximum(r-y,0),np.maximum(r+y-(2-w),0)])
    neg=np.maximum(-r,0)
    f=-r.sum()+mu*((v*v).sum()+(b*b).sum()+(neg*neg).sum())
    c=2*mu*v
    gx=np.bincount(I,-c*dx/d,minlength=N)+np.bincount(J,c*dx/d,minlength=N)
    gy=np.bincount(I,-c*dy/d,minlength=N)+np.bincount(J,c*dy/d,minlength=N)
    gr=-np.ones(N)+np.bincount(I,c,minlength=N)+np.bincount(J,c,minlength=N)
    gr+=2*mu*(b.sum(axis=0)-neg)
    gx+=2*mu*(-b[0]+b[1])
    gy+=2*mu*(-b[2]+b[3])
    return f,np.r_[gx,gy,gr,2*mu*(-b[1].sum()+b[3].sum())]

def lp_radii(p, remaining):
    m=len(I)
    a=np.zeros((m+4*N+1,N+4))
    b=np.zeros(m+4*N+1)
    a[np.arange(m),I]=1
    a[np.arange(m),J]=1
    b[:m]=distances(p)
    rows=m+np.arange(4*N).reshape(4,N)
    a[rows,np.arange(N)[None,:]]=1
    a[rows[0],N]=1
    a[rows[1],N+1]=-1
    a[rows[2],N+2]=1
    a[rows[3],N+3]=-1
    b[rows]=np.array([p[:,0],-p[:,0],p[:,1],-p[:,1]])
    a[-1,N:]=[-1,1,-1,1]
    b[-1]=2
    result=linprog(-np.r_[np.ones(N),np.zeros(4)],A_ub=a,b_ub=b,
                   bounds=[(0,None)]*N+[(None,None)]*4,method='highs',
                   options={'time_limit':max(0.001,remaining)})
    return result.x[:N] if result.success else None

def repair(p, r, deadline):
    r=np.maximum(r,1e-12)*(1-1e-12)
    d=distances(p)
    for unused in range(5):
        if time.monotonic()>=deadline:
            return None
        if not (d<r[I]+r[J]).any():
            break
        for i,j,dd in zip(I,J,d):
            if time.monotonic()>=deadline:
                return None
            if r[i]+r[j]>dd*(1-1e-12):
                s=dd*(1-1e-12)/(r[i]+r[j])
                r[i]*=s
                r[j]*=s
    for unused in range(60):
        if time.monotonic()>=deadline:
            return None
        excess=perimeter(p,r)-2
        if excess<=0:
            return r
        r*=1-max(2*float(excess)/float(r.sum()),1e-12)
    return r

def construct(params, seed, budget_s, init=None):
    start=time.monotonic()
    deadline=start+0.70*budget_s
    optimize_end=start+0.57*budget_s
    rng=np.random.default_rng(seed)
    best=None
    best_score=-np.inf
    if init is not None:
        a=np.asarray(init['circles'],dtype=float)
        p0,r0=a[:,:2].copy(),a[:,2].copy()
        if valid(p0,r0):
            best=init
            best_score=float(r0.sum())
        lo=(p0-np.abs(r0)[:,None]).min(axis=0)
        side=(p0+np.abs(r0)[:,None]).max(axis=0)-lo
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
        cols=int(np.ceil(N/rows))
        points=[]
        for k in range(rows):
            if time.monotonic()>=deadline:
                break
            for j in range(cols):
                if time.monotonic()>=deadline or len(points)>=N:
                    break
                points.append(((j+0.5)/cols+(0.25/cols if k%2 else 0),(k+0.5)/rows))
        if len(points)<N:
            p=np.column_stack((np.linspace(0.02,1.98,N),np.zeros(N)))
            r=np.full(N,1e-5)
            return {'circles':np.column_stack((p,r)).tolist()}
        p=np.clip(np.asarray(points)+rng.normal(0,float(params['jitter']),(N,2)),0.02,0.98)*[w,2-w]
        r=np.full(N,0.05)
        candidates=[]
    z=np.r_[p[:,0],p[:,1],r,w]
    latest=z.copy()
    def callback(zk):
        nonlocal latest
        latest=zk.copy()
        if time.monotonic()>=optimize_end:
            raise Deadline()
    def objective(zk,mu):
        if time.monotonic()>=optimize_end:
            raise Deadline()
        return penalty(zk,mu)
    mu=float(params['mu0'])
    for unused in range(int(params['stages'])):
        if time.monotonic()>=optimize_end:
            break
        try:
            result=minimize(objective,z,args=(mu,),jac=True,method='L-BFGS-B',
                            callback=callback,options={'maxiter':int(params['maxiter'])})
            z=result.x
            latest=z.copy()
        except Deadline:
            z=latest
            break
        mu*=float(params['mu_growth'])
    candidates.append(np.column_stack((z[:N],z[N:2*N])))
    for pc in candidates:
        remaining=deadline-time.monotonic()
        if remaining<0.025:
            break
        rc=lp_radii(pc,remaining*0.65)
        if rc is None:
            continue
        rc=repair(pc,rc,deadline)
        if rc is not None and valid(pc,rc):
            score=float(np.sum(rc))
            if score>best_score:
                best_score=score
                best={'circles':np.column_stack((pc,rc)).tolist()}
    if best is not None:
        return best
    p=np.column_stack((np.linspace(0.02,1.98,N),np.zeros(N)))
    return {'circles':np.column_stack((p,np.full(N,1e-5))).tolist()}
