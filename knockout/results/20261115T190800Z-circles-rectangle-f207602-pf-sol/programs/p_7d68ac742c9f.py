import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'pole_samples': (160, 20, 500),
    'relaxations': (6, 1, 20),
    'iterations': (95, 20, 250),
    'near_radius': (0.65, 0.2, 2.0),
    'far_radius': (2.4, 1.0, 6.0)
}
N = 21
I, J = np.triu_indices(N, 1)

class StopSearch(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0] - a[J, 0])**2 + (a[I, 1] - a[J, 1])**2)
    if np.any(d < a[I, 2] + a[J, 2]):
        return -np.inf
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    q = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
    q += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
    if q > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))

def normalize(a):
    a = a.copy()
    lo = np.min(a[:, :2]-a[:, 2,None], axis=0)
    hi = np.max(a[:, :2]+a[:, 2,None], axis=0)
    scale = 2.0 / np.sum(hi-lo)
    a[:, :2] = (a[:, :2]-lo)*scale
    a[:, 2] *= scale
    return a, (hi[0]-lo[0])*scale

def pack(a, w):
    return np.r_[a[:,0], a[:,1], a[:,2], w]

def unpack(z):
    return np.column_stack((z[:N], z[N:2*N], z[2*N:3*N]))

def constraints(z):
    x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
    dx,dy=x[I]-x[J],y[I]-y[J]
    d=np.sqrt(dx*dx+dy*dy+1e-30)
    return np.r_[d-r[I]-r[J], x-r,w-x-r,y-r,2-w-y-r]

def jacobian(z):
    x,y=z[:N],z[N:2*N]
    dx,dy=x[I]-x[J],y[I]-y[J]
    d=np.sqrt(dx*dx+dy*dy+1e-30)
    m=len(I)
    a=np.zeros((m+4*N,3*N+1))
    k=np.arange(m)
    a[k,I]=dx/d; a[k,J]=-dx/d
    a[k,N+I]=dy/d; a[k,N+J]=-dy/d
    a[k,2*N+I]=-1; a[k,2*N+J]=-1
    t=np.arange(N)
    a[m+t,t]=1; a[m+t,2*N+t]=-1
    a[m+N+t,t]=-1; a[m+N+t,2*N+t]=-1; a[m+N+t,-1]=1
    a[m+2*N+t,N+t]=1; a[m+2*N+t,2*N+t]=-1
    a[m+3*N+t,N+t]=-1; a[m+3*N+t,2*N+t]=-1; a[m+3*N+t,-1]=-1
    return a

def construct(params, seed, budget_s, init=None):
    start=time.monotonic()
    end=start+0.70*budget_s
    rng=np.random.default_rng(seed)
    if init is None:
        base=np.array([[(i%5+.5)/5,(i//5+.5)/5,.07] for i in range(N)],float)
        base,_=normalize(base)
        base[:,2]*=.9
        best={'circles':base.tolist()}
    else:
        best=init
        base=np.asarray(init['circles'],float).copy()
    best_score=exact_score(base)
    base,w=normalize(base)
    center=(np.min(base[:,:2]-base[:,2,None],axis=0)+np.max(base[:,:2]+base[:,2,None],axis=0))/2
    pool=[]
    for k in range(int(params['pole_samples'])):
        if time.monotonic()>end:
            break
        theta=2*np.pi*((k*.6180339887498949+ rng.uniform(-.025,.025))%1)
        radius=np.exp(rng.uniform(np.log(float(params['near_radius'])),np.log(float(params['far_radius']))))
        pole=center+radius*np.array([np.cos(theta),np.sin(theta)])
        v=base[:,:2]-pole
        denom=np.sum(v*v,axis=1)-base[:,2]**2
        if np.min(denom)<=1e-7:
            continue
        a=np.column_stack((v/denom[:,None],base[:,2]/denom))
        # A rotation can replace all four rectangle-support contacts.
        angle=rng.uniform(-.35,.35)
        c,s=np.cos(angle),np.sin(angle)
        a[:,:2]=a[:,:2]@np.array([[c,-s],[s,c]])
        a,aw=normalize(a)
        value=float(a[:,2].sum())
        signature=a[:,2]/np.linalg.norm(a[:,2])
        pool.append((value,a,aw,signature))
    pool.sort(key=lambda q:q[0],reverse=True)
    selected=[]
    for item in pool:
        if time.monotonic()>end:
            break
        if all(np.linalg.norm(item[3]-old[3])>.045 for old in selected):
            selected.append(item)
        if len(selected)>=int(params['relaxations']):
            break
    objective_grad=np.r_[np.zeros(2*N),-np.ones(N),0.]
    bounds=[(None,None)]*(2*N)+[(1e-7,.8)]*N+[(.12,1.88)]
    def objective(z):
        if time.monotonic()>end:
            raise StopSearch
        return -float(z[2*N:3*N].sum())
    def consider(z):
        nonlocal best,best_score
        if time.monotonic()>end:
            raise StopSearch
        a=unpack(z)
        if np.any(a[:,2]<=0):
            return
        # Uniform contraction of radii repairs floating contact residuals
        # without moving centers or relying on verifier tolerances.
        p=a[:,:2]
        d=np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
        factor=min(1.,float(np.min(d/(a[I,2]+a[J,2]))))
        a[:,2]*=max(0.,factor)*(1-4e-14)
        a,_=normalize(a)
        a*=1-8e-14
        score=exact_score(a)
        if score>best_score:
            best_score=score
            best={'circles':a.tolist()}
    for _,a,aw,_ in selected:
        if time.monotonic()>end-.03:
            break
        z=pack(a,aw)
        local_end=min(end,time.monotonic()+max(.12,(end-time.monotonic())/max(1,len(selected))))
        def callback(v):
            if time.monotonic()>local_end:
                consider(v)
                raise StopSearch
            if np.min(constraints(v))>-1e-8:
                consider(v)
        try:
            res=minimize(objective,z,jac=lambda z:objective_grad,method='SLSQP',bounds=bounds,
                         constraints={'type':'ineq','fun':constraints,'jac':jacobian},callback=callback,
                         options={'maxiter':int(params['iterations']),'ftol':2e-12,'disp':False})
            consider(res.x)
        except StopSearch:
            pass
    return best
