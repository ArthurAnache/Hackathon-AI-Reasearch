import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'population': (12, 4, 24),
    'slip': (0.19, 0.03, 0.4),
    'steps': (650, 100, 1500),
    'mu_start': (18.0, 2.0, 100.0, 'log'),
    'mu_end': (40000.0, 1000.0, 1000000.0, 'log')
}
N = 21
I, J = np.triu_indices(N, 1)

class StopSearch(Exception):
    pass

def valid(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return False
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return False
    x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
    return (max(u+t for u,t in zip(x,r))-min(u-t for u,t in zip(x,r)) +
            max(u+t for u,t in zip(y,r))-min(u-t for u,t in zip(y,r))) <= 2

def repair(z):
    a = np.column_stack((z[:21], z[21:42], z[42:63])).copy()
    if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return None
    p, r = a[:, :2], a[:, 2]
    d = np.sqrt((p[I, 0]-p[J, 0])**2+(p[I, 1]-p[J, 1])**2)
    rat = np.min(d/(r[I]+r[J]))
    r *= min(1.0, float(rat))*(1-2e-14)
    lo = np.min(p-r[:, None], axis=0)
    hi = np.max(p+r[:, None], axis=0)
    a[:, :2] -= lo
    a *= min(1.0, 2/float(np.sum(hi-lo)))*(1-2e-14)
    return a if valid(a) else None

def batch_gradient(z, mu):
    x, y, r = z[:, :21], z[:, 21:42], z[:, 42:63]
    w = z[:, 63:64]
    dx, dy = x[:, I]-x[:, J], y[:, I]-y[:, J]
    d = np.sqrt(dx*dx+dy*dy+1e-24)
    v = np.maximum(r[:, I]+r[:, J]-d, 0)
    c = 2*mu*v
    gx, gy, gr = np.zeros_like(x), np.zeros_like(y), -np.ones_like(r)
    # bincount-style accumulation across the entire population.
    off = np.arange(len(z))[:, None]*21
    ii, jj = (I[None, :]+off).ravel(), (J[None, :]+off).ravel()
    for g, q in ((gx, -c*dx/d), (gy, -c*dy/d)):
        g[:] = (np.bincount(ii, weights=q.ravel(), minlength=len(z)*21) + np.bincount(jj, weights=-q.ravel(), minlength=len(z)*21)).reshape(-1,21)
    gr += (np.bincount(ii, weights=c.ravel(), minlength=len(z)*21)+np.bincount(jj, weights=c.ravel(), minlength=len(z)*21)).reshape(-1,21)
    b0 = np.maximum(r-x,0)
    b1 = np.maximum(r+x-w,0)
    b2 = np.maximum(r-y,0)
    b3 = np.maximum(r+y-(2-w),0)
    gx += 2*mu*(-b0+b1)
    gy += 2*mu*(-b2+b3)
    gr += 2*mu*(b0+b1+b2+b3+np.minimum(r-0.00001,0))
    gw = 2*mu*(-b1.sum(axis=1)+b3.sum(axis=1))
    return np.column_stack((gx,gy,gr,gw))

def constraints(z):
    x,y,r,w = z[:21],z[21:42],z[42:63],z[63]
    dx,dy = x[I]-x[J],y[I]-y[J]
    d = np.sqrt(dx*dx+dy*dy+1e-30)
    return np.r_[d-r[I]-r[J],x-r,w-x-r,y-r,2-w-y-r]

def jacobian(z):
    x,y = z[:21],z[21:42]
    dx,dy = x[I]-x[J],y[I]-y[J]
    d = np.sqrt(dx*dx+dy*dy+1e-30)
    m=len(I)
    a=np.zeros((m+84,64))
    q=np.arange(m)
    a[q,I]=dx/d; a[q,J]=-dx/d
    a[q,I+21]=dy/d; a[q,J+21]=-dy/d
    a[q,I+42]=-1; a[q,J+42]=-1
    k=np.arange(21)
    for side in range(4):
        row=m+side*21+k
        a[row,42+k]=-1
        a[row,k+(21 if side>=2 else 0)]=1 if side%2==0 else -1
        if side==1: a[row,63]=1
        if side==3: a[row,63]=-1
    return a

def construct(params, seed, budget_s, init=None):
    start=time.monotonic()
    end=start+0.70*budget_s
    population_end=start+0.37*budget_s
    rng=np.random.default_rng(seed)
    best=init
    best_score=-np.inf
    if init is not None:
        a=np.asarray(init['circles'],dtype=float).copy()
        if valid(a): best_score=float(a[:,2].sum())
    else:
        a=np.array([[(i%5+0.5)/5,(i//5+0.5)/5,0.075] for i in range(21)])
        best={'circles':a.tolist()}
        best_score=float(a[:,2].sum()) if valid(a) else -np.inf
    lo=np.min(a[:,:2]-a[:,2,None],axis=0)
    hi=np.max(a[:,:2]+a[:,2,None],axis=0)
    scale=2/np.sum(hi-lo)
    p=(a[:,:2]-lo)*scale
    r=a[:,2]*scale
    base=np.r_[p[:,0],p[:,1],r,(hi[0]-lo[0])*scale]
    count=int(params['population'])
    z=np.tile(base,(count,1))
    for k in range(count):
        if time.monotonic()>=population_end: break
        axis=k%2
        cross=1-axis
        threshold=np.quantile(p[:,cross],rng.uniform(.25,.75))
        mask=p[:,cross]>threshold
        displacement=float(params['slip'])*rng.choice([-1,1])*rng.uniform(.6,1.4)
        z[k,axis*21:axis*21+21] += displacement*(mask-mask.mean())
        z[k,42:63] *= np.exp(rng.normal(0,.12,21))
        z[k,:42] += rng.normal(0,.006,42)
    velocity=np.zeros_like(z)
    dt=np.full((count,1),.004)
    alpha=np.full((count,1),.1)
    positive=np.zeros((count,1),int)
    steps=int(params['steps'])
    for step in range(steps):
        if time.monotonic()>=population_end: break
        fraction=step/max(1,steps-1)
        mu=float(params['mu_start'])*(float(params['mu_end'])/float(params['mu_start']))**fraction
        force=-batch_gradient(z,mu)
        power=np.sum(velocity*force,axis=1,keepdims=True)
        good=power>0
        positive=np.where(good,positive+1,0)
        dt=np.where(good & (positive>5),np.minimum(dt*1.08,.035/np.sqrt(mu)),np.where(good,dt,dt*.5))
        dt=np.maximum(dt,1e-7)
        alpha=np.where(good & (positive>5),alpha*.99,np.where(good,alpha,.1))
        velocity=np.where(good,velocity,0)
        fn=np.linalg.norm(force,axis=1,keepdims=True)+1e-30
        vn=np.linalg.norm(velocity,axis=1,keepdims=True)
        velocity=(1-alpha)*velocity+alpha*force*(vn/fn)
        velocity+=dt*force
        delta=dt*velocity
        delta*=np.minimum(1,.025/(np.max(np.abs(delta),axis=1,keepdims=True)+1e-30))
        z+=delta
        z[:,42:63]=np.maximum(z[:,42:63],.0001)
        z[:,63]=np.clip(z[:,63],.35,1.65)
    rankings=[]
    for k in range(count):
        if time.monotonic()>=end: break
        candidate=repair(z[k])
        score=float(candidate[:,2].sum()) if candidate is not None else -np.inf
        rankings.append((score,k))
        if score>best_score:
            best_score=score; best={'circles':candidate.tolist()}
    objective_gradient=np.r_[np.zeros(42),-np.ones(21),0.]
    def objective(q):
        if time.monotonic()>=end: raise StopSearch()
        return -q[42:63].sum(),objective_gradient
    for _,k in sorted(rankings,reverse=True)[:3]:
        if time.monotonic()>=end: break
        last=[z[k].copy()]
        def callback(q):
            last[0]=q.copy()
            if time.monotonic()>=end: raise StopSearch()
        try:
            result=minimize(objective,z[k],jac=True,method='SLSQP',constraints={'type':'ineq','fun':constraints,'jac':jacobian},bounds=[(None,None)]*42+[(1e-7,.5)]*21+[(.2,1.8)],callback=callback,options={'maxiter':85,'ftol':2e-12})
            last[0]=result.x
        except StopSearch:
            pass
        candidate=repair(last[0])
        if candidate is not None:
            score=float(candidate[:,2].sum())
            if score>best_score:
                best_score=score; best={'circles':candidate.tolist()}
    return best
