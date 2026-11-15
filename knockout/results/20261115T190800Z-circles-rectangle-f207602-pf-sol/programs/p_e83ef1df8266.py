import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'population': (16, 4, 32),
    'mutation': (0.032, 0.001, 0.12),
    'steps': (600, 100, 1500),
    'finish_count': (4, 1, 8),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class StopSearch(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    p, r = a[:, :2], a[:, 2]
    d = np.sqrt((p[I, 0]-p[J, 0])**2 + (p[I, 1]-p[J, 1])**2)
    if np.any(d < r[I]+r[J]):
        return -np.inf
    x, y, q = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
    hp = (max(u+v for u,v in zip(x,q))-min(u-v for u,v in zip(x,q))
          + max(u+v for u,v in zip(y,q))-min(u-v for u,v in zip(y,q)))
    if hp > 2:
        return -np.inf
    return float(np.sum(r))

def repair(z):
    p = np.column_stack((z[:N], z[N:2*N])).copy()
    r = np.maximum(z[2*N:3*N], 1e-10).copy()
    if not np.isfinite(p).all() or not np.isfinite(r).all():
        return None
    d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
    factor = min(1., float(np.min(d/(r[I]+r[J]))))
    r *= factor*(1.-4e-14)
    lo = np.min(p-r[:,None],axis=0)
    hi = np.max(p+r[:,None],axis=0)
    s = (2.-2e-13)/float(np.sum(hi-lo))
    p = (p-lo)*s
    r *= s
    return np.column_stack((p,r))

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + .70*budget_s
    batch_end = start + .34*budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a = np.array([[(i%7+.5)*.2, (i//7+.5)*.2, .1*(1.-1e-12)] for i in range(N)])
        best = {'circles': a.tolist()}
    else:
        best = init
        a = np.array(init['circles'],dtype=float)
    best_score = exact_score(a)
    lo = np.min(a[:,:2]-a[:,2,None],axis=0)
    sides = np.max(a[:,:2]+a[:,2,None],axis=0)-lo
    scale = 2./sides.sum()
    p = (a[:,:2]-lo)*scale
    r = a[:,2]*scale
    base = np.r_[p[:,0],p[:,1],r,sides[0]*scale]
    pop = int(params['population'])
    z = np.repeat(base[None,:],pop,axis=0)
    for k in range(1,pop):
        if time.monotonic() >= batch_end:
            break
        amp = float(params['mutation'])*(0.3+1.7*(k/(pop-1))**2)
        noise = rng.normal(size=(N,2))
        # Correlated regional motion releases several contacts together.
        direction = rng.normal(size=2)
        direction /= np.linalg.norm(direction)
        side = np.tanh((p-p.mean(axis=0))@direction/.12)
        motion = rng.normal(size=2)
        noise = .45*noise + side[:,None]*motion
        z[k,:N] += amp*noise[:,0]
        z[k,N:2*N] += amp*noise[:,1]
        z[k,2*N:3*N] *= np.exp(rng.normal(0,.12+.2*k/pop,N))
        z[k,-1] += rng.normal(0,amp)
    velocity = np.zeros_like(z)
    dt = np.full(pop,.0006)
    alpha = np.full(pop,.1)
    positive = np.zeros(pop,dtype=int)
    last_stage = -1
    steps = int(params['steps'])
    for it in range(steps):
        if time.monotonic() >= batch_end:
            break
        stage = min(3,4*it//steps)
        mu = (20.,100.,600.,4000.)[stage]
        if stage != last_stage:
            velocity *= .1
            dt[:] = .035/np.sqrt(mu)
            positive[:] = 0
            last_stage = stage
        x,y,rr,w = z[:,:N],z[:,N:2*N],z[:,2*N:3*N],z[:,-1:]
        dx,dy = x[:,I]-x[:,J],y[:,I]-y[:,J]
        d = np.sqrt(dx*dx+dy*dy+1e-24)
        v = np.maximum(rr[:,I]+rr[:,J]-d,0.)
        b0,b1 = np.maximum(rr-x,0.),np.maximum(rr+x-w,0.)
        b2,b3 = np.maximum(rr-y,0.),np.maximum(rr+y-(2.-w),0.)
        gx = 2*mu*(-b0+b1)
        gy = 2*mu*(-b2+b3)
        gr = -np.ones_like(rr)+2*mu*(b0+b1+b2+b3)
        c = 2*mu*v
        # A short loop over circles avoids slow batched scatter-add operations.
        for j in range(N):
            if time.monotonic() >= batch_end:
                break
            left,right = I==j,J==j
            gx[:,j] += np.sum(-c[:,left]*dx[:,left]/d[:,left],axis=1)+np.sum(c[:,right]*dx[:,right]/d[:,right],axis=1)
            gy[:,j] += np.sum(-c[:,left]*dy[:,left]/d[:,left],axis=1)+np.sum(c[:,right]*dy[:,right]/d[:,right],axis=1)
            gr[:,j] += np.sum(c[:,left],axis=1)+np.sum(c[:,right],axis=1)
        g = np.concatenate((gx,gy,gr,(2*mu*(-b1.sum(axis=1)+b3.sum(axis=1)))[:,None]),axis=1)
        force = -g
        velocity += dt[:,None]*force
        power = np.sum(velocity*force,axis=1)
        good = power > 0
        positive = np.where(good,positive+1,0)
        dt = np.where(good & (positive>5),np.minimum(dt*1.07,.16/np.sqrt(mu)),dt)
        dt = np.where(good,dt,dt*.5)
        alpha = np.where(good & (positive>5),alpha*.99,alpha)
        alpha = np.where(good,alpha,.1)
        velocity[~good] = 0.
        vn = np.linalg.norm(velocity,axis=1)
        fn = np.maximum(np.linalg.norm(force,axis=1),1e-30)
        velocity = (1-alpha[:,None])*velocity+alpha[:,None]*(vn/fn)[:,None]*force
        delta = dt[:,None]*velocity
        delta *= np.minimum(1.,.015/np.maximum(np.max(np.abs(delta),axis=1),1e-30))[:,None]
        z += delta
        z[:,2*N:3*N] = np.clip(z[:,2*N:3*N],1e-5,.45)
        z[:,-1] = np.clip(z[:,-1],.25,1.75)
    candidates = []
    for zz in z:
        if time.monotonic() >= end:
            break
        aa = repair(zz)
        if aa is not None:
            score = exact_score(aa)
            candidates.append((score,zz.copy()))
            if score > best_score:
                best_score,best = score,{'circles':aa.tolist()}
    candidates.sort(key=lambda t:t[0],reverse=True)
    def check():
        if time.monotonic() >= end:
            raise StopSearch()
    def objective(t):
        check()
        return -np.sum(t[2*N:3*N])
    grad = np.r_[np.zeros(2*N),-np.ones(N),0.]
    def constraints(t):
        check()
        x,y,r,w = t[:N],t[N:2*N],t[2*N:3*N],t[-1]
        dx,dy = x[I]-x[J],y[I]-y[J]
        return np.r_[np.sqrt(dx*dx+dy*dy)-r[I]-r[J],x-r,w-x-r,y-r,2-w-y-r]
    def jac(t):
        check()
        x,y = t[:N],t[N:2*N]
        dx,dy = x[I]-x[J],y[I]-y[J]
        d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-30)
        a = np.zeros((M+4*N,3*N+1))
        rows = np.arange(M)
        a[rows,I],a[rows,J] = dx/d,-dx/d
        a[rows,N+I],a[rows,N+J] = dy/d,-dy/d
        a[rows,2*N+I],a[rows,2*N+J] = -1.,-1.
        ids = np.arange(N)
        for k in range(4):
            check()
            q = M+k*N+ids
            a[q,2*N+ids] = -1
            a[q,(0 if k<2 else N)+ids] = 1 if k%2==0 else -1
            if k==1: a[q,-1] = 1
            if k==3: a[q,-1] = -1
        return a
    bounds = [(None,None)]*(2*N)+[(1e-7,.6)]*N+[(.2,1.8)]
    for _, zz in candidates[:int(params['finish_count'])]:
        if time.monotonic() >= end:
            break
        latest = [zz]
        def callback(t):
            latest[0] = t.copy()
            check()
        try:
            result = minimize(objective,zz,jac=lambda t:grad,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':constraints,'jac':jac},callback=callback,
                options={'maxiter':110,'ftol':2e-12,'disp':False})
            latest[0] = result.x
        except StopSearch:
            pass
        aa = repair(latest[0])
        if aa is not None:
            score = exact_score(aa)
            if score > best_score:
                best_score,best = score,{'circles':aa.tolist()}
    return best
