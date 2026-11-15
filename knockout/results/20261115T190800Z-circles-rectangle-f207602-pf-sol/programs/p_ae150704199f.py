import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'entropy_power': (-1.0, -3.0, 0.5),
    'transition_power': (0.45, 0.1, 0.9),
    'iterations': (85, 20, 200),
    'radius_scale': (0.11, 0.05, 0.2),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class Stop(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return -np.inf
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    w = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
    h = max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
    if w+h > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    if init is not None:
        a = np.asarray(init['circles'], dtype=float).copy()
        best = init
        score = exact_score(a)
    else:
        a = np.array([[(i%5+.5)*.2, (i//5+.5)*.2, .095] for i in range(N)])
        best = {'circles': a.tolist()}
        score = exact_score(a)
    lo = np.min(a[:, :2]-a[:, 2,None], axis=0)
    side = np.max(a[:, :2]+a[:, 2,None], axis=0)-lo
    factor = 2/side.sum()
    a[:, :2] = (a[:, :2]-lo)*factor
    a[:, 2] *= factor
    R = float(params['radius_scale'])
    z = np.r_[a[:,0], a[:,1], (a[:,2]/R)**2, side[0]*factor]
    bounds = [(0.,2.)]*(2*N)+[(1e-8,100.)]*N+[(.15,1.85)]
    last = z.copy()

    def unpack(v):
        q = np.maximum(v[2*N:3*N], 1e-12)
        r = R*np.sqrt(q)
        return v[:N],v[N:2*N],r,R/(2*np.sqrt(q)),v[-1]

    def constraints(v):
        if time.monotonic() >= deadline:
            raise Stop()
        x,y,r,dr,w = unpack(v)
        dx=x[I]-x[J]; dy=y[I]-y[J]
        dist=np.sqrt(dx*dx+dy*dy+1e-30)
        return np.r_[dist-r[I]-r[J],x-r,w-x-r,y-r,2-w-y-r]

    def jacobian(v):
        if time.monotonic() >= deadline:
            raise Stop()
        x,y,r,dr,w=unpack(v)
        dx=x[I]-x[J];dy=y[I]-y[J]
        d=np.sqrt(dx*dx+dy*dy+1e-30)
        g=np.zeros((M+4*N,3*N+1))
        rows=np.arange(M)
        g[rows,I]=dx/d;g[rows,J]=-dx/d
        g[rows,N+I]=dy/d;g[rows,N+J]=-dy/d
        g[rows,2*N+I]=-dr[I];g[rows,2*N+J]=-dr[J]
        ii=np.arange(N)
        for k in range(4):
            if time.monotonic() >= deadline:
                raise Stop()
            rr=M+k*N+ii
            g[rr,2*N+ii]=-dr
            g[rr,(0 if k<2 else N)+ii]=1 if k%2==0 else -1
            if k==1:g[rr,-1]=1
            if k==3:g[rr,-1]=-1
        return g

    def consider(v):
        nonlocal best,score
        if time.monotonic() >= deadline:
            return
        x,y,r,dr,w=unpack(v)
        p=np.column_stack((x,y))
        d=np.sqrt((x[I]-x[J])**2+(y[I]-y[J])**2)
        ratio=min(1.,float(np.min(d/(r[I]+r[J]))))
        r=r*max(0.,ratio)*(1-3e-14)
        lo=np.min(p-r[:,None],axis=0)
        hi=np.max(p+r[:,None],axis=0)
        s=min(1.,2/float(np.sum(hi-lo)))*(1-3e-14)
        aa=np.column_stack(((p-lo)*s,r*s))
        value=exact_score(aa)
        if value>score:
            best={'circles':aa.tolist()};score=value

    def callback(v):
        nonlocal last
        last=v.copy()
        if time.monotonic() >= stage_end:
            raise Stop()

    # Power is applied to normalized area; p=1/2 restores sum of radii.
    stages=[(float(params['entropy_power']),.25),
            (float(params['transition_power']),.19),(.5,.56)]
    total=max(0.,deadline-time.monotonic())
    for power, share in stages:
        if time.monotonic() >= deadline-.015:
            break
        stage_end=min(deadline-.008,time.monotonic()+share*total)
        def objective(v):
            if time.monotonic() >= stage_end:
                raise Stop()
            q=np.maximum(v[2*N:3*N],1e-12)
            if abs(power)<1e-12:
                value=-R*.5*np.log(q).sum()
                deriv=-R*.5/q
            else:
                value=-R*.5*np.sum(q**power)/power
                deriv=-R*.5*q**(power-1)
            g=np.zeros(3*N+1);g[2*N:3*N]=deriv
            return value,g
        last=z.copy()
        try:
            res=minimize(objective,z,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':constraints,'jac':jacobian},
                callback=callback,
                options={'maxiter':int(params['iterations']),'ftol':2e-12,'disp':False})
            z=res.x
        except Stop:
            z=last
        consider(z)
    return best
