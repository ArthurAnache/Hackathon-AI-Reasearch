import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'strength': (1.25, 0.65, 4.0),
    'attempts': (10, 1, 30),
    'maxiter': (105, 20, 250),
}
N = 21
I, J = np.triu_indices(N, 1)
K = np.arange(N)

class StopSearch(Exception):
    pass

def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return -np.inf
    x = list(map(Fraction, map(float, a[:, 0])))
    y = list(map(Fraction, map(float, a[:, 1])))
    r = list(map(Fraction, map(float, a[:, 2])))
    perimeter = (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
                 +max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)))
    return float(np.sum(a[:, 2])) if perimeter <= 2 else -np.inf

def normalize(a):
    a = a.copy()
    lo = np.min(a[:, :2]-a[:, 2,None], axis=0)
    hi = np.max(a[:, :2]+a[:, 2,None], axis=0)
    sides = hi-lo
    scale = 2.0/np.sum(sides)
    a[:, :2] = (a[:, :2]-lo)*scale
    a[:, 2] *= scale
    return a, sides[0]*scale

def pack(a, w):
    return np.r_[a[:,0], a[:,1], a[:,2], w]

def unpack(z):
    return np.column_stack((z[:N],z[N:2*N],z[2*N:3*N]))

def repair(z, deadline):
    a = unpack(z)
    if not np.isfinite(a).all() or np.any(a[:,2] <= 0):
        return None
    a, _ = normalize(a)
    d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
    factor = min(1.0, float(np.min(d/(a[I,2]+a[J,2]))))
    a[:,2] *= factor*(1-8e-15)
    a, _ = normalize(a)
    a[:, :2] *= 1-8e-15
    a[:, 2] *= 1-1.6e-14
    if time.monotonic() >= deadline:
        return a
    return a

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a = np.array([[(k%5+.5)*.2,(k//5+.5)*.2,.085] for k in range(N)])
        a, _ = normalize(a)
        a[:,2] *= 1-1e-12
        best = {'circles': a.tolist()}
    else:
        best = init
        a = np.asarray(init['circles'], dtype=float).copy()
    best_score = exact_score(a)
    base, base_w = normalize(a)
    last = None

    def constraints(z):
        if time.monotonic() >= deadline:
            raise StopSearch()
        x,y,r,w = z[:N], z[N:2*N], z[2*N:3*N],z[-1]
        dx,dy=x[I]-x[J],y[I]-y[J]
        d=np.sqrt(dx*dx+dy*dy+1e-30)
        return np.r_[d-r[I]-r[J], x-r,w-x-r,y-r,2-w-y-r]

    def jacobian(z):
        if time.monotonic() >= deadline:
            raise StopSearch()
        x,y=z[:N],z[N:2*N]
        dx,dy=x[I]-x[J],y[I]-y[J]
        d=np.sqrt(dx*dx+dy*dy+1e-30)
        m=len(I)
        out=np.zeros((m+4*N,3*N+1))
        row=np.arange(m)
        out[row,I]=dx/d
        out[row,J]=-dx/d
        out[row,N+I]=dy/d
        out[row,N+J]=-dy/d
        out[row,2*N+I]=-1
        out[row,2*N+J]=-1
        out[m+K,K]=1
        out[m+N+K,K]=-1
        out[m+2*N+K,N+K]=1
        out[m+3*N+K,N+K]=-1
        out[m+K,2*N+K]=-1
        out[m+N+K,2*N+K]=-1
        out[m+2*N+K,2*N+K]=-1
        out[m+3*N+K,2*N+K]=-1
        out[m+N+K,-1]=1
        out[m+3*N+K,-1]=-1
        return out

    grad=np.r_[np.zeros(2*N),-np.ones(N),0.0]
    def objective(z):
        if time.monotonic() >= deadline:
            raise StopSearch()
        return -float(z[2*N:3*N].sum())
    def callback(z):
        nonlocal last
        last=z.copy()
        if time.monotonic() >= deadline:
            raise StopSearch()

    angles = rng.uniform(0,2*np.pi) + np.arange(int(params['attempts']))*2.399963229728653
    for attempt, theta in enumerate(angles):
        if time.monotonic() >= deadline-0.025:
            break
        if attempt == 0:
            trial=base.copy()
            w=base_w
        else:
            center=np.array([base_w/2,(2-base_w)/2])
            direction=np.array([np.cos(theta),np.sin(theta)])
            distance=float(params['strength'])*(0.75+0.35*(attempt%4))
            pole=center+distance*direction
            v=base[:,:2]-pole
            den=np.sum(v*v,axis=1)-base[:,2]**2
            if np.min(den)<=0.015:
                continue
            trial=np.column_stack((v/den[:,None],base[:,2]/den))
            trial,w=normalize(trial)
        z=pack(trial,w)
        last=z.copy()
        try:
            result=minimize(objective,z,jac=lambda z:grad,method='SLSQP',
                constraints={'type':'ineq','fun':constraints,'jac':jacobian},
                bounds=[(None,None)]*(2*N)+[(1e-7,0.9)]*N+[(0.05,1.95)],
                callback=callback,
                options={'maxiter':int(params['maxiter']),'ftol':2e-12,'disp':False})
            last=result.x
        except StopSearch:
            pass
        if last is not None:
            candidate=repair(last,deadline)
            if candidate is not None:
                score=exact_score(candidate)
                if score>best_score:
                    best_score=score
                    best={'circles':candidate.tolist()}
        if time.monotonic() >= deadline:
            break
    return best
