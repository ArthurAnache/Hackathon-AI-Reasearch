import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'attempts': (12, 1, 40),
    'iterations': (95, 20, 250),
    'split_fraction': (0.51, 0.35, 0.7),
    'merge_growth': (1.0, 0.8, 1.2),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class StopSearch(Exception):
    pass

def score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    r = a[:, 2]
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < r[I]+r[J]):
        return -np.inf
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    q = [Fraction(float(v)) for v in r]
    perimeter = (max(v+t for v,t in zip(x,q))-min(v-t for v,t in zip(x,q))
                 +max(v+t for v,t in zip(y,q))-min(v-t for v,t in zip(y,q)))
    if perimeter > 2:
        return -np.inf
    return float(np.sum(r))

def repair(z, deadline):
    if time.monotonic() >= deadline:
        return None
    a = np.column_stack((z[:N], z[N:2*N], z[2*N:3*N])).copy()
    if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return None
    r = a[:, 2]
    d = np.sqrt((a[I, 0]-a[J, 0])**2+(a[I, 1]-a[J, 1])**2)
    factor = min(1., float(np.min(d/(r[I]+r[J]))))
    a[:, 2] *= factor*(1.-3e-15)
    lo = np.min(a[:, :2]-a[:, 2,None], axis=0)
    hi = np.max(a[:, :2]+a[:, 2,None], axis=0)
    a[:, :2] -= lo
    a *= (2./np.sum(hi-lo))*(1.-3e-15)
    for k in range(5):
        if time.monotonic() >= deadline:
            return None
        if np.isfinite(score(a)):
            return a
        a[:, 2] *= 1.-2e-14*(10**k)
    return None

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + .69*budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a = np.array([[(j+.5)/5., (i+.5)/5., .075]
                      for i in range(5) for j in range(5)][:21], dtype=float)
        best = {'circles': a.tolist()}
    else:
        a = np.asarray(init['circles'], dtype=float).copy()
        best = init
    best_score = score(a)
    lo = np.min(a[:,:2]-a[:,2,None],axis=0)
    hi = np.max(a[:,:2]+a[:,2,None],axis=0)
    scale = 2./np.sum(hi-lo)
    a[:,:2] = (a[:,:2]-lo)*scale
    a[:,2] *= scale
    width = (hi[0]-lo[0])*scale
    base = a.copy()
    dist = np.sqrt(np.sum((base[I,:2]-base[J,:2])**2,axis=1))
    gaps = dist-base[I,2]-base[J,2]
    contact_ids = np.flatnonzero(gaps < .008)
    if len(contact_ids) == 0:
        contact_ids = np.argsort(gaps)[:40]
    # Prefer mergers that transfer population between spatially separated regions.
    priority = base[I[contact_ids],2]+base[J[contact_ids],2]
    ordered = contact_ids[np.argsort(priority)]
    bound = [(None,None)]*(2*N)+[(1e-6,.7)]*N+[(.12,1.88)]
    last = None
    def check():
        if time.monotonic() >= deadline:
            raise StopSearch()
    def objective(z):
        nonlocal last
        check()
        last = z.copy()
        return -float(np.sum(z[2*N:3*N]))
    grad = np.r_[np.zeros(2*N),-np.ones(N),0.]
    def constraints(z):
        check()
        x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
        dx,dy = x[I]-x[J],y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        return np.r_[d-r[I]-r[J],x-r,w-x-r,y-r,2.-w-y-r]
    def jacobian(z):
        check()
        x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
        dx,dy = x[I]-x[J],y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        A = np.zeros((M+4*N,3*N+1))
        ix = np.arange(M)
        A[ix,I] = dx/d
        A[ix,J] = -dx/d
        A[ix,N+I] = dy/d
        A[ix,N+J] = -dy/d
        A[ix,2*N+I] = -1.
        A[ix,2*N+J] = -1.
        for k in range(4):
            check()
            rows = M+k*N+np.arange(N)
            A[rows,2*N+np.arange(N)] = -1.
            A[rows,(0 if k<2 else N)+np.arange(N)] = 1. if k%2==0 else -1.
            if k==1:
                A[rows,-1] = 1.
            elif k==3:
                A[rows,-1] = -1.
        return A
    for attempt in range(int(params['attempts'])):
        if time.monotonic() >= deadline-.025:
            break
        c = base.copy()
        # Cycle through small and large adjacent pairs, changing the two regions together.
        if attempt%3 == 0:
            edge = ordered[(attempt//3)%len(ordered)]
        elif attempt%3 == 1:
            edge = ordered[-1-(attempt//3)%len(ordered)]
        else:
            edge = ordered[int(rng.integers(len(ordered)))]
        i,j = int(I[edge]),int(J[edge])
        delta = c[j,:2]-c[i,:2]
        distance = float(np.linalg.norm(delta))
        center = c[i,:2] + delta*((distance+c[j,2]-c[i,2])/(2.*max(distance,1e-15)))
        remote = np.linalg.norm(c[:,:2]-center,axis=1)
        merit = c[:,2]*np.minimum(remote,.7)
        merit[[i,j]] = -1
        eligible = np.argsort(merit)[-5:]
        k = int(eligible[-1-(attempt//3)%len(eligible)])
        oldcenter = c[k,:2].copy()
        oldr = c[k,2]
        angle = float(rng.uniform(0.,np.pi))
        if attempt%2 == 0:
            angle = float(np.arctan2(delta[1],delta[0])+np.pi/2)
        direction = np.array([np.cos(angle),np.sin(angle)])
        c[i,:2] = center
        c[i,2] = .5*(distance+c[i,2]+c[j,2])*float(params['merge_growth'])
        f = float(params['split_fraction'])
        c[j,:2] = oldcenter-direction*oldr*.52
        c[k,:2] = oldcenter+direction*oldr*.52
        c[j,2] = oldr*f
        c[k,2] = oldr*f
        z = np.r_[c[:,0],c[:,1],c[:,2],width]
        last = z
        try:
            result = minimize(objective,z,jac=lambda z:grad,method='SLSQP',
                              bounds=bound,constraints={'type':'ineq','fun':constraints,'jac':jacobian},
                              options={'maxiter':int(params['iterations']),'ftol':2e-11,'disp':False})
            z = result.x
        except StopSearch:
            break
        candidate = repair(z,deadline)
        if candidate is not None:
            value = score(candidate)
            if value > best_score:
                best_score = value
                best = {'circles':candidate.tolist()}
    return best
