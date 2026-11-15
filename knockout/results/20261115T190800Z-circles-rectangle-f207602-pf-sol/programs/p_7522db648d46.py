import time
from fractions import Fraction
import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import minimize

PARAMS = {
    'active_gap': (3e-6, 1e-9, 1e-3, 'log'),
    'perturbation': (0.022, 0.001, 0.15, 'log'),
    'iterations': (90, 10, 300),
    'time_fraction': (0.70, 0.4, 0.75),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class StopSearch(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    r = a[:, 2]
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < r[I]+r[J]):
        return -np.inf
    x, y, q = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
    hp = max(u+v for u,v in zip(x,q))-min(u-v for u,v in zip(x,q))
    hp += max(u+v for u,v in zip(y,q))-min(u-v for u,v in zip(y,q))
    return float(np.sum(r)) if hp <= 2 else -np.inf

def perimeter(a):
    x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
    return (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
            +max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)))

def repair(z, deadline):
    a = np.column_stack((z[:N], z[N:2*N], z[2*N:3*N])).copy()
    if not np.isfinite(a).all() or np.any(a[:,2] <= 0):
        return None
    a[:, :2] -= np.mean(a[:, :2], axis=0)
    for k in range(8):
        if time.monotonic() >= deadline:
            return None
        d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
        ratios = d/(a[I,2]+a[J,2])
        s = min(1.0, float(np.min(ratios)))
        if s < 1.0 or k == 0:
            a[:,2] *= s*(1.0-8e-16)
        hp = perimeter(a)
        if hp > 2:
            a *= (2.0/float(hp))*(1.0-8e-16)
        if exact_score(a) > -np.inf:
            return a
        a[:,2] *= 1.0-2e-15
    return None

def pack(a):
    lo = np.min(a[:,:2]-a[:,2,None], axis=0)
    hi = np.max(a[:,:2]+a[:,2,None], axis=0)
    scale = 2.0/np.sum(hi-lo)
    p = (a[:,:2]-lo)*scale
    return np.r_[p[:,0],p[:,1],a[:,2]*scale,(hi[0]-lo[0])*scale]

def constraint(z, jac=False):
    x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
    dx,dy = x[I]-x[J],y[I]-y[J]
    d = np.sqrt(dx*dx+dy*dy)
    c = np.r_[d-r[I]-r[J],x-r,w-x-r,y-r,2-w-y-r]
    if not jac:
        return c
    A = np.zeros((M+4*N,3*N+1))
    rows = np.arange(M)
    ds = np.maximum(d,1e-100)
    A[rows,I],A[rows,J] = dx/ds,-dx/ds
    A[rows,N+I],A[rows,N+J] = dy/ds,-dy/ds
    A[rows,2*N+I] = -1
    A[rows,2*N+J] = -1
    ar = np.arange(N)
    for k in range(4):
        rr = M+k*N+ar
        A[rr,2*N+ar] = -1
        A[rr,(0 if k < 2 else N)+ar] = 1 if k%2 == 0 else -1
    A[M+N:M+2*N,-1] = 1
    A[M+3*N:M+4*N,-1] = -1
    return c,A

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start+float(params['time_fraction'])*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf
    if init is not None:
        initial = np.asarray(init['circles'],dtype=float)
        best_score = exact_score(initial)
    else:
        initial = np.array([[(i%5+0.5)/5,(i//5+0.5)/5,0.045] for i in range(N)])
    zbase = pack(initial)
    objective = np.zeros(3*N+1)
    objective[2*N:3*N] = -1

    def check():
        if time.monotonic() >= end:
            raise StopSearch()

    def consider(z):
        nonlocal best,best_score
        if time.monotonic() >= end:
            return
        a = repair(z,end)
        if a is not None:
            score = exact_score(a)
            if score > best_score:
                best_score = score
                best = {'circles':a.tolist()}

    def polish(z):
        z = z.copy()
        c,A = constraint(z,True)
        active = np.flatnonzero(c < float(params['active_gap']))
        if len(active) < 3*N-2:
            return z
        for iteration in range(9):
            check()
            c,A = constraint(z,True)
            B = A[active]
            dz = lstsq(B,-c[active],cond=1e-12,lapack_driver='gelsy')[0]
            if np.linalg.norm(dz) > 0.05:
                break
            z += dz
            if np.max(np.abs(dz)) < 2e-15:
                break
        consider(z)
        return z

    last = zbase.copy()
    def fun(z):
        check()
        return float(objective@z)
    def cj(z):
        check()
        return constraint(z,True)[1]
    def cf(z):
        check()
        return constraint(z)
    def callback(z):
        nonlocal last
        check()
        last = z.copy()

    bounds = [(0.0,2.0)]*(2*N)+[(1e-7,0.6)]*N+[(0.15,1.85)]
    try:
        consider(zbase)
        polished = polish(zbase)
        # Optimize the original smooth inequalities, without penalty bias.
        last = polished.copy()
        result = minimize(fun,polished,jac=lambda z:objective,method='SLSQP',
            bounds=bounds,constraints={'type':'ineq','fun':cf,'jac':cj},
            callback=callback,options={'maxiter':int(params['iterations']),'ftol':2e-14})
        consider(result.x)
        if time.monotonic() < end:
            polished = polish(result.x)
        attempt = 0
        while time.monotonic() < end:
            attempt += 1
            z = zbase.copy()
            amp = float(params['perturbation'])*(1.0+0.45*((attempt-1)%3))
            displacement = rng.normal(size=(N,2))
            # Perturb a spatial cluster together, changing several contacts at once.
            anchor = rng.integers(N)
            p = np.column_stack((z[:N],z[N:2*N]))
            dist = np.sum((p-p[anchor])**2,axis=1)
            cluster = np.argsort(dist)[:7]
            displacement *= 0.12
            displacement[cluster] += rng.normal(size=(1,2))
            p += amp*displacement
            z[:N],z[N:2*N] = p[:,0],p[:,1]
            z[2*N:3*N] *= 0.98
            last = z.copy()
            result = minimize(fun,z,jac=lambda z:objective,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':cf,'jac':cj},callback=callback,
                options={'maxiter':int(params['iterations']),'ftol':3e-13})
            consider(result.x)
            if result.success and time.monotonic() < end:
                polish(result.x)
    except StopSearch:
        pass
    if best is None:
        # Safe cold-start fallback with substantial separation.
        a = np.array([[(i%5)*0.18,(i//5)*0.18,0.04] for i in range(N)])
        best = {'circles':a.tolist()}
    return best
