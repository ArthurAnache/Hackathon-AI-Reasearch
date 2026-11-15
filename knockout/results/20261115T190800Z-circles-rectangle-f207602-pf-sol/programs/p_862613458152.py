import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'concave_power': (0.20, 0.05, 0.8),
    'convex_power': (3.0, 1.2, 5.0),
    'excursion_iterations': (32, 10, 80),
    'return_iterations': (85, 20, 160),
    'radius_floor': (0.008, 0.00001, 0.03, 'log')
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
AR = np.arange(N)

class StopSearch(Exception):
    pass

def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return -np.inf
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    perimeter = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
    perimeter += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
    return float(np.sum(a[:, 2])) if perimeter <= 2 else -np.inf

def constraints(z):
    x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
    dx,dy = x[I]-x[J],y[I]-y[J]
    d = np.sqrt(dx*dx+dy*dy+1e-30)
    return np.concatenate((d-r[I]-r[J],x-r,w-x-r,y-r,2-w-y-r))

def constraint_jac(z):
    x,y = z[:N],z[N:2*N]
    dx,dy = x[I]-x[J],y[I]-y[J]
    d = np.sqrt(dx*dx+dy*dy+1e-30)
    a = np.zeros((M+4*N,3*N+1))
    rows = np.arange(M)
    a[rows,I] = dx/d
    a[rows,J] = -dx/d
    a[rows,N+I] = dy/d
    a[rows,N+J] = -dy/d
    a[rows,2*N+I] = -1
    a[rows,2*N+J] = -1
    a[M+AR,AR] = 1
    a[M+AR,2*N+AR] = -1
    a[M+N+AR,AR] = -1
    a[M+N+AR,2*N+AR] = -1
    a[M+N+AR,-1] = 1
    a[M+2*N+AR,N+AR] = 1
    a[M+2*N+AR,2*N+AR] = -1
    a[M+3*N+AR,N+AR] = -1
    a[M+3*N+AR,2*N+AR] = -1
    a[M+3*N+AR,-1] = -1
    return a

def encode(a):
    p,r = a[:,:2].copy(),a[:,2].copy()
    lo = np.min(p-r[:,None],axis=0)
    sides = np.max(p+r[:,None],axis=0)-lo
    scale = 2/sides.sum()
    p = (p-lo)*scale
    return np.r_[p[:,0],p[:,1],r*scale,sides[0]*scale]

def repaired(z):
    p = np.column_stack((z[:N],z[N:2*N]))
    r = np.maximum(z[2*N:3*N],1e-12).copy()
    d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
    ratio = float(np.min(d/(r[I]+r[J])))
    r *= min(1.0,ratio)*(1-3e-14)
    lo = np.min(p-r[:,None],axis=0)
    sides = np.max(p+r[:,None],axis=0)-lo
    scale = (2-2e-13)/sides.sum()
    p = (p-lo)*scale
    r *= scale
    return np.column_stack((p,r))

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.69*budget_s
    rng = np.random.default_rng(seed)
    if init is not None:
        initial = np.asarray(init['circles'],dtype=float)
        best = init
        best_score = exact_score(initial)
    else:
        initial = np.array([[(k%5+0.5)*0.2,(k//5+0.5)*0.2,0.075] for k in range(N)])
        best = {'circles':initial.tolist()}
        best_score = exact_score(initial)
    base = encode(initial)
    cons = {'type':'ineq','fun':constraints,'jac':constraint_jac}
    latest = [base.copy()]

    def consider(z):
        nonlocal best,best_score
        if time.monotonic() >= deadline:
            return
        a = repaired(z)
        score = exact_score(a)
        if score > best_score:
            best,best_score = {'circles':a.tolist()},score

    def run(z,power,iterations,floor,local_deadline):
        def objective(v):
            if time.monotonic() >= local_deadline:
                raise StopSearch
            latest[0] = v.copy()
            u = np.maximum(v[2*N:3*N]/0.11,1e-15)
            grad = np.zeros(3*N+1)
            grad[2*N:3*N] = -u**(power-1)
            return -0.11*np.sum(u**power)/power,grad
        def callback(v):
            if time.monotonic() >= local_deadline:
                raise StopSearch
        bounds = [(0,2)]*(2*N)+[(floor,0.6)]*N+[(0.15,1.85)]
        latest[0] = z.copy()
        try:
            result = minimize(objective,z,jac=True,bounds=bounds,constraints=cons,
                              method='SLSQP',callback=callback,
                              options={'maxiter':iterations,'ftol':2e-12,'disp':False})
            return result.x
        except StopSearch:
            return latest[0].copy()

    powers = [float(params['concave_power']),float(params['convex_power'])]
    for k,power in enumerate(powers):
        now = time.monotonic()
        if now >= deadline-0.04:
            break
        remaining = deadline-now
        path_end = now + remaining/(len(powers)-k)
        z = base.copy()
        if k:
            # Break exact symmetry only in the convex excursion.
            z[2*N:3*N] *= 1+rng.normal(0,0.003,N)
        floor = float(params['radius_floor'])
        z[2*N:3*N] = np.maximum(z[2*N:3*N],floor)
        excursion_end = min(path_end-0.04,now+0.35*(path_end-now))
        if time.monotonic() < excursion_end:
            z = run(z,power,int(params['excursion_iterations']),floor,excursion_end)
        if time.monotonic() < path_end-0.015:
            z = run(z,1.0,int(params['return_iterations']),1e-7,path_end-0.008)
        consider(z)
    return best
