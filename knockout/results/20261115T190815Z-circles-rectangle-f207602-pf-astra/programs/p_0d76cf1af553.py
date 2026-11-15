import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'bias': (0.65, 0.1, 1.5),
    'biased_iterations': (90, 20, 200),
    'recovery_iterations': (85, 20, 200),
    'polish_iterations': (90, 20, 200),
    'cluster_scale': (0.28, 0.08, 0.6),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class StopSearch(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return -np.inf
    x = list(map(lambda v: Fraction(float(v)), a[:, 0]))
    y = list(map(lambda v: Fraction(float(v)), a[:, 1]))
    r = list(map(lambda v: Fraction(float(v)), a[:, 2]))
    extent = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
    extent += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
    return float(np.sum(a[:, 2])) if extent <= 2 else -np.inf

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.69*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf
    if init is not None:
        initial = np.asarray(init['circles'], dtype=float)
        best_score = exact_score(initial)
    else:
        initial = np.array([[(i % 7 + 0.5)/7, (i // 7 + 0.5)/3, 0.06] for i in range(21)])
        best = {'circles': initial.tolist()}
        best_score = exact_score(initial)
    lo = np.min(initial[:, :2]-initial[:, 2,None], axis=0)
    sides = np.max(initial[:, :2]+initial[:, 2,None], axis=0)-lo
    scale = 2/np.sum(sides)
    p = (initial[:, :2]-lo)*scale
    base = np.r_[p[:, 0], p[:, 1], initial[:, 2]*scale, sides[0]*scale]
    ones = np.ones(N)
    objective_gradient = np.r_[np.zeros(2*N), -ones, 0.0]
    lower = np.r_[np.zeros(2*N), np.full(N, 1e-8), 0.05]
    upper = np.r_[np.full(2*N, 2.0), np.full(N, 0.99), 1.95]
    bounds = list(zip(lower, upper))

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def geometry(z):
        x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
        dx,dy = x[I]-x[J],y[I]-y[J]
        d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-20)
        c = np.r_[d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r]
        return c,dx,dy,d

    def constraints(z):
        check()
        return geometry(z)[0]

    def jacobian(z):
        check()
        c,dx,dy,d = geometry(z)
        a = np.zeros((M+4*N,3*N+1))
        k = np.arange(M)
        a[k,I] = dx/d
        a[k,J] = -dx/d
        a[k,N+I] = dy/d
        a[k,N+J] = -dy/d
        a[k,2*N+I] = -1
        a[k,2*N+J] = -1
        q = np.arange(N)
        a[M+q,q] = 1
        a[M+N+q,q] = -1
        a[M+N+q,-1] = 1
        a[M+2*N+q,N+q] = 1
        a[M+3*N+q,N+q] = -1
        a[M+3*N+q,-1] = -1
        a[M+q,2*N+q] = -1
        a[M+N+q,2*N+q] = -1
        a[M+2*N+q,2*N+q] = -1
        a[M+3*N+q,2*N+q] = -1
        return a

    def penalty(z, mu, weights):
        check()
        c,dx,dy,d = geometry(z)
        v = np.maximum(-c,0)
        t = 2*mu*v[:M]
        gx = np.bincount(I,weights=-t*dx/d,minlength=N)+np.bincount(J,weights=t*dx/d,minlength=N)
        gy = np.bincount(I,weights=-t*dy/d,minlength=N)+np.bincount(J,weights=t*dy/d,minlength=N)
        gr = -weights+np.bincount(I,weights=t,minlength=N)+np.bincount(J,weights=t,minlength=N)
        b = (2*mu*v[M:]).reshape(4,N)
        gx += -b[0]+b[1]
        gy += -b[2]+b[3]
        gr += b.sum(axis=0)
        grad = np.r_[gx,gy,gr,-b[1].sum()+b[3].sum()]
        return -float(np.dot(weights,z[2*N:3*N]))+mu*float(np.dot(v,v)),grad

    def save(z):
        nonlocal best,best_score
        if time.monotonic() >= deadline:
            return
        a = np.column_stack((z[:N],z[N:2*N],z[2*N:3*N])).copy()
        if not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
        shrink = min(1.0,float(np.min(d/(a[I,2]+a[J,2]))))
        a[:,2] *= shrink*(1-8e-15)
        low = np.min(a[:,:2]-a[:,2,None],axis=0)
        high = np.max(a[:,:2]+a[:,2,None],axis=0)
        a[:,:2] -= (low+high)/2
        a *= (2/float(np.sum(high-low)))*(1-8e-15)
        score = exact_score(a)
        if score > best_score:
            best_score = score
            best = {'circles':a.tolist()}

    last = [base.copy()]
    def callback(z):
        check()
        last[0] = z.copy()

    def fun(z):
        check()
        return -float(np.sum(z[2*N:3*N]))

    attempt = 0
    try:
        while time.monotonic() < deadline:
            check()
            z = base.copy()
            centers = np.column_stack((z[:N],z[N:2*N]))
            anchor = int(rng.integers(N))
            distance = np.sum((centers-centers[anchor])**2,axis=1)
            field = np.exp(-distance/(2*float(params['cluster_scale'])**2))
            field -= np.mean(field)
            field /= max(float(np.max(np.abs(field))),1e-8)
            sign = 1 if attempt % 2 == 0 else -1
            weights = np.maximum(0.05,1+sign*float(params['bias'])*field)
            last[0] = z.copy()
            res = minimize(penalty,z,args=(120.0,weights),method='L-BFGS-B',jac=True,bounds=bounds,callback=callback,
                           options={'maxiter':int(params['biased_iterations']),'ftol':1e-11,'gtol':1e-6,'maxls':15})
            check()
            z = res.x
            res = minimize(penalty,z,args=(900.0,ones),method='L-BFGS-B',jac=True,bounds=bounds,callback=callback,
                           options={'maxiter':int(params['recovery_iterations']),'ftol':1e-12,'gtol':1e-7,'maxls':15})
            check()
            z = res.x
            save(z)
            res = minimize(fun,z,jac=lambda z: objective_gradient,method='SLSQP',bounds=bounds,
                           constraints={'type':'ineq','fun':constraints,'jac':jacobian},callback=callback,
                           options={'maxiter':int(params['polish_iterations']),'ftol':2e-13,'disp':False})
            check()
            save(res.x)
            attempt += 1
    except StopSearch:
        pass
    return best
