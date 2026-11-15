import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

PARAMS = {
    'temperature': (0.025, 0.003, 0.08, 'log'),
    'cooling': (0.16, 0.05, 0.5, 'log'),
    'penalty': (3.0, 1.0, 15.0),
    'iterations': (90, 20, 200),
    'jitter': (0.004, 0.0, 0.03)
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
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    q = [Fraction(float(v)) for v in r]
    p = max(v+t for v,t in zip(x,q))-min(v-t for v,t in zip(x,q))
    p += max(v+t for v,t in zip(y,q))-min(v-t for v,t in zip(y,q))
    return float(np.sum(r)) if p <= 2 else -np.inf

def unpack(z):
    return z[:N], z[N:2*N], z[2*N:3*N], z[-1]

def geometry(z, jac=False):
    x,y,r,w = unpack(z)
    dx,dy = x[I]-x[J], y[I]-y[J]
    d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-30)
    g = np.concatenate((d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r))
    if not jac:
        return g
    a = np.zeros((M+4*N, 3*N+1))
    rows = np.arange(M)
    a[rows,I] = dx/d
    a[rows,J] = -dx/d
    a[rows,N+I] = dy/d
    a[rows,N+J] = -dy/d
    a[rows,2*N+I] = -1
    a[rows,2*N+J] = -1
    k = np.arange(N)
    a[M+k,k] = 1
    a[M+N+k,k] = -1
    a[M+N+k,-1] = 1
    a[M+2*N+k,N+k] = 1
    a[M+3*N+k,N+k] = -1
    a[M+3*N+k,-1] = -1
    a[M+k,2*N+k] = -1
    a[M+N+k,2*N+k] = -1
    a[M+2*N+k,2*N+k] = -1
    a[M+3*N+k,2*N+k] = -1
    return a

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    anneal_end = start + 0.46*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf
    if init is not None:
        original = np.asarray(init['circles'], dtype=float)
        best_score = exact_score(original)
        a = original.copy()
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        sides = np.max(a[:,:2]+a[:,2,None],axis=0)-lo
        scale = 2/sides.sum()
        a[:,:2] = (a[:,:2]-lo)*scale
        a[:,2] *= scale
        w = sides[0]*scale
    else:
        k = np.arange(N)
        a = np.column_stack(((k%5+0.5)/5, (k//5+0.5)/5, np.full(N,0.07)))
        w = 1.0
        fallback = a.copy()
        fallback[:,2] = 0.03
        best = {'circles':fallback.tolist()}
        best_score = exact_score(fallback)
    z = np.r_[a[:,0],a[:,1],a[:,2],w]
    z[:2*N] += rng.normal(0,float(params['jitter']),2*N)
    bounds = [(0.0,2.0)]*(2*N)+[(1e-7,0.6)]*N+[(0.15,1.85)]

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def consider(v):
        nonlocal best,best_score
        check()
        x,y,r,_ = unpack(v)
        p = np.column_stack((x,y))
        r = np.maximum(r.copy(),1e-14)
        d = np.sqrt((x[I]-x[J])**2+(y[I]-y[J])**2)
        factor = min(1.0,float(np.min(d/(r[I]+r[J]))))
        r *= max(0.0,factor)*(1-4e-15)
        for _ in range(8):
            check()
            candidate = np.column_stack((p,r))
            s = exact_score(candidate)
            if s > best_score:
                best_score = s
                best = {'circles':candidate.tolist()}
            if np.isfinite(s):
                break
            lo = np.min(p-r[:,None],axis=0)
            hi = np.max(p+r[:,None],axis=0)
            excess = float(np.sum(hi-lo))-2
            r *= 1-max(2e-14, min(0.1,max(0.0,excess)/(2*max(float(r.max()),1e-10))))

    last = z.copy()
    try:
        tau = float(params['temperature'])
        penalty = float(params['penalty'])
        for stage in range(6):
            check()
            if time.monotonic() >= anneal_end:
                break
            def energy(v):
                nonlocal last
                check()
                if time.monotonic() >= anneal_end:
                    raise StopSearch()
                last = v.copy()
                g = geometry(v)
                c = -penalty*expit(-g/tau)
                x,y,r,w = unpack(v)
                dx,dy = x[I]-x[J],y[I]-y[J]
                d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-30)
                cp = c[:M]
                gx = np.bincount(I,weights=cp*dx/d,minlength=N)-np.bincount(J,weights=cp*dx/d,minlength=N)
                gy = np.bincount(I,weights=cp*dy/d,minlength=N)-np.bincount(J,weights=cp*dy/d,minlength=N)
                gr = -np.ones(N)-np.bincount(I,weights=cp,minlength=N)-np.bincount(J,weights=cp,minlength=N)
                b = c[M:].reshape(4,N)
                gx += b[0]-b[1]
                gy += b[2]-b[3]
                gr -= b.sum(axis=0)
                value = -r.sum()+penalty*tau*np.logaddexp(0,-g/tau).sum()
                return value,np.r_[gx,gy,gr,b[1].sum()-b[3].sum()]
            try:
                result = minimize(energy,z,jac=True,method='L-BFGS-B',bounds=bounds,
                                  options={'maxiter':int(params['iterations']),'ftol':1e-12,'gtol':1e-8,'maxls':18})
                z = result.x
            except StopSearch:
                z = last.copy()
                break
            consider(z)
            tau *= float(params['cooling'])
        check()
        objjac = np.r_[np.zeros(2*N),-np.ones(N),0.0]
        def objective(v):
            nonlocal last
            check()
            last = v.copy()
            return -v[2*N:3*N].sum()
        def cons(v):
            check()
            return geometry(v)
        def jac(v):
            check()
            return geometry(v,True)
        result = minimize(objective,z,jac=lambda v:objjac,method='SLSQP',bounds=bounds,
                          constraints=[{'type':'ineq','fun':cons,'jac':jac}],
                          options={'maxiter':120,'ftol':2e-13})
        consider(result.x)
    except StopSearch:
        pass
    if time.monotonic() < deadline:
        try:
            consider(last)
        except StopSearch:
            pass
    return best
