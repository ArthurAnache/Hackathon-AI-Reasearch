import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

PARAMS = {
    'orientations': (40, 12, 100),
    'iterations': (100, 20, 250),
    'restarts': (7, 1, 20),
    'random_poles': (60, 0, 200),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class Stop(Exception):
    pass

def perimeter(p, r):
    x = [Fraction(float(v)) for v in p[:, 0]]
    y = [Fraction(float(v)) for v in p[:, 1]]
    q = [Fraction(float(v)) for v in r]
    return (max(a+b for a,b in zip(x,q))-min(a-b for a,b in zip(x,q))
            +max(a+b for a,b in zip(y,q))-min(a-b for a,b in zip(y,q)))

def exact_score(p, r):
    if p.shape != (21,2) or r.shape != (21,):
        return -np.inf
    if not np.isfinite(p).all() or not np.isfinite(r).all() or np.any(r <= 0):
        return -np.inf
    d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
    if np.any(d < r[I]+r[J]) or perimeter(p,r) > 2:
        return -np.inf
    return float(np.sum(r))

def repair(p, r):
    p = p.copy()
    r = np.maximum(r.copy(), 1e-14)
    d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
    factor = min(1.0, float(np.min(d/(r[I]+r[J]))))
    r *= factor*(1-4e-15)
    hp = perimeter(p,r)
    scale = min(1.0, 2/float(hp))*(1-4e-15)
    p *= scale
    r *= scale
    return p,r

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        p0 = np.array([[(i+.5)*.2,(j+.5)*.2] for j in range(3) for i in range(7)])
        r0 = np.full(N,.1*(1-1e-12))
        init = {'circles': np.column_stack((p0,r0)).tolist()}
    a = np.asarray(init['circles'],dtype=float)
    p0,r0 = a[:,:2].copy(),a[:,2].copy()
    best = init
    bestval = exact_score(p0,r0)
    lo = np.min(p0-r0[:,None],axis=0)
    hi = np.max(p0+r0[:,None],axis=0)
    scale = 2/np.sum(hi-lo)
    p0 = (p0-(lo+hi)/2)*scale
    r0 *= scale
    angles = np.arange(int(params['orientations']))*np.pi/(2*int(params['orientations']))
    cs,ss = np.cos(angles),np.sin(angles)
    pool = []

    def orient(p,r,tag):
        if time.monotonic() >= deadline:
            raise Stop()
        xx = p[:,0,None]*cs-p[:,1,None]*ss
        yy = p[:,0,None]*ss+p[:,1,None]*cs
        widths = np.max(xx+r[:,None],axis=0)-np.min(xx-r[:,None],axis=0)
        heights = np.max(yy+r[:,None],axis=0)-np.min(yy-r[:,None],axis=0)
        k = int(np.argmin(widths+heights))
        s = 2/(widths[k]+heights[k])
        pp = np.column_stack((xx[:,k],yy[:,k]))
        pp -= np.min(pp-r[:,None],axis=0)
        pool.append((float(r.sum()*s),pp*s,r*s,float(widths[k]*s),tag))

    try:
        orient(p0,r0,-1)
        poles = []
        try:
            tri = Delaunay(p0)
            for face in tri.simplices:
                if time.monotonic() >= deadline:
                    raise Stop()
                pts = p0[face]
                poles.append(pts.mean(axis=0))
                # Side-length weights move the pole toward a gap's incenter.
                weights = np.linalg.norm(pts[[1,2,0]]-pts[[2,0,1]],axis=1)
                poles.append(np.sum(pts*weights[:,None],axis=0)/weights.sum())
        except Stop:
            raise
        except Exception:
            pass
        for k in range(int(params['random_poles'])):
            if time.monotonic() >= deadline:
                raise Stop()
            poles.append(rng.uniform(np.min(p0,axis=0)-.15,np.max(p0,axis=0)+.15))
        for k,pole in enumerate(poles):
            if time.monotonic() >= start+.19*budget_s:
                break
            v = p0-pole
            den = np.sum(v*v,axis=1)-r0*r0
            if np.min(den) <= 1e-7:
                continue
            orient(v/den[:,None],r0/den,k)
    except Stop:
        return best

    pool.sort(key=lambda t:t[0],reverse=True)
    chosen = []
    for item in pool:
        if time.monotonic() >= deadline:
            break
        # Suppress multiple nearly identical inversion poles.
        if all(np.linalg.norm(np.sort(item[2])-np.sort(old[2])) > .012 for old in chosen):
            chosen.append(item)
        if len(chosen) >= int(params['restarts']):
            break
    if not chosen:
        return best

    objective_grad = np.r_[np.zeros(2*N),-np.ones(N),0.]
    base_jac = np.zeros((M+4*N,3*N+1))
    rows = np.arange(M)
    base_jac[rows,2*N+I] = -1
    base_jac[rows,2*N+J] = -1
    k = np.arange(N)
    for b in range(4):
        rr = M+b*N+k
        base_jac[rr,2*N+k] = -1
    base_jac[M+k,k] = 1
    base_jac[M+N+k,k] = -1
    base_jac[M+N+k,-1] = 1
    base_jac[M+2*N+k,N+k] = 1
    base_jac[M+3*N+k,N+k] = -1
    base_jac[M+3*N+k,-1] = -1
    cache_z = None
    cache_v = None
    cache_j = None

    def constraints(z):
        nonlocal cache_z,cache_v,cache_j
        if time.monotonic() >= deadline:
            raise Stop()
        if cache_z is not None and np.array_equal(z,cache_z):
            return cache_v,cache_j
        x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
        dx,dy = x[I]-x[J],y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        val = np.r_[d-r[I]-r[J],x-r,w-x-r,y-r,2-w-y-r]
        jac = base_jac.copy()
        jac[rows,I] = dx/d
        jac[rows,J] = -dx/d
        jac[rows,N+I] = dy/d
        jac[rows,N+J] = -dy/d
        cache_z,cache_v,cache_j = z.copy(),val,jac
        return val,jac

    def consider(z):
        nonlocal best,bestval
        if time.monotonic() >= deadline:
            raise Stop()
        p = np.column_stack((z[:N],z[N:2*N]))
        r = z[2*N:3*N]
        if np.sum(r) <= bestval:
            return
        p,r = repair(p,r)
        score = exact_score(p,r)
        if score > bestval:
            bestval = score
            best = {'circles':np.column_stack((p,r)).tolist()}

    bounds = [(None,None)]*(2*N)+[(1e-10,1.)]*N+[(.02,1.98)]
    for _,p,r,w,tag in chosen:
        if time.monotonic() >= deadline:
            break
        z = np.r_[p[:,0],p[:,1],r,w]
        cache_z = None
        try:
            result = minimize(lambda z:-float(np.sum(z[2*N:3*N])),z,
                jac=lambda z:objective_grad,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda z:constraints(z)[0],
                             'jac':lambda z:constraints(z)[1]},
                callback=consider,
                options={'maxiter':int(params['iterations']),'ftol':2e-12,'disp':False})
            consider(result.x)
        except Stop:
            break
        except (ValueError,FloatingPointError):
            continue
    return best
