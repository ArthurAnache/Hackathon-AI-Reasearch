import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'starts': (4, 1, 8),
    'iterations': (160, 40, 300),
    'jitter': (0.035, 0.0, 0.12),
    'refine_fraction': (0.22, 0.1, 0.4),
}
N = 21
II, JJ = np.triu_indices(N, 1)

class StopSearch(Exception):
    pass

def valid(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return False
    d = np.sqrt((a[II, 0]-a[JJ, 0])**2 + (a[II, 1]-a[JJ, 1])**2)
    if np.any(d < a[II, 2] + a[JJ, 2]):
        return False
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    return max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r)) + max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)) <= 2

def repair(a):
    a = np.array(a, dtype=float, copy=True)
    if a.shape != (21, 3) or not np.isfinite(a).all():
        return None
    a[:, 2] = np.maximum(a[:, 2], 1e-10)
    d = np.sqrt((a[II, 0]-a[JJ, 0])**2 + (a[II, 1]-a[JJ, 1])**2)
    ratio = np.min(d/(a[II, 2]+a[JJ, 2]))
    if ratio <= 0:
        return None
    a[:, 2] *= min(1.0, ratio)*(1-3e-12)
    lo = (a[:, :2]-a[:, 2,None]).min(axis=0)
    hi = (a[:, :2]+a[:, 2,None]).max(axis=0)
    scale = (2-2e-12)/float(np.sum(hi-lo))
    a[:, :2] = (a[:, :2]-lo)*scale
    a[:, 2] *= scale
    return a if valid(a) else None

def model(symmetric):
    if symmetric:
        dim = 34
        ids = np.r_[np.arange(12), np.arange(3,12)]
        signs = np.r_[np.ones(12), -np.ones(9)]
        X = np.zeros((21,dim)); Y = X.copy(); R = X.copy()
        X[np.arange(21), ids] = 1
        mask = ids >= 3
        Y[np.arange(21)[mask], 12+ids[mask]-3] = signs[mask]
        R[np.arange(21), 21+ids] = 1
        pairs = [(i,j) for i in range(12) for j in range(i+1,12)]
        pairs += [(i,12+j-3) for i in range(3,12) for j in range(i,12)]
        pi = np.array([p[0] for p in pairs]); pj = np.array([p[1] for p in pairs])
        boundary = np.arange(12)
    else:
        dim = 64
        X = np.zeros((21,dim)); Y = X.copy(); R = X.copy()
        X[np.arange(21),np.arange(21)] = 1
        Y[np.arange(21),21+np.arange(21)] = 1
        R[np.arange(21),42+np.arange(21)] = 1
        pi,pj = II,JJ
        boundary = np.arange(21)
    W = np.zeros(dim); W[-1] = 1
    DX = X[pi]-X[pj]; DY = Y[pi]-Y[pj]; RS = R[pi]+R[pj]
    xb,yb,rb = X[boundary],Y[boundary],R[boundary]
    B = np.concatenate((xb-rb, W-xb-rb, -.5*W-yb-rb, -.5*W+yb-rb))
    offset = np.r_[np.zeros(2*len(boundary)),np.ones(2*len(boundary))]
    objective = -R.sum(axis=0)
    return X,Y,R,DX,DY,RS,B,offset,objective

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    end = started + 0.70*budget_s
    rng = np.random.default_rng(seed)
    best = None
    score = -float('inf')
    if init is not None:
        try:
            a = np.asarray(init['circles'],dtype=float)
            if valid(a):
                best = init
                score = float(np.sum(a[:,2]))
        except (ValueError, KeyError, TypeError):
            pass
    def offer(a):
        nonlocal best,score
        if time.monotonic() >= end:
            return
        b = repair(a)
        if b is not None:
            s = float(np.sum(b[:,2]))
            if s > score:
                best = {'circles': b.tolist()}
                score = s
    # Reliable fallback: a rectangular 3 by 7 packing.
    a = np.array([[(i+.5)*.2,(j+.5)*.2,.1*(1-1e-11)] for j in range(7) for i in range(3)])
    offer(a)
    sym = model(True)
    def optimize(z, m, deadline):
        X,Y,R,DX,DY,RS,B,offset,obj = m
        last = [z.copy()]
        cached = [None,None,None]
        def constraints(v):
            if time.monotonic() >= deadline:
                raise StopSearch()
            if cached[0] is not None and np.array_equal(v,cached[0]):
                return cached[1],cached[2]
            dx = DX@v; dy = DY@v
            d = np.sqrt(dx*dx+dy*dy+1e-28)
            c = np.r_[d-RS@v,B@v+offset]
            jac = np.concatenate(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RS,B))
            cached[:] = [v.copy(),c,jac]
            last[0] = v.copy()
            return c,jac
        def fun(v):
            if time.monotonic() >= deadline:
                raise StopSearch()
            return float(obj@v),obj
        bounds = [(None,None)]*len(z)
        if len(z)==34:
            for k in range(21,33): bounds[k]=(1e-7,.6)
        else:
            for k in range(42,63): bounds[k]=(1e-7,.6)
        bounds[-1] = (.2,1.8)
        try:
            result = minimize(fun,z,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda v:constraints(v)[0], 'jac':lambda v:constraints(v)[1]},
                options={'maxiter':int(params['iterations']),'ftol':2e-11,'disp':False})
            z = result.x
        except StopSearch:
            z = last[0]
        return np.column_stack((X@z,Y@z,R@z))
    starts = int(params['starts'])
    search_end = started + (0.70-float(params['refine_fraction'])*0.70)*budget_s
    for attempt in range(starts):
        now = time.monotonic()
        if now >= search_end:
            break
        # Independent symmetric row constructions, with varying box aspect.
        counts = [(5,4),(4,5),(3,3,3),(2,3,4)][attempt%4]
        rows = len(counts)
        w = [1.05,1.12,.65,.88][attempt%4]
        h = 2-w
        xs = list(np.linspace(.16*w,.84*w,3))
        ys = []
        for k,count in enumerate(counts):
            if time.monotonic() >= end:
                break
            xx = (np.arange(count)+.5)*w/count
            xx += rng.normal(0,float(params['jitter'])*w,count)
            xs.extend(xx.tolist())
            ys.extend([h*.5*(k+1)/(rows+.45)]*count)
        if len(xs)!=12:
            break
        z = np.r_[xs,ys,np.full(12,.075),w]
        deadline = min(search_end, now+(search_end-now)/(starts-attempt))
        candidate = optimize(z,sym,deadline)
        offer(candidate)
    if best is not None and time.monotonic() < end-.025:
        a = np.asarray(best['circles'],dtype=float)
        lo = (a[:,:2]-a[:,2,None]).min(0)
        hi = (a[:,:2]+a[:,2,None]).max(0)
        w = hi[0]-lo[0]
        p = a[:,:2]-lo
        p[:,1] -= (hi[1]-lo[1])*.5
        z = np.r_[p[:,0],p[:,1],a[:,2],w]
        candidate = optimize(z,model(False),end-.012)
        offer(candidate)
    return best
