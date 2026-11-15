import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'starts': (4, 2, 8),
    'jitter': (0.018, 0.0, 0.08),
    'search_fraction': (0.72, 0.45, 0.85),
    'iterations': (180, 50, 350),
}
N = 21
II, JJ = np.triu_indices(N, 1)

class Finished(Exception):
    pass

def valid(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return False
    d = np.sqrt((a[II, 0]-a[JJ, 0])**2 + (a[II, 1]-a[JJ, 1])**2)
    if np.any(d < a[II, 2]+a[JJ, 2]):
        return False
    x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
    return (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
            +max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))) <= 2

def repair(a):
    a = np.array(a, dtype=float, copy=True)
    if a.shape != (21,3) or not np.isfinite(a).all() or np.min(a[:,2]) <= 0:
        return None
    d = np.sqrt((a[II,0]-a[JJ,0])**2+(a[II,1]-a[JJ,1])**2)
    q = float(np.min(d/(a[II,2]+a[JJ,2])))
    if q <= 0:
        return None
    a[:,2] *= min(q,1.0)*(1-8e-14)
    lo = np.min(a[:,:2]-a[:,2,None],axis=0)
    hi = np.max(a[:,:2]+a[:,2,None],axis=0)
    scale = (2-3e-13)/float(np.sum(hi-lo))
    a[:,:2] = (a[:,:2]-(lo+hi)/2)*scale
    a[:,2] *= scale
    return a if valid(a) else None

def model(symmetric):
    if symmetric:
        dim = 34
        ids = np.r_[np.arange(12),np.arange(3,12)]
        signs = np.r_[np.ones(12),-np.ones(9)]
        X = np.zeros((21,dim)); Y = X.copy(); R = X.copy()
        X[np.arange(21),ids] = 1
        mask = ids >= 3
        Y[np.arange(21)[mask],12+ids[mask]-3] = signs[mask]
        R[np.arange(21),21+ids] = 1
        pairs = [(i,j) for i in range(12) for j in range(i+1,12)]
        pairs += [(i,12+j-3) for i in range(3,12) for j in range(i,12)]
        pi,pj = np.array(pairs).T
        bd = np.arange(12)
        radii = range(21,33)
    else:
        dim = 64
        X = np.zeros((21,dim)); Y = X.copy(); R = X.copy()
        X[np.arange(21),np.arange(21)] = 1
        Y[np.arange(21),21+np.arange(21)] = 1
        R[np.arange(21),42+np.arange(21)] = 1
        pi,pj = II,JJ
        bd = np.arange(21)
        radii = range(42,63)
    W = np.zeros(dim); W[-1] = 1
    xb,yb,rb = X[bd],Y[bd],R[bd]
    B = np.concatenate((xb-rb,W-xb-rb,-.5*W-yb-rb,-.5*W+yb-rb))
    offset = np.r_[np.zeros(2*len(bd)),np.ones(2*len(bd))]
    bounds = [(None,None)]*dim
    for k in radii:
        bounds[k] = (1e-7,.6)
    bounds[-1] = (.25,1.75)
    return X,Y,R,X[pi]-X[pj],Y[pi]-Y[pj],R[pi]+R[pj],B,offset,-R.sum(0),bounds

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + .695*budget_s
    rng = np.random.default_rng(seed)
    best = None
    score = -float('inf')
    if init is not None:
        try:
            a = np.asarray(init['circles'],dtype=float)
            if valid(a):
                best = init
                score = float(np.sum(a[:,2]))
        except (ValueError,TypeError,KeyError):
            pass
    def offer(a):
        nonlocal best,score
        if time.monotonic() >= end:
            return
        b = repair(a)
        if b is not None:
            s = float(np.sum(b[:,2]))
            if s > score:
                best = {'circles':b.tolist()}
                score = s
    offer(np.array([[(i+.5)*.2,(j+.5)*.2,.1] for j in range(7) for i in range(3)]))
    sym = model(True)
    full = model(False)
    archive = []

    def optimize(z,m,deadline):
        X,Y,R,DX,DY,RS,B,off,obj,bounds = m
        cache = [None,None,None]
        last = [z.copy()]
        count = [0]
        def con(v):
            if time.monotonic() >= deadline:
                raise Finished()
            if cache[0] is not None and np.array_equal(cache[0],v):
                return cache[1],cache[2]
            dx,dy = DX@v,DY@v
            d = np.sqrt(dx*dx+dy*dy+1e-30)
            c = np.r_[d-RS@v,B@v+off]
            J = np.concatenate(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RS,B))
            cache[:] = [v.copy(),c,J]
            return c,J
        def fun(v):
            if time.monotonic() >= deadline:
                raise Finished()
            return float(obj@v),obj
        def callback(v):
            if time.monotonic() >= deadline:
                raise Finished()
            last[0] = v.copy()
            count[0] += 1
            if count[0] % 5 == 0:
                offer(np.column_stack((X@v,Y@v,R@v)))
        try:
            result = minimize(fun,z,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda v:con(v)[0],'jac':lambda v:con(v)[1]},
                callback=callback,options={'maxiter':int(params['iterations']),'ftol':8e-12,'disp':False})
            v = result.x
        except Finished:
            v = last[0]
        a = np.column_stack((X@v,Y@v,R@v))
        offer(a)
        if len(v)==34 and time.monotonic()<end:
            b = repair(a)
            if b is not None:
                archive.append((float(np.sum(b[:,2])),v.copy()))
        return v

    def full_vector(a):
        lo = (a[:,:2]-a[:,2,None]).min(0)
        hi = (a[:,:2]+a[:,2,None]).max(0)
        p = a[:,:2]-lo
        p[:,1] -= (hi[1]-lo[1])/2
        return np.r_[p[:,0],p[:,1],a[:,2],hi[0]-lo[0]]

    if score > 2.2 and time.monotonic()<end:
        optimize(full_vector(np.asarray(best['circles'])),full,min(end-.02,start+.26*budget_s))
    search_end = start + .695*float(params['search_fraction'])*budget_s
    nstarts = int(params['starts'])
    for attempt in range(nstarts):
        now = time.monotonic()
        if now >= search_end-.015:
            break
        counts = [(5,4),(4,5),(3,3,3),(2,3,4)][attempt%4]
        w = [1.05,1.12,.66,.88][attempt%4]
        h = 2-w
        xs = list(np.linspace(.16*w,.84*w,3))
        ys = []
        for k,c in enumerate(counts):
            if time.monotonic()>=search_end:
                break
            xx = (np.arange(c)+.5)*w/c
            xx += rng.normal(0,float(params['jitter'])*w,c)
            xs.extend(xx.tolist())
            ys.extend([h*.5*(k+1)/(len(counts)+.45)]*c)
        if len(xs)!=12:
            break
        z = np.r_[xs,ys,np.full(12,.075),w]
        allowance = (search_end-now)/(nstarts-attempt)
        optimize(z,sym,min(search_end,now+allowance))
    if archive and time.monotonic()<end-.05:
        archive.sort(key=lambda t:t[0],reverse=True)
        z = archive[0][1].copy()
        optimize(z,sym,min(end-.025,time.monotonic()+.09*budget_s))
    if best is not None and time.monotonic()<end-.025:
        optimize(full_vector(np.asarray(best['circles'])),full,end-.012)
    return best
