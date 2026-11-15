import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'screen_iterations': (28, 12, 70),
    'screen_fraction': (0.32, 0.15, 0.48),
    'jitter': (0.025, 0.0, 0.10),
    'iterations': (220, 60, 350),
}
N = 21
II, JJ = np.triu_indices(N, 1)

class StopSearch(Exception):
    pass

def valid(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return False
    dx = a[II, 0] - a[JJ, 0]
    dy = a[II, 1] - a[JJ, 1]
    if np.any(np.sqrt(dx*dx + dy*dy) < a[II, 2] + a[JJ, 2]):
        return False
    xs = [Fraction(float(v)) for v in a[:, 0]]
    ys = [Fraction(float(v)) for v in a[:, 1]]
    rs = [Fraction(float(v)) for v in a[:, 2]]
    width = max(x+r for x,r in zip(xs,rs)) - min(x-r for x,r in zip(xs,rs))
    height = max(y+r for y,r in zip(ys,rs)) - min(y-r for y,r in zip(ys,rs))
    return width + height <= 2

def repair(a):
    a = np.array(a, dtype=float, copy=True)
    if a.shape != (21,3) or not np.isfinite(a).all():
        return None
    a[:,2] = np.maximum(a[:,2], 1e-10)
    dx = a[II,0]-a[JJ,0]
    dy = a[II,1]-a[JJ,1]
    ratio = float(np.min(np.sqrt(dx*dx+dy*dy)/(a[II,2]+a[JJ,2])))
    if ratio <= 0:
        return None
    a[:,2] *= min(1.0,ratio)*(1-2e-13)
    lo = np.min(a[:,:2]-a[:,2,None],axis=0)
    hi = np.max(a[:,:2]+a[:,2,None],axis=0)
    scale = (2-4e-13)/float(np.sum(hi-lo))
    a[:,:2] = (a[:,:2]-(hi+lo)*.5)*scale
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
        pi = np.array([p[0] for p in pairs]); pj = np.array([p[1] for p in pairs])
        boundary = np.arange(12)
        ridx = range(21,33)
    else:
        dim = 64
        X = np.zeros((21,dim)); Y = X.copy(); R = X.copy()
        X[np.arange(21),np.arange(21)] = 1
        Y[np.arange(21),21+np.arange(21)] = 1
        R[np.arange(21),42+np.arange(21)] = 1
        pi,pj = II,JJ
        boundary = np.arange(21)
        ridx = range(42,63)
    W = np.zeros(dim); W[-1] = 1
    xb,yb,rb = X[boundary],Y[boundary],R[boundary]
    B = np.concatenate((xb-rb,W-xb-rb,-.5*W-yb-rb,-.5*W+yb-rb))
    offset = np.r_[np.zeros(2*len(boundary)),np.ones(2*len(boundary))]
    bounds = [(None,None)]*dim
    for k in ridx:
        bounds[k] = (1e-8,.65)
    bounds[-1] = (.18,1.82)
    return X,Y,R,X[pi]-X[pj],Y[pi]-Y[pj],R[pi]+R[pj],B,offset,-R.sum(axis=0),bounds

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + .70*budget_s
    rng = np.random.default_rng(seed)
    best = None
    best_score = -float('inf')
    if init is not None:
        try:
            a = np.asarray(init['circles'],dtype=float)
            if valid(a):
                best = init
                best_score = float(np.sum(a[:,2]))
        except (ValueError,TypeError,KeyError):
            pass

    def offer(a):
        nonlocal best,best_score
        if time.monotonic() >= end:
            return -float('inf')
        b = repair(a)
        if b is None:
            return -float('inf')
        score = float(np.sum(b[:,2]))
        if score > best_score:
            best_score = score
            best = {'circles':b.tolist()}
        return score

    fallback = np.array([[(i+.5)*.2,(j+.5)*.2,.1*(1-1e-12)] for j in range(7) for i in range(3)])
    offer(fallback)
    sym = model(True)

    def optimize(z,m,deadline,maxiter):
        X,Y,R,DX,DY,RS,B,offset,obj,bounds = m
        last = z.copy()
        cache_v = None
        cache_c = None
        cache_j = None
        local_score = -float('inf')
        local_z = z.copy()
        calls = 0
        def constraints(v):
            nonlocal last,cache_v,cache_c,cache_j
            if time.monotonic() >= deadline:
                raise StopSearch()
            if cache_v is None or not np.array_equal(v,cache_v):
                dx = DX@v; dy = DY@v
                d = np.sqrt(dx*dx+dy*dy+1e-30)
                cache_c = np.r_[d-RS@v,B@v+offset]
                cache_j = np.concatenate(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RS,B))
                cache_v = v.copy()
                last = v.copy()
            return cache_c,cache_j
        def fun(v):
            if time.monotonic() >= deadline:
                raise StopSearch()
            return float(obj@v),obj
        def checkpoint(v):
            nonlocal calls,local_score,local_z
            if time.monotonic() >= deadline:
                raise StopSearch()
            calls += 1
            if calls % 7 == 0:
                s = offer(np.column_stack((X@v,Y@v,R@v)))
                if s > local_score:
                    local_score = s
                    local_z = v.copy()
        try:
            result = minimize(fun,z,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda v:constraints(v)[0],
                             'jac':lambda v:constraints(v)[1]},
                callback=checkpoint,
                options={'maxiter':int(maxiter),'ftol':8e-12,'disp':False})
            last = result.x
        except StopSearch:
            pass
        if time.monotonic() < end:
            s = offer(np.column_stack((X@last,Y@last,R@last)))
            if s > local_score:
                local_score = s
                local_z = last.copy()
        return local_z,local_score

    # A strong warm start receives the first trajectory; cold starts use
    # a small tournament of substantially different reflection quotients.
    if best_score > 2.25 and time.monotonic() < end-.04:
        a = np.asarray(best['circles'],dtype=float)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        p = a[:,:2]-lo
        p[:,1] -= (hi[1]-lo[1])*.5
        z = np.r_[p[:,0],p[:,1],a[:,2],hi[0]-lo[0]]
        optimize(z,model(False),end-.015,int(params['iterations']))
        return best

    screen_end = start + float(params['screen_fraction'])*budget_s
    archive = []
    patterns = [((5,4),1.05),((3,3,3),.65),((4,5),1.12),((2,3,4),.88)]
    for attempt,(counts,w) in enumerate(patterns):
        now = time.monotonic()
        if now >= screen_end:
            break
        h = 2-w
        xs = list(np.linspace(.16*w,.84*w,3))
        ys = []
        for k,count in enumerate(counts):
            if time.monotonic() >= screen_end:
                break
            xx = (np.arange(count)+.5)*w/count
            xx += rng.normal(0,float(params['jitter'])*w,count)
            xs.extend(xx.tolist())
            ys.extend([h*.5*(k+1)/(len(counts)+.45)]*count)
        if len(xs) != 12:
            break
        z = np.r_[xs,ys,np.full(12,.075),w]
        deadline = min(screen_end,now+(screen_end-now)/(4-attempt))
        zz,s = optimize(z,sym,deadline,int(params['screen_iterations']))
        archive.append((s,zz))
    archive.sort(key=lambda item:item[0],reverse=True)
    refine_end = end - min(.10*budget_s,.17)
    if archive and time.monotonic() < refine_end:
        z,s = optimize(archive[0][1],sym,refine_end,int(params['iterations']))
        archive[0] = (s,z)
    # Release reflection symmetry only after the reduced contact graph has
    # settled, rather than spending most of the budget in 64 dimensions.
    if best is not None and time.monotonic() < end-.025:
        a = np.asarray(best['circles'],dtype=float)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        p = a[:,:2]-lo
        p[:,1] -= (hi[1]-lo[1])*.5
        z = np.r_[p[:,0],p[:,1],a[:,2],hi[0]-lo[0]]
        optimize(z,model(False),end-.01,int(params['iterations']))
    return best
