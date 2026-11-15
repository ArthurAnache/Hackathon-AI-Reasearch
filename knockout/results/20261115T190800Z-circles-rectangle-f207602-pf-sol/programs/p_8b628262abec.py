import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {'race_iters': (58, 25, 100), 'polish_iters': (90, 40, 150), 'jitter': (0.022, 0.004, 0.08), 'release': (0.012, 0.003, 0.04)}

class StopSearch(Exception):
    pass

class EndSolve(Exception):
    pass

def canonical(obj):
    if 'circles' in obj:
        return obj
    rows = []
    for x,y,r,m in obj['orbits']:
        rows.append([x,y,r])
        if m == 2:
            rows.append([x,-y,r])
    return {'circles': rows}

def score(obj):
    try:
        a = np.asarray(canonical(obj)['circles'], dtype=float)
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return -float('inf')
        x,y,r = [[Fraction(float(v)) for v in a[:,k]] for k in range(3)]
        wh = max(u+t for u,t in zip(x,r))-min(u-t for u,t in zip(x,r))
        wh += max(u+t for u,t in zip(y,r))-min(u-t for u,t in zip(y,r))
        if wh > 2:
            return -float('inf')
        i,j = np.triu_indices(21,1)
        d = np.sqrt((a[i,0]-a[j,0])**2+(a[i,1]-a[j,1])**2)
        if np.any(d < a[i,2]+a[j,2]):
            return -float('inf')
        return float(np.sum(a[:,2]))
    except Exception:
        return -float('inf')

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + .69*budget_s
    rng = np.random.default_rng(seed)
    best = init
    bestval = score(init)
    def tick():
        if time.monotonic() >= deadline:
            raise StopSearch()
    fallback = np.array([[2*(i%7),2*(i//7),1.] for i in range(21)],float)/10.000000000001
    if bestval == -float('inf'):
        best = {'circles': fallback.tolist()}
        bestval = score(best)
    groups = np.r_[np.arange(3),np.repeat(np.arange(3,12),2)]
    signs = np.r_[np.zeros(3),np.tile([1.,-1.],9)]
    X = np.zeros((21,34)); Y = np.zeros_like(X); R = np.zeros_like(X)
    X[np.arange(21),groups] = 1.
    R[np.arange(21),21+groups] = 1.
    mask = groups >= 3
    Y[np.flatnonzero(mask),groups[mask]+9] = signs[mask]
    ai,aj = np.triu_indices(21,1)
    keys = np.column_stack((np.minimum(groups[ai],groups[aj]),np.maximum(groups[ai],groups[aj]),signs[ai]*signs[aj]))
    _,keep = np.unique(keys,axis=0,return_index=True)
    pi,pj = ai[keep],aj[keep]
    DX,DY,RS = X[pi]-X[pj],Y[pi]-Y[pj],R[pi]+R[pj]
    W = np.zeros((36,34)); off = np.zeros(36)
    q = np.arange(12)
    W[q,q] = 1.; W[q,21+q] = -1.; W[q,33] = .5
    W[12+q,q] = -1.; W[12+q,21+q] = -1.; W[12+q,33] = .5
    W[24+q,21+q] = -1.; W[24+q,33] = -.5
    W[27+np.arange(9),12+np.arange(9)] = -1.
    off[24:] = 1.
    weights = np.r_[np.ones(3),np.full(9,2.)]
    objective = np.zeros(34); objective[21:33] = -weights
    bounds = [(-1.,1.)]*12+[(0.,1.)]*9+[(1e-8,.5)]*12+[(.1,1.9)]
    release = np.zeros(len(pi))
    cache_z = None; cache_c = None; cache_j = None
    best_z = None

    def geometry(z):
        nonlocal cache_z,cache_c,cache_j
        tick()
        if cache_z is None or not np.array_equal(z,cache_z):
            dx,dy = DX@z,DY@z
            d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-25)
            cache_c = np.r_[d-RS@z,W@z+off]
            cache_j = np.vstack(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RS,W))
            cache_z = z.copy()
        return cache_c,cache_j

    def retain(z):
        nonlocal best,bestval,best_z
        tick()
        if not np.isfinite(z).all() or np.min(z[21:33]) <= 0:
            return
        a = np.column_stack((X@z,Y@z,R@z))
        d = np.sqrt((a[ai,0]-a[aj,0])**2+(a[ai,1]-a[aj,1])**2)
        shrink = min(1.,float(np.min(d/(a[ai,2]+a[aj,2]))))*(1.-8e-15)
        if shrink <= 0:
            return
        a[:,2] *= shrink
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        scale = 2./float(np.sum(hi-lo))*(1.-8e-15)
        a[:,:2] -= (lo+hi)/2
        a *= scale
        if float(np.sum(a[:,2])) <= bestval:
            return
        obj = {'circles':a.tolist()}
        val = score(obj)
        if val > bestval:
            best,bestval = obj,val
            best_z = z.copy()
            best_z[:12] -= (lo[0]+hi[0])/2
            best_z[:33] *= scale
            best_z[21:33] *= shrink
            best_z[33] = (hi[0]-lo[0])*scale

    def solve(z,n,seconds):
        tick()
        stop = min(deadline,time.monotonic()+seconds)
        last = z.copy()
        count = 0
        def check():
            tick()
            if time.monotonic() >= stop:
                raise EndSolve()
        def fun(v):
            check(); return float(objective@v)
        def jac(v):
            check(); return objective
        def con(v):
            check()
            c = geometry(v)[0].copy(); c[:len(pi)] -= release
            return c
        def cj(v):
            check(); return geometry(v)[1]
        def cb(v):
            nonlocal last,count
            last = v.copy(); count += 1
            if count % 5 == 0:
                retain(v)
            check()
        try:
            res = minimize(fun,z,jac=jac,method='SLSQP',bounds=bounds,constraints={'type':'ineq','fun':con,'jac':cj},callback=cb,options={'maxiter':int(n),'ftol':3e-14,'disp':False})
            last = res.x
        except EndSolve:
            pass
        retain(last)
        return last

    def refine(z):
        z = z.copy()
        for _ in range(5):
            tick()
            c,j = geometry(z)
            active = np.flatnonzero(c < 1e-7)
            if len(active) < 34:
                break
            step,_,rank,_ = np.linalg.lstsq(j[active],-c[active],rcond=1e-11)
            if rank < 34 or np.linalg.norm(step) > .002:
                break
            z += step
            retain(z)
            if np.max(np.abs(step)) < 2e-15:
                break
        return z

    def pack(g):
        tick()
        z = np.zeros(34)
        z[:12] = g[:,0]; z[12:21] = np.abs(g[3:,1]); z[21:33] = np.maximum(g[:,2],.001)
        lo = np.min(z[:12]-z[21:33]); hi = np.max(z[:12]+z[21:33])
        z[:12] -= (lo+hi)/2
        w = hi-lo
        h = 2*max(float(np.max(z[12:21]+z[24:33])),float(np.max(z[21:24])))
        s = 2./(w+h)
        z[:33] *= s; z[33] = w*s
        return z

    try:
        a = np.asarray(canonical(best)['circles'],float).copy()
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        a[:,:2] -= (lo+hi)/2
        projections = []
        for swap in (False,True):
            tick()
            b = a[:,[1,0,2]].copy() if swap else a.copy()
            order = np.argsort(b[:,1]); lower,axis,upper = order[:9],order[9:12],order[12:]
            reflected = b[lower].copy(); reflected[:,1] *= -1
            cost = np.sum((b[upper,None,:]-reflected[None,:,:])**2,axis=2)
            u,v = linear_sum_assignment(cost)
            g = np.vstack((b[axis],(b[upper[u]]+reflected[v])/2))
            g[:3,1] = 0.
            projections.append((float(np.sum(cost[u,v])+np.sum(b[axis,1]**2)),pack(g)))
        projections.sort(key=lambda p:p[0])
        seeds = [projections[0][1]]
        for counts in ((4,5),(3,3,3)):
            tick()
            rows = [[x,0.,1.] for x in (-2.,0.,2.)]
            for level,count in enumerate(counts,1):
                tick()
                offset = 1. if counts == (3,3,3) and level%2 else 0.
                rows.extend([[2*k-(count-1)+offset,level*np.sqrt(3.),1.] for k in range(count)])
            seeds.append(pack(np.asarray(rows,float)))
        results = []
        for z in seeds:
            tick()
            z = solve(z,params['race_iters'],.155*budget_s)
            c = geometry(z)[0]
            merit = float(weights@z[21:33])-30*max(0.,-float(np.min(c)))
            results.append((merit,z.copy()))
            if np.min(c) > -1e-7:
                refine(z)
        results.sort(key=lambda p:p[0],reverse=True)
        anchor = best_z.copy() if best_z is not None else results[0][1]
        anchor = solve(anchor,params['polish_iters'],.14*budget_s)
        if np.min(geometry(anchor)[0]) > -1e-7:
            refine(anchor)
        attempt = 0
        while time.monotonic() < deadline:
            tick()
            base = best_z.copy() if best_z is not None else anchor.copy()
            if attempt % 2 == 0:
                c,j = geometry(base)
                near = np.flatnonzero(c[:len(pi)] < .002)
                if len(near) >= 3:
                    chosen = rng.choice(near,min(4,len(near)),replace=False)
                    release[chosen] = float(params['release'])
                    probe = solve(base,15,.045*budget_s)
                    release[:] = 0.
                else:
                    probe = base
            else:
                probe = base.copy()
                probe[:21] += rng.normal(0.,params['jitter'],21)
                probe[12:21] = np.clip(probe[12:21],.001,.98)
                probe[:12] = np.clip(probe[:12],-.98,.98)
            z = solve(probe,params['polish_iters'],.12*budget_s)
            if np.min(geometry(z)[0]) > -1e-7:
                refine(z)
            attempt += 1
    except (StopSearch,ValueError,FloatingPointError,np.linalg.LinAlgError):
        pass
    return best
