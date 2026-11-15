import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {'polish': (85, 30, 150), 'probe': (42, 20, 80), 'jitter': (0.022, 0.003, 0.08), 'release': (0.012, 0.002, 0.04)}

class StopSearch(Exception):
    pass

class SliceEnd(Exception):
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
        best = {'circles':fallback.tolist()}
        bestval = score(best)
    ids = np.r_[np.arange(3),np.repeat(np.arange(3,12),2)]
    signs = np.r_[np.zeros(3),np.tile([1.,-1.],9)]
    X = np.zeros((21,34)); Y = X.copy(); R = X.copy()
    X[np.arange(21),ids] = 1
    R[np.arange(21),21+ids] = 1
    mask = ids >= 3
    Y[np.flatnonzero(mask),ids[mask]+9] = signs[mask]
    ai,aj = np.triu_indices(21,1)
    keys = np.column_stack((np.minimum(ids[ai],ids[aj]),np.maximum(ids[ai],ids[aj]),signs[ai]*signs[aj]))
    _,keep = np.unique(keys,axis=0,return_index=True)
    pi,pj = ai[keep],aj[keep]
    DX,DY,RS = X[pi]-X[pj],Y[pi]-Y[pj],R[pi]+R[pj]
    W = np.zeros((36,34)); off = np.zeros(36)
    q = np.arange(12)
    W[q,q] = 1; W[q,21+q] = -1; W[q,33] = .5
    W[12+q,q] = -1; W[12+q,21+q] = -1; W[12+q,33] = .5
    W[24+q,21+q] = -1; W[24+q,33] = -.5
    W[27+np.arange(9),12+np.arange(9)] = -1
    off[24:] = 1
    weights = np.r_[np.ones(3),np.full(9,2.)]
    objective = np.zeros(34); objective[21:33] = -weights
    bounds = [(-1.,1.)]*12+[(0.,1.)]*9+[(1e-8,.5)]*12+[(.1,1.9)]
    release = np.zeros(len(pi))
    cached_z = None
    cached_c = None
    cached_j = None
    best_z = None
    def geometry(z):
        nonlocal cached_z,cached_c,cached_j
        tick()
        if cached_z is None or not np.array_equal(z,cached_z):
            dx,dy = DX@z,DY@z
            d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-24)
            cached_c = np.r_[d-RS@z,W@z+off]
            cached_j = np.vstack(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RS,W))
            cached_z = z.copy()
        return cached_c,cached_j
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
    def fun(z):
        tick(); return float(objective@z)
    def jac(z):
        tick(); return objective
    def con(z):
        c = geometry(z)[0].copy()
        c[:len(pi)] -= release
        return c
    def cj(z):
        return geometry(z)[1]
    def solve(z,n,seconds):
        tick()
        until = min(deadline,time.monotonic()+seconds)
        last = z.copy()
        counter = 0
        def cb(v):
            nonlocal last,counter
            tick()
            last = v.copy(); counter += 1
            if counter % 4 == 0:
                retain(v)
            if time.monotonic() >= until:
                raise SliceEnd()
        try:
            res = minimize(fun,z,jac=jac,bounds=bounds,method='SLSQP',constraints={'type':'ineq','fun':con,'jac':cj},callback=cb,options={'maxiter':int(n),'ftol':2e-14,'disp':False})
            last = res.x
        except SliceEnd:
            pass
        retain(last)
        return last
    def refine(z):
        z = z.copy()
        for _ in range(4):
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
    def lattice(which):
        tick()
        if which == 0:
            xx = np.r_[[-2.,0.,2.],[-3.,-1.,1.,3.],[-4.,-2.,0.,2.,4.]]
            yy = np.r_[np.full(4,np.sqrt(3.)),np.full(5,2*np.sqrt(3.))]
        else:
            xx = np.r_[[-2.,0.,2.],[-1.,1.,3.],[-2.,0.,2.],[-1.,1.,3.]]
            yy = np.repeat(np.arange(1,4)*np.sqrt(3.),3)
        w = np.max(xx+1)-np.min(xx-1)
        h = 2*(np.max(yy)+1)
        s = 2/(w+h)
        z = np.zeros(34)
        z[:12] = (xx-(np.max(xx)+np.min(xx))/2)*s
        z[12:21] = yy*s; z[21:33] = s; z[33] = w*s
        return z
    try:
        tick()
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
            z = np.zeros(34)
            z[:12] = g[:,0]; z[12:21] = np.abs(g[3:,1]); z[21:33] = np.maximum(g[:,2],1e-6)
            z[33] = hi[int(swap)]-lo[int(swap)]
            projections.append((float(np.sum(cost[u,v])+np.sum(b[axis,1]**2)),z))
        projections.sort(key=lambda p:p[0])
        candidates = []
        seeds = [projections[0][1],lattice(0),lattice(1)]
        if projections[1][0] < .002:
            seeds.append(projections[1][1])
        for z in seeds:
            tick()
            z = solve(z,params['polish'],.17*budget_s)
            c = geometry(z)[0]
            merit = float(weights@z[21:33])-30*max(0.,-float(np.min(c)))
            candidates.append((merit,z.copy()))
            if np.min(c) > -1e-7:
                refine(z)
        candidates.sort(key=lambda t:t[0],reverse=True)
        anchor = solve(candidates[0][1],params['polish'],.12*budget_s)
        refine(anchor)
        attempt = 0
        while time.monotonic() < deadline:
            tick()
            base = best_z.copy() if best_z is not None else anchor.copy()
            if attempt % 3 == 2:
                c,j = geometry(base)
                near = np.flatnonzero(c[:len(pi)] < .002)
                if len(near) >= 3:
                    chosen = rng.choice(near,min(4,len(near)),replace=False)
                    release[chosen] = float(params['release'])*rng.uniform(.6,1.4,len(chosen))
                    base = solve(base,16,.045*budget_s)
                    release[:] = 0
            else:
                base[:21] += rng.normal(0.,float(params['jitter']),21)
                base[12:21] = np.maximum(base[12:21],.001)
            z = solve(base,params['polish'],.13*budget_s)
            if np.min(geometry(z)[0]) > -1e-7:
                refine(z)
            attempt += 1
    except (StopSearch,ValueError,FloatingPointError,np.linalg.LinAlgError):
        pass
    return best
