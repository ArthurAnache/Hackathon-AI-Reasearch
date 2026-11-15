import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {'initial_iters': (85, 40, 140), 'polish_iters': (100, 40, 180), 'release': (0.007, 0.001, 0.03), 'contacts': (3, 3, 5)}

class StopSearch(Exception):
    pass

class StopSlice(Exception):
    pass

def canonical(obj):
    if 'circles' in obj:
        return obj
    rows = []
    for x, y, r, m in obj['orbits']:
        rows.append([x, y, r])
        if m == 2:
            rows.append([x, -y, r])
    return {'circles': rows}

def score(obj):
    try:
        a = np.asarray(canonical(obj)['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
        span = max(u+t for u,t in zip(x,r))-min(u-t for u,t in zip(x,r))
        span += max(u+t for u,t in zip(y,r))-min(u-t for u,t in zip(y,r))
        if span > 2:
            return -float('inf')
        i, j = np.triu_indices(21, 1)
        if np.any(np.sqrt((a[i,0]-a[j,0])**2+(a[i,1]-a[j,1])**2) < a[i,2]+a[j,2]):
            return -float('inf')
        return float(np.sum(a[:,2]))
    except Exception:
        return -float('inf')

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + .695*budget_s
    rng = np.random.default_rng(seed)
    best = init
    bestval = score(init)
    fallback = {'circles': [[(i%7)*.199999999999, (i//7)*.199999999999, .099999999999] for i in range(21)]}
    fv = score(fallback)
    if fv > bestval:
        best, bestval = fallback, fv
    def tick():
        if time.monotonic() >= deadline:
            raise StopSearch()
    groups = np.r_[np.arange(3), np.repeat(np.arange(3,12),2)]
    signs = np.r_[np.zeros(3), np.tile([1.,-1.],9)]
    X = np.zeros((21,34)); Y = X.copy(); R = X.copy()
    X[np.arange(21),groups] = 1.
    R[np.arange(21),21+groups] = 1.
    mask = groups >= 3
    Y[np.flatnonzero(mask),groups[mask]+9] = signs[mask]
    ai, aj = np.triu_indices(21,1)
    keys = np.column_stack((np.minimum(groups[ai],groups[aj]),np.maximum(groups[ai],groups[aj]),signs[ai]*signs[aj]))
    _, keep = np.unique(keys,axis=0,return_index=True)
    pi,pj = ai[keep],aj[keep]
    DX,DY,RS = X[pi]-X[pj],Y[pi]-Y[pj],R[pi]+R[pj]
    W = np.zeros((36,34)); off = np.zeros(36)
    q = np.arange(12)
    W[q,q]=1.; W[q,21+q]=-1.; W[q,33]=.5
    W[12+q,q]=-1.; W[12+q,21+q]=-1.; W[12+q,33]=.5
    W[24+q,21+q]=-1.; W[24+q,33]=-.5
    W[27+np.arange(9),12+np.arange(9)]=-1.
    off[24:]=1.
    weights=np.r_[np.ones(3),np.full(9,2.)]
    objective=np.zeros(34); objective[21:33]=-weights
    bounds=[(-1.,1.)]*12+[(0.,1.)]*9+[(1e-8,.6)]*12+[(.1,1.9)]
    release=np.zeros(len(pi))
    anchor=None
    local_end=deadline
    cached_v=None; cached_c=None; cached_j=None
    last=None

    def retain(v):
        nonlocal best,bestval,anchor
        tick()
        if v is None or not np.isfinite(v).all():
            return
        a=np.column_stack((X@v,Y@v,R@v))
        if np.any(a[:,2]<=0):
            return
        d=np.sqrt((a[ai,0]-a[aj,0])**2+(a[ai,1]-a[aj,1])**2)
        shrink=min(1.,float(np.min(d/(a[ai,2]+a[aj,2]))))*(1.-1.2e-14)
        if shrink<=0:
            return
        a[:,2]*=shrink
        lo=np.min(a[:,:2]-a[:,2,None],axis=0)
        hi=np.max(a[:,:2]+a[:,2,None],axis=0)
        scale=2./float(np.sum(hi-lo))*(1.-1.2e-14)
        a*=scale
        if float(np.sum(a[:,2]))<=bestval:
            return
        obj={'circles':a.tolist()}
        val=score(obj)
        if val>bestval:
            best,bestval=obj,val
            anchor=v.copy()
            anchor[:33]*=scale
            anchor[21:33]*=shrink
            anchor[:12]-=(lo[0]+hi[0])*.5*scale
            anchor[33]=(hi[0]-lo[0])*scale

    def check():
        tick()
        if time.monotonic()>=local_end:
            raise StopSlice()
    def update(v):
        nonlocal cached_v,cached_c,cached_j
        check()
        if cached_v is None or not np.array_equal(v,cached_v):
            dx,dy=DX@v,DY@v
            d=np.sqrt(dx*dx+dy*dy)
            inv=1./np.maximum(d,1e-24)
            cached_v=v.copy()
            cached_c=np.r_[d-RS@v,W@v+off]
            cached_j=np.vstack(((dx*inv)[:,None]*DX+(dy*inv)[:,None]*DY-RS,W))
    def fun(v):
        check()
        return float(objective@v)
    def jac(v):
        check()
        return objective
    def con(v):
        update(v)
        c=cached_c.copy(); c[:len(pi)]-=release
        return c
    def cjac(v):
        update(v)
        return cached_j
    def solve(v,iterations,seconds):
        nonlocal local_end,last
        tick()
        local_end=min(deadline-.003,time.monotonic()+max(.005,seconds))
        last=np.asarray(v).copy()
        count=0
        def cb(z):
            nonlocal last,count
            last=z.copy(); count+=1
            check()
            if count%5==0:
                retain(z)
        try:
            res=minimize(fun,last,jac=jac,method='SLSQP',bounds=bounds,constraints={'type':'ineq','fun':con,'jac':cjac},callback=cb,options={'maxiter':int(iterations),'ftol':7e-14,'disp':False})
            last=res.x.copy()
        except StopSlice:
            pass
        retain(last)
        return last.copy()
    def polish(v):
        nonlocal local_end
        local_end=deadline
        update(v)
        if np.min(cached_c)<-2e-6:
            return
        active=np.flatnonzero(np.abs(cached_c)<3e-6)
        if len(active)<34:
            return
        for unused in range(5):
            tick()
            update(v)
            delta,_,rank,_=np.linalg.lstsq(cached_j[active],-cached_c[active],rcond=1e-11)
            if rank<34 or np.max(np.abs(delta))>.002:
                return
            v=v+delta
            retain(v)
            if np.max(np.abs(delta))<2e-16:
                return
    def pack(g):
        tick()
        z=np.zeros(34)
        z[:12]=g[:,0]; z[12:21]=np.abs(g[3:,1]); z[21:33]=np.maximum(g[:,2],.002)
        w=np.max(g[:,0]+z[21:33])-np.min(g[:,0]-z[21:33])
        h=2*np.max(np.abs(g[:,1])+z[21:33])
        s=1.98/max(w+h,1e-9)
        z[:33]*=s; z[33]=np.clip(w*s,.1,1.9)
        return z
    def lattice(counts):
        rows=[[x,0.,1.] for x in (-2.,0.,2.)]
        for level,n in enumerate(counts,1):
            tick()
            rows.extend([[2*k-(n-1),level*np.sqrt(3.),1.] for k in range(n)])
        return pack(np.asarray(rows,float))
    try:
        projections=[]
        try:
            a=np.asarray(canonical(init)['circles'],float).copy()
            if a.shape==(21,3) and np.isfinite(a).all():
                lo=np.min(a[:,:2]-a[:,2,None],axis=0)
                hi=np.max(a[:,:2]+a[:,2,None],axis=0)
                a[:,:2]-=(lo+hi)*.5
                for swap in (False,True):
                    tick()
                    b=a[:,[1,0,2]].copy() if swap else a.copy()
                    order=np.argsort(b[:,1]); lower,axis,upper=order[:9],order[9:12],order[12:]
                    reflected=b[lower].copy(); reflected[:,1]*=-1
                    cost=np.sum((b[upper,None,:]-reflected[None,:,:])**2,axis=2)
                    u,v=linear_sum_assignment(cost)
                    g=np.vstack((b[axis],(b[upper[u]]+reflected[v])/2))
                    g[:3,1]=0.
                    error=float(np.sum(cost[u,v])+np.sum(b[axis,1]**2))
                    projections.append((error,pack(g)))
        except (TypeError,KeyError,ValueError):
            pass
        projections.sort(key=lambda t:t[0])
        endpoints=[]
        if projections:
            z=solve(projections[0][1],params['initial_iters'],.235*budget_s)
            endpoints.append(z)
            polish(z)
        z=solve(lattice((4,5)),params['initial_iters'],.225*budget_s)
        endpoints.append(z)
        polish(z)
        if anchor is not None:
            z=solve(anchor.copy(),params['polish_iters'],.13*budget_s)
            polish(z)
        if bestval<2.36 and time.monotonic()<deadline-.06*budget_s:
            z=projections[1][1] if len(projections)>1 else lattice((3,3,3))
            z=solve(z,params['polish_iters'],.075*budget_s)
            polish(z)
        attempt=0
        while time.monotonic()<deadline-.025 and anchor is not None:
            tick()
            base=anchor.copy()
            gaps=np.hypot(DX@base,DY@base)-RS@base
            near=np.flatnonzero(gaps<.0005)
            if len(near)<3:
                break
            pivot=int(rng.choice(near))
            gp=[groups[pi[pivot]],groups[pj[pivot]]]
            affinity=np.isin(groups[pi[near]],gp)|np.isin(groups[pj[near]],gp)
            chosen=near[np.argsort(rng.random(len(near))-.8*affinity)[:params['contacts']]]
            release[:]=0.
            release[chosen]=params['release']*(1.5**(attempt%3-1))
            remaining=deadline-time.monotonic()
            displaced=solve(base,22,.33*remaining)
            release[:]=0.
            z=solve(displaced,params['polish_iters'],max(.005,deadline-time.monotonic()-.006))
            polish(z)
            attempt+=1
    except (StopSearch,StopSlice,ValueError,FloatingPointError,np.linalg.LinAlgError):
        pass
    return best
