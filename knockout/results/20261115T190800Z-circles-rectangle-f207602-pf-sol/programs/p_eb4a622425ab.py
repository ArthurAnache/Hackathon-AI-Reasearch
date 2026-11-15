import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {'release': (0.009, 0.001, 0.04), 'contacts': (4, 3, 5), 'probe_iterations': (24, 10, 45), 'polish_iterations': (65, 25, 120)}

class StopSearch(Exception):
    pass

class StopProbe(Exception):
    pass

def canonical(obj):
    if 'circles' in obj:
        return obj
    out = []
    for x,y,r,m in obj['orbits']:
        out.append([x,y,r])
        if m == 2:
            out.append([x,-y,r])
    return {'circles': out}

def score(obj):
    try:
        a = np.asarray(canonical(obj)['circles'], dtype=float)
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return -float('inf')
        x,y,r = [[Fraction(float(v)) for v in a[:,k]] for k in range(3)]
        p = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        p += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        if p > 2:
            return -float('inf')
        i,j = np.triu_indices(21,1)
        if np.any(np.sqrt((a[i,0]-a[j,0])**2+(a[i,1]-a[j,1])**2) < a[i,2]+a[j,2]):
            return -float('inf')
        return float(np.sum(a[:,2]))
    except Exception:
        return -float('inf')

def construct(params, seed, budget_s, init=None):
    deadline = time.monotonic()+0.69*budget_s
    rng = np.random.default_rng(seed)
    best = init
    bestval = score(init)
    def tick():
        if time.monotonic() >= deadline:
            raise StopSearch()
    groups = np.r_[np.arange(3),np.repeat(np.arange(3,12),2)]
    signs = np.r_[np.zeros(3),np.tile([1.,-1.],9)]
    X = np.zeros((21,34)); Y = X.copy(); R = X.copy()
    X[np.arange(21),groups] = 1
    R[np.arange(21),21+groups] = 1
    mask = groups >= 3
    Y[np.flatnonzero(mask),groups[mask]+9] = signs[mask]
    ai,aj = np.triu_indices(21,1)
    keys = np.column_stack((np.minimum(groups[ai],groups[aj]),np.maximum(groups[ai],groups[aj]),signs[ai]*signs[aj]))
    _,keep = np.unique(keys,axis=0,return_index=True)
    pi,pj = ai[keep],aj[keep]
    DX,DY,RS = X[pi]-X[pj],Y[pi]-Y[pj],R[pi]+R[pj]
    W = np.zeros((36,34)); off = np.zeros(36)
    q = np.arange(12)
    W[q,q]=1; W[q,21+q]=-1; W[q,33]=.5
    W[12+q,q]=-1; W[12+q,21+q]=-1; W[12+q,33]=.5
    W[24+q,21+q]=-1; W[24+q,33]=-.5
    W[27+np.arange(9),12+np.arange(9)]=-1
    off[24:]=1
    weights=np.r_[np.ones(3),np.full(9,2.)]
    objective=np.zeros(34); objective[21:33]=-weights
    bounds=[(-1.,1.)]*12+[(0.,1.)]*9+[(1e-8,.6)]*12+[(.1,1.9)]
    release=np.zeros(len(pi))
    anchor_best=None
    local_end=deadline
    last=None
    cached_v=None; cached_c=None; cached_j=None

    def retain(v):
        nonlocal best,bestval,anchor_best
        tick()
        if v is None or not np.isfinite(v).all():
            return
        a=np.column_stack((X@v,Y@v,R@v))
        if np.any(a[:,2]<=0):
            return
        d=np.sqrt((a[ai,0]-a[aj,0])**2+(a[ai,1]-a[aj,1])**2)
        shrink=min(1.,float(np.min(d/(a[ai,2]+a[aj,2]))))*(1.-4e-15)
        if shrink<=0:
            return
        a[:,2]*=shrink
        span=np.max(a[:,:2]+a[:,2,None],axis=0)-np.min(a[:,:2]-a[:,2,None],axis=0)
        scale=2./float(np.sum(span))*(1.-4e-15)
        a*=scale
        if float(np.sum(a[:,2]))<=bestval:
            return
        obj={'circles':a.tolist()}
        val=score(obj)
        if val>bestval:
            best,bestval=obj,val
            anchor_best=v.copy()
            anchor_best[:33]*=scale
            anchor_best[21:33]*=shrink
            lo=np.min(a[:,0]-a[:,2]); hi=np.max(a[:,0]+a[:,2])
            anchor_best[:12]-=(lo+hi)*.5
            anchor_best[33]=hi-lo

    def check():
        tick()
        if time.monotonic()>=local_end:
            raise StopProbe()
    def fun(v):
        check(); return float(objective@v)
    def jac(v):
        check(); return objective
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
    def con(v):
        update(v)
        c=cached_c.copy(); c[:len(pi)]-=release
        return c
    def cjac(v):
        update(v); return cached_j
    def callback(v):
        nonlocal last
        last=v.copy(); check()
    def solve(v,iterations,seconds):
        nonlocal local_end,last
        tick()
        local_end=min(deadline,time.monotonic()+seconds)
        last=v.copy()
        try:
            res=minimize(fun,v,jac=jac,method='SLSQP',bounds=bounds,constraints={'type':'ineq','fun':con,'jac':cjac},callback=callback,options={'maxiter':int(iterations),'ftol':8e-14,'disp':False})
            last=res.x.copy()
        except StopProbe:
            pass
        retain(last)
        return last.copy()

    try:
        starts=[]
        if init is not None:
            a=np.asarray(canonical(init)['circles'],float).copy()
            low=np.min(a[:,:2]-a[:,2,None],axis=0)
            high=np.max(a[:,:2]+a[:,2,None],axis=0)
            a[:,:2]-=(low+high)/2
            for swap in (False,True):
                tick()
                b=a[:,[1,0,2]].copy() if swap else a.copy()
                order=np.argsort(b[:,1]); lower,axis,upper=order[:9],order[9:12],order[12:]
                reflected=b[lower].copy(); reflected[:,1]*=-1
                cost=np.sum((b[upper,None,:]-reflected[None,:,:])**2,axis=2)
                u,v=linear_sum_assignment(cost)
                g=np.vstack((b[axis],(b[upper[u]]+reflected[v])/2))
                z=np.zeros(34); z[:12]=g[:,0]; z[12:21]=np.abs(g[3:,1]); z[21:33]=g[:,2]
                z[33]=high[int(swap)]-low[int(swap)]
                error=float(np.sum(cost[u,v])+np.sum(b[axis,1]**2))
                starts.append((error,z))
        if not starts:
            z=np.zeros(34)
            xx=np.r_[[-2.,0.,2.],[-3.,-1.,1.,3.],[-4.,-2.,0.,2.,4.]]
            yy=np.r_[np.full(4,np.sqrt(3.)),np.full(5,2*np.sqrt(3.))]
            s=2./(10+2*(1+2*np.sqrt(3.)))
            z[:12]=xx*s; z[12:21]=yy*s; z[21:33]=s; z[33]=10*s
            starts=[(0.,z)]
        z=min(starts,key=lambda p:p[0])[1]
        retain(z)
        polished=solve(z,params['polish_iterations'],.23*budget_s)
        anchor=anchor_best.copy() if anchor_best is not None else polished.copy()
        attempt=0
        while time.monotonic()<deadline:
            tick()
            base=anchor_best.copy() if anchor_best is not None else anchor.copy()
            gaps=np.hypot(DX@base,DY@base)-RS@base
            near=np.flatnonzero(gaps<.001)
            if len(near)<3:
                break
            # Cluster by shared orbit, allowing simultaneous local graph surgery.
            pivot=int(rng.choice(near))
            gp={int(groups[pi[pivot]]),int(groups[pj[pivot]])}
            affinity=np.isin(groups[pi[near]],list(gp)) | np.isin(groups[pj[near]],list(gp))
            priority=rng.random(len(near))-.8*affinity
            chosen=near[np.argsort(priority)[:int(params['contacts'])]]
            release[:]=0
            amplitude=float(params['release'])*(1.7**(attempt%3-1))
            release[chosen]=amplitude*rng.uniform(.75,1.25,len(chosen))
            displaced=solve(base,params['probe_iterations'],.105*budget_s)
            release[:]=0
            solve(displaced,params['polish_iterations'],.19*budget_s)
            attempt+=1
    except (StopSearch,ValueError,FloatingPointError):
        pass
    return best
