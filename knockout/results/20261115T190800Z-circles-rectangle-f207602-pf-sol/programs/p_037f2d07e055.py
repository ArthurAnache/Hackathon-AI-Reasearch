import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {"initial_iterations": (100, 40, 180), "release": (0.012, 0.002, 0.05), "contacts": (4, 3, 6), "portfolio": (3, 1, 5)}

class StopSearch(Exception):
    pass

def canonical(obj):
    if obj is None:
        return None
    if 'circles' in obj:
        return obj
    rows = []
    for x,y,r,m in obj['orbits']:
        rows.append([x,y,r])
        if m == 2:
            rows.append([x,-y,r])
    return {'circles': rows}

def exact_score(obj):
    try:
        a = np.asarray(canonical(obj)['circles'], dtype=float)
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return -float('inf')
        i,j = np.triu_indices(21,1)
        if np.any(np.sqrt((a[i,0]-a[j,0])**2+(a[i,1]-a[j,1])**2) < a[i,2]+a[j,2]):
            return -float('inf')
        x = list(map(Fraction, map(float,a[:,0])))
        y = list(map(Fraction, map(float,a[:,1])))
        r = list(map(Fraction, map(float,a[:,2])))
        width = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        height = max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        if width+height > 2:
            return -float('inf')
        return float(np.sum(a[:,2]))
    except (ValueError,TypeError,KeyError,OverflowError):
        return -float('inf')

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + .70*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = exact_score(init)
    best_z = None
    def tick():
        if time.monotonic() >= deadline:
            raise StopSearch()
    # A strictly feasible fallback independent of the incoming representation.
    fallback = np.array([[.2*(k%7-3),.2*(k//7-1),.09999999999999] for k in range(21)])
    fb = {'circles':fallback.tolist()}
    fs = exact_score(fb)
    if fs > best_score:
        best,best_score = fb,fs
    gids = np.r_[np.arange(3),np.repeat(np.arange(3,12),2)]
    signs = np.r_[np.zeros(3),np.tile([1.,-1.],9)]
    X = np.zeros((21,34)); Y = np.zeros_like(X); R = np.zeros_like(X)
    X[np.arange(21),gids] = 1
    R[np.arange(21),21+gids] = 1
    ind = np.flatnonzero(gids >= 3)
    Y[ind,9+gids[ind]] = signs[ind]
    pairs = []
    seen = set()
    try:
        for i in range(21):
            tick()
            for j in range(i):
                tick()
                key = (min(gids[i],gids[j]),max(gids[i],gids[j]),signs[i]*signs[j])
                if key not in seen:
                    seen.add(key); pairs.append((i,j))
        pi,pj = np.array(pairs).T
        DX,DY,RS = X[pi]-X[pj],Y[pi]-Y[pj],R[pi]+R[pj]
        W = np.zeros((36,34)); off = np.zeros(36)
        for g in range(12):
            tick()
            W[g,g]=1; W[g,21+g]=-1; W[g,33]=.5
            W[12+g,g]=-1; W[12+g,21+g]=-1; W[12+g,33]=.5
            W[24+g,21+g]=-1; W[24+g,33]=-.5; off[24+g]=1
            if g>=3:
                W[24+g,9+g]=-1
        weights=np.r_[np.ones(3),np.full(9,2.)]
        grad=np.zeros(34); grad[21:33]=-weights
        release=np.zeros(len(pi))
        bounds=[(-1.,1.)]*12+[(0.,1.)]*9+[(1e-7,.5)]*12+[(.05,1.95)]
        ai,aj=np.triu_indices(21,1)
        def pack(groups):
            v=np.zeros(34)
            v[:12]=groups[:,0]; v[12:21]=np.abs(groups[3:,1]); v[21:33]=groups[:,2]
            v[33]=np.clip(np.ptp(groups[:,0])+2*np.max(groups[:,2]),.05,1.95)
            return v
        def retain(v):
            nonlocal best,best_score,best_z
            tick()
            if not np.isfinite(v).all() or np.min(v[21:33])<=0:
                return
            a=np.column_stack((X@v,Y@v,R@v))
            d=np.sqrt((a[ai,0]-a[aj,0])**2+(a[ai,1]-a[aj,1])**2)
            a[:,2]*=min(1.,float(np.min(d/(a[ai,2]+a[aj,2]))))*(1.-8e-15)
            if np.min(a[:,2])<=0:
                return
            wh=np.max(a[:,:2]+a[:,2,None],axis=0)-np.min(a[:,:2]-a[:,2,None],axis=0)
            scale=min(1.,2./float(np.sum(wh)))*(1.-8e-15)
            a*=scale
            if float(np.sum(a[:,2]))<=best_score:
                return
            obj={'circles':a.tolist()}
            val=exact_score(obj)
            if val>best_score:
                best,best_score=obj,val
                best_z=v.copy()
        def fun(v):
            tick(); return float(grad@v)
        def jac(v):
            tick(); return grad
        def con(v):
            tick()
            dx,dy=DX@v,DY@v
            return np.r_[np.sqrt(dx*dx+dy*dy)-RS@v-release,W@v+off]
        def cj(v):
            tick()
            dx,dy=DX@v,DY@v
            d=np.maximum(np.sqrt(dx*dx+dy*dy),1e-24)
            return np.vstack(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RS,W))
        def solve(v,n):
            tick()
            count=[0]
            def cb(z):
                tick(); count[0]+=1
                if count[0]%5==0:
                    retain(z)
            result=minimize(fun,v,jac=jac,bounds=bounds,method='SLSQP',constraints={'type':'ineq','fun':con,'jac':cj},callback=cb,options={'maxiter':n,'ftol':2e-13,'disp':False})
            retain(result.x)
            return result.x
        starts=[]
        raw=canonical(init)
        if raw is not None:
            a=np.asarray(raw['circles'],float).copy()
            if a.shape==(21,3) and np.isfinite(a).all():
                lo=np.min(a[:,:2]-a[:,2,None],axis=0)
                hi=np.max(a[:,:2]+a[:,2,None],axis=0)
                a[:,:2]-=(lo+hi)/2
                projections=[]
                for swap in (False,True):
                    tick()
                    b=a.copy()
                    if swap:
                        b[:,[0,1]]=b[:,[1,0]]
                    order=np.argsort(b[:,1]); lower=order[:9]; upper=order[12:]; singles=order[9:12]
                    reflected=b[lower].copy(); reflected[:,1]*=-1
                    cost=np.sum((b[upper,None,:]-reflected[None,:,:])**2,axis=2)
                    u,v=linear_sum_assignment(cost)
                    groups=np.zeros((12,3)); groups[:3]=b[singles]; groups[:3,1]=0
                    groups[3:]=(b[upper[u]]+reflected[v])/2
                    err=float(np.sum(cost[u,v])+np.sum(b[singles,1]**2))
                    projections.append((err,pack(groups)))
                projections.sort(key=lambda p:p[0])
                starts.extend(p[1] for p in projections)
        # Staggered symmetric seeds complement the inferred cold-start geometry.
        for kind in range(2):
            tick()
            groups=np.zeros((12,3)); groups[:,2]=.07
            groups[:3,0]=[-.32,0,.32]
            if kind==0:
                groups[3:7,0]=np.linspace(-.42,.42,4); groups[3:7,1]=.16
                groups[7:,0]=np.linspace(-.42,.42,5); groups[7:,1]=.34
            else:
                groups[3:,0]=np.tile([-.28,0,.28],3)
                groups[3:,1]=np.repeat([.14,.28,.42],3)
            starts.append(pack(groups))
        # First discover basins; then concentrate all remaining work on the winner.
        for k,z in enumerate(starts[:int(params['portfolio'])]):
            tick()
            if k and time.monotonic()-start>.43*budget_s:
                break
            release[:]=0
            z=solve(z,int(params['initial_iterations']))
        if best_z is not None:
            release[:]=0
            anchor=solve(best_z.copy(),100)
            attempt=0
            while time.monotonic()<deadline:
                tick()
                release[:]=0
                if best_z is not None:
                    anchor=best_z.copy()
                dx,dy=DX@anchor,DY@anchor
                gaps=np.sqrt(dx*dx+dy*dy)-RS@anchor
                near=np.flatnonzero(gaps<.004)
                if len(near)<3:
                    break
                k=min(int(params['contacts']),len(near))
                chosen=rng.choice(near,k,replace=False)
                strength=float(params['release'])*(1.+.4*(attempt%3))
                release[chosen]=strength*rng.uniform(.7,1.3,k)
                moved=solve(anchor,18)
                release[:]=0
                solve(moved,65)
                attempt+=1
    except (StopSearch,ValueError,FloatingPointError):
        pass
    return best
