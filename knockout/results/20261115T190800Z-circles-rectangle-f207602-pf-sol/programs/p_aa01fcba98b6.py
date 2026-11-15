import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {
    'fraction': (0.70, 0.5, 0.75),
    'amplitude': (0.045, 0.005, 0.15),
    'iterations': (65, 20, 150),
    'cluster': (4, 3, 8),
}

class StopSearch(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    i, j = np.triu_indices(21, 1)
    d = np.sqrt((a[i, 0]-a[j, 0])**2 + (a[i, 1]-a[j, 1])**2)
    if np.any(d < a[i, 2]+a[j, 2]):
        return -np.inf
    x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
    perimeter = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
    perimeter += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
    if perimeter > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + float(params['fraction'])*budget_s
    rng = np.random.default_rng(seed)
    best = init
    score = -np.inf
    if init is not None:
        original = np.array(init['circles'], dtype=float)
        score = exact_score(original)
        a = original.copy()
    else:
        a = np.array([[0.17+0.33*k, 0.08+0.14*j, 0.065] for j in range(7) for k in range(3)])
    lo = np.min(a[:, :2]-a[:, 2,None], axis=0)
    hi = np.max(a[:, :2]+a[:, 2,None], axis=0)
    a[:, :2] -= (lo+hi)/2
    a *= 2/np.sum(hi-lo)

    # Discover which coordinate is the reflection direction.
    options = []
    for swap in (False, True):
        if time.monotonic() >= end:
            return best
        b = a.copy()
        if swap:
            b[:, :2] = b[:, [1,0]]
        singles = np.argsort(np.abs(b[:,1]))[:3]
        rem = np.array([i for i in range(21) if i not in singles])
        rem = rem[np.argsort(b[rem,1])]
        low, high = rem[:9], rem[9:]
        costs = (b[low,None,0]-b[high,0])**2 + (b[low,None,1]+b[high,1])**2 + (b[low,None,2]-b[high,2])**2
        ii,jj = linear_sum_assignment(costs)
        err = costs[ii,jj].sum()+np.sum(b[singles,1]**2)
        options.append((err,b,singles,low[ii],high[jj]))
    _, a, singles, low, high = min(options, key=lambda v:v[0])
    width = np.max(a[:,0]+a[:,2])-np.min(a[:,0]-a[:,2])
    a[:,0] -= np.min(a[:,0]-a[:,2])
    # q = twelve horizontal positions, twelve radii, nine heights, width.
    X = np.zeros((21,34)); Y = X.copy(); R = X.copy()
    q = np.zeros(34)
    groups = [[int(i)] for i in singles] + [[int(i),int(j)] for i,j in zip(low,high)]
    for g, members in enumerate(groups):
        if time.monotonic() >= end:
            return best
        X[members,g] = 1
        R[members,12+g] = 1
        q[g] = np.mean(a[members,0])
        q[12+g] = np.mean(a[members,2])
        if g >= 3:
            Y[members[0],24+g-3] = -1
            Y[members[1],24+g-3] = 1
            q[24+g-3] = np.mean(np.abs(a[members,1]))
    q[33] = width
    W = np.zeros(34); W[33] = 1
    wallA = np.vstack((X-R, W-X-R, -0.5*W-Y-R, -0.5*W+Y-R))
    wallb = np.r_[np.zeros(42),np.ones(42)]
    # Reflected wall and pair constraints are duplicates; retain one each.
    _, ids = np.unique(np.column_stack((wallA,wallb)),axis=0,return_index=True)
    wallA, wallb = wallA[ids],wallb[ids]
    pi,pj = np.triu_indices(21,1)
    dxs=[]; dys=[]; rss=[]; seen=set()
    for i,j in zip(pi,pj):
        if time.monotonic() >= end:
            return best
        dx = X[i]-X[j]; dy = Y[i]-Y[j]; rs = R[i]+R[j]
        def canonical(v):
            nz=np.flatnonzero(v)
            return v if len(nz)==0 or v[nz[0]]>0 else -v
        dx=canonical(dx); dy=canonical(dy)
        key=tuple(np.r_[dx,dy,rs])
        if key not in seen:
            seen.add(key); dxs.append(dx); dys.append(dy); rss.append(rs)
    DX=np.array(dxs); DY=np.array(dys); RS=np.array(rss)
    objective=-R.sum(axis=0)
    bounds=[(0.0,2.0)]*12+[(1e-6,0.4)]*12+[(0.0,1.0)]*9+[(0.35,1.65)]
    cache_q=None; cache_v=None; cache_j=None
    def constraint(v):
        nonlocal cache_q,cache_v,cache_j
        if time.monotonic() >= end:
            raise StopSearch()
        if cache_q is None or not np.array_equal(cache_q,v):
            dx=DX@v; dy=DY@v
            d=np.sqrt(dx*dx+dy*dy+1e-30)
            cache_v=np.r_[d-RS@v,wallA@v+wallb]
            cache_j=np.vstack(((dx[:,None]*DX+dy[:,None]*DY)/d[:,None]-RS,wallA))
            cache_q=v.copy()
        return cache_v
    def jac(v):
        constraint(v)
        return cache_j
    last=q.copy()
    def callback(v):
        nonlocal last
        last=v.copy()
        if time.monotonic() >= end:
            raise StopSearch()
    def consider(v):
        nonlocal best,score
        if time.monotonic() >= end+0.06:
            return
        c=np.column_stack((X@v,Y@v,R@v))
        if not np.isfinite(c).all() or np.min(c[:,2])<=0:
            return
        dist=np.sqrt((c[pi,0]-c[pj,0])**2+(c[pi,1]-c[pj,1])**2)
        factor=min(1.0,float(np.min(dist/(c[pi,2]+c[pj,2]))))
        c[:,2] *= factor*(1-3e-15)
        per=np.sum(np.max(c[:,:2]+c[:,2,None],axis=0)-np.min(c[:,:2]-c[:,2,None],axis=0))
        c *= min(1.0,2.0/per)*(1-3e-15)
        value=exact_score(c)
        if value>score:
            score=value
            best={'circles':c.tolist()}
    base=q.copy()
    attempt=0
    while time.monotonic()<end:
        if attempt==0:
            trial=base.copy()
        else:
            trial=base.copy()
            center=int(rng.integers(12))
            locations=np.column_stack((base[:12],np.r_[np.zeros(3),base[24:33]]))
            near=np.argsort(np.sum((locations-locations[center])**2,axis=1))[:int(params['cluster'])]
            amp=float(params['amplitude'])*(0.5,1.0,1.8,2.8)[(attempt-1)%4]
            trial[near]+=rng.normal(0,amp,len(near))
            trial[12+near]*=np.exp(rng.normal(0,0.12,len(near)))
            paired=near[near>=3]
            trial[24+paired-3]+=rng.normal(0,amp,len(paired))
            trial[33]+=rng.normal(0,amp*0.4)
            trial=np.array([np.clip(t,l,h) for t,(l,h) in zip(trial,bounds)])
        last=trial.copy()
        try:
            result=minimize(lambda v:float(objective@v),trial,jac=lambda v:objective,
                bounds=bounds,constraints={'type':'ineq','fun':constraint,'jac':jac},
                method='SLSQP',callback=callback,
                options={'maxiter':int(params['iterations']),'ftol':2e-13,'disp':False})
            last=result.x
        except StopSearch:
            consider(last)
            break
        consider(last)
        if attempt==0 and np.min(constraint(last))>-1e-7:
            base=last.copy()
        attempt+=1
    if best is None:
        c=np.array([[0.08+0.19*(i%5),0.08+0.19*(i//5),0.03] for i in range(21)])
        best={'circles':c.tolist()}
    return best
