import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {
    'fraction': (0.70, 0.5, 0.75),
    'first_iterations': (85, 30, 160),
    'reduced_iterations': (85, 30, 160),
    'amplitude': (0.018, 0.001, 0.10),
}

class StopSearch(Exception):
    pass

def exact_score(a):
    if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
        return -np.inf
    i,j = np.triu_indices(21,1)
    d = np.sqrt((a[i,0]-a[j,0])**2+(a[i,1]-a[j,1])**2)
    if np.any(d < a[i,2]+a[j,2]):
        return -np.inf
    x,y,r = [[Fraction(float(v)) for v in a[:,k]] for k in range(3)]
    p = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
    p += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
    return float(np.sum(a[:,2])) if p <= 2 else -np.inf

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + float(params['fraction'])*budget_s
    rng = np.random.default_rng(seed)
    pi,pj = np.triu_indices(21,1)
    fallback = np.array([[0.1+0.19*(i%5),0.1+0.19*(i//5),0.08] for i in range(21)])
    best = {'circles':fallback.tolist()}
    score = exact_score(fallback)
    a = fallback.copy()
    if init is not None:
        original = np.asarray(init['circles'],dtype=float)
        s = exact_score(original)
        if s >= score:
            best,score = init,s
        if original.shape == (21,3) and np.isfinite(original).all() and np.all(original[:,2]>0):
            a = original.copy()
    best_array = np.asarray(best['circles'],dtype=float).copy()

    def normalize(c):
        c = c.copy()
        lo = np.min(c[:,:2]-c[:,2,None],axis=0)
        hi = np.max(c[:,:2]+c[:,2,None],axis=0)
        c[:,:2] -= (lo+hi)/2
        c *= 2/np.sum(hi-lo)
        return c

    def consider(c):
        nonlocal best,score,best_array
        if time.monotonic() > end+0.035:
            return
        if not np.isfinite(c).all() or np.min(c[:,2])<=0:
            return
        c = c.copy()
        d = np.sqrt((c[pi,0]-c[pj,0])**2+(c[pi,1]-c[pj,1])**2)
        f = min(1.0,float(np.min(d/(c[pi,2]+c[pj,2]))))
        c[:,2] *= f*(1-8e-15)
        if np.min(c[:,2])<=0:
            return
        c = normalize(c)*(1-8e-15)
        s = exact_score(c)
        if s > score:
            score,best_array = s,c.copy()
            best = {'circles':c.tolist()}

    a = normalize(a)
    consider(a)

    def optimize(X,Y,R,q,iterations,deadline):
        m = len(q)
        W = np.zeros(m); W[-1]=1
        A = np.vstack((X-R,W-X-R,-0.5*W-Y-R,-0.5*W+Y-R))
        b = np.r_[np.zeros(42),np.ones(42)]
        _,ids = np.unique(np.column_stack((A,b)),axis=0,return_index=True)
        A,b = A[ids],b[ids]
        DX,DY,RS = X[pi]-X[pj],Y[pi]-Y[pj],R[pi]+R[pj]
        # Reflection duplicates differ only in the signs of coordinate differences.
        keys = np.column_stack((np.abs(DX),np.abs(DY),RS))
        _,ids = np.unique(keys,axis=0,return_index=True)
        DX,DY,RS = DX[ids],DY[ids],RS[ids]
        obj = -np.sum(R,axis=0)
        cached = None
        cv = cj = None
        last = q.copy()
        count = 0
        def constraints(v):
            nonlocal cached,cv,cj
            if time.monotonic() >= deadline:
                raise StopSearch()
            if cached is None or not np.array_equal(cached,v):
                dx,dy = DX@v,DY@v
                dist = np.sqrt(dx*dx+dy*dy+1e-30)
                cv = np.r_[dist-RS@v,A@v+b]
                cj = np.vstack(((dx[:,None]*DX+dy[:,None]*DY)/dist[:,None]-RS,A))
                cached = v.copy()
            return cv
        def jac(v):
            constraints(v)
            return cj
        def callback(v):
            nonlocal last,count
            last = v.copy(); count += 1
            if time.monotonic() >= deadline:
                raise StopSearch()
            if count % 12 == 0:
                consider(np.column_stack((X@v,Y@v,R@v)))
        bounds = [(-2.,2.)]*m
        for k in range(m):
            if time.monotonic() >= deadline:
                return last
            if np.any(R[:,k]):
                bounds[k]=(1e-7,0.6)
        bounds[-1]=(0.2,1.8)
        try:
            res = minimize(lambda v:float(obj@v),q,jac=lambda v:obj,
                bounds=bounds,constraints={'type':'ineq','fun':constraints,'jac':jac},
                method='SLSQP',callback=callback,
                options={'maxiter':iterations,'ftol':3e-13,'disp':False})
            last=res.x
        except StopSearch:
            pass
        consider(np.column_stack((X@last,Y@last,R@last)))
        return last

    # First preserve all degrees of freedom, rather than guessing orbit assignments.
    X=np.zeros((21,64)); Y=X.copy(); R=X.copy()
    X[:,:21]=np.eye(21); Y[:,21:42]=np.eye(21); R[:,42:63]=np.eye(21)
    width=np.ptp(np.r_[a[:,0]-a[:,2],a[:,0]+a[:,2]])
    q=np.r_[a[:,0]+width/2,a[:,1],a[:,2],width]
    first_end=min(end,start+0.36*budget_s)
    if time.monotonic()<first_end:
        optimize(X,Y,R,q,int(params['first_iterations']),first_end)
    if time.monotonic()>=end:
        return best
    a=normalize(best_array)
    options=[]
    for swap in (False,True):
        if time.monotonic()>=end:
            return best
        c=a.copy()
        if swap:
            c[:,:2]=c[:,[1,0]]
        singles=np.argsort(np.abs(c[:,1]))[:3]
        rest=np.setdiff1d(np.arange(21),singles)
        rest=rest[np.argsort(c[rest,1])]
        low,high=rest[:9],rest[9:]
        cost=(c[low,None,0]-c[high,0])**2+(c[low,None,1]+c[high,1])**2+(c[low,None,2]-c[high,2])**2
        ii,jj=linear_sum_assignment(cost)
        err=float(cost[ii,jj].sum()+np.sum(c[singles,1]**2))
        options.append((err,c,singles,low[ii],high[jj]))
    _,a,singles,low,high=min(options,key=lambda v:v[0])
    X=np.zeros((21,34)); Y=X.copy(); R=X.copy(); q=np.zeros(34)
    width=np.max(a[:,0]+a[:,2])-np.min(a[:,0]-a[:,2])
    a[:,0]-=np.min(a[:,0]-a[:,2])
    groups=[[int(i)] for i in singles]+[[int(i),int(j)] for i,j in zip(low,high)]
    for g,members in enumerate(groups):
        if time.monotonic()>=end:
            return best
        X[members,g]=1; R[members,12+g]=1
        q[g]=np.mean(a[members,0]); q[12+g]=np.mean(a[members,2])
        if g>=3:
            Y[members[0],24+g-3]=-1; Y[members[1],24+g-3]=1
            q[24+g-3]=np.mean(np.abs(a[members,1]))
    q[-1]=width
    attempt=0
    base=q.copy()
    base_value=-np.inf
    while time.monotonic()<end:
        trial=base.copy()
        if attempt:
            center=int(rng.integers(12))
            loc=np.column_stack((base[:12],np.r_[np.zeros(3),base[24:33]]))
            near=np.argsort(np.sum((loc-loc[center])**2,axis=1))[:4]
            amp=float(params['amplitude'])*(1,2,0.5,3)[(attempt-1)%4]
            trial[near]+=rng.normal(0,amp,len(near))
            paired=near[near>=3]
            trial[24+paired-3]+=rng.normal(0,amp,len(paired))
        result=optimize(X,Y,R,trial,int(params['reduced_iterations']),end)
        c=np.column_stack((X@result,Y@result,R@result))
        d=np.sqrt((c[pi,0]-c[pj,0])**2+(c[pi,1]-c[pj,1])**2)
        violation=np.min(d-c[pi,2]-c[pj,2])
        value=float(np.sum(c[:,2]))
        if violation>-1e-7 and value>base_value:
            base=result.copy(); base_value=value
        attempt+=1
    return best
