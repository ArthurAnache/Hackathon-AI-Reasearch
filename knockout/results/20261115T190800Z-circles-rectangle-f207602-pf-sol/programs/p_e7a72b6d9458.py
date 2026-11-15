import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"iterations": (180, 60, 300), "exploration": (0.69, 0.5, 0.72)}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + float(params['exploration']) * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -float('inf')

    def exact_score(obj):
        try:
            if 'circles' in obj:
                c = np.asarray(obj['circles'], dtype=float)
            else:
                c = np.column_stack((obj['centers'], obj['radii']))
            if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
                return -float('inf')
            xx = []; yy = []; rr = []
            for a,b,r in c:
                time.monotonic()
                xx.append(Fraction(float(a))); yy.append(Fraction(float(b))); rr.append(Fraction(float(r)))
            w = max(a+r for a,r in zip(xx,rr))-min(a-r for a,r in zip(xx,rr))
            h = max(a+r for a,r in zip(yy,rr))-min(a-r for a,r in zip(yy,rr))
            if w+h > 2:
                return -float('inf')
            for i,j in itertools.combinations(range(21),2):
                time.monotonic()
                d = np.sqrt((c[i,0]-c[j,0])**2+(c[i,1]-c[j,1])**2)
                if d < c[i,2]+c[j,2]:
                    return -float('inf')
            return float(np.sum(c[:,2]))
        except Exception:
            return -float('inf')

    if init is not None:
        best_score = exact_score(init)

    # q = axis x (3), paired x (9), paired y (9), radii (12), width.
    X = np.zeros((21,34)); Y = np.zeros_like(X); R = np.zeros_like(X)
    for i in range(3):
        time.monotonic()
        X[i,i] = 1; R[i,21+i] = 1
    for k in range(9):
        time.monotonic()
        i=3+k; j=12+k
        X[i,3+k]=X[j,3+k]=1
        Y[i,12+k]=1; Y[j,12+k]=-1
        R[i,24+k]=R[j,24+k]=1
    mirror = np.r_[np.arange(3),np.arange(12,21),np.arange(3,12)]
    pairs=[]
    for i,j in itertools.combinations(range(21),2):
        time.monotonic()
        if (i,j) <= tuple(sorted((int(mirror[i]),int(mirror[j])))):
            pairs.append((i,j))
    ia=np.array([p[0] for p in pairs]); ib=np.array([p[1] for p in pairs])
    DX=X[ia]-X[ib]; DY=Y[ia]-Y[ib]; SR=R[ia]+R[ib]
    e=np.zeros(34); e[-1]=0.5
    # x+r <= W/2; -x+r <= W/2; upper y+r <= (2-W)/2.
    L=np.vstack((e-X[:12]-R[:12],e+X[:12]-R[:12],-e-Y[:12]-R[:12]))
    offset=np.r_[np.zeros(24),np.ones(12)]
    weights=np.zeros(34); weights[21:24]=1; weights[24:33]=2
    bounds=[(None,None)]*12+[(0,None)]*9+[(1.e-7,0.49)]*12+[(0.12,1.88)]
    cache_q=None; cache_val=None; cache_jac=None

    def calculate(q):
        nonlocal cache_q,cache_val,cache_jac
        if cache_q is not None and np.array_equal(cache_q,q):
            return
        dx=DX@q; dy=DY@q
        d=np.sqrt(dx*dx+dy*dy)
        den=np.maximum(d,1.e-18)
        cache_val=np.r_[d-SR@q,L@q+offset]
        cache_jac=np.vstack((DX*(dx/den)[:,None]+DY*(dy/den)[:,None]-SR,L))
        cache_q=q.copy()
    def fun(q):
        if time.monotonic() > deadline:
            raise StopSearch()
        return -float(weights@q)
    def cf(q):
        calculate(q)
        return cache_val
    def cj(q):
        calculate(q)
        return cache_jac

    def offer(q):
        nonlocal best,best_score
        c=np.column_stack((X@q,Y@q,R@q))
        if not np.isfinite(c).all() or np.min(c[:,2])<=0:
            return
        delta=c[ia,:2]-c[ib,:2]
        ds=np.sqrt(np.sum(delta*delta,axis=1))
        factor=min(1.,float(np.min(ds/(c[ia,2]+c[ib,2]))))
        if factor<=0:
            return
        c[:,2]*=factor*(1.-3.e-14)
        lo=np.min(c[:,:2]-c[:,2,None],axis=0)
        hi=np.max(c[:,:2]+c[:,2,None],axis=0)
        p=float(np.sum(hi-lo))
        if p<=0:
            return
        c[:,:2]-=(hi+lo)/2
        c*=2./p*(1.-3.e-14)
        obj={'centers':c[:,:2].tolist(),'radii':c[:,2].tolist()}
        sc=exact_score(obj)
        if sc>best_score:
            best_score=sc; best=obj

    def layout(counts):
        axis=[]; upper=[]
        for k,n in enumerate(counts):
            if time.monotonic()>deadline:
                raise StopSearch()
            x=k*np.sqrt(3.)
            if n%2:
                axis.append(x)
                upper.extend((x,float(2*j)) for j in range(1,(n+1)//2))
            else:
                upper.extend((x,float(2*j+1)) for j in range(n//2))
        a=np.array(upper)
        q=np.zeros(34)
        q[:3]=axis; q[3:12]=a[:,0]; q[12:21]=a[:,1]; q[21:33]=1
        c=np.column_stack((X@q,Y@q,R@q))
        lo=np.min(c[:,:2]-c[:,2,None],axis=0); hi=np.max(c[:,:2]+c[:,2,None],axis=0)
        q[:12]-=(lo[0]+hi[0])/2
        p=np.sum(hi-lo)
        q[:33]*=2/p
        q[-1]=2*(hi[0]-lo[0])/p
        return q

    families=[[3,4,3,4,3,4],[5,4,5,4,3],[4,5,4,5,3],[5,4,3,4,5],[7,7,7],[3,6,3,6,3],[3,4,5,4,5]]
    saved=[]
    last=None
    def callback(q):
        nonlocal last
        last=q.copy()
        if time.monotonic()>deadline:
            raise StopSearch()
    for counts in families:
        if time.monotonic()>deadline-0.035:
            break
        try:
            q=layout(counts)
            offer(q)
            last=q.copy()
            res=minimize(fun,q,jac=lambda z:-weights,method='SLSQP',bounds=bounds,
                         constraints={'type':'ineq','fun':cf,'jac':cj},callback=callback,
                         options={'maxiter':int(params['iterations']),'ftol':2.e-12,'disp':False})
            offer(res.x)
            saved.append(res.x.copy())
        except StopSearch:
            if last is not None:
                offer(last)
            break
    # Use any remaining time to try coordinated symmetry-preserving changes.
    attempt=0
    while saved and time.monotonic()<deadline-0.06:
        q=saved[int(rng.integers(len(saved)))].copy()
        q[:21]+=rng.normal(0,0.015 if attempt%2==0 else 0.04,21)
        q[12:21]=np.maximum(q[12:21],0.005)
        last=q.copy()
        try:
            res=minimize(fun,q,jac=lambda z:-weights,method='SLSQP',bounds=bounds,
                         constraints={'type':'ineq','fun':cf,'jac':cj},callback=callback,
                         options={'maxiter':100,'ftol':2.e-12,'disp':False})
            offer(res.x)
        except StopSearch:
            offer(last)
            break
        attempt+=1
    if best_score == -float('inf'):
        c=np.array([[i%7*0.2,i//7*0.2,0.09] for i in range(21)])
        c*=2/(1.2+0.4+0.36)*(1-1.e-12)
        return {'centers':c[:,:2].tolist(),'radii':c[:,2].tolist()}
    return best
