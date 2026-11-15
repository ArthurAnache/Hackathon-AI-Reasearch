import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {
    'first_fraction': (0.30, 0.15, 0.40),
    'second_fraction': (0.20, 0.10, 0.30),
    'jitter': (0.012, 0.001, 0.04, 'log'),
    'iterations': (180, 60, 300)
}

class StopSearch(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best_obj = None
    best_score = -float('inf')

    def exact_score(obj):
        try:
            a = np.asarray(obj['circles'], dtype=float)
            if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
                return -float('inf')
            f = [[Fraction(float(v)) for v in row] for row in a]
            w = max(x+r for x,y,r in f)-min(x-r for x,y,r in f)
            h = max(y+r for x,y,r in f)-min(y-r for x,y,r in f)
            if w+h > 2:
                return -float('inf')
            for i in range(21):
                if time.monotonic() > start + 0.94*budget_s:
                    return -float('inf')
                for j in range(i):
                    d = np.sqrt((a[i,0]-a[j,0])**2+(a[i,1]-a[j,1])**2)
                    if d < a[i,2]+a[j,2]:
                        return -float('inf')
            return float(np.sum(a[:,2]))
        except Exception:
            return -float('inf')

    if init is not None:
        if 'circles' in init:
            canonical = init
        else:
            cc = []
            for x,y,r,m in init.get('orbits', []):
                cc.append([x,y,r])
                if m == 2:
                    cc.append([x,-y,r])
            canonical = {'circles': cc}
        best_score = exact_score(canonical)
        if np.isfinite(best_score):
            best_obj = init
    else:
        canonical = None

    ii, jj = np.triu_indices(21, 1)
    def repaired(a):
        a = np.array(a, dtype=float, copy=True)
        if a.shape != (21,3) or not np.isfinite(a).all() or np.min(a[:,2]) <= 0:
            return None, -float('inf')
        a[:,:2] -= np.mean(a[:,:2], axis=0)
        dist = np.sqrt(np.sum((a[ii,:2]-a[jj,:2])**2, axis=1))
        ratio = float(np.min(dist/(a[ii,2]+a[jj,2])))
        a[:,2] *= min(1., ratio)*(1.-3e-14)
        wh = np.ptp(a[:,0,None]+np.array([-1.,1.])*a[:,2,None]) if False else 0.
        w = np.max(a[:,0]+a[:,2])-np.min(a[:,0]-a[:,2])
        h = np.max(a[:,1]+a[:,2])-np.min(a[:,1]-a[:,2])
        if w+h <= 0:
            return None, -float('inf')
        a *= (2.-8e-14)/(w+h)
        return {'circles': a.tolist()}, float(np.sum(a[:,2]))

    # Three axis orbits followed by nine two-circle orbits.
    mult = np.array([1.]*3+[2.]*9)
    ids = []
    signs = []
    for k in range(12):
        ids.append(k); signs.append(0. if k < 3 else 1.)
        if k >= 3:
            ids.append(k); signs.append(-1.)
    ids = np.array(ids)
    signs = np.array(signs)
    def expand(z):
        yy = np.r_[np.zeros(3), z[12:21]]
        return np.column_stack((z[:12][ids], yy[ids]*signs, z[21:33][ids]))

    # Unique orbit-pair constraints, including both relative reflection signs.
    pi=[]; pj=[]; sg=[]
    for i in range(12):
        for j in range(i+1,12):
            pi.append(i); pj.append(j); sg.append(1.)
            if i >= 3:
                pi.append(i); pj.append(j); sg.append(-1.)
    pi=np.array(pi); pj=np.array(pj); sg=np.array(sg)
    n=len(pi)
    DX=np.zeros((n,34)); DY=np.zeros((n,34)); RR=np.zeros((n,34))
    rows=np.arange(n)
    DX[rows,pi]=1.; DX[rows,pj]=-1.
    for k in range(n):
        if pi[k]>=3: DY[k,12+pi[k]-3]=1.
        if pj[k]>=3: DY[k,12+pj[k]-3]=-sg[k]
    RR[rows,21+pi]=1.; RR[rows,21+pj]=1.
    linear=[]
    for k in range(12):
        v=np.zeros(34); v[k]=1.; v[21+k]=-1.; linear.append(v)
        v=np.zeros(34); v[33]=1.; v[k]=-1.; v[21+k]=-1.; linear.append(v)
        v=np.zeros(34); v[33]=-.5; v[21+k]=-1.
        if k>=3: v[12+k-3]=-1.
        linear.append(v)
        if k>=3:
            v=np.zeros(34); v[12+k-3]=1.; v[21+k]=-1.; linear.append(v)
    L=np.array(linear)
    offset=np.zeros(len(L))
    p=0
    for k in range(12):
        offset[p+2]=1.
        p += 4 if k>=3 else 3
    grad=np.zeros(34); grad[21:33]=-mult
    bounds=[(-1.,3.)]*12+[(0.,1.)]*9+[(1e-7,.8)]*12+[(.15,1.85)]
    cache_x=None; cache_c=None; cache_j=None
    local_deadline=deadline
    def calc(z):
        nonlocal cache_x,cache_c,cache_j
        if time.monotonic() >= local_deadline:
            raise StopSearch()
        if cache_x is None or not np.array_equal(z,cache_x):
            dx=DX@z; dy=DY@z; rr=RR@z
            d=np.sqrt(dx*dx+dy*dy)
            safe=np.maximum(d,1e-18)
            cache_c=np.r_[d-rr,L@z+offset]
            cache_j=np.vstack(((dx/safe)[:,None]*DX+(dy/safe)[:,None]*DY-RR,L))
            cache_x=z.copy()
        return cache_c,cache_j

    pool=[]
    def register(z):
        nonlocal best_score,best_obj
        obj,score=repaired(expand(z))
        if obj is not None:
            pool.append((score,z.copy()))
            if score > best_score and time.monotonic() < deadline+.05*budget_s:
                checked=exact_score(obj)
                if checked>best_score:
                    best_score=checked; best_obj=obj

    def solve(z, seconds, maxiter=None):
        nonlocal local_deadline,cache_x
        if time.monotonic() >= deadline:
            return z
        local_deadline=min(deadline,time.monotonic()+seconds)
        cache_x=None
        last=z.copy()
        promising=z.copy()
        quality=-float('inf')
        def callback(x):
            nonlocal last,promising,quality
            last=x.copy()
            c,_=calc(x)
            q=-float(grad@x)-12.*max(0.,-float(np.min(c)))
            if q>quality:
                quality=q; promising=x.copy()
        try:
            result=minimize(lambda x: float(grad@x),z,jac=lambda x:grad,
                method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda x:calc(x)[0], 'jac':lambda x:calc(x)[1]},
                callback=callback, options={'maxiter':maxiter or params['iterations'],'ftol':2e-12,'disp':False})
            last=result.x
        except StopSearch:
            pass
        except Exception:
            pass
        register(last)
        register(promising)
        return last

    def project(a,swap):
        a=np.asarray(a,dtype=float).copy()
        if swap: a[:,[0,1]]=a[:,[1,0]]
        a[:,:2]-=(np.max(a[:,:2]+a[:,2,None],axis=0)+np.min(a[:,:2]-a[:,2,None],axis=0))/2
        order=np.argsort(np.abs(a[:,1]))
        singles=order[:3]
        rest=order[3:]
        rest=rest[np.argsort(a[rest,1])]
        low=rest[:9]; high=rest[9:]
        cost=(a[low,0,None]-a[high,0][None,:])**2+(a[low,1,None]+a[high,1][None,:])**2+2*(a[low,2,None]-a[high,2][None,:])**2
        u,v=linear_sum_assignment(cost)
        x=np.r_[a[singles,0],(a[low[u],0]+a[high[v],0])/2]
        y=(np.abs(a[low[u],1])+np.abs(a[high[v],1]))/2
        r=np.r_[a[singles,2],(a[low[u],2]+a[high[v],2])/2]
        w=np.max(x+r)-np.min(x-r)
        h=2*max(np.max(y+r[3:]),np.max(r[:3]))
        scale=2/(w+h)
        x=(x-np.min(x-r))*scale
        return np.r_[x,y*scale,r*scale,w*scale]

    projections=[]
    if canonical is not None:
        try:
            a=np.asarray(canonical['circles'],dtype=float)
            if a.shape==(21,3) and np.isfinite(a).all():
                for swap in (False,True):
                    z=project(a,swap)
                    projections.append(z)
        except Exception:
            pass
    if projections:
        solve(projections[0],params['first_fraction']*budget_s)
        if len(projections)>1:
            solve(projections[1],params['second_fraction']*budget_s)
    else:
        x=np.r_[np.arange(3)*2.,np.tile(np.arange(3)*2.,3)+np.repeat([1.,0.,1.],3)]
        y=np.repeat(np.arange(1,4)*np.sqrt(3.),3)
        r=np.ones(12)
        w=7.; h=2*(3*np.sqrt(3.)+1.)
        z=np.r_[(x+1)*2/(w+h),y*2/(w+h),r*2/(w+h),w*2/(w+h)]
        solve(z,.45*budget_s)

    attempt=0
    while time.monotonic() < deadline-.035*budget_s:
        attempt+=1
        if not pool:
            break
        pool.sort(key=lambda t:t[0],reverse=True)
        z=pool[0][1].copy()
        if attempt>1:
            z[:21]+=rng.normal(0.,params['jitter'],21)
            z[12:21]=np.maximum(z[12:21],.005)
            z[21:33]*=rng.uniform(.95,1.02,12)
        solve(z,min(.18*budget_s,deadline-time.monotonic()))
    if best_obj is not None:
        return best_obj
    # A guaranteed valid emergency construction.
    a=np.array([[.047+(.094)*(i%7),.047+.094*(i//7),.047] for i in range(21)])
    obj,_=repaired(a)
    return obj
