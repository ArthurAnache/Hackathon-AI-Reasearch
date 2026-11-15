import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import time
import math
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'family_time': (0.105, 0.04, 0.18),
    'iterations': (110, 40, 180),
    'perturbation': (0.014, 0.001, 0.08),
}

class StopSearch(Exception):
    pass

def exact_score(data):
    try:
        a = np.asarray(data['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        xs = [Fraction(float(v)) for v in a[:, 0]]
        ys = [Fraction(float(v)) for v in a[:, 1]]
        rs = [Fraction(float(v)) for v in a[:, 2]]
        w = max(x+r for x,r in zip(xs,rs))-min(x-r for x,r in zip(xs,rs))
        h = max(y+r for y,r in zip(ys,rs))-min(y-r for y,r in zip(ys,rs))
        if w+h > 2:
            return -float('inf')
        for i,j in itertools.combinations(range(21), 2):
            d = np.sqrt((a[i,0]-a[j,0])**2 + (a[i,1]-a[j,1])**2)
            if d < a[i,2]+a[j,2]:
                return -float('inf')
        return float(np.sum(a[:,2]))
    except Exception:
        return -float('inf')

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + 0.69*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = exact_score(init) if init is not None else -float('inf')
    ii,jj = np.triu_indices(21,1)

    def retain(a):
        nonlocal best, best_score
        a = np.asarray(a, dtype=float).copy()
        if a.shape != (21,3) or not np.isfinite(a).all() or np.min(a[:,2]) <= 0:
            return
        dx = a[ii,0]-a[jj,0]
        dy = a[ii,1]-a[jj,1]
        distances = np.sqrt(dx*dx+dy*dy)
        factor = min(1.0, float(np.min(distances/(a[ii,2]+a[jj,2]))))
        if factor <= 0:
            return
        a[:,2] *= factor*(1.0-3e-13)
        low = np.min(a[:,:2]-a[:,2,None],axis=0)
        high = np.max(a[:,:2]+a[:,2,None],axis=0)
        perimeter = float(np.sum(high-low))
        if not perimeter > 0:
            return
        a[:,:2] -= (low+high)/2
        a *= (2.0/perimeter)*(1.0-3e-13)
        candidate = {'circles': a.tolist()}
        score = exact_score(candidate)
        if score > best_score:
            best, best_score = candidate, score

    # A valid baseline is available even if a very small budget interrupts search.
    fallback = np.array([[2*(k%5), 2*(k//5), 1.0] for k in range(21)],dtype=float)
    retain(fallback)

    def model(rows):
        if time.monotonic() >= deadline:
            raise StopSearch
        groups=[]
        for row,k in enumerate(rows):
            if time.monotonic() >= deadline:
                raise StopSearch
            y=(row-(len(rows)-1)/2)*math.sqrt(3)
            if k%2:
                groups.append((0.0,y,False))
            for x in range(1 if k%2==0 else 2,k,2):
                if time.monotonic() >= deadline:
                    raise StopSearch
                groups.append((float(x),y,True))
        dim=1+sum(3 if pair else 2 for x,y,pair in groups)
        X=np.zeros((21,dim)); Y=X.copy(); R=X.copy()
        z=np.zeros(dim); bounds=[(0.15,1.85)]
        wraw=2*max(rows)
        hraw=2+(len(rows)-1)*math.sqrt(3)
        scale=2/(wraw+hraw)
        z[0]=wraw*scale
        primary=[]
        cursor=1
        count=0
        for x,y,pair in groups:
            if time.monotonic() >= deadline:
                raise StopSearch
            if pair:
                xi,yi,ri=cursor,cursor+1,cursor+2
                cursor+=3
                z[xi]=x*scale; z[yi]=y*scale; z[ri]=scale
                bounds.extend([(0.0,0.95),(-0.95,0.95),(1e-6,0.5)])
                X[count,xi]=1; X[count+1,xi]=-1
                Y[count:count+2,yi]=1; R[count:count+2,ri]=1
                primary.append(count)
                count+=2
            else:
                yi,ri=cursor,cursor+1
                cursor+=2
                z[yi]=y*scale; z[ri]=scale
                bounds.extend([(-0.95,0.95),(1e-6,0.5)])
                Y[count,yi]=1; R[count,ri]=1
                primary.append(count)
                count+=1
        A=X[ii]-X[jj]; B=Y[ii]-Y[jj]; C=R[ii]+R[jj]
        keep=[]; seen=set()
        for k in range(len(ii)):
            if time.monotonic() >= deadline:
                raise StopSearch
            aa=A[k]; bb=B[k]
            ia=np.flatnonzero(aa); ib=np.flatnonzero(bb)
            ca=aa if len(ia)==0 or aa[ia[0]]>0 else -aa
            cb=bb if len(ib)==0 or bb[ib[0]]>0 else -bb
            key=(ca.tobytes(),cb.tobytes(),C[k].tobytes())
            # Normalize signed zero before constructing a duplicate key.
            key=(tuple(ca),tuple(cb),tuple(C[k]))
            if key not in seen:
                seen.add(key); keep.append(k)
        A=A[keep]; B=B[keep]; C=C[keep]
        walls=[]; constants=[]
        for k in primary:
            if time.monotonic() >= deadline:
                raise StopSearch
            line=-X[k]-R[k]; line=line.copy(); line[0]+=0.5
            walls.append(line); constants.append(0.)
            line=-Y[k]-R[k]; line=line.copy(); line[0]-=0.5
            walls.append(line); constants.append(1.)
            line=Y[k]-R[k]; line=line.copy(); line[0]-=0.5
            walls.append(line); constants.append(1.)
        L=np.array(walls); offset=np.array(constants)
        objective=-np.sum(R,axis=0)
        return (X,Y,R,A,B,C,L,offset,objective,bounds,z)

    records=[]
    def optimize(m,z,local_deadline,maxiter):
        X,Y,R,A,B,C,L,offset,obj,bounds,_=m
        last=z.copy()
        cache_z=None; cache_c=None; cache_j=None
        def update(v):
            nonlocal cache_z,cache_c,cache_j
            if time.monotonic() >= local_deadline:
                raise StopSearch
            if cache_z is None or not np.array_equal(cache_z,v):
                dx=A@v; dy=B@v
                dist=np.sqrt(np.maximum(dx*dx+dy*dy,1e-28))
                cache_c=np.concatenate((dist-C@v,L@v+offset))
                jac=(dx/dist)[:,None]*A+(dy/dist)[:,None]*B-C
                cache_j=np.vstack((jac,L))
                cache_z=v.copy()
        def fun(v):
            if time.monotonic() >= local_deadline:
                raise StopSearch
            return float(obj@v)
        def con(v):
            update(v)
            return cache_c
        def jac(v):
            update(v)
            return cache_j
        def callback(v):
            nonlocal last
            last=v.copy()
            if time.monotonic() >= local_deadline:
                raise StopSearch
        try:
            result=minimize(fun,z,jac=lambda v:obj,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':con,'jac':jac},callback=callback,
                options={'maxiter':maxiter,'ftol':2e-12,'disp':False})
            last=result.x.copy()
        except StopSearch:
            pass
        a=np.column_stack((X@last,Y@last,R@last))
        retain(a)
        return last

    families=[(5,4,5,4,3),(5,4,3,4,5),(4,5,4,5,3),
              (4,3,4,3,4,3),(3,4,5,4,3,2),(5,6,5,5)]
    for rows in families:
        if time.monotonic() >= deadline-0.035:
            break
        try:
            m=model(rows)
        except StopSearch:
            break
        z=m[-1].copy()
        local=min(deadline-0.015,time.monotonic()+params['family_time']*budget_s)
        z=optimize(m,z,local,int(params['iterations']))
        X,Y,R=m[:3]
        a=np.column_stack((X@z,Y@z,R@z))
        dx=a[ii,0]-a[jj,0]; dy=a[ii,1]-a[jj,1]
        factor=min(1.,float(np.min(np.sqrt(dx*dx+dy*dy)/(a[ii,2]+a[jj,2]))))
        merit=float(np.sum(a[:,2]))*max(0.,factor)
        records.append((merit,m,z.copy()))
        if best_score > 2.3658323758:
            break

    records.sort(key=lambda item:item[0],reverse=True)
    iteration=0
    while records and time.monotonic() < deadline-0.025:
        if time.monotonic() >= deadline:
            break
        merit,m,z=records[iteration%min(2,len(records))]
        trial=z.copy()
        if iteration>=1:
            amplitude=params['perturbation']*(1+0.35*(iteration%3))
            trial+=rng.normal(0,amplitude,len(trial))
            for k,(lo,hi) in enumerate(m[-2]):
                if time.monotonic() >= deadline:
                    break
                trial[k]=np.clip(trial[k],lo+1e-8,hi-1e-8)
        local=min(deadline-0.008,time.monotonic()+0.11*budget_s)
        newz=optimize(m,trial,local,int(params['iterations']))
        if iteration==0:
            records[0]=(merit,m,newz)
        iteration+=1
        if best_score > 2.3658323758:
            break
    return best
