import time
import math
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"iterations": (150, 40, 250), "safety": (0.70, 0.5, 0.8)}

class StopSearch(Exception):
    pass

def exact_score(obj):
    try:
        a = np.asarray(obj['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        xx = list(map(Fraction, map(float, a[:, 0])))
        yy = list(map(Fraction, map(float, a[:, 1])))
        rr = list(map(Fraction, map(float, a[:, 2])))
        width = max(x+r for x,r in zip(xx,rr))-min(x-r for x,r in zip(xx,rr))
        height = max(y+r for y,r in zip(yy,rr))-min(y-r for y,r in zip(yy,rr))
        if width+height > 2:
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
    deadline = start + float(params['safety'])*budget_s
    rng = np.random.default_rng(seed)
    best = init
    if init is not None and 'orbits' in init:
        a=[]
        for x,y,r,m in init['orbits']:
            if time.monotonic() >= deadline:
                break
            a.append([x,y,r])
            if m == 2:
                a.append([x,-y,r])
        best={'circles':a}
    best_score = exact_score(best) if best is not None else -float('inf')
    ii,jj=np.triu_indices(21,1)

    def consider(a):
        nonlocal best,best_score
        if not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        a=a.copy()
        d=np.sqrt((a[ii,0]-a[jj,0])**2+(a[ii,1]-a[jj,1])**2)
        factor=min(1.0,float(np.min(d/(a[ii,2]+a[jj,2]))))
        if factor <= 0:
            return
        a[:,2] *= factor*(1-2e-13)
        extent=np.max(a[:,:2]+a[:,2,None],axis=0)-np.min(a[:,:2]-a[:,2,None],axis=0)
        a *= min(1.0,2/float(np.sum(extent)))*(1-2e-13)
        obj={'circles':a.tolist()}
        value=exact_score(obj)
        if value > best_score:
            best,best_score=obj,value

    # A universally feasible fallback.
    fallback=np.array([[0.05+0.1*(k%7),0.05+0.1*(k//7),0.049999999999] for k in range(21)])
    consider(fallback)

    # x[0:12], positive y[12:21], radii[21:33], width[33].
    pi=[]; pj=[]; sg=[]
    for i in range(12):
        if time.monotonic() >= deadline:
            return best
        for j in range(i+1,12):
            if time.monotonic() >= deadline:
                return best
            pi.append(i); pj.append(j); sg.append(1.)
            if i>=3:
                pi.append(i); pj.append(j); sg.append(-1.)
    pi=np.array(pi); pj=np.array(pj); sg=np.array(sg)
    m=len(pi)
    rows=np.arange(m)
    ymap=np.maximum(np.arange(12)-3,0)+12
    L=np.zeros((45,34)); c=np.zeros(45)
    z=np.arange(12)
    L[z,z]=1; L[z,21+z]=-1
    L[12+z,z]=-1; L[12+z,21+z]=-1; L[12+z,33]=1
    L[24+z,21+z]=-1; L[24+z,33]=-.5; c[24+z]=1
    L[27+np.arange(9),12+np.arange(9)]=-1
    L[36+np.arange(9),12+np.arange(9)]=1
    L[36+np.arange(9),24+np.arange(9)]=-1
    objective=np.zeros(34); objective[21:24]=-1; objective[24:33]=-2
    cache_z=None; cache_g=None; cache_j=None

    def calc(v):
        nonlocal cache_z,cache_g,cache_j
        if time.monotonic() >= deadline:
            raise StopSearch()
        if cache_z is not None and np.array_equal(v,cache_z):
            return cache_g,cache_j
        y=np.zeros(12); y[3:]=v[12:21]
        dx=v[pi]-v[pj]
        dy=y[pi]-sg*y[pj]
        dist=np.sqrt(dx*dx+dy*dy+1e-40)
        g=dist-v[21+pi]-v[21+pj]
        J=np.zeros((m,34))
        J[rows,pi]=dx/dist; J[rows,pj]=-dx/dist
        mask=pi>=3
        J[rows[mask],ymap[pi[mask]]]=dy[mask]/dist[mask]
        mask=pj>=3
        J[rows[mask],ymap[pj[mask]]]=-sg[mask]*dy[mask]/dist[mask]
        J[rows,21+pi]=-1; J[rows,21+pj]=-1
        cache_z=v.copy()
        cache_g=np.concatenate((g,L@v+c))
        cache_j=np.vstack((J,L))
        return cache_g,cache_j

    def expand(v):
        a=np.zeros((21,3)); a[:3,0]=v[:3]; a[:3,2]=v[21:24]
        a[3::2,0]=v[3:12]; a[4::2,0]=v[3:12]
        a[3::2,1]=v[12:21]; a[4::2,1]=-v[12:21]
        a[3::2,2]=v[24:33]; a[4::2,2]=v[24:33]
        return a

    bounds=[(0,2)]*12+[(0,1)]*9+[(1e-6,.5)]*12+[(.15,1.85)]
    layouts=[[4,5],[5,4],[3,3,3],[4,3,2],[2,3,4],[3,6],[6,3]]
    attempt=0
    while time.monotonic() < deadline-.055:
        counts=layouts[attempt%len(layouts)]
        v=np.zeros(34)
        nrows=len(counts)
        maxcount=max([3]+counts)
        spacing=math.sqrt(3)
        wunits=2*maxcount+(.6 if maxcount==3 else 0)
        hunits=2+2*nrows*spacing
        r=2/(wunits+hunits)
        w=wunits*r
        v[33]=w
        v[:3]=w/2+2*r*(np.arange(3)-1)
        cursor=3
        for row,count in enumerate(counts):
            if time.monotonic() >= deadline:
                return best
            shift=0.0
            if count==3 and maxcount==3:
                shift=(.5 if row%2==0 else -.5)*r
            v[cursor:cursor+count]=w/2+2*r*(np.arange(count)-(count-1)/2)+shift
            v[12+cursor-3:12+cursor-3+count]=(row+1)*spacing*r
            cursor+=count
        v[21:33]=r*.94
        if attempt>=len(layouts):
            v[:12]+=rng.normal(0,.025,12)
            v[12:21]+=rng.normal(0,.02,9)
        last=[v.copy()]
        local_end=min(deadline-.025,time.monotonic()+max(.12,(deadline-time.monotonic())*.31))
        def callback(q):
            last[0]=q.copy()
            if time.monotonic() >= local_end:
                raise StopSearch()
        try:
            res=minimize(lambda q:float(objective@q),v,jac=lambda q:objective,
                bounds=bounds, constraints={'type':'ineq','fun':lambda q:calc(q)[0], 'jac':lambda q:calc(q)[1]},
                method='SLSQP',callback=callback,
                options={'maxiter':int(params['iterations']),'ftol':2e-12,'disp':False})
            last[0]=res.x
        except StopSearch:
            pass
        except (ValueError,FloatingPointError):
            pass
        consider(expand(last[0]))
        attempt+=1
    return best
