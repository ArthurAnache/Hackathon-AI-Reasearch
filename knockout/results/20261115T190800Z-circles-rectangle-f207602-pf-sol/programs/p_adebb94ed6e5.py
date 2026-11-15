import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"iterations": (120, 30, 250), "safety": (0.70, 0.5, 0.75)}


def exact_score(obj):
    try:
        if 'circles' in obj:
            a = np.asarray(obj['circles'], dtype=float)
        else:
            a = np.column_stack((obj['centers'], obj['radii']))
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        xx = [Fraction(float(v)) for v in a[:, 0]]
        yy = [Fraction(float(v)) for v in a[:, 1]]
        rr = [Fraction(float(v)) for v in a[:, 2]]
        p = max(x+r for x,r in zip(xx,rr))-min(x-r for x,r in zip(xx,rr))
        p += max(y+r for y,r in zip(yy,rr))-min(y-r for y,r in zip(yy,rr))
        if p > 2:
            return -float('inf')
        for i in range(21):
            for j in range(i):
                if np.sqrt((a[i,0]-a[j,0])**2+(a[i,1]-a[j,1])**2) < a[i,2]+a[j,2]:
                    return -float('inf')
        return float(np.sum(a[:,2]))
    except Exception:
        return -float('inf')


class StopSearch(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + float(params['safety']) * budget_s
    best = init
    best_score = exact_score(init) if init is not None else -float('inf')
    rng = np.random.default_rng(seed)

    def save(a):
        nonlocal best, best_score
        if time.monotonic() >= deadline or not np.isfinite(a).all():
            return
        a = np.array(a, dtype=float, copy=True)
        if np.any(a[:,2] <= 0):
            return
        ii,jj = np.triu_indices(21,1)
        dd = np.sqrt(np.sum((a[ii,:2]-a[jj,:2])**2,axis=1))
        rat = float(np.min(dd/(a[ii,2]+a[jj,2])))
        if rat <= 0:
            return
        a[:,2] *= min(1.0,rat)*(1-4e-14)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        a[:,:2] -= (lo+hi)/2
        a *= (2.0/float(np.sum(hi-lo)))*(1-4e-14)
        if np.sum(a[:,2]) <= best_score:
            return
        obj = {'centers':a[:,:2].tolist(),'radii':a[:,2].tolist()}
        val = exact_score(obj)
        if val > best_score:
            best, best_score = obj, val

    # Guaranteed feasible fallback.
    a = np.array([[i*.18,j*.18,.089] for j in range(3) for i in range(7)],float)
    save(a)

    patterns = [[5,4,3,4,5], [4,5,3,5,4], [5,4,5,4,3],
                [4,5,4,5,3], [6,5,6,4], [5,6,5,5], [7,7,7],
                [3,6,3,6,3], [4,4,5,4,4]]
    for attempt, rows in enumerate(patterns):
        if time.monotonic() >= deadline - .035:
            break
        reps=[]
        owners=[]
        signs=[]
        mirror=[]
        for row,n in enumerate(rows):
            if time.monotonic() >= deadline:
                break
            if n%2:
                k=len(reps)
                reps.append((0.0, row*np.sqrt(3), True))
                owners.append(k); signs.append(0.0); mirror.append(len(owners)-1)
            for q in range(n//2):
                k=len(reps)
                x=2*(q+1) if n%2 else 2*q+1
                reps.append((float(x),row*np.sqrt(3),False))
                u=len(owners)
                owners.extend([k,k]); signs.extend([1.,-1.]); mirror.extend([u+1,u])
        if len(owners)!=21:
            break
        m=len(reps)
        free=[k for k,p in enumerate(reps) if not p[2]]
        nx=len(free)
        d=nx+2*m+1
        X=np.zeros((21,d)); Y=np.zeros((21,d)); R=np.zeros((21,d))
        xi={k:t for t,k in enumerate(free)}
        z=np.zeros(d)
        for k,(x,y,axis) in enumerate(reps):
            if not axis:
                z[xi[k]]=x
            z[nx+k]=y-(len(rows)-1)*np.sqrt(3)/2
            z[nx+m+k]=1
        for i,k in enumerate(owners):
            if k in xi:
                X[i,xi[k]]=signs[i]
            Y[i,nx+k]=1
            R[i,nx+m+k]=1
        width=2*(max(p[0] for p in reps)+1)
        height=2+(len(rows)-1)*np.sqrt(3)
        z[:-1] *= 2/(width+height)
        z[-1]=2*width/(width+height)
        pairs=[]
        for i in range(21):
            if time.monotonic() >= deadline:
                break
            for j in range(i):
                edge=(j,i)
                reflected=tuple(sorted((mirror[i],mirror[j])))
                if edge <= reflected:
                    pairs.append((j,i))
        if time.monotonic() >= deadline:
            break
        ii=np.array([p[0] for p in pairs]); jj=np.array([p[1] for p in pairs])
        DX=X[ii]-X[jj]; DY=Y[ii]-Y[jj]; RR=R[ii]+R[jj]
        # One positive-x boundary per orbit; upper and lower y boundaries.
        unique=[owners.index(k) for k in range(m)]
        B=[]; offsets=[]
        for i in unique:
            b=-X[i]-R[i]; b[-1]+=.5
            B.append(b); offsets.append(0.)
            b=-Y[i]-R[i]; b[-1]-=.5
            B.append(b); offsets.append(1.)
            b=Y[i]-R[i]; b[-1]-=.5
            B.append(b); offsets.append(1.)
        B=np.array(B); offsets=np.array(offsets)
        objective=-np.sum(R,axis=0)
        bounds=[(0.,None)]*nx + [(None,None)]*m + [(1e-7,None)]*m + [(.12,1.88)]
        last=[z.copy()]
        cache=[None,None,None]
        def compute(v):
            if cache[0] is not None and np.array_equal(v,cache[0]):
                return cache[1],cache[2]
            dx=DX@v; dy=DY@v
            dist=np.maximum(np.sqrt(dx*dx+dy*dy),1e-15)
            c=np.concatenate((dist-RR@v,B@v+offsets))
            jac=np.vstack(((dx/dist)[:,None]*DX+(dy/dist)[:,None]*DY-RR,B))
            cache[:]=[v.copy(),c,jac]
            return c,jac
        def fun(v):
            if time.monotonic() >= deadline:
                raise StopSearch()
            return float(objective@v)
        def callback(v):
            last[0]=v.copy()
            if time.monotonic() >= deadline-.012:
                raise StopSearch()
        try:
            result=minimize(fun,z,jac=lambda v:objective,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda v:compute(v)[0],
                             'jac':lambda v:compute(v)[1]},
                callback=callback,
                options={'maxiter':int(params['iterations']),'ftol':2e-12,'disp':False})
            last[0]=result.x
        except StopSearch:
            pass
        v=last[0]
        if time.monotonic() < deadline:
            save(np.column_stack((X@v,Y@v,R@v)))
    return best
