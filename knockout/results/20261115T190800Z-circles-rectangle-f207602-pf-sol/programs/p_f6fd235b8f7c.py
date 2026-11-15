import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"iterations": (110, 40, 250), "screen_fraction": (0.24, 0.1, 0.3), "radius_floor": (0.00001, 0.000001, 0.001)}

class StopSearch(Exception):
    pass

def exact_score(data):
    try:
        c = np.asarray(data['circles'], dtype=float)
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return -float('inf')
        x = [Fraction(float(v)) for v in c[:, 0]]
        y = [Fraction(float(v)) for v in c[:, 1]]
        r = [Fraction(float(v)) for v in c[:, 2]]
        if max(a+b for a,b in zip(x,r))-min(a-b for a,b in zip(x,r))+max(a+b for a,b in zip(y,r))-min(a-b for a,b in zip(y,r)) > 2:
            return -float('inf')
        for i in range(21):
            for j in range(i):
                if np.sqrt((c[i,0]-c[j,0])**2+(c[i,1]-c[j,1])**2) < c[i,2]+c[j,2]:
                    return -float('inf')
        return float(np.sum(c[:,2]))
    except Exception:
        return -float('inf')

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = exact_score(init) if init is not None else -float('inf')
    ii, jj = np.triu_indices(21, 1)
    saved = None

    def retain(c):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        c = np.array(c, dtype=float, copy=True)
        if not np.isfinite(c).all() or np.min(c[:,2]) <= 0:
            return
        d = np.sqrt(np.sum((c[ii,:2]-c[jj,:2])**2, axis=1))
        ratios = d / (c[ii,2]+c[jj,2])
        f = min(1.0, float(np.min(ratios)))
        if f <= 0:
            return
        c[:,2] *= f * (1.0-3e-14)
        lo = np.min(c[:,:2]-c[:,2,None], axis=0)
        hi = np.max(c[:,:2]+c[:,2,None], axis=0)
        c[:,:2] -= (hi+lo)/2
        scale = 2.0 / float(np.sum(hi-lo)) * (1.0-3e-14)
        c *= scale
        if float(c[:,2].sum()) <= best_score:
            return
        data = {'circles': c.tolist()}
        s = exact_score(data)
        if s > best_score:
            best_score, best = s, data

    def model(rows):
        reps = []
        for row, count in enumerate(rows):
            if time.monotonic() >= deadline:
                raise StopSearch
            if count % 2:
                reps.append((0.0, row, False))
            for k in range(count//2):
                reps.append((float(2*k+1 if count%2 == 0 else 2*k+2), row, True))
        nvar = sum(3 if p else 2 for x,y,p in reps)+1
        X=[]; Y=[]; R=[]; z=np.zeros(nvar); walls=[]; bounds=[]
        radius = 2.0/(2*max(rows)+2+(len(rows)-1)*np.sqrt(3.0))
        height = radius*(2+(len(rows)-1)*np.sqrt(3.0))
        p=0
        for x,y,paired in reps:
            if time.monotonic() >= deadline:
                raise StopSearch
            mx=np.zeros(nvar); my=np.zeros(nvar); mr=np.zeros(nvar)
            if paired:
                mx[p]=1; z[p]=x*radius; bounds.append((0,None)); p+=1
            my[p]=1; z[p]=radius+y*np.sqrt(3.0)*radius; bounds.append((None,None)); p+=1
            mr[p]=1; z[p]=radius*.98; bounds.append((params['radius_floor'],None)); p+=1
            X.append(mx); Y.append(my); R.append(mr)
            if paired:
                X.append(-mx); Y.append(my); R.append(mr)
            a=-mx-mr; a[-1]-=.5; walls.append((a,1.0))
            walls.append((my-mr,0.0))
            a=-my-mr; a[-1]+=1; walls.append((a,0.0))
        bounds.append((.15,1.85)); z[-1]=height
        X=np.array(X); Y=np.array(Y); R=np.array(R)
        dx=X[ii]-X[jj]; dy=Y[ii]-Y[jj]; sr=R[ii]+R[jj]
        keep=[]; seen=set()
        for k in range(len(ii)):
            if time.monotonic() >= deadline:
                raise StopSearch
            ax=dx[k]; ay=dy[k]
            nz=np.flatnonzero(ax)
            if len(nz) and ax[nz[0]]<0: ax=-ax
            nz=np.flatnonzero(ay)
            if len(nz) and ay[nz[0]]<0: ay=-ay
            key=(ax.tobytes(),ay.tobytes(),sr[k].tobytes())
            if key not in seen:
                seen.add(key); keep.append(k)
        dx=dx[keep]; dy=dy[keep]; sr=sr[keep]
        A=np.array([a for a,b in walls]); b=np.array([b for a,b in walls])
        obj=-R.sum(axis=0)
        return (X,Y,R,dx,dy,sr,A,b,obj,bounds),z

    def solve(m,z,until,maxiter):
        nonlocal saved
        X,Y,R,dx,dy,sr,A,b,obj,bounds=m
        last=[z.copy()]
        cache=[None,None,None]
        def check():
            if time.monotonic() >= until:
                raise StopSearch
        def objective(v):
            check()
            return float(obj@v)
        def calc(v):
            check()
            if cache[0] is None or not np.array_equal(v,cache[0]):
                xx=dx@v; yy=dy@v
                dist=np.sqrt(xx*xx+yy*yy+1e-30)
                val=np.concatenate((dist-sr@v,A@v+b))
                jac=np.vstack(((xx/dist)[:,None]*dx+(yy/dist)[:,None]*dy-sr,A))
                cache[:]=[v.copy(),val,jac]
            return cache[1],cache[2]
        def callback(v):
            check()
            last[0]=v.copy()
            if -obj@v > best_score:
                retain(np.column_stack((X@v,Y@v,R@v)))
        try:
            res=minimize(objective,z,jac=lambda v:obj,method='SLSQP',bounds=bounds,
                         constraints={'type':'ineq','fun':lambda v:calc(v)[0],'jac':lambda v:calc(v)[1]},
                         callback=callback,options={'maxiter':maxiter,'ftol':2e-11,'disp':False})
            last[0]=res.x
        except StopSearch:
            pass
        v=last[0]
        retain(np.column_stack((X@v,Y@v,R@v)))
        merit=float(-obj@v)
        violation=float(np.min(np.concatenate((np.sqrt((dx@v)**2+(dy@v)**2)-sr@v,A@v+b))))
        if violation > -1e-5 and (saved is None or merit>saved[0]):
            saved=(merit,m,v.copy())
        return v

    patterns=[[5,4,5,4,3],[4,5,3,5,4],[5,4,3,4,5]]
    for rows in patterns:
        if time.monotonic() >= deadline-0.04:
            break
        try:
            m,z=model(rows)
        except StopSearch:
            break
        end=min(deadline-0.03,time.monotonic()+params['screen_fraction']*budget_s)
        solve(m,z,end,int(params['iterations']))
    if saved is not None and time.monotonic()<deadline-0.025:
        _,m,z=saved
        solve(m,z,deadline-0.01,200)
    if best is None or not np.isfinite(best_score):
        r=0.09
        c=np.array([[(k%7)*2*r,(k//7)*2*r,r] for k in range(21)])
        best={'circles':c.tolist()}
    return best
