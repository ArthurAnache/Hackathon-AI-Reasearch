import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"search_fraction": (0.69, 0.5, 0.75), "seed_slice": (0.205, 0.08, 0.35)}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + float(params['search_fraction']) * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    best = init
    best_score = -float('inf')

    def exact(c):
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return -float('inf')
        d = np.sqrt((c[ii, 0]-c[jj, 0])**2 + (c[ii, 1]-c[jj, 1])**2)
        if np.any(d < c[ii, 2]+c[jj, 2]):
            return -float('inf')
        x = [Fraction(float(v)) for v in c[:, 0]]
        y = [Fraction(float(v)) for v in c[:, 1]]
        r = [Fraction(float(v)) for v in c[:, 2]]
        p = max(a+b for a,b in zip(x,r))-min(a-b for a,b in zip(x,r))
        p += max(a+b for a,b in zip(y,r))-min(a-b for a,b in zip(y,r))
        return float(np.sum(c[:, 2])) if p <= 2 else -float('inf')

    if init is not None:
        try:
            best_score = exact(np.asarray(init['circles'], dtype=float))
        except Exception:
            pass

    def retain(c):
        nonlocal best, best_score
        c = np.asarray(c, dtype=float).copy()
        if not np.isfinite(c).all() or np.any(c[:,2] <= 0):
            return
        d = np.sqrt((c[ii,0]-c[jj,0])**2+(c[ii,1]-c[jj,1])**2)
        f = min(1.0, float(np.min(d/(c[ii,2]+c[jj,2]))))
        if f <= 0:
            return
        c[:,2] *= f * (1.0-3e-14)
        lo = np.min(c[:,:2]-c[:,2,None], axis=0)
        hi = np.max(c[:,:2]+c[:,2,None], axis=0)
        p = float(np.sum(hi-lo))
        if p <= 0:
            return
        c[:,:2] -= (hi+lo)/2
        c *= (2.0/p)*(1.0-4e-14)
        if float(np.sum(c[:,2])) <= best_score:
            return
        score = exact(c)
        if score > best_score:
            best_score = score
            best = {'circles': c.tolist()}

    def lattice(counts):
        rows = []
        for k, m in enumerate(counts):
            if time.monotonic() >= end:
                break
            for t in range(m):
                rows.append([k*np.sqrt(3.0)+1, 2*t-(m-1), 1.0])
        c = np.array(rows)
        w = (len(counts)-1)*np.sqrt(3.0)+2
        h = 2*max(counts)
        c *= 2/(w+h)
        return c, w*2/(w+h)

    def optimize(c, width, symmetric, until):
        if time.monotonic() >= until:
            return
        # Each physical coordinate maps to one reduced variable.
        ax=[]; ay=[]; ar=[]; signs=[]; values=[]; lower=[]; upper=[]
        if symmetric:
            used = np.zeros(21, bool)
            for i in range(21):
                if time.monotonic() >= until:
                    return
                if used[i]:
                    continue
                if abs(c[i,1]) < 1e-9:
                    k=len(values); values.extend([c[i,0],c[i,2]])
                    lower.extend([0.0,1e-7]); upper.extend([2.0,1.0])
                    ax.append((i,k)); ay.append((i,-1)); ar.append((i,k+1)); signs.append((i,0.0)); used[i]=True
                else:
                    target=c[i].copy(); target[1]*=-1
                    ds=np.sum((c-target)**2,axis=1); ds[used]=np.inf; ds[i]=np.inf
                    j=int(np.argmin(ds))
                    k=len(values); values.extend([c[i,0],abs(c[i,1]),c[i,2]])
                    lower.extend([0.0,1e-7,1e-7]); upper.extend([2.0,1.0,1.0])
                    for h,s in ((i,np.sign(c[i,1])),(j,-np.sign(c[i,1]))):
                        ax.append((h,k)); ay.append((h,k+1)); ar.append((h,k+2)); signs.append((h,s)); used[h]=True
        else:
            for i in range(21):
                k=len(values); values.extend(c[i].tolist())
                lower.extend([0.0,-1.0,1e-7]); upper.extend([2.0,1.0,1.0])
                ax.append((i,k)); ay.append((i,k+1)); ar.append((i,k+2)); signs.append((i,1.0))
        values.append(width); lower.append(0.08); upper.append(1.92)
        z=np.array(values); nv=len(z)
        E=np.zeros((63,nv))
        for i,k in ax: E[3*i,k]=1
        sm=dict(signs)
        for i,k in ay:
            if k>=0: E[3*i+1,k]=sm[i]
        for i,k in ar: E[3*i+2,k]=1
        X=E[0::3]; Y=E[1::3]; R=E[2::3]
        B=np.vstack((X-R,-X-R,-Y-R,Y-R))
        B[21:42,-1]+=1
        B[42:,-1]-=0.5
        const=np.r_[np.zeros(42),np.ones(42)]
        DX=X[ii]-X[jj]; DY=Y[ii]-Y[jj]; RS=R[ii]+R[jj]
        objective=-np.sum(R,axis=0)
        cache_z=None; cache_val=None; cache_jac=None
        def calculate(v):
            nonlocal cache_z,cache_val,cache_jac
            if time.monotonic() >= until:
                raise StopSearch()
            if cache_z is None or not np.array_equal(v,cache_z):
                dx=DX@v; dy=DY@v
                dist=np.maximum(np.sqrt(dx*dx+dy*dy),1e-15)
                cache_val=np.r_[B@v+const,dist-RS@v]
                cache_jac=np.vstack((B,DX*(dx/dist)[:,None]+DY*(dy/dist)[:,None]-RS))
                cache_z=v.copy()
            return cache_val
        last=z.copy(); iterations=0
        def cb(v):
            nonlocal last,iterations
            last=v.copy(); iterations+=1
            if iterations%4==0:
                retain((E@v).reshape(21,3))
            if time.monotonic() >= until:
                raise StopSearch()
        try:
            res=minimize(lambda v: float(objective@v),z,jac=lambda v: objective,
                         method='SLSQP',bounds=list(zip(lower,upper)),
                         constraints={'type':'ineq','fun':calculate,'jac':lambda v:(calculate(v),cache_jac)[1]},
                         callback=cb,options={'maxiter':160,'ftol':2e-11,'disp':False})
            last=res.x
        except StopSearch:
            pass
        retain((E@last).reshape(21,3))

    families = [[4,5,4,5,3], [4,5,3,5,4], [5,4,3,4,5], [3,5,5,5,3]]
    # Always establish a substantial valid fallback before numerical work.
    c,w=lattice(families[0])
    if c.shape==(21,3): retain(c)
    if best_score > 2.2 and init is not None:
        c0=np.asarray(init['circles'],float).copy()
        lo=np.min(c0[:,:2]-c0[:,2,None],axis=0); hi=np.max(c0[:,:2]+c0[:,2,None],axis=0)
        c0[:,0]-=lo[0]; c0[:,1]-=(hi[1]+lo[1])/2
        optimize(c0,float(hi[0]-lo[0]),False,min(end,time.monotonic()+0.28*budget_s))
    for counts in families:
        if time.monotonic() >= end-0.23*budget_s:
            break
        c,w=lattice(counts)
        if c.shape!=(21,3): break
        optimize(c,w,True,min(end-0.16*budget_s,time.monotonic()+float(params['seed_slice'])*budget_s))
    if best is not None and time.monotonic() < end-0.015:
        c=np.asarray(best['circles'],float).copy()
        lo=np.min(c[:,:2]-c[:,2,None],axis=0); hi=np.max(c[:,:2]+c[:,2,None],axis=0)
        c[:,0]-=lo[0]; c[:,1]-=(lo[1]+hi[1])/2
        optimize(c,float(hi[0]-lo[0]),False,end)
    return best
