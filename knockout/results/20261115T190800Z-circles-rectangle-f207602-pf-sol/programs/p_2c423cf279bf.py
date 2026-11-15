import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"iterations": (135, 60, 250), "safety": (0.70, 0.55, 0.75)}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    end = started + float(params['safety']) * budget_s
    ii, jj = np.triu_indices(21, 1)
    best_exact = -float('inf')
    original = None

    def canonical(obj):
        if 'circles' in obj:
            return np.asarray(obj['circles'], dtype=float).copy()
        z = np.asarray(obj['state'], dtype=float)
        return np.column_stack((z[:21], z[21:42], z[42:63]))

    def exact(c):
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return -float('inf')
        x = [Fraction(float(v)) for v in c[:, 0]]
        y = [Fraction(float(v)) for v in c[:, 1]]
        r = [Fraction(float(v)) for v in c[:, 2]]
        perimeter = max(a+b for a,b in zip(x,r))-min(a-b for a,b in zip(x,r))
        perimeter += max(a+b for a,b in zip(y,r))-min(a-b for a,b in zip(y,r))
        if perimeter > 2:
            return -float('inf')
        dx = c[ii,0]-c[jj,0]
        dy = c[ii,1]-c[jj,1]
        if np.any(np.sqrt(dx*dx+dy*dy) < c[ii,2]+c[jj,2]):
            return -float('inf')
        return float(np.sum(c[:,2]))

    if init is not None:
        try:
            original = canonical(init)
            best_exact = exact(original)
        except Exception:
            original = None
    saved = []
    best_proxy = -float('inf')
    best_c = None

    def repair(c):
        c = np.asarray(c, dtype=float).copy()
        if not np.isfinite(c).all() or np.min(c[:,2]) <= 0:
            return None
        d = np.hypot(c[ii,0]-c[jj,0], c[ii,1]-c[jj,1])
        factor = min(1.0, float(np.min(d/(c[ii,2]+c[jj,2]))))
        if factor <= 0:
            return None
        c[:,2] *= factor * (1.0-3e-14)
        lo = np.min(c[:,:2]-c[:,2,None], axis=0)
        hi = np.max(c[:,:2]+c[:,2,None], axis=0)
        scale = 2.0 / float(np.sum(hi-lo)) * (1.0-3e-14)
        c[:,:2] -= (lo+hi)*0.5
        c *= scale
        return c

    def save(c):
        nonlocal best_proxy, best_c
        q = repair(c)
        if q is None:
            return
        score = float(np.sum(q[:,2]))
        if score > best_proxy:
            best_proxy, best_c = score, q
            saved.append(q)
            if len(saved) > 5:
                del saved[0]

    if original is not None:
        save(original)

    def make_model(symmetric):
        if symmetric:
            nvar = 34
            X = np.zeros((21,nvar)); Y = X.copy(); R = X.copy()
            reflection = np.arange(21)
            for k in range(3):
                Y[k,9+k] = 1
                R[k,21+k] = 1
            for k in range(9):
                a,b = 3+2*k,4+2*k
                X[a,k] = 1; X[b,k] = -1
                Y[a,12+k] = Y[b,12+k] = 1
                R[a,24+k] = R[b,24+k] = 1
                reflection[a],reflection[b] = b,a
            keep=[]
            for a,b in zip(ii,jj):
                if time.monotonic() >= end:
                    raise StopSearch()
                mirror=tuple(sorted((int(reflection[a]),int(reflection[b]))))
                if (int(a),int(b)) <= mirror:
                    keep.append((a,b))
            a=np.array([v[0] for v in keep]); b=np.array([v[1] for v in keep])
        else:
            nvar=64
            X=np.zeros((21,nvar)); Y=X.copy(); R=X.copy()
            X[np.arange(21),np.arange(21)] = 1
            Y[np.arange(21),21+np.arange(21)] = 1
            R[np.arange(21),42+np.arange(21)] = 1
            a,b=ii,jj
        DX=X[a]-X[b]; DY=Y[a]-Y[b]; RR=R[a]+R[b]
        walls=[]; offsets=[]; seen=set()
        for mat, vertical in ((X,False),(Y,True)):
            for k in range(21):
                if time.monotonic() >= end:
                    raise StopSearch()
                for sign in (-1,1):
                    row=sign*mat[k]-R[k]
                    row=row.copy(); row[-1] += -0.5 if vertical else 0.5
                    off=1.0 if vertical else 0.0
                    key=(tuple(row),off)
                    if key not in seen:
                        walls.append(row); offsets.append(off); seen.add(key)
        W=np.array(walls); offset=np.array(offsets)
        obj=-np.sum(R,axis=0)
        bounds=[(None,None)]*nvar
        if symmetric:
            for k in range(9): bounds[k]=(0.0,None)
            for k in range(21,33): bounds[k]=(1e-7,None)
        else:
            for k in range(42,63): bounds[k]=(1e-7,None)
        bounds[-1]=(0.12,1.88)
        return X,Y,R,DX,DY,RR,W,offset,obj,bounds

    def solve(z, model, deadline, iterations):
        X,Y,R,DX,DY,RR,W,offset,obj,bounds=model
        last=z.copy()
        def check():
            if time.monotonic() >= deadline:
                raise StopSearch()
        def fun(v):
            check()
            return float(obj@v)
        def cons(v):
            check()
            return np.concatenate((np.hypot(DX@v,DY@v)-RR@v, W@v+offset))
        def jac(v):
            check()
            dx=DX@v; dy=DY@v
            d=np.maximum(np.hypot(dx,dy),1e-20)
            return np.vstack(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RR,W))
        def callback(v):
            nonlocal last
            check()
            last=v.copy()
            save(np.column_stack((X@v,Y@v,R@v)))
        try:
            out=minimize(fun,z,jac=lambda v:obj,method='SLSQP',bounds=bounds,
                         constraints={'type':'ineq','fun':cons,'jac':jac},callback=callback,
                         options={'maxiter':int(iterations),'ftol':2e-12,'disp':False})
            last=out.x
        except StopSearch:
            pass
        save(np.column_stack((X@last,Y@last,R@last)))
        return last

    def seed_state(rows):
        singles=[]; pairs=[]
        for row,n in enumerate(rows):
            if time.monotonic() >= end:
                raise StopSearch()
            yy=(row-(len(rows)-1)/2)*np.sqrt(3.0)
            if n%2:
                singles.append((0.0,yy,0.97))
                for k in range(1,(n+1)//2): pairs.append((2.0*k,yy,0.97))
            else:
                for k in range(n//2): pairs.append((2.0*k+1,yy,0.97))
        c=np.array(singles+[p for a in pairs for p in (a,(-a[0],a[1],a[2]))])
        c=repair(c)
        z=np.zeros(34)
        z[:9]=c[3::2,0]
        z[9:12]=c[:3,1]; z[12:21]=c[3::2,1]
        z[21:24]=c[:3,2]; z[24:33]=c[3::2,2]
        z[33]=float(np.max(c[:,0]+c[:,2])-np.min(c[:,0]-c[:,2]))
        save(c)
        return z

    try:
        model=make_model(True)
        families=[(3,4,3,4,3,4),(5,4,3,4,5),(3,4,5,4,5)]
        fractions=(0.28,0.52,0.73)
        for rows,fraction in zip(families,fractions):
            if time.monotonic() >= started+fraction*(end-started):
                continue
            z=seed_state(rows)
            solve(z,model,started+fraction*(end-started),params['iterations'])
        if best_c is not None and time.monotonic() < end-0.04:
            full=make_model(False)
            c=best_c.copy()
            w=float(np.max(c[:,0]+c[:,2])-np.min(c[:,0]-c[:,2]))
            h=float(np.max(c[:,1]+c[:,2])-np.min(c[:,1]-c[:,2]))
            w += max(0.0,2-w-h)*0.5
            z=np.concatenate((c[:,0],c[:,1],c[:,2],[w]))
            solve(z,full,end,params['iterations'])
    except StopSearch:
        pass

    answer=original if best_exact > -float('inf') else None
    for c in reversed(saved):
        if time.monotonic() >= started+0.91*budget_s:
            break
        score=exact(c)
        if score > best_exact:
            best_exact=score; answer=c
    if answer is None:
        r=1.0/22.0*(1-1e-12)
        answer=np.array([[(k-10)*2*r,0.0,r] for k in range(21)])
    return {'circles':answer.tolist()}
