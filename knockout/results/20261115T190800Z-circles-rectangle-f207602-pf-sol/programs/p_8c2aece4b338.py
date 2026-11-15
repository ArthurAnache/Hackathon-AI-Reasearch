import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"search_fraction": (0.69, 0.5, 0.75), "stage_seconds": (0.235, 0.08, 0.4), "maxiter": (150, 50, 300)}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + budget_s * params['search_fraction']
    best = init
    best_score = -float('inf')

    def exact(data):
        try:
            a = np.asarray(data['circles'], dtype=float)
            if a.shape != (21, 3) or not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
                return -float('inf')
            xs = [Fraction(float(v)) for v in a[:, 0]]
            ys = [Fraction(float(v)) for v in a[:, 1]]
            rs = [Fraction(float(v)) for v in a[:, 2]]
            w = max(x+r for x,r in zip(xs,rs))-min(x-r for x,r in zip(xs,rs))
            h = max(y+r for y,r in zip(ys,rs))-min(y-r for y,r in zip(ys,rs))
            if w+h > 2:
                return -float('inf')
            for i,j in itertools.combinations(range(21),2):
                if time.monotonic() > started + budget_s * 0.94:
                    return -float('inf')
                dx = a[i,0]-a[j,0]
                dy = a[i,1]-a[j,1]
                if np.sqrt(dx*dx+dy*dy) < a[i,2]+a[j,2]:
                    return -float('inf')
            return float(np.sum(a[:,2]))
        except Exception:
            return -float('inf')

    if init is not None:
        best_score = exact(init)

    def retain(a):
        nonlocal best, best_score
        if time.monotonic() >= deadline or not np.isfinite(a).all():
            return
        a = a.copy()
        if np.min(a[:,2]) <= 0:
            return
        ii,jj = np.triu_indices(21,1)
        distances = np.sqrt(np.sum((a[ii,:2]-a[jj,:2])**2,axis=1))
        ratio = min(1.0,float(np.min(distances/(a[ii,2]+a[jj,2]))))
        if ratio <= 0:
            return
        a[:,2] *= ratio * (1.0-5e-14)
        low = np.min(a[:,:2]-a[:,2,None],axis=0)
        high = np.max(a[:,:2]+a[:,2,None],axis=0)
        size = float(np.sum(high-low))
        if size <= 0:
            return
        a[:,:2] -= (low+high)*0.5
        a *= (2.0-1e-13)/size
        if float(np.sum(a[:,2])) <= best_score:
            return
        candidate = {'circles':a.tolist()}
        score = exact(candidate)
        if score > best_score:
            best_score, best = score, candidate

    families = [(4,5,3,5,4), (5,4,3,4,5), (4,5,4,5,3), (3,4,5,4,5), (5,4,5,4,3)]
    saved = []

    def model(rows):
        reps = []
        for row,n in enumerate(rows):
            if time.monotonic() >= deadline:
                raise StopSearch()
            if n % 2:
                reps.append((0.0,float(row)*np.sqrt(3.0),False))
            for k in range(n//2):
                reps.append((float(2*k+1 if n%2==0 else 2*k+2),float(row)*np.sqrt(3.0),True))
        m = len(reps)
        pcount = sum(int(t[2]) for t in reps)
        d = pcount+2*m+1
        X = np.zeros((21,d)); Y = X.copy(); R = X.copy()
        mirror=[]; representatives=[]; q=np.zeros(d)
        pos=0; xp=0
        factor=2.0/(2*max(rows)+2+(len(rows)-1)*np.sqrt(3.0))
        for k,(x,y,paired) in enumerate(reps):
            if time.monotonic() >= deadline:
                raise StopSearch()
            yi=pcount+k; ri=pcount+m+k
            representatives.append(pos)
            Y[pos,yi]=1; R[pos,ri]=1
            q[yi]=(y+1)*factor; q[ri]=factor
            if paired:
                X[pos,xp]=1; X[pos+1,xp]=-1
                Y[pos+1,yi]=1; R[pos+1,ri]=1
                q[xp]=x*factor
                mirror.extend([pos+1,pos]); pos+=2; xp+=1
            else:
                mirror.append(pos); pos+=1
        q[-1]=2*max(rows)*factor
        pairs=[]
        for i,j in itertools.combinations(range(21),2):
            if time.monotonic() >= deadline:
                raise StopSearch()
            reflected=tuple(sorted((mirror[i],mirror[j])))
            if (i,j) <= reflected:
                pairs.append((i,j))
        ii=np.array([p[0] for p in pairs]); jj=np.array([p[1] for p in pairs])
        DX=X[ii]-X[jj]; DY=Y[ii]-Y[jj]; SR=R[ii]+R[jj]
        ids=np.array(representatives)
        B=np.concatenate((-X[ids]-R[ids],Y[ids]-R[ids],-Y[ids]-R[ids]),axis=0)
        B[:m,-1]+=0.5
        B[2*m:,-1]-=1.0
        bc=np.zeros(3*m); bc[2*m:]=2.0
        obj=-np.sum(R,axis=0)
        bounds=[(0.0,1.0)]*pcount+[(0.0,2.0)]*m+[(1e-7,0.5)]*m+[(0.15,1.85)]
        return (q,X,Y,R,DX,DY,SR,B,bc,obj,bounds)

    def optimize(mod, q, seconds):
        q0,X,Y,R,DX,DY,SR,B,bc,obj,bounds=mod
        stage_end=min(deadline,time.monotonic()+seconds)
        last=q.copy()
        cache_q=None; cache_c=None; cache_j=None
        def compute(z):
            nonlocal cache_q,cache_c,cache_j
            if time.monotonic() >= stage_end:
                raise StopSearch()
            if cache_q is not None and np.array_equal(z,cache_q):
                return
            dx=DX@z; dy=DY@z
            dist=np.maximum(np.sqrt(dx*dx+dy*dy),1e-20)
            cache_c=np.concatenate((dist-SR@z,B@z+bc))
            cache_j=np.concatenate(((dx/dist)[:,None]*DX+(dy/dist)[:,None]*DY-SR,B),axis=0)
            cache_q=z.copy()
        def fun(z):
            compute(z)
            return cache_c
        def jac(z):
            compute(z)
            return cache_j
        def callback(z):
            nonlocal last
            last=z.copy()
            if time.monotonic() >= stage_end:
                raise StopSearch()
        try:
            res=minimize(lambda z:float(obj@z),q,jac=lambda z:obj,method='SLSQP',bounds=bounds,
                         constraints={'type':'ineq','fun':fun,'jac':jac},callback=callback,
                         options={'maxiter':int(params['maxiter']),'ftol':2e-12,'disp':False})
            last=res.x
        except StopSearch:
            pass
        except (ValueError, FloatingPointError):
            pass
        a=np.column_stack((X@last,Y@last,R@last))
        retain(a)
        return last,float(np.sum(a[:,2]))

    for rows in families:
        if time.monotonic() >= deadline-0.025:
            break
        try:
            mod=model(rows)
            remaining=deadline-time.monotonic()
            q,score=optimize(mod,mod[0],min(params['stage_seconds'],remaining*0.75))
            saved.append((score,mod,q))
        except StopSearch:
            break
    if saved and time.monotonic() < deadline-0.035:
        saved.sort(key=lambda v:v[0],reverse=True)
        _,mod,q=saved[0]
        optimize(mod,q,max(0.001,deadline-time.monotonic()-0.012))
    if best is None or best_score == -float('inf'):
        a=[]
        for j in range(3):
            if time.monotonic() >= started+budget_s*0.98:
                break
            for i in range(7):
                a.append([(i-3)*0.19,(j-1)*0.19,0.09])
        if len(a)==21:
            fallback={'circles':a}
            if exact(fallback)>best_score:
                best=fallback
    return best
