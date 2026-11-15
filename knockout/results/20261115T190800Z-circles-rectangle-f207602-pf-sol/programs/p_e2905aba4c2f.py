import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linprog

PARAMS = {
    'noise': (0.035, 0.005, 0.12),
    'iterations': (65, 15, 150),
    'lp_steps': (3, 0, 12),
    'safety': (3e-15, 1e-15, 1e-12, 'log')
}
N=21
I,J=np.triu_indices(N,1)
M=len(I)
C=np.zeros(64); C[42:63]=-1
A=np.zeros((84,64)); B=np.zeros(84)
for k in range(N):
    A[k,k]=-1; A[k,42+k]=1
    A[21+k,k]=1; A[21+k,42+k]=1; A[21+k,63]=-1
    A[42+k,21+k]=-1; A[42+k,42+k]=1
    A[63+k,21+k]=1; A[63+k,42+k]=1; A[63+k,63]=1; B[63+k]=2

class Stop(Exception): pass

def perimeter(p,r):
    x=[Fraction(float(v)) for v in p[:,0]]
    y=[Fraction(float(v)) for v in p[:,1]]
    q=[Fraction(float(v)) for v in r]
    return (max(a+b for a,b in zip(x,q))-min(a-b for a,b in zip(x,q))+
            max(a+b for a,b in zip(y,q))-min(a-b for a,b in zip(y,q)))

def valid(p,r):
    if not np.isfinite(p).all() or not np.isfinite(r).all() or np.min(r)<=0: return False
    d=np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
    return not np.any(d<r[I]+r[J]) and perimeter(p,r)<=2

def construct(params,seed,budget_s,init=None):
    start=time.monotonic(); end=start+0.70*budget_s
    rng=np.random.default_rng(seed)
    best=init; score=-np.inf
    def check():
        if time.monotonic()>=end: raise Stop()
    if init is None:
        arr=np.array([[0.12+0.25*(k%7),0.12+0.25*(k//7),0.09] for k in range(21)])
    else:
        arr=np.array(init['circles'],dtype=float)
        if valid(arr[:,:2],arr[:,2]): score=float(np.sum(arr[:,2]))
    def save(z,swap=False):
        nonlocal best,score
        check()
        p=np.column_stack((z[:21],z[21:42])).copy(); r=z[42:63].copy()
        if np.min(r)<=0 or not np.isfinite(z).all(): return
        p-=np.min(p,axis=0)
        d=np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
        r*=min(1.0,float(np.min(d/(r[I]+r[J]))))*(1-float(params['safety']))
        per=perimeter(p,r)
        if per>2:
            f=2/float(per)*(1-float(params['safety'])); p*=f; r*=f
        s=float(np.sum(r))
        if s>score and valid(p,r):
            if swap: p=p[:,::-1]
            if valid(p,r):
                best={'circles':np.column_stack((p,r)).tolist()}; score=s
    p=arr[:,:2].copy(); r=arr[:,2].copy()
    lo=np.min(p-r[:,None],axis=0); sides=np.max(p+r[:,None],axis=0)-lo
    p-=lo
    # Select the more accurate reflection axis, then use a horizontal axis internally.
    errors=[]
    for axis in range(2):
        check()
        pp=p.copy(); pp[:,axis]=sides[axis]-pp[:,axis]
        costs=np.sum((pp[:,None,:]-p[None,:,:])**2,axis=2)+(r[:,None]-r[None,:])**2
        errors.append(float(np.sum(np.min(costs,axis=1))))
    swap=errors[0]<errors[1]
    if swap: p=p[:,::-1]; sides=sides[::-1]
    z=np.r_[p[:,0],p[:,1],r,sides[0]]
    def gj(z):
        check()
        dx=z[I]-z[J]; dy=z[21+I]-z[21+J]
        d=np.maximum(np.sqrt(dx*dx+dy*dy),1e-25)
        g=np.r_[d-z[42+I]-z[42+J], B-A@z]
        j=np.zeros((M+84,64)); rows=np.arange(M)
        j[rows,I]=dx/d; j[rows,J]=-dx/d
        j[rows,21+I]=dy/d; j[rows,21+J]=-dy/d
        j[rows,42+I]=-1; j[rows,42+J]=-1; j[M:]=-A
        return g,j
    def refine(v):
        # A rigid contact graph can be polished directly, without an optimizer's stopping tolerances.
        check(); g,j=gj(v); active=np.flatnonzero(g<2e-7)
        if len(active)<64: return v
        w=v.copy()
        for _ in range(5):
            check(); g,j=gj(w)
            step=np.linalg.lstsq(j[active],-g[active],rcond=1e-11)[0]
            if np.max(np.abs(step))>0.002: break
            w+=step
            save(w,swap)
            if np.max(np.abs(step))<2e-16: break
        return w
    try:
        save(z,swap); z=refine(z)
        # Build the reflection quotient from nearest reflected partners.
        remaining=set(range(N)); groups=[]
        while remaining:
            check()
            i=min(remaining); remaining.remove(i)
            target=np.array([p[i,0],sides[1]-p[i,1]])
            if abs(p[i,1]-sides[1]/2)<1e-5:
                groups.append((i,))
            elif remaining:
                j=min(remaining,key=lambda k:float(np.sum((p[k]-target)**2)+(r[k]-r[i])**2))
                if np.linalg.norm(p[j]-target)>1e-4: raise ValueError()
                remaining.remove(j)
                groups.append((i,j) if p[i,1]>p[j,1] else (j,i))
            else: raise ValueError()
        d=2*len(groups)+sum(len(g)==2 for g in groups)+1
        T=np.zeros((64,d)); base=np.zeros(64); u=np.zeros(d); bounds=[]; at=0
        for group in groups:
            check()
            for i in group: T[i,at]=1
            u[at]=np.mean([z[i] for i in group]); bounds.append((0,2)); at+=1
            if len(group)==2:
                T[21+group[0],at]=1; T[21+group[1],at]=-1
                u[at]=abs(z[21+group[0]]-z[21+group[1]])/2
                bounds.append((0,1)); at+=1
            for i in group: T[42+i,at]=1
            u[at]=np.mean([z[42+i] for i in group]); bounds.append((1e-6,1)); at+=1
        for i in range(N):
            base[21+i]=1; T[21+i,-1]=-0.5
        T[63,-1]=1; u[-1]=z[63]; bounds.append((0.1,1.9))
        objective=C@T
        # Remove algebraically duplicated inequalities in the quotient.
        test=u+rng.normal(0,0.003,d)
        gg,jj=gj(base+T@test); jj=jj@T
        keys=np.round(np.column_stack((gg,jj)),11)
        _,keep=np.unique(keys,axis=0,return_index=True); keep=np.sort(keep)
        cache_x=None; cache_g=None; cache_j=None
        def reduced(v):
            nonlocal cache_x,cache_g,cache_j
            check()
            if cache_x is None or not np.array_equal(v,cache_x):
                g,j=gj(base+T@v); cache_x=v.copy(); cache_g=g[keep]; cache_j=(j@T)[keep]
            return cache_g,cache_j
        def callback(v):
            check()
            if -float(objective@v)>score: save(base+T@v,swap)
        anchor=u.copy(); attempt=0
        while time.monotonic()<end:
            check()
            v=anchor.copy()
            if attempt:
                v+=rng.normal(0,float(params['noise'])*(1+0.35*((attempt-1)%3)),d)
                v=np.array([np.clip(x,b[0],b[1]) for x,b in zip(v,bounds)])
                # Feasible separating planes move several contacts in concert.
                for step in range(int(params['lp_steps'])):
                    check()
                    zz=base+T@v; g,j=gj(zz)
                    mat=-j@T; rhs=g-j@(zz-base)
                    objective_lp=objective.copy()
                    if step==0: objective_lp+=rng.normal(0,0.13,d)
                    left=end-time.monotonic()
                    if left<0.12: break
                    res=linprog(objective_lp,A_ub=mat,b_ub=rhs,bounds=bounds,method='highs',options={'time_limit':min(0.10,left)})
                    check()
                    if not res.success: break
                    v=res.x; callback(v)
            result=minimize(lambda q:(float(objective@q),objective),v,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda q:reduced(q)[0],'jac':lambda q:reduced(q)[1]},callback=callback,
                options={'maxiter':int(params['iterations']),'ftol':2e-14,'disp':False})
            check(); callback(result.x); refine(base+T@result.x)
            attempt+=1
    except (Stop,ValueError,np.linalg.LinAlgError):
        pass
    if best is not None: return best
    return {'circles':[[0.1+0.25*(i%7),0.1+0.25*(i//7),0.04] for i in range(21)]}
