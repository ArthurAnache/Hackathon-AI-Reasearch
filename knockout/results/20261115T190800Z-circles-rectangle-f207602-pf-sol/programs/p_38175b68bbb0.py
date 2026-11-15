import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {'iterations': (90, 40, 160), 'probe_iterations': (48, 20, 90), 'perturbation': (0.012, 0.001, 0.05)}

class Halt(Exception):
    pass

def canonical(obj):
    if obj is None:
        return None
    if 'circles' in obj:
        return obj
    a=[]
    for x,y,r,m in obj['orbits']:
        a.append([x,y,r])
        if m==2:
            a.append([x,-y,r])
    return {'circles':a}

def score(obj):
    try:
        a=np.asarray(canonical(obj)['circles'],float)
        if a.shape!=(21,3) or not np.isfinite(a).all() or np.any(a[:,2]<=0):
            return -np.inf
        i,j=np.triu_indices(21,1)
        if np.any(np.sqrt((a[i,0]-a[j,0])**2+(a[i,1]-a[j,1])**2)<a[i,2]+a[j,2]):
            return -np.inf
        x=[Fraction(float(v)) for v in a[:,0]]
        y=[Fraction(float(v)) for v in a[:,1]]
        r=[Fraction(float(v)) for v in a[:,2]]
        wh=max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        wh+=max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        return float(np.sum(a[:,2])) if wh<=2 else -np.inf
    except (TypeError,ValueError,KeyError,OverflowError):
        return -np.inf

def construct(params, seed, budget_s, init=None):
    t0=time.monotonic()
    deadline=t0+0.70*budget_s
    rng=np.random.default_rng(seed)
    best=init
    bestscore=score(init)
    def tick():
        if time.monotonic()>=deadline:
            raise Halt()
    fallback={'circles':[[.2*(k%7-3),.2*(k//7-1),.09999999999998] for k in range(21)]}
    fs=score(fallback)
    if fs>bestscore:
        best,bestscore=fallback,fs
    pool=[]
    try:
        tick()
        gids=np.r_[np.arange(3),np.repeat(np.arange(3,12),2)]
        signs=np.r_[np.zeros(3),np.tile([1.,-1.],9)]
        X=np.zeros((21,34)); Y=X.copy(); R=X.copy()
        X[np.arange(21),gids]=1
        R[np.arange(21),21+gids]=1
        ix=np.flatnonzero(gids>=3)
        Y[ix,9+gids[ix]]=signs[ix]
        seen=set(); pairs=[]
        for i in range(21):
            tick()
            for j in range(i):
                tick()
                key=(min(gids[i],gids[j]),max(gids[i],gids[j]),signs[i]*signs[j])
                if key not in seen:
                    seen.add(key); pairs.append((i,j))
        pi,pj=np.array(pairs).T
        DX=X[pi]-X[pj]; DY=Y[pi]-Y[pj]; RS=R[pi]+R[pj]
        W=np.zeros((36,34)); off=np.zeros(36)
        for g in range(12):
            tick()
            W[g,g]=1; W[g,21+g]=-1; W[g,33]=.5
            W[g+12,g]=-1; W[g+12,21+g]=-1; W[g+12,33]=.5
            W[g+24,21+g]=-1; W[g+24,33]=-.5; off[g+24]=1
            if g>=3:
                W[g+24,g+9]=-1
        grad=np.zeros(34); grad[21:33]=-np.r_[np.ones(3),np.full(9,2.)]
        bounds=[(-1.,1.)]*12+[(0.,1.)]*9+[(1e-6,.5)]*12+[(.1,1.9)]
        ai,aj=np.triu_indices(21,1)
        def retain(z):
            nonlocal best,bestscore
            tick()
            a=np.column_stack((X@z,Y@z,R@z))
            if not np.isfinite(a).all() or np.min(a[:,2])<=0:
                return
            d=np.sqrt((a[ai,0]-a[aj,0])**2+(a[ai,1]-a[aj,1])**2)
            shrink=min(1.,float(np.min(d/(a[ai,2]+a[aj,2]))))*(1.-2e-14)
            a[:,2]*=shrink
            wh=np.max(a[:,:2]+a[:,2,None],axis=0)-np.min(a[:,:2]-a[:,2,None],axis=0)
            a*=min(1.,2./float(np.sum(wh)))*(1.-2e-14)
            val=float(np.sum(a[:,2]))
            if val>bestscore:
                obj={'circles':a.tolist()}
                exact=score(obj)
                if exact>bestscore:
                    best,bestscore=obj,exact
        cache=[None,None,None]
        def calc(z):
            tick()
            if cache[0] is None or not np.array_equal(z,cache[0]):
                dx=DX@z; dy=DY@z
                d=np.maximum(np.sqrt(dx*dx+dy*dy),1e-25)
                cache[0]=z.copy()
                cache[1]=np.r_[d-RS@z,W@z+off]
                cache[2]=np.vstack(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RS,W))
            return cache[1],cache[2]
        def fun(z):
            tick(); return float(grad@z)
        def jac(z):
            tick(); return grad
        def solve(z,n,seconds):
            tick()
            local=min(deadline,time.monotonic()+seconds)
            last=[z.copy()]; count=[0]
            class SliceEnd(Exception):
                pass
            def cb(v):
                tick()
                last[0]=v.copy(); count[0]+=1
                if count[0]%8==0:
                    retain(v)
                if time.monotonic()>=local:
                    raise SliceEnd()
            try:
                res=minimize(fun,z,jac=jac,method='SLSQP',bounds=bounds,
                    constraints={'type':'ineq','fun':lambda v:calc(v)[0],'jac':lambda v:calc(v)[1]},
                    callback=cb,options={'maxiter':n,'ftol':3e-13,'disp':False})
                last[0]=res.x
            except SliceEnd:
                pass
            retain(last[0])
            pool.append((float(-grad@last[0])-100*max(0.,-float(np.min(calc(last[0])[0]))),last[0].copy()))
            return last[0]
        def pack(g):
            z=np.zeros(34); z[:12]=g[:,0]; z[12:21]=abs(g[3:,1]); z[21:33]=g[:,2]
            z[33]=np.max(g[:,0]+g[:,2])-np.min(g[:,0]-g[:,2])
            return z
        starts=[]
        for kind in range(4):
            tick()
            g=np.zeros((12,3)); g[:,2]=.095
            g[:3,0]=[-.2,0,.2]
            g[3:7,0]=[-.3,-.1,.1,.3]; g[3:7,1]=.1732050807569
            g[7:,0]=[-.4,-.2,0,.2,.4]; g[7:,1]=.3464101615138
            if kind==1:
                g[:3,0]*=1.35; g[3:7,0]*=1.35; g[7:,0]*=1.05
            elif kind==2:
                g[:3,0]=[-.32,0,.32]
                g[3:7,0]=np.linspace(-.42,.42,4); g[3:7,1]=.16
                g[7:,0]=np.linspace(-.42,.42,5); g[7:,1]=.34
            elif kind==3:
                g[:,0]*=.85; g[3:,1]*=1.2
            starts.append(pack(g))
        projected=[]
        if score(init)>2.30:
            a=np.asarray(canonical(init)['circles'],float).copy()
            lo=np.min(a[:,:2]-a[:,2,None],axis=0); hi=np.max(a[:,:2]+a[:,2,None],axis=0)
            a[:,:2]-=(lo+hi)/2
            for swap in (False,True):
                tick()
                b=a.copy()
                if swap:
                    b[:,[0,1]]=b[:,[1,0]]
                order=np.argsort(b[:,1]); lower=order[:9]; upper=order[12:]; singles=order[9:12]
                ref=b[lower].copy(); ref[:,1]*=-1
                cost=np.sum((b[upper,None,:]-ref[None,:,:])**2,axis=2)
                u,v=linear_sum_assignment(cost)
                g=np.zeros((12,3)); g[:3]=b[singles]; g[:3,1]=0; g[3:]=(b[upper[u]]+ref[v])/2
                err=float(np.sum(cost[u,v])+np.sum(b[singles,1]**2))
                projected.append((err,pack(g)))
            projected.sort(key=lambda p:p[0])
        solve(starts[0],int(params['iterations']),.23*budget_s)
        if projected:
            solve(projected[0][1],int(params['iterations']),.16*budget_s)
        for z in starts[1:]:
            tick()
            if time.monotonic()>t0+.53*budget_s:
                break
            solve(z,int(params['probe_iterations']),.10*budget_s)
        attempt=0
        while time.monotonic()<deadline:
            tick()
            pool.sort(key=lambda item:item[0],reverse=True)
            z=pool[0][1].copy()
            if attempt:
                z[:12]+=rng.normal(0,float(params['perturbation']),12)
                z[12:21]=np.maximum(.005,z[12:21]+rng.normal(0,float(params['perturbation']),9))
            solve(z,int(params['iterations']),.12*budget_s)
            attempt+=1
    except (Halt,ValueError,FloatingPointError):
        pass
    return best
