import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {'iterations': (130, 40, 250), 'jitter': (0.013, 0.001, 0.05), 'safety': (4e-14, 1e-14, 1e-10, 'log')}

class Halt(Exception):
    pass

class SliceEnd(Exception):
    pass

def canonical(obj):
    if obj is None:
        return None
    if 'circles' in obj:
        return obj
    rows=[]
    for x,y,r,m in obj['orbits']:
        rows.append([x,y,r])
        if m==2:
            rows.append([x,-y,r])
    return {'circles':rows}

def construct(params, seed, budget_s, init=None):
    begin=time.monotonic()
    deadline=begin+0.70*budget_s
    rng=np.random.default_rng(seed)
    ai,aj=np.triu_indices(21,1)
    best=init
    bestscore=-np.inf
    safety=float(params['safety'])

    def tick():
        if time.monotonic()>=deadline:
            raise Halt()

    def exact(a):
        if a.shape!=(21,3) or not np.isfinite(a).all() or np.any(a[:,2]<=0):
            return -np.inf
        d=a[ai,:2]-a[aj,:2]
        if np.any(np.sqrt(d[:,0]**2+d[:,1]**2)<a[ai,2]+a[aj,2]):
            return -np.inf
        x=[Fraction(float(v)) for v in a[:,0]]
        y=[Fraction(float(v)) for v in a[:,1]]
        r=[Fraction(float(v)) for v in a[:,2]]
        w=max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        h=max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        return float(np.sum(a[:,2])) if w+h<=2 else -np.inf

    def retain(a):
        nonlocal best,bestscore
        a=np.array(a,dtype=float,copy=True)
        if a.shape!=(21,3) or not np.isfinite(a).all() or np.any(a[:,2]<=0):
            return
        d=a[ai,:2]-a[aj,:2]
        rat=float(np.min(np.sqrt(d[:,0]**2+d[:,1]**2)/(a[ai,2]+a[aj,2])))
        if rat<=0:
            return
        a[:,2]*=min(1.,rat)*(1.-safety)
        lo=np.min(a[:,:2]-a[:,2,None],axis=0)
        hi=np.max(a[:,:2]+a[:,2,None],axis=0)
        wh=float(np.sum(hi-lo))
        if wh<=0:
            return
        a[:,:2]-=(lo+hi)/2
        a*=(2./wh)*(1.-safety)
        if float(np.sum(a[:,2]))>bestscore:
            val=exact(a)
            if val>bestscore:
                bestscore=val
                best={'circles':a.tolist()}

    initial=None
    try:
        initial=np.asarray(canonical(init)['circles'],float)
        bestscore=exact(initial)
        retain(initial)
    except (TypeError,ValueError,KeyError,OverflowError):
        initial=None
    fallback=np.array([[.2*(k%7-3),.2*(k//7-1),.0999999999999] for k in range(21)])
    retain(fallback)

    try:
        tick()
        # Both representations use a centered rectangle of width w and height 2-w.
        def model(symmetric):
            tick()
            if symmetric:
                dim=34
                ids=np.r_[np.arange(3),np.repeat(np.arange(3,12),2)]
                signs=np.r_[np.zeros(3),np.tile([1.,-1.],9)]
                X=np.zeros((21,dim)); Y=X.copy(); R=X.copy()
                X[np.arange(21),ids]=1
                R[np.arange(21),21+ids]=1
                ix=np.flatnonzero(ids>=3)
                Y[ix,9+ids[ix]]=signs[ix]
                seen=set(); edges=[]
                for i in range(21):
                    tick()
                    for j in range(i):
                        tick()
                        key=(min(ids[i],ids[j]),max(ids[i],ids[j]),signs[i]*signs[j])
                        if key not in seen:
                            seen.add(key); edges.append((i,j))
                pi,pj=np.asarray(edges).T
                bounds=[(-1.,1.)]*12+[(0.,1.)]*9+[(1e-7,.7)]*12+[(.1,1.9)]
            else:
                dim=64
                X=np.zeros((21,dim)); Y=X.copy(); R=X.copy()
                X[np.arange(21),np.arange(21)]=1
                Y[np.arange(21),21+np.arange(21)]=1
                R[np.arange(21),42+np.arange(21)]=1
                pi,pj=ai,aj
                bounds=[(-1.,1.)]*42+[(1e-7,.7)]*21+[(.1,1.9)]
            DX=X[pi]-X[pj]; DY=Y[pi]-Y[pj]; RS=R[pi]+R[pj]
            W=np.vstack((X-R,-X-R,Y-R,-Y-R))
            W[:42,-1]+=.5
            W[42:,-1]-=.5
            off=np.r_[np.zeros(42),np.ones(42)]
            grad=-np.sum(R,axis=0)
            return X,Y,R,DX,DY,RS,W,off,grad,bounds

        sym=model(True)
        full=model(False)
        pool=[]

        def expand(z,m):
            return np.column_stack((m[0]@z,m[1]@z,m[2]@z))

        def solve(z,m,seconds):
            tick()
            local=min(deadline-.007,time.monotonic()+seconds)
            if local<=time.monotonic():
                return z
            X,Y,R,DX,DY,RS,W,off,grad,bounds=m
            z=np.asarray(z,float)
            last=[z.copy()]; cache=[None,None,None]; steps=[0]
            def calc(v):
                tick()
                if time.monotonic()>=local:
                    raise SliceEnd()
                if cache[0] is None or not np.array_equal(v,cache[0]):
                    dx=DX@v; dy=DY@v
                    dist=np.maximum(np.sqrt(dx*dx+dy*dy),1e-25)
                    cache[0]=v.copy()
                    cache[1]=np.r_[dist-RS@v,W@v+off]
                    cache[2]=np.vstack(((dx/dist)[:,None]*DX+(dy/dist)[:,None]*DY-RS,W))
                return cache[1],cache[2]
            def cb(v):
                last[0]=v.copy(); steps[0]+=1
                tick()
                if steps[0]%12==0:
                    retain(expand(v,m))
                if time.monotonic()>=local:
                    raise SliceEnd()
            try:
                result=minimize(lambda v:float(grad@v),z,jac=lambda v:grad,
                    method='SLSQP',bounds=bounds,
                    constraints={'type':'ineq','fun':lambda v:calc(v)[0],'jac':lambda v:calc(v)[1]},
                    callback=cb,options={'maxiter':int(params['iterations']),'ftol':8e-14,'disp':False})
                last[0]=result.x
            except SliceEnd:
                pass
            retain(expand(last[0],m))
            if m is sym:
                v=last[0]
                dx=DX@v; dy=DY@v
                violation=max(0.,-float(np.min(np.r_[np.sqrt(dx*dx+dy*dy)-RS@v,W@v+off])))
                pool.append((float(-grad@v)-100*violation,v.copy()))
            return last[0]

        def centered(a):
            a=np.array(a,float,copy=True)
            lo=np.min(a[:,:2]-a[:,2,None],axis=0)
            hi=np.max(a[:,:2]+a[:,2,None],axis=0)
            a[:,:2]-=(lo+hi)/2
            a*=2/float(np.sum(hi-lo))
            return a

        def packfull(a):
            a=centered(a)
            return np.r_[a[:,0],a[:,1],a[:,2],np.ptp(a[:,0]+0.)+0.] if False else np.r_[a[:,0],a[:,1],a[:,2],np.max(a[:,0]+a[:,2])-np.min(a[:,0]-a[:,2])]

        def packorbit(g):
            z=np.zeros(34)
            z[:12]=g[:,0]; z[12:21]=np.abs(g[3:,1]); z[21:33]=g[:,2]
            z[33]=np.max(g[:,0]+g[:,2])-np.min(g[:,0]-g[:,2])
            return z

        projections=[]
        if initial is not None and initial.shape==(21,3) and np.isfinite(initial).all():
            a=centered(initial)
            for swap in (False,True):
                tick()
                b=a.copy()
                if swap:
                    b[:,[0,1]]=b[:,[1,0]]
                order=np.argsort(b[:,1]); low=order[:9]; high=order[12:]; singles=order[9:12]
                ref=b[low].copy(); ref[:,1]*=-1
                cost=np.sum((b[high,None,:]-ref[None,:,:])**2,axis=2)
                u,v=linear_sum_assignment(cost)
                g=np.zeros((12,3)); g[:3]=b[singles]; g[:3,1]=0
                g[3:]=(b[high[u]]+ref[v])/2
                err=float(np.sum(cost[u,v])+np.sum(b[singles,1]**2))
                projections.append((err,packorbit(g)))
            projections.sort(key=lambda p:p[0])
            solve(packfull(initial),full,.12*budget_s)
            if projections:
                solve(projections[0][1],sym,.19*budget_s)

        # A compact 5-4-3-4-5 triangular family complements the incumbent topology.
        g=np.zeros((12,3)); g[:,2]=.095
        g[:3,0]=[-.2,0,.2]
        g[3:7,0]=[-.3,-.1,.1,.3]; g[3:7,1]=np.sqrt(3)*.1
        g[7:,0]=[-.4,-.2,0,.2,.4]; g[7:,1]=np.sqrt(3)*.2
        solve(packorbit(g),sym,.23*budget_s)
        attempt=0
        while time.monotonic()<deadline-.025:
            tick()
            remaining=deadline-time.monotonic()
            if attempt==0 and best is not None:
                solve(packfull(np.asarray(canonical(best)['circles'])),full,min(.09*budget_s,remaining-.012))
            elif pool:
                pool.sort(key=lambda item:item[0],reverse=True)
                z=pool[0][1].copy()
                if attempt>1:
                    z[:12]+=rng.normal(0,float(params['jitter']),12)
                    z[12:21]=np.clip(z[12:21]+rng.normal(0,float(params['jitter']),9),.002,.85)
                solve(z,sym,min(.12*budget_s,remaining-.012))
            else:
                break
            attempt+=1
    except (Halt,ValueError,FloatingPointError):
        pass
    return best
