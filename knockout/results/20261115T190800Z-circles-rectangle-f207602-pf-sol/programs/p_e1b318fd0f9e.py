import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {'iterations': (110, 60, 180), 'initial_jitter': (0.014, 0.001, 0.06), 'restart': (0.035, 0.005, 0.12)}


def construct(params, seed, budget_s, init=None):
    deadline = time.monotonic() + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    best = init
    score_best = -float('inf')
    champion = None

    class Stop(Exception):
        pass

    def check():
        if time.monotonic() >= deadline:
            raise Stop()

    def perimeter(a):
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        return (max(t+s for t,s in zip(x,r))-min(t-s for t,s in zip(x,r))
                +max(t+s for t,s in zip(y,r))-min(t-s for t,s in zip(y,r)))

    def exact(a):
        if a.shape != (21,3) or not np.isfinite(a).all() or np.min(a[:,2]) <= 0:
            return -float('inf')
        if perimeter(a) > 2:
            return -float('inf')
        d = a[ii,:2]-a[jj,:2]
        if np.any(np.sqrt(d[:,0]**2+d[:,1]**2) < a[ii,2]+a[jj,2]):
            return -float('inf')
        return float(np.sum(a[:,2]))

    if init is not None:
        try:
            a = np.asarray(init['circles'], float) if 'circles' in init else np.column_stack((init['centers'],init['radii']))
            score_best = exact(a)
        except (ValueError, TypeError, KeyError):
            pass

    def retain(model,z):
        nonlocal best, score_best, champion
        X,Y,R = model[:3]
        a = np.column_stack((X@z,Y@z,R@z))
        if not np.isfinite(a).all() or np.min(a[:,2]) <= 0:
            return
        a[:,:2] -= np.mean(a[:,:2],axis=0)
        d = a[ii,:2]-a[jj,:2]
        ratio = np.min(np.sqrt(d[:,0]**2+d[:,1]**2)/(a[ii,2]+a[jj,2]))
        if ratio <= 0:
            return
        a[:,2] *= min(1.,float(ratio))*(1-3e-14)
        p = perimeter(a)
        if p <= 0:
            return
        a *= float(Fraction(2)/p)*(1-3e-14)
        s = exact(a)
        if s > score_best:
            score_best = s
            best = {'centers':a[:,:2].tolist(),'radii':a[:,2].tolist()}
            champion = (model,z.copy())

    def make(counts):
        check()
        nv = 2*counts[0]+3*sum(counts[1:])+1
        X = np.zeros((21,nv)); Y = np.zeros_like(X); R = np.zeros_like(X)
        z = np.zeros(nv)
        mirror = np.arange(21)
        unique=[]; xc=[]; yc=[]; rc=[]
        row=0; col=0
        for level,count in enumerate(counts):
            check()
            for k in range(count):
                check()
                x=(k-(count-1)/2)*.20
                if level == 0 and count == 3:
                    x *= 1.45
                unique.append(row); xc.append(col)
                if level == 0:
                    X[row,col]=1; R[row,col+1]=1
                    z[col:col+2]=[x,.095]
                    rc.append(col+1); row+=1; col+=2
                else:
                    X[row:row+2,col]=1
                    Y[row,col+1]=1; Y[row+1,col+1]=-1
                    R[row:row+2,col+2]=1
                    z[col:col+3]=[x,level*.177,.095]
                    mirror[row],mirror[row+1]=row+1,row
                    yc.append(col+1); rc.append(col+2)
                    row+=2; col+=3
        a=np.column_stack((X@z,Y@z,R@z))
        lo=np.min(a[:,:2]-a[:,2,None],axis=0)
        hi=np.max(a[:,:2]+a[:,2,None],axis=0)
        scale=2/np.sum(hi-lo)
        z*=scale
        z[xc]-=lo[0]*scale
        z[-1]=(hi[0]-lo[0])*scale
        pairs=[]
        for i,j in zip(ii,jj):
            check()
            if (int(i),int(j)) <= tuple(sorted((int(mirror[i]),int(mirror[j])))):
                pairs.append((i,j))
        pairs=np.asarray(pairs)
        DX=X[pairs[:,0]]-X[pairs[:,1]]
        DY=Y[pairs[:,0]]-Y[pairs[:,1]]
        RS=R[pairs[:,0]]+R[pairs[:,1]]
        u=np.asarray(unique); m=len(u)
        L=np.concatenate((X[u]-R[u],-X[u]-R[u],-Y[u]-R[u]),axis=0)
        L[m:2*m,-1]=1
        L[2*m:,-1]=-.5
        c=np.zeros(3*m); c[2*m:]=1
        bounds=[(None,None)]*nv
        for k in rc:
            check()
            bounds[k]=(1e-7,.5)
        for k in yc:
            check()
            bounds[k]=(0.,1.)
        bounds[-1]=(.05,1.95)
        grad=-np.sum(R,axis=0)
        return (X,Y,R,DX,DY,RS,L,c,bounds,grad,np.array(xc),np.array(yc)),z

    def solve(model,z,nit):
        check()
        X,Y,R,DX,DY,RS,L,c,bounds,grad=model[:10]
        cache=[None,None,None]
        latest=[z.copy()]
        counter=[0]
        def calc(t):
            check()
            if cache[0] is None or not np.array_equal(t,cache[0]):
                dx=DX@t; dy=DY@t
                d=np.sqrt(dx*dx+dy*dy)
                den=np.maximum(d,1e-16)
                cache[0]=t.copy()
                cache[1]=np.concatenate((d-RS@t,L@t+c))
                cache[2]=np.vstack((DX*(dx/den)[:,None]+DY*(dy/den)[:,None]-RS,L))
            return cache[1],cache[2]
        def callback(t):
            check()
            latest[0]=t.copy()
            counter[0]+=1
            if counter[0]%25 == 0:
                retain(model,t)
        def objective(t):
            check()
            return float(grad@t)
        try:
            res=minimize(objective,z,jac=lambda t:grad,method='SLSQP',bounds=bounds,
                         constraints={'type':'ineq','fun':lambda t:calc(t)[0],'jac':lambda t:calc(t)[1]},
                         callback=callback,options={'maxiter':nit,'ftol':5e-13,'disp':False})
            latest[0]=res.x
        except Stop:
            retain(model,latest[0])
            raise
        retain(model,latest[0])
        return latest[0]

    try:
        model,z=make((3,5,4))
        retain(model,z)
        # The horizontal reflection remains exact; the other reflection is deliberately removed.
        z[model[10]]+=rng.normal(0,params['initial_jitter'],len(model[10]))
        z[model[11]]+=rng.normal(0,params['initial_jitter']*.4,len(model[11]))
        solve(model,z,params['iterations'])
        for attempt in range(20):
            check()
            if attempt in (2,5,9):
                family={2:(3,4,5),5:(5,4,4),9:(5,5,3)}[attempt]
                model,z=make(family)
                z[model[10]]+=rng.normal(0,params['initial_jitter'],len(model[10]))
            elif champion is not None:
                model,z0=champion
                z=z0.copy()
                xc,yc=model[10:12]
                amp=params['restart']*(1.7 if attempt%3==2 else 1.)
                chosen=rng.choice(len(xc),min(5,len(xc)),replace=False)
                z[xc[chosen]]+=rng.normal(0,amp,len(chosen))
                z[yc]=np.maximum(1e-5,z[yc]+rng.normal(0,amp*.5,len(yc)))
            else:
                model,z=make((3,5,4))
            solve(model,z,params['iterations'])
    except (Stop,ValueError,np.linalg.LinAlgError):
        pass
    if score_best == -float('inf'):
        a=np.array([[(i%7)*.2,(i//7)*.2,.0999999999999] for i in range(21)])
        if exact(a)>score_best:
            return {'circles':a.tolist()}
    return best
