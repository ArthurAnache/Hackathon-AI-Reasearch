import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {'iterations': (85, 30, 160), 'release': (0.012, 0.002, 0.05), 'contacts': (4, 3, 6)}

class StopSearch(Exception):
    pass

def canonical(obj):
    if 'circles' in obj:
        return obj
    rows = []
    for x,y,r,m in obj['orbits']:
        rows.append([x,y,r])
        if m == 2:
            rows.append([x,-y,r])
    return {'circles': rows}

def exact_score(obj):
    try:
        a = np.asarray(canonical(obj)['circles'], dtype=float)
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return -np.inf
        i,j = np.triu_indices(21,1)
        d = a[i,:2]-a[j,:2]
        if np.any(np.sqrt(d[:,0]**2+d[:,1]**2) < a[i,2]+a[j,2]):
            return -np.inf
        x = [Fraction(float(v)) for v in a[:,0]]
        y = [Fraction(float(v)) for v in a[:,1]]
        r = [Fraction(float(v)) for v in a[:,2]]
        perimeter = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        perimeter += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        if perimeter > 2:
            return -np.inf
        return float(np.sum(a[:,2]))
    except Exception:
        return -np.inf

def construct(params, seed, budget_s, init=None):
    deadline = time.monotonic()+0.69*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = exact_score(init) if init is not None else -np.inf
    def tick():
        if time.monotonic() >= deadline:
            raise StopSearch

    # A fully valid fallback independent of the input's geometry.
    grid = np.array([[2*(i%7),2*(i//7),1.] for i in range(21)],float)/10.000000000001
    fallback = {'circles':grid.tolist()}
    fs = exact_score(fallback)
    if fs > best_score:
        best,best_score = fallback,fs

    groups = np.r_[np.arange(3),np.repeat(np.arange(3,12),2)]
    signs = np.r_[np.zeros(3),np.tile([1.,-1.],9)]
    X = np.zeros((21,34)); Y = X.copy(); R = X.copy()
    X[np.arange(21),groups] = 1
    R[np.arange(21),21+groups] = 1
    mask = groups >= 3
    Y[np.where(mask)[0],9+groups[mask]] = signs[mask]
    pairs=[]; seen=set()
    for i in range(21):
        if time.monotonic() >= deadline:
            return best
        for j in range(i):
            tick()
            key=(min(groups[i],groups[j]),max(groups[i],groups[j]),signs[i]*signs[j])
            if key not in seen:
                seen.add(key); pairs.append((i,j))
    pi,pj=np.array(pairs).T
    DX=X[pi]-X[pj]; DY=Y[pi]-Y[pj]; RS=R[pi]+R[pj]
    walls=np.zeros((36,34)); offsets=np.zeros(36)
    for g in range(12):
        tick()
        walls[g,g]=1; walls[g,21+g]=-1; walls[g,33]=.5
        walls[12+g,g]=-1; walls[12+g,21+g]=-1; walls[12+g,33]=.5
        walls[24+g,21+g]=-1; walls[24+g,33]=-.5; offsets[24+g]=1
        if g>=3:
            walls[24+g,9+g]=-1
    weights=np.r_[np.ones(3),np.full(9,2.)]
    grad=np.zeros(34); grad[21:33]=-weights
    bounds=[(-1.,1.)]*12+[(0.,1.)]*9+[(1e-7,.5)]*12+[(.12,1.88)]
    ii,jj=np.triu_indices(21,1)
    release=np.zeros(len(pi))
    best_z=None

    def retain(v):
        nonlocal best,best_score,best_z
        tick()
        if not np.isfinite(v).all() or np.any(v[21:33]<=0):
            return
        a=np.column_stack((X@v,Y@v,R@v))
        delta=a[ii,:2]-a[jj,:2]
        ratio=float(np.min(np.sqrt(delta[:,0]**2+delta[:,1]**2)/(a[ii,2]+a[jj,2])))
        a[:,2]*=max(0.,min(1.,ratio))*(1.-2e-14)
        if np.any(a[:,2]<=0):
            return
        lower=np.min(a[:,:2]-a[:,2,None],axis=0)
        upper=np.max(a[:,:2]+a[:,2,None],axis=0)
        a[:,:2]-=(lower+upper)/2
        scale=min(1.,2./float(np.sum(upper-lower)))*(1.-2e-14)
        a*=scale
        if float(np.sum(a[:,2]))<=best_score:
            return
        obj={'circles':a.tolist()}
        s=exact_score(obj)
        if s>best_score:
            best,best_score,best_z=obj,s,v.copy()

    def fun(v):
        tick(); return float(grad@v)
    def jac(v):
        tick(); return grad
    def con(v):
        tick()
        dx=DX@v; dy=DY@v
        return np.r_[np.sqrt(dx*dx+dy*dy)-RS@v-release,walls@v+offsets]
    def cjac(v):
        tick()
        dx=DX@v; dy=DY@v
        d=np.maximum(np.sqrt(dx*dx+dy*dy),1e-24)
        return np.vstack(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RS,walls))
    def solve(v,n):
        tick()
        calls=0
        def callback(w):
            nonlocal calls
            tick(); calls+=1
            if calls%4==0:
                retain(w)
        res=minimize(fun,v,jac=jac,method='SLSQP',bounds=bounds,
                     constraints={'type':'ineq','fun':con,'jac':cjac},callback=callback,
                     options={'maxiter':n,'ftol':8e-13,'disp':False})
        retain(res.x)
        return res.x

    def row_seed(counts):
        tick()
        z=np.zeros(34)
        z[:3]=[-.28,0.,.28]
        z[21:24]=.145
        k=3
        nr=len(counts)
        for row,count in enumerate(counts):
            tick()
            spacing=.235 if count>=4 else .27
            xs=(np.arange(count)-(count-1)/2)*spacing
            for x in xs:
                tick()
                z[k]=x
                z[9+k]=(row+1)*(.205 if nr==2 else .215)
                z[21+k]=.105 if count>=5 else .117
                k+=1
        width=2*np.max(np.abs(z[:12])+z[21:33])
        height=2*np.max(np.r_[np.zeros(3),z[12:21]]+z[21:33])
        scale=2/(width+height)
        z[:33]*=scale
        z[33]=width*scale
        return z

    try:
        seeds=[]
        if init is not None and exact_score(init)>1.8:
            a=np.asarray(canonical(init)['circles'],float).copy()
            lo=np.min(a[:,:2]-a[:,2,None],axis=0)
            hi=np.max(a[:,:2]+a[:,2,None],axis=0)
            a[:,:2]-=(lo+hi)/2
            projections=[]
            for swap in (False,True):
                tick()
                b=a.copy()
                if swap:
                    b[:,[0,1]]=b[:,[1,0]]
                order=np.argsort(b[:,1]); low=order[:9]; singles=order[9:12]; up=order[12:]
                reflected=b[low].copy(); reflected[:,1]*=-1
                cost=np.sum((b[up,None,:]-reflected[None,:,:])**2,axis=2)
                p,q=linear_sum_assignment(cost)
                g=np.vstack((b[singles],(b[up[p]]+reflected[q])/2))
                error=np.sum(cost[p,q])+np.sum(g[:3,1]**2)
                z=np.zeros(34); z[:12]=g[:,0]; z[12:21]=np.abs(g[3:,1]); z[21:33]=g[:,2]
                z[33]=hi[int(swap)]-lo[int(swap)]
                projections.append((error,z))
            seeds.append(min(projections,key=lambda t:t[0])[1])
        seeds.extend([row_seed((5,4)),row_seed((4,5)),row_seed((3,3,3))])
        anchors=[]
        for z in seeds:
            tick()
            v=solve(z,int(params['iterations']))
            anchors.append(v.copy())
        if best_z is not None:
            anchor=best_z.copy()
        else:
            anchor=min(anchors,key=lambda v:float(grad@v)).copy()
        while time.monotonic()<deadline:
            tick()
            release[:]=0
            anchor=solve(anchor,35)
            dx=DX@anchor; dy=DY@anchor
            gaps=np.sqrt(dx*dx+dy*dy)-RS@anchor
            near=np.flatnonzero(gaps<.004)
            if len(near)<3:
                break
            chosen=rng.choice(near,min(len(near),int(params['contacts'])),replace=False)
            release[chosen]=float(params['release'])*rng.uniform(.7,1.4,len(chosen))
            displaced=solve(anchor.copy(),20)
            release[:]=0
            recovered=solve(displaced,55)
            if np.min(con(recovered))>=-1e-8 and grad@recovered<grad@anchor:
                anchor=recovered.copy()
            elif best_z is not None:
                anchor=best_z.copy()
    except (StopSearch,ValueError,FloatingPointError):
        pass
    return best
