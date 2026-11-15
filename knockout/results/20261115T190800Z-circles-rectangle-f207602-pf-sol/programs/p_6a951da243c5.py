import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {'iterations': (125, 40, 220), 'slice': (0.27, 0.12, 0.4), 'release': (0.008, 0.002, 0.025)}

class Stop(Exception):
    pass

def canonical(obj):
    if obj is None or 'circles' in obj:
        return obj
    out = []
    for x,y,r,m in obj['orbits']:
        out.append([x,y,r])
        if m == 2:
            out.append([x,-y,r])
    return {'circles': out}

def score(obj):
    try:
        a = np.asarray(canonical(obj)['circles'], dtype=np.float64)
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return -float('inf')
        x,y,r = [[Fraction(float(v)) for v in a[:,k]] for k in range(3)]
        w = max(u+t for u,t in zip(x,r))-min(u-t for u,t in zip(x,r))
        h = max(u+t for u,t in zip(y,r))-min(u-t for u,t in zip(y,r))
        if w+h > 2:
            return -float('inf')
        i,j = np.triu_indices(21,1)
        if np.any(np.sqrt((a[i,0]-a[j,0])**2+(a[i,1]-a[j,1])**2) < a[i,2]+a[j,2]):
            return -float('inf')
        return float(np.sum(a[:,2]))
    except Exception:
        return -float('inf')

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + .70*budget_s
    rng = np.random.default_rng(seed)
    best = init
    bestscore = score(init)
    ai,aj = np.triu_indices(21,1)
    anchor = None
    anchor_score = -float('inf')

    def retain(a,z=None):
        nonlocal best,bestscore,anchor,anchor_score
        if time.monotonic() >= deadline+.015:
            return
        if not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        a = a.copy()
        d = np.sqrt((a[ai,0]-a[aj,0])**2+(a[ai,1]-a[aj,1])**2)
        f = min(1.,float(np.min(d/(a[ai,2]+a[aj,2]))))
        if f <= 0:
            return
        a[:,2] *= f*(1.-8e-15)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        a[:,:2] -= (hi+lo)/2
        a *= (2./float(np.sum(hi-lo)))*(1.-8e-15)
        obj = {'circles':a.tolist()}
        s = score(obj)
        if s > bestscore:
            best,bestscore = obj,s
        if z is not None and s > anchor_score:
            anchor,anchor_score = z.copy(),s

    retain(np.array([[2*(k%7),2*(k//7),1.] for k in range(21)],float)/10.000000001)
    groups = np.r_[np.arange(3),np.repeat(np.arange(3,12),2)]
    signs = np.r_[np.zeros(3),np.tile([1.,-1.],9)]
    pi=[]; pj=[]; sg=[]
    for i in range(12):
        if time.monotonic() >= deadline:
            return best
        for j in range(i+1,12):
            if time.monotonic() >= deadline:
                return best
            pi.append(i); pj.append(j); sg.append(1.)
            if i >= 3:
                pi.append(i); pj.append(j); sg.append(-1.)
    pi=np.array(pi); pj=np.array(pj); sg=np.array(sg)
    rows=np.arange(len(pi)); ym=np.maximum(np.arange(12)-3,0)+12
    L=np.zeros((45,34)); c=np.zeros(45); q=np.arange(12)
    L[q,q]=1.; L[q,21+q]=-1.
    L[12+q,q]=-1.; L[12+q,21+q]=-1.; L[12+q,33]=1.
    L[24+q,21+q]=-1.; L[24+q,33]=-.5; c[24+q]=1.
    L[27+np.arange(9),12+np.arange(9)]=-1.
    L[36+np.arange(9),12+np.arange(9)]=1.
    L[36+np.arange(9),24+np.arange(9)]=-1.
    grad=np.zeros(34); grad[21:24]=-1.; grad[24:33]=-2.
    bounds=[(0.,2.)]*12+[(0.,1.)]*9+[(1e-6,.5)]*12+[(.15,1.85)]
    release=np.zeros(len(pi))
    cache=[None,None,None]

    def calc(v):
        if time.monotonic() >= deadline:
            raise Stop()
        if cache[0] is not None and np.array_equal(v,cache[0]):
            return cache[1],cache[2]
        y=np.zeros(12); y[3:]=v[12:21]
        dx=v[pi]-v[pj]; dy=y[pi]-sg*y[pj]
        d=np.maximum(np.sqrt(dx*dx+dy*dy),1e-30)
        J=np.zeros((len(pi),34))
        J[rows,pi]=dx/d; J[rows,pj]=-dx/d
        mask=pi>=3; J[rows[mask],ym[pi[mask]]]=dy[mask]/d[mask]
        mask=pj>=3; J[rows[mask],ym[pj[mask]]]=-sg[mask]*dy[mask]/d[mask]
        J[rows,21+pi]=-1.; J[rows,21+pj]=-1.
        g=np.r_[d-v[21+pi]-v[21+pj],L@v+c]
        cache[:]=[v.copy(),g,np.vstack((J,L))]
        return cache[1],cache[2]

    def expand(v):
        y=np.r_[np.zeros(3),v[12:21]]
        return np.column_stack((v[groups],y[groups]*signs,v[21+groups]))

    def con(v):
        g=calc(v)[0].copy(); g[:len(pi)]-=release
        return g

    def solve(v,seconds,iters=None):
        end=min(deadline-.012,time.monotonic()+seconds)
        last=[v.copy()]
        def cb(z):
            last[0]=z.copy()
            if time.monotonic() >= end:
                raise Stop()
        try:
            res=minimize(lambda z:float(grad@z),v,jac=lambda z:grad,
                bounds=bounds,method='SLSQP',
                constraints={'type':'ineq','fun':con,'jac':lambda z:calc(z)[1]},
                callback=cb,options={'maxiter':int(iters or params['iterations']),'ftol':5e-14,'disp':False})
            last[0]=res.x
        except (Stop,ValueError,FloatingPointError):
            pass
        retain(expand(last[0]),last[0])
        return last[0]

    projections=[]
    try:
        a=np.asarray(canonical(init)['circles'],float).copy()
        lo=np.min(a[:,:2]-a[:,2,None],axis=0); hi=np.max(a[:,:2]+a[:,2,None],axis=0)
        a[:,:2]-=(lo+hi)/2
        for swap in (False,True):
            if time.monotonic() >= deadline:
                return best
            b=a[:,[1,0,2]].copy() if swap else a.copy()
            order=np.argsort(b[:,1]); lower,axis,upper=order[:9],order[9:12],order[12:]
            reflected=b[lower].copy(); reflected[:,1]*=-1
            costs=np.sum((b[upper,None,:]-reflected[None,:,:])**2,axis=2)
            u,v=linear_sum_assignment(costs)
            g=np.vstack((b[axis],(b[upper[u]]+reflected[v])/2))
            z=np.zeros(34); z[33]=hi[int(swap)]-lo[int(swap)]
            z[:12]=g[:,0]+z[33]/2; z[12:21]=np.abs(g[3:,1]); z[21:33]=g[:,2]
            err=float(np.sum(costs[u,v])+np.sum(b[axis,1]**2))
            projections.append((err,z))
        projections.sort(key=lambda t:t[0])
    except (TypeError,KeyError,ValueError,IndexError):
        pass

    if projections and time.monotonic()<deadline-.04:
        solve(projections[0][1],budget_s*params['slice'])

    for counts in ([4,5],[5,4],[3,3,3]):
        if time.monotonic() >= deadline-.06:
            break
        n=len(counts); maximum=max(3,max(counts))
        wu=2*maximum+(.6 if maximum==3 else 0.)
        r=2/(wu+2+2*n*np.sqrt(3.)); w=wu*r
        z=np.zeros(34); z[33]=w; z[:3]=w/2+2*r*(np.arange(3)-1)
        cursor=3
        for row,count in enumerate(counts):
            if time.monotonic() >= deadline:
                return best
            shift=((.5 if row%2==0 else -.5)*r) if maximum==3 else 0.
            z[cursor:cursor+count]=w/2+2*r*(np.arange(count)-(count-1)/2)+shift
            z[12+cursor-3:12+cursor-3+count]=(row+1)*np.sqrt(3.)*r
            cursor+=count
        z[21:33]=r*.94
        solve(z,min(.30*budget_s,max(.08,(deadline-time.monotonic())*.55)))

    attempt=0
    while anchor is not None and time.monotonic()<deadline-.055:
        z=anchor.copy(); release[:]=0.
        if attempt>0:
            try:
                near=np.flatnonzero(calc(z)[0][:len(pi)]<.002)
                if len(near)>=3:
                    chosen=rng.choice(near,min(4,len(near)),replace=False)
                    release[chosen]=params['release']
                    z=solve(z,min(.09*budget_s,(deadline-time.monotonic())*.3),22)
                    release[:]=0.
            except Stop:
                break
        solve(z,deadline-time.monotonic()-.015)
        attempt+=1
    return best
