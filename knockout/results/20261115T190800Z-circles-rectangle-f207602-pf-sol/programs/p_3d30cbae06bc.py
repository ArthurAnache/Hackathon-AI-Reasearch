import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {
    'stage_fraction': (0.19, 0.10, 0.24),
    'ftol': (2e-12, 1e-14, 1e-8, 'log'),
    'safety': (3e-14, 1e-15, 1e-10, 'log')
}


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    best = init
    best_score = -float('inf')

    def exact(data):
        try:
            a = np.asarray(data['circles'], dtype=float)
            if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
                return -float('inf')
            xx = [Fraction(float(v)) for v in a[:, 0]]
            yy = [Fraction(float(v)) for v in a[:, 1]]
            rr = [Fraction(float(v)) for v in a[:, 2]]
            p = max(x+r for x,r in zip(xx,rr))-min(x-r for x,r in zip(xx,rr))
            p += max(y+r for y,r in zip(yy,rr))-min(y-r for y,r in zip(yy,rr))
            if p > 2:
                return -float('inf')
            for i in range(21):
                if time.monotonic() > deadline + 0.06:
                    return -float('inf')
                for j in range(i):
                    d = np.sqrt((a[i,0]-a[j,0])**2 + (a[i,1]-a[j,1])**2)
                    if d < a[i,2]+a[j,2]:
                        return -float('inf')
            return float(np.sum(a[:,2]))
        except (ValueError, KeyError, TypeError):
            return -float('inf')

    if init is not None:
        best_score = exact(init)

    # z = twelve x coordinates, nine positive y coordinates, twelve radii, width.
    n = 34
    ex = np.zeros((12,n)); ey = np.zeros((12,n)); er = np.zeros((12,n))
    ex[np.arange(12),np.arange(12)] = 1
    ey[np.arange(3,12),np.arange(12,21)] = 1
    er[np.arange(12),np.arange(21,33)] = 1
    ew = np.zeros(n); ew[33] = 1
    ia=[]; ib=[]; signs=[]
    for i in range(12):
        if time.monotonic() > deadline:
            break
        if i >= 3:
            ia.append(i); ib.append(i); signs.append(-1)
        for j in range(i):
            ia.append(i); ib.append(j); signs.append(1)
            if j >= 3:
                ia.append(i); ib.append(j); signs.append(-1)
    ia=np.array(ia); ib=np.array(ib); signs=np.array(signs)
    dxmat=ex[ia]-ex[ib]
    dymat=ey[ia]-signs[:,None]*ey[ib]
    rmat=er[ia]+er[ib]
    wall = np.concatenate((0.5*ew-ex-er, 0.5*ew+ex-er,
                           -0.5*ew-ey-er),axis=0)
    offset=np.r_[np.zeros(24),np.ones(12)]
    weights=np.r_[np.ones(3),np.full(9,2.)]
    objective=np.zeros(n); objective[21:33]=-weights
    bounds=[(-1.,1.)]*12+[(0.,1.)]*9+[(1e-7,0.5)]*12+[(0.2,1.8)]
    cached_z=None; cached_c=None; cached_j=None

    def constraints(z):
        nonlocal cached_z,cached_c,cached_j
        if cached_z is None or not np.array_equal(z,cached_z):
            dx=dxmat@z; dy=dymat@z
            d=np.maximum(np.sqrt(dx*dx+dy*dy),1e-15)
            cached_c=np.r_[d-rmat@z, wall@z+offset]
            cached_j=np.concatenate(((dx/d)[:,None]*dxmat+(dy/d)[:,None]*dymat-rmat,wall))
            cached_z=z.copy()
        return cached_c

    def jac(z):
        constraints(z)
        return cached_j

    def expand(z):
        x=z[:12]; y=np.r_[np.zeros(3),z[12:21]]; r=z[21:33]
        out=[]
        for i in range(12):
            if time.monotonic() > deadline + 0.04:
                return None
            out.append([x[i],y[i],r[i]])
            if i>=3:
                out.append([x[i],-y[i],r[i]])
        return np.array(out)

    def consider(z):
        nonlocal best,best_score
        if time.monotonic() > deadline + 0.02:
            return
        a=expand(z)
        if a is None or np.any(a[:,2]<=0) or not np.isfinite(a).all():
            return
        u,v=np.triu_indices(21,1)
        dist=np.hypot(a[u,0]-a[v,0],a[u,1]-a[v,1])
        f=min(1.,float(np.min(dist/(a[u,2]+a[v,2]))))
        a[:,2] *= max(1e-12, f*(1-params['safety']))
        lo=np.min(a[:,:2]-a[:,2,None],axis=0)
        hi=np.max(a[:,:2]+a[:,2,None],axis=0)
        a[:,:2]-=(lo+hi)/2
        a*= (2./np.sum(hi-lo))*(1-params['safety'])
        obj={'circles':a.tolist()}
        score=exact(obj)
        if score>best_score:
            best_score=score; best=obj

    def projection(a,swap):
        a=a.copy()
        if swap:
            a[:,[0,1]]=a[:,[1,0]]
        lo=np.min(a[:,:2]-a[:,2,None],axis=0)
        hi=np.max(a[:,:2]+a[:,2,None],axis=0)
        a[:,:2]-=(lo+hi)/2
        order=np.argsort(a[:,1]); low=order[:9]; axis=order[9:12]; high=order[12:]
        cost=(a[low,0,None]-a[high,0][None,:])**2
        cost+=(a[low,1,None]+a[high,1][None,:])**2
        cost+=0.3*(a[low,2,None]-a[high,2][None,:])**2
        l,h=linear_sum_assignment(cost)
        lower=a[low[l]]; upper=a[high[h]]
        z=np.zeros(n)
        z[:3]=a[axis,0]
        z[3:12]=(lower[:,0]+upper[:,0])/2
        z[12:21]=(upper[:,1]-lower[:,1])/2
        z[21:24]=a[axis,2]
        z[24:33]=(lower[:,2]+upper[:,2])/2
        width=np.max(z[:12]+z[21:33])-np.min(z[:12]-z[21:33])
        height=2*np.max(np.r_[np.zeros(3),z[12:21]]+z[21:33])
        scale=2/(width+height)
        z[:33]*=scale; z[33]=np.clip(width*scale,0.2,1.8)
        return z

    seeds=[]
    if init is not None:
        try:
            a=np.asarray(init['circles'],dtype=float)
            if a.shape==(21,3) and np.isfinite(a).all():
                for swap in (False,True):
                    if time.monotonic()>deadline:
                        break
                    seeds.append(projection(a,swap))
        except (ValueError,KeyError):
            pass
    z=np.zeros(n)
    z[:3]=[-2,0,2]
    z[3:7]=[-3,-1,1,3]
    z[7:12]=[-4,-2,0,2,4]
    z[12:16]=np.sqrt(3)
    z[16:21]=2*np.sqrt(3)
    z[21:33]=1
    scale=2/(12+4*np.sqrt(3))
    z[:33]*=scale; z[33]=10*scale
    seeds.append(z)

    class Timeout(Exception):
        pass

    endpoints=[]
    def solve(z,until,iterations):
        last=z.copy()
        def cb(v):
            nonlocal last
            last=v.copy()
            if time.monotonic()>until:
                raise Timeout()
        try:
            res=minimize(lambda v:float(objective@v),z,jac=lambda v:objective,
                         bounds=bounds,constraints={'type':'ineq','fun':constraints,'jac':jac},
                         method='SLSQP',callback=cb,
                         options={'maxiter':iterations,'ftol':params['ftol'],'disp':False})
            last=res.x
        except Timeout:
            pass
        consider(last)
        cv=constraints(last)
        penalty=max(0.,-float(np.min(cv)))
        endpoints.append((float(-objective@last)-30*penalty,last.copy()))
        return last

    for z in seeds:
        if time.monotonic()>deadline-0.10:
            break
        until=min(deadline-0.13,time.monotonic()+params['stage_fraction']*budget_s)
        solve(z,until,90)
    if endpoints and time.monotonic()<deadline-0.05:
        endpoints.sort(key=lambda p:p[0],reverse=True)
        solve(endpoints[0][1],deadline-0.025,220)
    if best_score == -float('inf'):
        # Universally feasible fallback, used only for missing or malformed starts.
        r=1/22
        a=[[(-10+i)*2*r,0.,r*(1-1e-12)] for i in range(21)]
        return {'circles':a}
    return best
