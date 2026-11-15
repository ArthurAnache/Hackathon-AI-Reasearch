import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"iterations": (115, 40, 220), "slack": (2e-14, 1e-15, 1e-11), "restart_noise": (0.016, 0.001, 0.06)}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    best = init
    best_score = -float('inf')
    best_array = None

    def exact(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        if max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r)) + max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)) > 2:
            return -float('inf')
        d = np.sqrt((a[ii,0]-a[jj,0])**2+(a[ii,1]-a[jj,1])**2)
        if np.any(d < a[ii,2]+a[jj,2]):
            return -float('inf')
        return float(np.sum(a[:,2]))

    if init is not None:
        try:
            ar = np.asarray(init['circles'], dtype=float)
            best_score = exact(ar)
            if np.isfinite(best_score):
                best_array = ar.copy()
        except (ValueError, KeyError, TypeError):
            pass

    def retain(a):
        nonlocal best, best_score, best_array
        if time.monotonic() >= deadline:
            raise StopSearch()
        a = np.array(a, dtype=float, copy=True)
        if not np.isfinite(a).all() or np.min(a[:,2]) <= 0:
            return
        dist = np.sqrt(np.sum((a[ii,:2]-a[jj,:2])**2, axis=1))
        factor = min(1.0, float(np.min(dist/(a[ii,2]+a[jj,2]))))
        if factor <= 0:
            return
        a[:,2] *= factor * (1.0-params['slack'])
        lo = np.min(a[:,:2]-a[:,2,None], axis=0)
        hi = np.max(a[:,:2]+a[:,2,None], axis=0)
        perimeter = float(np.sum(hi-lo))
        if perimeter <= 0:
            return
        a[:,:2] -= (hi+lo)*0.5
        a *= (2.0/perimeter)*(1.0-params['slack'])
        if float(np.sum(a[:,2])) <= best_score:
            return
        score = exact(a)
        if score > best_score:
            best_score = score
            best_array = a.copy()
            best = {'circles': a.tolist()}

    def model(rows):
        # Three axis circles and nine reflected pairs, in row order.
        groups = []
        for row, count in enumerate(rows):
            if time.monotonic() >= deadline:
                raise StopSearch()
            if count % 2:
                groups.append((0.0, row*np.sqrt(3.0), True))
            for k in range(count//2):
                x = 2.0*(k+1) if count % 2 else 2.0*k+1.0
                groups.append((x, row*np.sqrt(3.0), False))
        dim = 1 + sum(2 if g[2] else 3 for g in groups)
        X=[]; Y=[]; R=[]; reflect=[]; initial=np.zeros(dim)
        initial[0]=1.0
        pos=1
        for x,y,axis in groups:
            if time.monotonic() >= deadline:
                raise StopSearch()
            if axis:
                xv=np.zeros(dim); yv=np.zeros(dim); rv=np.zeros(dim)
                yv[pos]=1; rv[pos+1]=1
                initial[pos:pos+2]=[y,0.9]
                idx=len(X); X.append(xv);Y.append(yv);R.append(rv);reflect.append(idx)
                pos+=2
            else:
                xv=np.zeros(dim); yv=np.zeros(dim); rv=np.zeros(dim)
                xv[pos]=1;yv[pos+1]=1;rv[pos+2]=1
                initial[pos:pos+3]=[x,y,0.9]
                idx=len(X)
                X.extend([xv,-xv]);Y.extend([yv,yv]);R.extend([rv,rv]);reflect.extend([idx+1,idx])
                pos+=3
        X=np.asarray(X);Y=np.asarray(Y);R=np.asarray(R)
        raw=np.column_stack((X@initial,Y@initial,R@initial))
        lo=np.min(raw[:,:2]-raw[:,2,None],axis=0)
        hi=np.max(raw[:,:2]+raw[:,2,None],axis=0)
        scale=2.0/np.sum(hi-lo)
        initial *= scale
        # Shift the row coordinate so the lower boundary is zero.
        for k in range(21):
            if time.monotonic() >= deadline:
                raise StopSearch()
            inds=np.flatnonzero(Y[k])
            if len(inds):
                initial[inds[0]] = (raw[k,1]-lo[1])*scale
        initial[0]=(hi[0]-lo[0])*scale
        pairs=[]
        for i,j in zip(ii,jj):
            if time.monotonic() >= deadline:
                raise StopSearch()
            mirrored=tuple(sorted((reflect[i],reflect[j])))
            if (int(i),int(j)) <= mirrored:
                pairs.append((i,j))
        pi=np.array([p[0] for p in pairs]);pj=np.array([p[1] for p in pairs])
        DX=X[pi]-X[pj];DY=Y[pi]-Y[pj];DR=R[pi]+R[pj]
        W=np.zeros(dim);W[0]=1
        B=[];C=[]
        seen=set()
        for k in range(21):
            if time.monotonic() >= deadline:
                raise StopSearch()
            for row,c in ((0.5*W-X[k]-R[k],0),(0.5*W+X[k]-R[k],0),(Y[k]-R[k],0),(-W-Y[k]-R[k],2)):
                key=tuple(row)+(c,)
                if key not in seen:
                    seen.add(key);B.append(row);C.append(c)
        B=np.asarray(B);C=np.asarray(C)
        objective=-np.sum(R,axis=0)
        def fun(z):
            return np.concatenate(((DX@z)**2+(DY@z)**2-(DR@z)**2,B@z+C))
        def jac(z):
            return np.vstack((2*(DX@z)[:,None]*DX+2*(DY@z)[:,None]*DY-2*(DR@z)[:,None]*DR,B))
        def unpack(z):
            return np.column_stack((X@z,Y@z,R@z))
        bounds=[(0.15,1.85)] + [(None,None)]*(dim-1)
        for k in range(dim):
            if time.monotonic() >= deadline:
                raise StopSearch()
            if np.any(R[:,k]):
                bounds[k]=(1e-7,0.5)
        return initial,objective,fun,jac,unpack,bounds

    families=[(5,4,5,4,3),(4,5,4,5,3),(5,4,3,4,5),(3,6,3,6,3)]
    records=[]
    try:
        for rows in families:
            if time.monotonic() >= deadline:
                break
            z,obj,con,jac,unpack,bounds=model(rows)
            retain(unpack(z))
            count=[0]
            def callback(v):
                if time.monotonic() >= deadline:
                    raise StopSearch()
                count[0]+=1
                if count[0]%6==0:
                    retain(unpack(v))
            res=minimize(lambda v: float(obj@v),z,jac=lambda v: obj,
                bounds=bounds,constraints={'type':'ineq','fun':con,'jac':jac},
                method='SLSQP',callback=callback,
                options={'maxiter':int(params['iterations']),'ftol':2e-12,'disp':False})
            retain(unpack(res.x))
            records.append((float(-obj@res.x),res.x.copy(),obj,con,jac,unpack,bounds))
        records.sort(key=lambda q:q[0],reverse=True)
        trial=0
        while records and time.monotonic() < deadline:
            _,base,obj,con,jac,unpack,bounds=records[trial%min(2,len(records))]
            z=base.copy()
            z[1:] += rng.normal(0,params['restart_noise'],len(z)-1)
            for k,(low,high) in enumerate(bounds):
                if time.monotonic() >= deadline:
                    raise StopSearch()
                if low is not None: z[k]=max(z[k],low)
                if high is not None: z[k]=min(z[k],high)
            def cb(v):
                if time.monotonic() >= deadline: raise StopSearch()
            res=minimize(lambda v:float(obj@v),z,jac=lambda v:obj,bounds=bounds,
                constraints={'type':'ineq','fun':con,'jac':jac},method='SLSQP',callback=cb,
                options={'maxiter':80,'ftol':1e-12,'disp':False})
            retain(unpack(res.x))
            trial+=1
    except StopSearch:
        pass
    if best is None:
        # A deterministic valid fallback, also usable with a missing warm start.
        a=[]
        for k in range(21):
            a.append([(k%7)*0.14,(k//7)*0.14,0.06999999999999])
        return {'circles':a}
    return best
