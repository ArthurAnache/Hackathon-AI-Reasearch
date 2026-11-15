import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"remove": (3, 2, 4), "relax_iterations": (22, 8, 45), "repair_iterations": (65, 20, 120), "void_samples": (350, 100, 900)}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.69 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -float('inf')

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def exact(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        w = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        h = max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        if w+h > 2:
            return -float('inf')
        i,j = np.triu_indices(21,1)
        d = np.sqrt((a[i,0]-a[j,0])**2+(a[i,1]-a[j,1])**2)
        if np.any(d < a[i,2]+a[j,2]):
            return -float('inf')
        return float(np.sum(a[:,2]))

    if init is not None:
        base = np.array(init['circles'], dtype=float)
        best_score = exact(base)
    else:
        base = np.array([[0.11+0.21*(i%5),0.11+0.21*(i//5),0.09] for i in range(21)])

    def retain(a):
        nonlocal best, best_score
        if len(a) != 21 or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        b = a.copy()
        ii,jj = np.triu_indices(21,1)
        d = np.sqrt(np.sum((b[ii,:2]-b[jj,:2])**2,axis=1))
        q = min(1.0, float(np.min(d/(b[ii,2]+b[jj,2]))))
        b[:,2] *= q*(1.0-3e-14)
        lo = np.min(b[:,:2]-b[:,2,None],axis=0)
        b[:,:2] -= lo
        ext = np.max(b[:,:2]+b[:,2,None],axis=0)-np.min(b[:,:2]-b[:,2,None],axis=0)
        b *= min(1.0, 2.0/float(np.sum(ext)))*(1.0-3e-14)
        if float(np.sum(b[:,2])) <= best_score:
            return
        s = exact(b)
        if s > best_score:
            best_score = s
            best = {'circles': b.tolist()}

    def optimize(a, iterations, local_end=None):
        check()
        n = len(a)
        a = a.copy()
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        a[:,:2] -= lo
        wh = np.max(a[:,:2]+a[:,2,None],axis=0)
        scale = 2.0 / np.sum(wh)
        a *= scale
        w = wh[0]*scale
        z = np.r_[a.ravel(),w]
        ii,jj = np.triu_indices(n,1)
        m = len(ii)
        rows = np.arange(m)
        ar = np.arange(n)
        grad = np.zeros(3*n+1)
        grad[2:3*n:3] = -1
        latest = z.copy()
        def tick():
            check()
            if local_end is not None and time.monotonic() >= local_end:
                raise StopSearch()
        def fun(v):
            tick()
            return float(grad @ v)
        def con(v):
            tick()
            c = v[:-1].reshape(n,3)
            x,y,r = c.T
            dx=x[ii]-x[jj]; dy=y[ii]-y[jj]; rr=r[ii]+r[jj]
            return np.r_[dx*dx+dy*dy-rr*rr,x-r,v[-1]-x-r,y-r,2-v[-1]-y-r]
        def jac(v):
            tick()
            c=v[:-1].reshape(n,3)
            x,y,r=c.T
            dx=2*(x[ii]-x[jj]); dy=2*(y[ii]-y[jj]); rr=-2*(r[ii]+r[jj])
            out=np.zeros((m+4*n,3*n+1))
            out[rows,3*ii]=dx; out[rows,3*jj]=-dx
            out[rows,3*ii+1]=dy; out[rows,3*jj+1]=-dy
            out[rows,3*ii+2]=rr; out[rows,3*jj+2]=rr
            for k in range(4):
                tick()
                rrw=m+k*n+ar
                out[rrw,3*ar+2]=-1
                out[rrw,3*ar+(0 if k<2 else 1)]=1 if k%2==0 else -1
            out[m+n:m+2*n,-1]=1
            out[m+3*n:,-1]=-1
            return out
        def callback(v):
            nonlocal latest
            tick()
            latest=v.copy()
            if n==21:
                retain(v[:-1].reshape(n,3))
        bounds=[(None,None),(None,None),(1e-7,0.7)]*n+[(0.15,1.85)]
        try:
            res=minimize(fun,z,jac=lambda v:grad,method='SLSQP',bounds=bounds,
                         constraints={'type':'ineq','fun':con,'jac':jac},callback=callback,
                         options={'maxiter':int(iterations),'ftol':2e-12,'disp':False})
            latest=res.x
        except StopSearch:
            pass
        ans=latest[:-1].reshape(n,3).copy()
        if n==21 and time.monotonic()<deadline:
            retain(ans)
        return ans

    try:
        retain(base)
        # A short polish also removes any small numerical slack in the incumbent.
        polished=optimize(base,12,min(deadline,start+0.16*budget_s))
        if best is not None:
            base=np.array(best['circles'],dtype=float)
        attempt=0
        while time.monotonic()<deadline:
            check()
            a=base.copy()
            k=int(params['remove'])
            anchor=int(rng.integers(21))
            d=np.linalg.norm(a[:,:2]-a[anchor,:2],axis=1)
            # Randomized locality changes the removed connected patch between attempts.
            order=np.argsort(d+rng.uniform(0,0.08,21))
            removed=order[:k]
            survivors=np.delete(a,removed,axis=0)
            survivors=optimize(survivors,int(params['relax_iterations']),min(deadline,time.monotonic()+0.20*budget_s))
            check()
            low=np.min(survivors[:,:2]-survivors[:,2,None],axis=0)
            high=np.max(survivors[:,:2]+survivors[:,2,None],axis=0)
            survivors[:,:2]-=low
            high-=low
            work=survivors
            for slot in range(k):
                check()
                points=rng.random((int(params['void_samples']),2))*high
                old=a[removed,:2]-low
                points=np.vstack((points,old))
                wall=np.minimum(points,high-points).min(axis=1)
                clearance=np.linalg.norm(points[:,None,:]-work[None,:,:2],axis=2)-work[None,:,2]
                radii=np.minimum(wall,clearance.min(axis=1))
                # Usually choose the largest void, occasionally a competitive alternative.
                ranking=np.argsort(radii)
                rank=1 if attempt%3==0 else int(rng.integers(1,5))
                idx=ranking[-rank]
                rr=max(1e-5,float(radii[idx])*0.96)
                work=np.vstack((work,[points[idx,0],points[idx,1],rr]))
            optimize(work,int(params['repair_iterations']))
            attempt+=1
    except StopSearch:
        pass
    if best is None:
        # This branch is only needed for a cold start with an exceptionally short budget.
        a=base.copy()
        i,j=np.triu_indices(21,1)
        ratio=np.min(np.linalg.norm(a[i,:2]-a[j,:2],axis=1)/(a[i,2]+a[j,2]))
        a[:,2]*=min(1.0,ratio)*0.999999
        extent=np.ptp(a[:,:2],axis=0)+2*np.max(a[:,2])
        a*=min(1.0,1.999999/np.sum(extent))
        best={'circles':a.tolist()}
    return best
