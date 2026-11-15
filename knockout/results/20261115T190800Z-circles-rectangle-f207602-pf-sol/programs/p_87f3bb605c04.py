import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'remove': (3, 2, 5),
    'relax_iterations': (35, 10, 100),
    'finish_iterations': (85, 20, 180),
    'grid': (25, 12, 40),
    'relax_fraction': (0.65, 0.1, 1.0)
}

class Deadline(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf

    def check():
        if time.monotonic() >= end:
            raise Deadline()

    def score(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -np.inf
        i, j = np.triu_indices(21, 1)
        d = np.sqrt((a[i, 0]-a[j, 0])**2 + (a[i, 1]-a[j, 1])**2)
        if np.any(d < a[i, 2]+a[j, 2]):
            return -np.inf
        x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
        perimeter = (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
                     + max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)))
        return float(a[:,2].sum()) if perimeter <= 2 else -np.inf

    def accept(a):
        nonlocal best, best_score
        # Uniform scaling preserves all geometry; a tiny radius contraction
        # then protects against rounding in the strict overlap test.
        a = a.copy()
        if np.any(a[:,2] <= 0):
            return
        i,j = np.triu_indices(21,1)
        dist = np.sqrt(((a[i,:2]-a[j,:2])**2).sum(axis=1))
        factor = min(1.0, float(np.min(dist/(a[i,2]+a[j,2]))))
        a[:,2] *= factor * (1-3e-14)
        lo = (a[:,:2]-a[:,2,None]).min(axis=0)
        hi = (a[:,:2]+a[:,2,None]).max(axis=0)
        scale = min(1.0, 2.0/float((hi-lo).sum())) * (1-3e-14)
        a[:,:2] = (a[:,:2]-lo)*scale
        a[:,2] *= scale
        s = score(a)
        if s > best_score:
            best_score = s
            best = {'circles': a.tolist()}

    def optimize(a, w, iterations):
        n = len(a)
        ii,jj = np.triu_indices(n,1)
        m = len(ii)
        ar = np.arange(n)
        z0 = np.r_[a[:,0],a[:,1],a[:,2],w]
        objective_grad = np.r_[np.zeros(2*n),-np.ones(n),0.0]
        def fun(z):
            check()
            return -z[2*n:3*n].sum(), objective_grad
        def constraints(z):
            check()
            x,y,r = z[:n],z[n:2*n],z[2*n:3*n]
            dx,dy = x[ii]-x[jj],y[ii]-y[jj]
            d = np.sqrt(dx*dx+dy*dy+1e-30)
            return np.r_[d-r[ii]-r[jj],x-r,z[-1]-x-r,y-r,2-z[-1]-y-r]
        def jac(z):
            check()
            x,y = z[:n],z[n:2*n]
            dx,dy = x[ii]-x[jj],y[ii]-y[jj]
            d = np.sqrt(dx*dx+dy*dy+1e-30)
            q = np.zeros((m+4*n,3*n+1))
            rows = np.arange(m)
            q[rows,ii]=dx/d; q[rows,jj]=-dx/d
            q[rows,n+ii]=dy/d; q[rows,n+jj]=-dy/d
            q[rows,2*n+ii]=-1; q[rows,2*n+jj]=-1
            for k in range(4):
                rr=m+k*n+ar
                q[rr,2*n+ar]=-1
                q[rr,(0 if k<2 else n)+ar]=1 if k%2==0 else -1
            q[m+n:m+2*n,-1]=1
            q[m+3*n:,-1]=-1
            return q
        result = minimize(fun,z0,jac=True,method='SLSQP',
            bounds=[(None,None)]*(2*n)+[(1e-7,0.6)]*n+[(0.15,1.85)],
            constraints={'type':'ineq','fun':constraints,'jac':jac},
            options={'maxiter':iterations,'ftol':3e-12,'disp':False})
        z=result.x
        return np.column_stack((z[:n],z[n:2*n],z[2*n:3*n])),float(z[-1])

    def insert(a,w):
        check()
        h=2-w
        g=int(params['grid'])
        xx,yy=np.meshgrid(np.linspace(0,w,g),np.linspace(0,h,g))
        p=np.column_stack((xx.ravel(),yy.ravel()))
        clearance=np.minimum.reduce([p[:,0],w-p[:,0],p[:,1],h-p[:,1]])
        clearance=np.minimum(clearance,(np.sqrt(((p[:,None,:]-a[None,:,:2])**2).sum(axis=2))-a[None,:,2]).min(axis=1))
        # Optimize several competing empty regions, not only the grid maximum.
        order=np.argsort(clearance)[::-1]
        chosen=[]
        answer=None
        for idx in order:
            check()
            p0=p[idx]
            if any(np.linalg.norm(p0-q)<0.09 for q in chosen):
                continue
            chosen.append(p0)
            def f(z):
                check()
                return -z[2],np.array([0.,0.,-1.])
            def c(z):
                check()
                d=np.sqrt(((z[:2]-a[:,:2])**2).sum(axis=1)+1e-30)
                return np.r_[d-a[:,2]-z[2],z[0]-z[2],w-z[0]-z[2],z[1]-z[2],h-z[1]-z[2]]
            def cj(z):
                check()
                delta=z[:2]-a[:,:2]
                d=np.sqrt((delta*delta).sum(axis=1)+1e-30)
                return np.vstack((np.column_stack((delta/d[:,None],-np.ones(len(a)))),[[1,0,-1],[-1,0,-1],[0,1,-1],[0,-1,-1]]))
            res=minimize(f,np.r_[p0,max(1e-7,clearance[idx])],jac=True,method='SLSQP',
                bounds=[(0,w),(0,h),(1e-8,0.5)],constraints={'type':'ineq','fun':c,'jac':cj},
                options={'maxiter':24,'ftol':1e-11})
            if answer is None or res.x[2]>answer[2]:
                answer=res.x.copy()
            if len(chosen)>=3:
                break
        return np.vstack((a,answer))

    if init is not None:
        original=np.array(init['circles'],dtype=float)
        best_score=score(original)
    else:
        original=np.array([[(i%5+.5)/5,(i//5+.5)/5,0.055] for i in range(21)])
        accept(original)
    lo=(original[:,:2]-original[:,2,None]).min(axis=0)
    span=(original[:,:2]+original[:,2,None]).max(axis=0)-lo
    original[:,:2]-=lo
    width=float(span[0])
    try:
        attempt=0
        while time.monotonic()<end:
            check()
            k=int(params['remove'])
            if attempt%3==0:
                # Delete a connected neighborhood, opening a substantial cavity.
                pivot=int(rng.integers(21))
                distance=np.linalg.norm(original[:,:2]-original[pivot,:2],axis=1)
                remove=np.argsort(distance)[:k]
            elif attempt%3==1:
                # Distant deletions permit radius mass to migrate across the box.
                remove=rng.choice(21,k,replace=False)
            else:
                remove=np.argsort(original[:,2]+rng.uniform(0,0.04,21))[:k]
            a=np.delete(original,remove,axis=0)
            relaxed,w=optimize(a,width,int(params['relax_iterations']))
            fraction=float(params['relax_fraction'])
            a=(1-fraction)*a+fraction*relaxed
            w=(1-fraction)*width+fraction*w
            for unused in range(k):
                check()
                a=insert(a,w)
            a,w=optimize(a,w,int(params['finish_iterations']))
            accept(a)
            attempt+=1
    except Deadline:
        pass
    return best
