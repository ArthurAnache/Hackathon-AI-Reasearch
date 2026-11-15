import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'backbone_iterations': (75, 20, 160),
    'final_iterations': (110, 30, 220),
    'grid_size': (31, 15, 61),
    'attempts': (5, 1, 12),
}

class StopSearch(Exception):
    pass


def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    i, j = np.triu_indices(21, 1)
    d = np.sqrt((a[i, 0]-a[j, 0])**2 + (a[i, 1]-a[j, 1])**2)
    if np.any(d < a[i, 2] + a[j, 2]):
        return -np.inf
    x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
    perimeter = (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
                 + max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)))
    if perimeter > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf
    if init is not None:
        original = np.asarray(init['circles'], dtype=float)
        best_score = exact_score(original)
        base = original.copy()
    else:
        base = np.array([[(k % 7 + .5)/3.5, (k//7+.5)/3.5, .065] for k in range(21)])

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def normalize(a):
        a = a.copy()
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        scale = 2.0 / np.sum(hi-lo)
        a[:, :2] = (a[:, :2]-lo)*scale
        a[:, 2] *= scale
        return a, float((hi[0]-lo[0])*scale)

    base, base_w = normalize(base)

    def offer(a):
        nonlocal best, best_score
        check()
        if a.shape != (21,3) or not np.isfinite(a).all():
            return
        a = a.copy()
        a[:,2] = np.maximum(a[:,2], 1e-13)
        ii,jj = np.triu_indices(21,1)
        d = np.sqrt((a[ii,0]-a[jj,0])**2+(a[ii,1]-a[jj,1])**2)
        ratio = float(np.min(d/(a[ii,2]+a[jj,2])))
        if ratio < 1:
            a[:,2] *= max(0,ratio)*(1-2e-14)
        a,w = normalize(a)
        a[:, :2] *= 1-3e-14
        a[:, 2] *= 1-6e-14
        score = exact_score(a)
        if score > best_score:
            best_score = score
            best = {'circles': a.tolist()}

    def relax(a, w, iterations):
        n = len(a)
        ii,jj = np.triu_indices(n,1)
        m = len(ii)
        z = np.r_[a[:,0],a[:,1],a[:,2],w]
        c = np.r_[np.zeros(2*n),-np.ones(n),0.]
        rows = np.arange(m)
        idx = np.arange(n)
        def fun(z):
            check()
            return float(c@z), c
        def con(z):
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
            q[rows,ii] = dx/d
            q[rows,jj] = -dx/d
            q[rows,n+ii] = dy/d
            q[rows,n+jj] = -dy/d
            q[rows,2*n+ii] = -1
            q[rows,2*n+jj] = -1
            q[m+idx,idx] = 1
            q[m+n+idx,idx] = -1
            q[m+2*n+idx,n+idx] = 1
            q[m+3*n+idx,n+idx] = -1
            q[m+idx,2*n+idx] = -1
            q[m+n+idx,2*n+idx] = -1
            q[m+2*n+idx,2*n+idx] = -1
            q[m+3*n+idx,2*n+idx] = -1
            q[m+n+idx,-1] = 1
            q[m+3*n+idx,-1] = -1
            return q
        bounds = [(0,2)]*(2*n)+[(1e-9,.65)]*n+[(.15,1.85)]
        res = minimize(fun,z,jac=True,method='SLSQP',bounds=bounds,
                       constraints={'type':'ineq','fun':con,'jac':jac},
                       options={'maxiter':iterations,'ftol':3e-12,'disp':False})
        z = res.x
        return np.column_stack((z[:n],z[n:2*n],z[2*n:3*n])),float(z[-1])

    def insert(a,w):
        check()
        h = 2-w
        g = int(params['grid_size'])
        xx,yy = np.meshgrid(np.linspace(.001,w-.001,g),np.linspace(.001,h-.001,g))
        p = np.column_stack((xx.ravel(),yy.ravel()))
        clearance = np.min(np.sqrt(np.sum((p[:,None,:]-a[None,:,:2])**2,axis=2))-a[None,:,2],axis=1)
        clearance = np.minimum(clearance,np.min(np.column_stack((p[:,0],w-p[:,0],p[:,1],h-p[:,1])),axis=1))
        order = np.argsort(clearance)[::-1]
        starts = []
        for k in order:
            check()
            if all(np.linalg.norm(p[k]-v) > .08 for v in starts):
                starts.append(p[k])
            if len(starts) >= 3:
                break
        chosen = np.r_[p[order[0]],max(1e-9,clearance[order[0]])]
        def cf(v):
            check()
            d = np.sqrt(np.sum((v[:2]-a[:,:2])**2,axis=1)+1e-30)
            return np.r_[d-a[:,2]-v[2],v[0]-v[2],w-v[0]-v[2],v[1]-v[2],h-v[1]-v[2]]
        def cj(v):
            check()
            delta = v[:2]-a[:,:2]
            d = np.sqrt(np.sum(delta*delta,axis=1)+1e-30)
            return np.vstack((np.column_stack((delta/d[:,None],-np.ones(len(a)))),[1,0,-1],[-1,0,-1],[0,1,-1],[0,-1,-1]))
        for p0 in starts:
            check()
            res = minimize(lambda v:(-v[2],np.array([0.,0.,-1.])),np.r_[p0,.01],jac=True,
                           method='SLSQP',bounds=[(0,w),(0,h),(1e-10,.5)],
                           constraints={'type':'ineq','fun':cf,'jac':cj},
                           options={'maxiter':30,'ftol':1e-12})
            if res.x[2] > chosen[2] and np.min(cf(res.x)) > -1e-8:
                chosen = res.x
        return np.vstack((a,chosen))

    try:
        offer(base)
        rank = np.argsort(base[:,2])[::-1]
        for trial in range(int(params['attempts'])):
            check()
            count = 1 if trial % 3 != 2 else 2
            if trial == 0:
                removed = [int(rank[0])]
            elif trial == 1:
                removed = [int(rank[len(rank)//2])]
            else:
                removed = rng.choice(21,size=count,replace=False)
            a = np.delete(base,removed,axis=0)
            a,w = relax(a,base_w,int(params['backbone_iterations']))
            for k in range(count):
                check()
                a = insert(a,w)
            offer(a)
            a,w = relax(a,w,int(params['final_iterations']))
            offer(a)
    except StopSearch:
        pass
    if best is None:
        a = base.copy()
        a[:,2] *= .1
        best = {'circles':a.tolist()}
    return best
