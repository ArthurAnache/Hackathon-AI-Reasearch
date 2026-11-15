import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'removed': (4, 3, 7),
    'grid': (35, 21, 61),
    'iterations': (100, 30, 200),
    'exclusion': (0.085, 0.02, 0.2)
}
N = 21
I, J = np.triu_indices(N, 1)

class Expired(Exception):
    pass


def exact_score(c):
    a = np.asarray(c['circles'], dtype=float)
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return -np.inf
    x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
    perimeter = (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
                 + max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)))
    return float(a[:, 2].sum()) if perimeter <= 2 else -np.inf


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    def check(*args):
        if time.monotonic() >= deadline:
            raise Expired()
    if init is None:
        a = np.array([[(k%5+.5)/5, (k//5+.5)/5, .085] for k in range(N)])
        init = {'circles': a.tolist()}
    best = init
    best_score = exact_score(init)
    a = np.array(init['circles'], dtype=float)
    low = np.min(a[:, :2]-a[:, 2,None], axis=0)
    sides = np.max(a[:, :2]+a[:, 2,None], axis=0)-low
    a[:, :2] -= low
    scale = 2/sides.sum()
    a *= scale
    w = sides[0]*scale
    h = 2-w
    m = len(I)
    objective = np.r_[np.zeros(42), -np.ones(21), 0.]
    def unpack(z):
        return z[:21], z[21:42], z[42:63], z[63]
    def constraints(z):
        check()
        x,y,r,b = unpack(z)
        d = np.sqrt((x[I]-x[J])**2+(y[I]-y[J])**2+1e-30)
        return np.r_[d-r[I]-r[J], x-r, b-x-r, y-r, 2-b-y-r]
    def jac(z):
        check()
        x,y,r,b = unpack(z)
        dx,dy = x[I]-x[J], y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        out = np.zeros((m+84,64))
        t = np.arange(m)
        out[t,I]=dx/d
        out[t,J]=-dx/d
        out[t,21+I]=dy/d
        out[t,21+J]=-dy/d
        out[t,42+I]=-1
        out[t,42+J]=-1
        k = np.arange(21)
        out[m+k,k]=1
        out[m+21+k,k]=-1
        out[m+21+k,63]=1
        out[m+42+k,21+k]=1
        out[m+63+k,21+k]=-1
        out[m+63+k,63]=-1
        out[m+k,42+k]=-1
        out[m+21+k,42+k]=-1
        out[m+42+k,42+k]=-1
        out[m+63+k,42+k]=-1
        return out
    def consider(z):
        nonlocal best,best_score
        if time.monotonic() >= deadline:
            return
        x,y,r,b = unpack(z)
        p = np.column_stack((x,y))
        if np.any(r <= 0) or not np.isfinite(z).all():
            return
        d = np.sqrt((x[I]-x[J])**2+(y[I]-y[J])**2)
        factor = min(1., float(np.min(d/(r[I]+r[J]))))
        rr = r*factor*(1-3e-14)
        widthheight = np.ptp(np.array([np.min(x-rr),np.max(x+rr)])) + np.ptp(np.array([np.min(y-rr),np.max(y+rr)]))
        s = min(1.,2/widthheight)*(1-3e-14)
        cand = {'circles': np.column_stack((p*s,rr*s)).tolist()}
        val = exact_score(cand)
        if val > best_score:
            best,best_score = cand,val
    grid = int(params['grid'])
    gx,gy = np.meshgrid(np.linspace(0,w,grid),np.linspace(0,h,grid))
    samples = np.column_stack((gx.ravel(),gy.ravel()))
    attempt = 0
    try:
        while time.monotonic() < deadline:
            check()
            k = int(params['removed'])
            # Alternate a connected vacancy cluster with spatially dispersed vacancies.
            if attempt % 2 == 0:
                anchor = int(rng.integers(N))
                metric = np.linalg.norm(a[:,:2]-a[anchor,:2],axis=1)
                metric += rng.uniform(0,.035,N)
                removed = np.argsort(metric)[:k]
            else:
                removed = rng.choice(N,k,replace=False)
            keep = np.ones(N,dtype=bool)
            keep[removed]=False
            work = a[keep].copy()
            for insertion in range(k):
                check()
                q = np.vstack((samples,a[removed,:2]))
                walls = np.minimum.reduce((q[:,0],w-q[:,0],q[:,1],h-q[:,1]))
                clearance = np.minimum(walls, np.min(np.linalg.norm(q[:,None,:]-work[None,:,:2],axis=2)-work[None,:,2],axis=1))
                if insertion == 0:
                    distold = np.min(np.linalg.norm(q[:,None,:]-a[removed,None,:2].reshape(1,k,2),axis=2),axis=1)
                    clearance[distold < float(params['exclusion'])] = -1
                idx = int(np.argmax(clearance))
                v0 = np.r_[q[idx],max(1e-5,clearance[idx])]
                def cavity(v):
                    check()
                    x,y,r = v
                    return np.r_[x-r,w-x-r,y-r,h-y-r,np.linalg.norm(work[:,:2]-v[:2],axis=1)-work[:,2]-r]
                res = minimize(lambda v:-v[2],v0,jac=lambda v:np.array([0.,0.,-1.]),
                    constraints={'type':'ineq','fun':cavity},method='SLSQP',
                    bounds=[(0,w),(0,h),(1e-7,min(w,h)/2)],
                    callback=check,options={'maxiter':22,'ftol':1e-10})
                work = np.vstack((work,res.x))
            z = np.r_[work[:,0],work[:,1],work[:,2],w]
            consider(z)
            res = minimize(lambda z:float(objective@z),z,jac=lambda z:objective,
                constraints={'type':'ineq','fun':constraints,'jac':jac},
                bounds=[(None,None)]*42+[(1e-7,.8)]*21+[(.12,1.88)],
                method='SLSQP',callback=check,
                options={'maxiter':int(params['iterations']),'ftol':3e-12})
            consider(res.x)
            attempt += 1
    except Expired:
        pass
    return best
