import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'barrier': (0.003, 0.0001, 0.03, 'log'),
    'decay': (0.16, 0.05, 0.5, 'log'),
    'stages': (11, 5, 16),
    'iterations': (115, 30, 300),
    'tilt': (0.025, 0.0, 0.2)
}
N = 21
I, J = np.triu_indices(N, 1)

class StopSearch(Exception):
    pass

def exact(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return -np.inf
    x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
    perimeter = (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
                 +max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)))
    if perimeter > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    best = init
    bestscore = -np.inf
    if init is not None:
        a = np.array(init['circles'], dtype=float)
        bestscore = exact(a)
    else:
        a = np.zeros((N,3))
        a[:,0] = (np.arange(N)%5 + 0.5)/5
        a[:,1] = (np.arange(N)//5 + 0.5)/5
        a[:,2] = 0.035
        best = {'circles': a.tolist()}
        bestscore = exact(a)
    lo = np.min(a[:,:2]-a[:,2,None], axis=0)
    sides = np.max(a[:,:2]+a[:,2,None], axis=0)-lo
    scale = 2/sides.sum()
    p = (a[:,:2]-lo)*scale
    r = a[:,2]*scale
    w = sides[0]*scale
    z = np.r_[p[:,0], p[:,1], r*0.68, w]
    tilt = rng.normal(size=(2,N))
    tilt -= tilt.mean(axis=1)[:,None]
    saved = [z.copy()]

    def consider(v):
        nonlocal best, bestscore
        if time.monotonic() >= deadline:
            return
        q = np.column_stack((v[:N],v[N:2*N],v[2*N:3*N]))
        if not np.isfinite(q).all() or np.min(q[:,2]) <= 0:
            return
        # Homogeneous dilation uses any unused perimeter, while preserving gaps.
        low = np.min(q[:,:2]-q[:,2,None],axis=0)
        high = np.max(q[:,:2]+q[:,2,None],axis=0)
        factor = 2/np.sum(high-low)
        q[:,:2] = (q[:,:2]-low)*factor
        q[:,2] *= factor
        distances = np.sqrt((q[I,0]-q[J,0])**2+(q[I,1]-q[J,1])**2)
        ratio = np.min(distances/(q[I,2]+q[J,2]))
        q[:,2] *= min(1.0,ratio)*(1-2e-14)
        score = exact(q)
        if score > bestscore:
            bestscore = score
            best = {'circles': q.tolist()}

    mu = float(params['barrier'])
    for stage in range(int(params['stages'])):
        if time.monotonic() >= deadline:
            break
        field = float(params['tilt'])*(0.15**stage)
        def fun(v):
            if time.monotonic() >= deadline:
                raise StopSearch()
            x,y,rr,ww = v[:N],v[N:2*N],v[2*N:3*N],v[-1]
            dx,dy = x[I]-x[J],y[I]-y[J]
            d = np.sqrt(dx*dx+dy*dy)
            g = d-rr[I]-rr[J]
            walls = np.concatenate((x-rr,ww-x-rr,y-rr,2-ww-y-rr,rr))
            if np.min(g) <= 0 or np.min(walls) <= 0:
                return 1e40, np.zeros(64)
            f = -rr.sum()-mu*(np.log(g).sum()+np.log(walls).sum())
            f += field*(np.dot(tilt[0],x)+np.dot(tilt[1],y))
            gx = field*tilt[0].copy()
            gy = field*tilt[1].copy()
            gr = -np.ones(N)
            c = mu/g
            ux,uy = dx/d,dy/d
            np.add.at(gx,I,-c*ux)
            np.add.at(gx,J,c*ux)
            np.add.at(gy,I,-c*uy)
            np.add.at(gy,J,c*uy)
            np.add.at(gr,I,c)
            np.add.at(gr,J,c)
            b0,b1,b2,b3,b4 = (mu/walls).reshape(5,N)
            gx += -b0+b1
            gy += -b2+b3
            gr += b0+b1+b2+b3-b4
            gw = -b1.sum()+b3.sum()
            return f,np.r_[gx,gy,gr,gw]
        def callback(v):
            if time.monotonic() >= deadline:
                raise StopSearch()
            saved[0] = v.copy()
        saved[0] = z.copy()
        try:
            result = minimize(fun,z,jac=True,method='L-BFGS-B',callback=callback,
                              options={'maxiter':int(params['iterations']), 'maxls':45,
                                       'ftol':1e-14,'gtol':1e-9,'maxcor':15})
            z = result.x
        except StopSearch:
            z = saved[0]
            break
        consider(z)
        mu *= float(params['decay'])
    if time.monotonic() < deadline:
        consider(z)
    return best
