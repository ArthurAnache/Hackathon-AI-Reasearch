import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'deformation': (0.16, 0.05, 0.35),
    'attempts': (5, 1, 12),
    'squeeze_iterations': (65, 15, 150),
    'release_iterations': (100, 20, 250),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class Deadline(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -float('inf')

    def check():
        if time.monotonic() >= end:
            raise Deadline()

    def perimeter(a):
        x = list(map(lambda v: Fraction(float(v)), a[:, 0]))
        y = list(map(lambda v: Fraction(float(v)), a[:, 1]))
        r = list(map(lambda v: Fraction(float(v)), a[:, 2]))
        return (max(u+v for u,v in zip(x,r)) - min(u-v for u,v in zip(x,r))
                + max(u+v for u,v in zip(y,r)) - min(u-v for u,v in zip(y,r)))

    def score(a):
        if a.shape != (N,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return -float('inf')
        if perimeter(a) > 2:
            return -float('inf')
        for i,j in zip(I,J):
            check()
            c,d = a[i],a[j]
            dist = np.sqrt((c[0]-d[0])**2 + (c[1]-d[1])**2)
            if dist < c[2]+d[2]:
                return -float('inf')
        return float(np.sum(a[:,2]))

    def consider(z):
        nonlocal best, best_score
        check()
        if np.sum(z[42:63]) <= best_score:
            return
        a = np.column_stack((z[:21],z[21:42],z[42:63])).copy()
        if not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
        ratio = min(1.0,float(np.min(d/(a[I,2]+a[J,2]))))
        a[:,2] *= max(0.0,ratio)*(1.0-3e-15)
        if np.sum(a[:,2]) <= best_score:
            return
        p = perimeter(a)
        if p > 2:
            a *= np.nextafter(float(Fraction(2)/p),0.0)*(1.0-3e-15)
        if np.sum(a[:,2]) <= best_score:
            return
        value = score(a)
        if value > best_score:
            best_score = value
            best = {'circles': a.tolist()}

    if init is None:
        a = np.zeros((N,3))
        a[:,0] = (np.arange(N)%5+0.5)/5
        a[:,1] = (np.arange(N)//5+0.5)/5
        a[:,2] = 0.09
        best = {'circles':a.tolist()}
    else:
        a = np.asarray(init['circles'],dtype=float).copy()
    try:
        best_score = score(a)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        side = np.max(a[:,:2]+a[:,2,None],axis=0)-lo
        a[:,:2] -= lo
        if side[0] < side[1]:
            a[:,:2] = a[:,[1,0]]
            side = side[::-1]
        scale = 2.0/np.sum(side)
        a *= scale
        width = side[0]*scale
        base = np.r_[a[:,0],a[:,1],a[:,2],width]

        # All constraints are smooth except coincident centres, avoided by bounds.
        jacbase = np.zeros((M+4*N,64))
        q = np.arange(M)
        jacbase[q,42+I] = -1.0
        jacbase[q,42+J] = -1.0
        k = np.arange(N)
        jacbase[M+k,k] = 1
        jacbase[M+k,42+k] = -1
        jacbase[M+N+k,k] = -1
        jacbase[M+N+k,42+k] = -1
        jacbase[M+N+k,63] = 1
        jacbase[M+2*N+k,21+k] = 1
        jacbase[M+2*N+k,42+k] = -1
        jacbase[M+3*N+k,21+k] = -1
        jacbase[M+3*N+k,42+k] = -1
        jacbase[M+3*N+k,63] = -1
        cache_z = None
        cache_c = None
        cache_j = None

        def calculations(z):
            nonlocal cache_z,cache_c,cache_j
            check()
            if cache_z is not None and np.array_equal(z,cache_z):
                return cache_c,cache_j
            x,y,r,w = z[:21],z[21:42],z[42:63],z[63]
            dx,dy = x[I]-x[J],y[I]-y[J]
            d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-30)
            c = np.r_[d-r[I]-r[J],x-r,w-x-r,y-r,2-w-y-r]
            j = jacbase.copy()
            j[q,I] = dx/d
            j[q,J] = -dx/d
            j[q,21+I] = dy/d
            j[q,21+J] = -dy/d
            cache_z,cache_c,cache_j = z.copy(),c,j
            return c,j

        g = np.r_[np.zeros(42),-np.ones(21),0.0]
        def objective(z):
            check()
            return -float(np.sum(z[42:63]))
        def gradient(z):
            check()
            return g
        def callback(z):
            check()
            consider(z)
        constraint = {'type':'ineq','fun':lambda z: calculations(z)[0],
                      'jac':lambda z: calculations(z)[1]}
        bounds = [(0.0,2.0)]*42+[(1e-6,0.5)]*21+[(0.35,1.65)]
        signs = [1,-1,1,-1,1,-1]
        if rng.random() < 0.5:
            signs = [-v for v in signs]
        for attempt in range(int(params['attempts'])):
            check()
            if end-time.monotonic() < 0.10:
                break
            delta = float(params['deformation'])*(1+0.55*(attempt//2))
            target = float(np.clip(width+signs[attempt%6]*delta,0.55,1.50))
            z = base.copy()
            sx,sy = target/width,(2-target)/(2-width)
            z[:21] *= sx
            z[21:42] *= sy
            z[42:63] *= min(sx,sy)*0.995
            z[63] = target
            fixed = bounds.copy()
            fixed[-1] = (target,target)
            result = minimize(objective,z,jac=gradient,method='SLSQP',
                bounds=fixed,constraints=constraint,callback=callback,
                options={'maxiter':int(params['squeeze_iterations']),'ftol':2e-10})
            check()
            z = result.x
            consider(z)
            result = minimize(objective,z,jac=gradient,method='SLSQP',
                bounds=bounds,constraints=constraint,callback=callback,
                options={'maxiter':int(params['release_iterations']),'ftol':3e-13})
            consider(result.x)
    except Deadline:
        pass
    return best
