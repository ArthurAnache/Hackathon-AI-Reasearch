import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'penalty': (120.0, 10.0, 1000.0, 'log'),
    'cluster_angle': (0.48, 0.05, 1.2),
    'inner_iterations': (48, 15, 120),
    'stages': (5, 2, 9),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class Deadline(Exception):
    pass

def exact_score(a):
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
    end = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    if init is not None:
        original = np.asarray(init['circles'], dtype=float)
        best = init
        score = exact_score(original)
    else:
        original = np.array([[(k%5+0.5)*0.2, (k//5+0.5)*0.2, 0.09] for k in range(N)])
        best = {'circles': original.tolist()}
        score = exact_score(original)
    lo = np.min(original[:, :2]-original[:, 2,None], axis=0)
    side = np.max(original[:, :2]+original[:, 2,None], axis=0)-lo
    scale = 2.0 / side.sum()
    p = (original[:, :2]-lo)*scale
    base = np.r_[p[:,0], p[:,1], original[:,2]*scale, side[0]*scale]
    last = base.copy()
    bounds = [(0.0, 2.0)]*42 + [(1e-7, 0.6)]*21 + [(0.15, 1.85)]
    objective_grad = np.r_[np.zeros(42), -np.ones(21), 0.0]

    def check():
        if time.monotonic() >= end:
            raise Deadline()

    def constraints(z, jac=False):
        check()
        x,y,r,w = z[:21], z[21:42], z[42:63], z[63]
        dx,dy = x[I]-x[J], y[I]-y[J]
        d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-15)
        g = np.r_[r[I]+r[J]-d, r-x, r+x-w, r-y, r+y+w-2]
        if not jac:
            return g
        a = np.zeros((M+84,64))
        rows = np.arange(M)
        a[rows,I] = -dx/d
        a[rows,J] = dx/d
        a[rows,21+I] = -dy/d
        a[rows,21+J] = dy/d
        a[rows,42+I] = 1
        a[rows,42+J] = 1
        k = np.arange(21)
        for b in range(4):
            check()
            t = M+21*b+k
            a[t,42+k] = 1
            a[t,k+(21 if b >= 2 else 0)] = -1 if b in (0,2) else 1
        a[M+21:M+42,63] = -1
        a[M+63:M+84,63] = 1
        return g,a

    def remember(z):
        nonlocal best, score
        if time.monotonic() >= end:
            return
        a = np.column_stack((z[:21],z[21:42],z[42:63])).copy()
        if np.any(a[:,2] <= 0) or not np.isfinite(a).all():
            return
        d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
        ratio = min(1.0,float(np.min(d/(a[I,2]+a[J,2]))))
        a[:,2] *= ratio*(1-2e-14)
        low = np.min(a[:,:2]-a[:,2,None],axis=0)
        high = np.max(a[:,:2]+a[:,2,None],axis=0)
        a[:,:2] -= low
        a *= (2.0/(high-low).sum())*(1-3e-14)
        s = exact_score(a)
        if s > score:
            best,score = {'circles':a.tolist()},s

    def finish(z, limit):
        check()
        cached_z = None
        cached = None
        def both(v):
            nonlocal cached_z,cached
            check()
            if cached_z is None or not np.array_equal(v,cached_z):
                cached_z=v.copy()
                cached=constraints(v,True)
            return cached
        result = minimize(lambda v: -float(v[42:63].sum()), z,
            jac=lambda v: objective_grad, method='SLSQP', bounds=bounds,
            constraints={'type':'ineq','fun':lambda v:-both(v)[0], 'jac':lambda v:-both(v)[1]},
            options={'maxiter':limit,'ftol':2e-13})
        remember(result.x)
        return result.x

    attempt = 0
    try:
        # Obtain an accurate reference before spending the remaining time on topology changes.
        last = finish(base, 12)
        while time.monotonic() < end:
            z = base.copy()
            x = np.column_stack((z[:21],z[21:42]))
            if attempt % 3 == 0:
                anchor = int(rng.integers(N))
                nearest = np.argsort(np.sum((x-x[anchor])**2,axis=1))[:int(rng.integers(5,10))]
                center = np.average(x[nearest],axis=0,weights=z[42+nearest])
                angle = float(params['cluster_angle'])*(1 if attempt%2 else -1)
                c,s = np.cos(angle),np.sin(angle)
                x[nearest] = (x[nearest]-center) @ np.array([[c,s],[-s,c]])+center
            elif attempt % 3 == 1:
                axis = attempt % 2
                t = x[:,1-axis]
                x[:,axis] += 0.075*np.sin(2*np.pi*t/max(np.ptp(t),1e-3)+rng.uniform(0,6.28))
                z[42:63] = 0.65*z[42:63]+0.35*np.mean(z[42:63])
            else:
                x += rng.normal(0,0.027,x.shape)
            z[:21],z[21:42] = x[:,0],x[:,1]
            z = np.clip(z,np.array([b[0] for b in bounds]),np.array([b[1] for b in bounds]))
            lam = np.zeros(M+84)
            mu = float(params['penalty'])
            for stage in range(int(params['stages'])):
                check()
                def augmented(v):
                    g,a = constraints(v,True)
                    q = np.maximum(lam+mu*g,0)
                    return (-v[42:63].sum()+(q@q-lam@lam)/(2*mu), objective_grad+q@a)
                def callback(v):
                    nonlocal last
                    check()
                    last = v.copy()
                res = minimize(augmented,z,jac=True,method='L-BFGS-B',bounds=bounds,
                    callback=callback,options={'maxiter':int(params['inner_iterations']),
                    'ftol':1e-12,'gtol':1e-7,'maxls':18,'maxcor':12})
                z = res.x
                lam = np.maximum(0,lam+mu*constraints(z))
                mu *= 3
            remember(z)
            if time.monotonic() < end-0.10:
                last = finish(z,65)
            attempt += 1
    except Deadline:
        pass
    return best
