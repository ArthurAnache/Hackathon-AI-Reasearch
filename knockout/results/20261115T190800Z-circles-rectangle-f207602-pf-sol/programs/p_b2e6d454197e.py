import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'excursion': (0.17, 0.04, 0.35),
    'forced_iterations': (36, 10, 100),
    'free_iterations': (75, 20, 180),
    'trials': (4, 1, 8)
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class StopSearch(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -float('inf')
    d = np.sqrt((a[I, 0] - a[J, 0])**2 + (a[I, 1] - a[J, 1])**2)
    if np.any(d < a[I, 2] + a[J, 2]):
        return -float('inf')
    x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
    perimeter = (max(u+v for u,v in zip(x,r)) - min(u-v for u,v in zip(x,r))
                 + max(u+v for u,v in zip(y,r)) - min(u-v for u,v in zip(y,r)))
    if perimeter > 2:
        return -float('inf')
    return float(np.sum(a[:, 2]))

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a = np.array([[(k % 5 + .5)*.2, (k // 5 + .5)*.2, .045] for k in range(N)])
        best = {'circles': a.tolist()}
    else:
        a = np.asarray(init['circles'], dtype=float).copy()
        best = init
    best_score = exact_score(a)
    lo = np.min(a[:, :2] - a[:, 2,None], axis=0)
    hi = np.max(a[:, :2] + a[:, 2,None], axis=0)
    side = hi-lo
    p = a[:, :2] - (hi+lo)/2
    if side[0] > side[1]:
        p = p[:, ::-1].copy()
        side = side[::-1].copy()
    scale = 2/np.sum(side)
    p *= scale
    z0 = np.r_[p[:,0], p[:,1], a[:,2]*scale, side[0]*scale]
    objective_gradient = np.r_[np.zeros(42), -np.ones(21), 0.]
    rows = np.arange(M)
    ix = np.arange(N)
    last = z0.copy()

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def fun(z):
        check()
        return -float(np.sum(z[42:63]))

    def constraints(z):
        check()
        x,y,r,w = z[:21],z[21:42],z[42:63],z[63]
        d = np.sqrt((x[I]-x[J])**2+(y[I]-y[J])**2+1e-30)
        return np.r_[d-r[I]-r[J], w/2+x-r, w/2-x-r,
                     (2-w)/2+y-r, (2-w)/2-y-r]

    def jacobian(z):
        check()
        x,y = z[:21],z[21:42]
        dx,dy = x[I]-x[J],y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        jac = np.zeros((M+84,64))
        jac[rows,I] = dx/d
        jac[rows,J] = -dx/d
        jac[rows,21+I] = dy/d
        jac[rows,21+J] = -dy/d
        jac[rows,42+I] = -1
        jac[rows,42+J] = -1
        for k in range(4):
            check()
            rr = M+k*21+ix
            jac[rr,42+ix] = -1
            jac[rr, (0 if k<2 else 21)+ix] = 1 if k%2==0 else -1
            jac[rr,63] = .5 if k<2 else -.5
        return jac

    def consider(z):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        b = np.column_stack((z[:21],z[21:42],z[42:63])).copy()
        if not np.isfinite(b).all() or np.any(b[:,2] <= 0):
            return
        # Uniform radius contraction repairs strict floating-point overlap.
        d = np.sqrt((b[I,0]-b[J,0])**2+(b[I,1]-b[J,1])**2)
        factor = min(1., float(np.min(d/(b[I,2]+b[J,2]))))
        b[:,2] *= factor*(1-3e-15)
        lo1 = np.min(b[:,:2]-b[:,2,None],axis=0)
        hi1 = np.max(b[:,:2]+b[:,2,None],axis=0)
        b[:,:2] -= (lo1+hi1)/2
        extent = float(np.sum(hi1-lo1))
        b *= (2/extent)*(1-3e-15)
        score = exact_score(b)
        if score > best_score:
            best_score = score
            best = {'circles': b.tolist()}

    def callback(z):
        nonlocal last
        check()
        last = z.copy()

    def solve(z, width, iterations):
        nonlocal last
        check()
        last = z.copy()
        bounds = [(None,None)]*42 + [(1e-7,.49)]*21 + [(0.15,1.85)]
        if width is not None:
            bounds[-1] = (width,width)
        out = minimize(fun,z,jac=lambda z: objective_gradient,
                       bounds=bounds,
                       constraints={'type':'ineq','fun':constraints,'jac':jacobian},
                       method='SLSQP',callback=callback,
                       options={'maxiter':iterations,'ftol':2e-12,'disp':False})
        last = out.x.copy()
        consider(last)
        return last

    try:
        for trial in range(int(params['trials'])):
            check()
            direction = -1 if trial%2==0 else 1
            amplitude = float(params['excursion'])*(1+0.45*(trial//2))
            target = float(np.clip(z0[63]*(1+direction*amplitude),.24,1.65))
            z = z0.copy()
            z[:21] *= target/z0[63]
            z[21:42] *= (2-target)/(2-z0[63])
            z[63] = target
            # A tiny deterministic asymmetry selects a branch at contact bifurcations.
            z[:42] += rng.normal(0,0.0002,42)
            z = solve(z,target,int(params['forced_iterations']))
            z = solve(z,None,int(params['free_iterations']))
    except StopSearch:
        pass
    return best
