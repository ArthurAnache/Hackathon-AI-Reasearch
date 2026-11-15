import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'power_low': (1.45, 1.1, 1.95),
    'power_high': (3.4, 2.1, 6.0),
    'angle': (0.29, 0.0, 0.7853981633974483),
    'excursion_iterations': (36, 10, 100),
    'recovery_iterations': (85, 20, 200)
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class Deadline(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = a[I, :2] - a[J, :2]
    if np.any(np.sqrt(d[:, 0]**2 + d[:, 1]**2) < a[I, 2] + a[J, 2]):
        return -np.inf
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    perimeter = (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
                 +max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)))
    return float(np.sum(a[:, 2])) if perimeter <= 2 else -np.inf

def repair(z, deadline):
    a = np.column_stack((z[:N], z[N:2*N], z[2*N:3*N])).copy()
    if not np.isfinite(a).all():
        return None
    a[:, 2] = np.maximum(a[:, 2], 1e-12)
    delta = a[I, :2]-a[J, :2]
    d = np.sqrt(delta[:, 0]**2+delta[:, 1]**2)
    ratio = np.min(d/(a[I, 2]+a[J, 2]))
    a[:, 2] *= min(1.0, ratio)*(1-3e-15)
    lo = np.min(a[:, :2]-a[:, 2,None], axis=0)
    a[:, :2] -= lo
    extent = np.max(a[:, :2]+a[:, 2,None], axis=0).sum()
    if extent <= 0:
        return None
    a *= (2.0/extent)*(1-3e-15)
    for k in range(5):
        if time.monotonic() >= deadline:
            break
        if np.isfinite(exact_score(a)):
            return a
        a[:, 2] *= 1-2e-14
    return a

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    solver_end = start + 0.65*budget_s
    rng = np.random.default_rng(seed)
    if init is not None:
        base = np.asarray(init['circles'], dtype=float).copy()
        best = init
        best_score = exact_score(base)
    else:
        base = np.array([[(i%5)*0.19+0.08, (i//5)*0.19+0.08, 0.075] for i in range(N)])
        best = {'circles': base.tolist()}
        best_score = exact_score(base)
    low = np.min(base[:, :2]-base[:, 2,None], axis=0)
    sides = np.max(base[:, :2]+base[:, 2,None], axis=0)-low
    scale = 2/sides.sum()
    base[:, :2] = (base[:, :2]-low)*scale
    base[:, 2] *= scale
    initial = np.r_[base[:,0], base[:,1], base[:,2], sides[0]*scale]
    objective_jac = np.r_[np.zeros(2*N), -np.ones(N), 0.0]
    bounds = [(0.0, 2.0)]*(2*N)+[(1e-7, 0.6)]*N+[(0.15, 1.85)]
    rows = np.arange(M)
    ids = np.arange(N)
    angle = float(params['angle']) + float(rng.uniform(-0.035,0.035))
    powers = [float(params['power_low']), float(params['power_high'])]
    for attempt, power in enumerate(powers):
        if time.monotonic() >= solver_end:
            break
        z = initial.copy()
        theta = angle if attempt == 0 else -angle
        co, si = np.cos(theta), np.sin(theta)
        last = [z.copy()]
        def objective(v):
            if time.monotonic() >= solver_end:
                raise Deadline()
            return -np.sum(v[2*N:3*N])
        def callback(v):
            last[0] = v.copy()
            if time.monotonic() >= solver_end:
                raise Deadline()
        for p, limit in [(power, int(params['excursion_iterations'])),
                         ((power+2)/2, 18),
                         (2.0, int(params['recovery_iterations']))]:
            if time.monotonic() >= solver_end:
                break
            cache = [None, None, None]
            def calculate(v):
                if time.monotonic() >= solver_end:
                    raise Deadline()
                if cache[0] is not None and np.array_equal(v, cache[0]):
                    return cache[1], cache[2]
                x,y,r,w = v[:N],v[N:2*N],v[2*N:3*N],v[-1]
                dx,dy = x[I]-x[J], y[I]-y[J]
                u, vv = co*dx+si*dy, -si*dx+co*dy
                au,av = np.maximum(np.abs(u),1e-24),np.maximum(np.abs(vv),1e-24)
                dist = (au**p+av**p)**(1/p)
                gu = np.sign(u)*(au/dist)**(p-1)
                gv = np.sign(vv)*(av/dist)**(p-1)
                gx,gy = co*gu-si*gv,si*gu+co*gv
                c = np.r_[dist-r[I]-r[J], x-r,w-x-r,y-r,2-w-y-r]
                jac = np.zeros((M+4*N,3*N+1))
                jac[rows,I]=gx
                jac[rows,J]=-gx
                jac[rows,N+I]=gy
                jac[rows,N+J]=-gy
                jac[rows,2*N+I]=-1
                jac[rows,2*N+J]=-1
                for k in range(4):
                    if time.monotonic() >= solver_end:
                        raise Deadline()
                    rr=M+k*N+ids
                    jac[rr,2*N+ids]=-1
                    jac[rr,(0 if k<2 else N)+ids]=1 if k%2==0 else -1
                jac[M+N:M+2*N,-1]=1
                jac[M+3*N:M+4*N,-1]=-1
                cache[:] = [v.copy(),c,jac]
                return c,jac
            try:
                res = minimize(objective,z,jac=lambda v: objective_jac,
                    constraints={'type':'ineq','fun':lambda v:calculate(v)[0],
                                 'jac':lambda v:calculate(v)[1]},
                    bounds=bounds,method='SLSQP',callback=callback,
                    options={'maxiter':limit,'ftol':2e-12,'disp':False})
                z=res.x
            except Deadline:
                z=last[0]
            if time.monotonic() < deadline:
                candidate=repair(z,deadline)
                if candidate is not None:
                    score=exact_score(candidate)
                    if score>best_score:
                        best_score=score
                        best={'circles':candidate.tolist()}
    return best
