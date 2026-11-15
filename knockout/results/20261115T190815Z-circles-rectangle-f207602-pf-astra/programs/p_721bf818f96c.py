import time
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize

PARAMS = {
    'trust': (0.15, 0.04, 0.35),
    'lp_steps': (10, 3, 30),
    'perturbation': (0.022, 0.001, 0.08),
    'safety': (0.70, 0.50, 0.75)
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
P = np.arange(M)
K = np.arange(N)

class Deadline(Exception):
    pass

def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    r = a[:, 2]
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < r[I]+r[J]):
        return -np.inf
    x = list(map(Fraction, map(float, a[:, 0])))
    y = list(map(Fraction, map(float, a[:, 1])))
    q = list(map(Fraction, map(float, r)))
    w = max(u+v for u,v in zip(x,q))-min(u-v for u,v in zip(x,q))
    h = max(u+v for u,v in zip(y,q))-min(u-v for u,v in zip(y,q))
    return float(np.sum(r)) if w+h <= 2 else -np.inf

def encode(a):
    return {'coordinate_rows': a.T.tolist()}

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + float(params['safety'])*budget_s
    rng = np.random.default_rng(seed)
    best_obj = None
    best_a = None
    best_score = -np.inf
    raw = None
    if init is not None:
        try:
            raw = np.asarray(init['circles'] if 'circles' in init else np.asarray(init['coordinate_rows']).T, dtype=float)
            s = exact_score(raw)
            if s > best_score:
                best_obj, best_a, best_score = init, raw.copy(), s
        except (KeyError, ValueError, TypeError, OverflowError):
            raw = None
    k = np.arange(N)
    fallback = np.column_stack(((k % 5 + 0.5)*0.2, (k//5+0.5)*0.2, np.full(N, 0.1*(1-1e-12))))
    s = exact_score(fallback)
    if s > best_score:
        best_obj, best_a, best_score = encode(fallback), fallback.copy(), s
    if raw is None or raw.shape != (N,3) or not np.isfinite(raw).all():
        raw = fallback.copy()

    obj = np.r_[np.zeros(2*N), -np.ones(N), 0.0]
    B = np.zeros((4*N, 3*N+1))
    B[K,K] = -1
    B[K,2*N+K] = 1
    B[N+K,K] = 1
    B[N+K,2*N+K] = 1
    B[N+K,-1] = -1
    B[2*N+K,N+K] = -1
    B[2*N+K,2*N+K] = 1
    B[3*N+K,N+K] = 1
    B[3*N+K,2*N+K] = 1
    B[3*N+K,-1] = 1
    bb = np.r_[np.zeros(3*N), np.full(N,2.0)]
    A = np.zeros((M+4*N,3*N+1))
    A[M:] = B
    A[P,2*N+I] = 1
    A[P,2*N+J] = 1
    rhs = np.r_[np.zeros(M),bb]
    stop = end

    def guard():
        if time.monotonic() >= stop:
            raise Deadline()

    def pack(a):
        a = a.copy()
        a[:,2] = np.maximum(np.abs(a[:,2]),1e-7)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        sides = hi-lo
        scale = 2/max(float(sides.sum()),1e-12)
        a[:,:2] = (a[:,:2]-lo)*scale
        a[:,2] *= scale
        return np.r_[a[:,0],a[:,1],a[:,2],sides[0]*scale]

    def consider(v):
        nonlocal best_obj, best_a, best_score
        guard()
        c = np.column_stack((v[:N],v[N:2*N],np.maximum(v[2*N:3*N],1e-10)))
        d = np.sqrt((c[I,0]-c[J,0])**2+(c[I,1]-c[J,1])**2)
        f = min(1.0,float(np.min(d/(c[I,2]+c[J,2]))))
        if not np.isfinite(f) or f <= 0:
            return
        c[:,2] *= f*(1-3e-12)
        lo = np.min(c[:,:2]-c[:,2,None],axis=0)
        hi = np.max(c[:,:2]+c[:,2,None],axis=0)
        c[:,:2] -= (lo+hi)/2
        c *= (2/float((hi-lo).sum()))*(1-3e-12)
        if float(c[:,2].sum()) <= best_score:
            return
        sc = exact_score(c)
        if sc > best_score:
            best_obj, best_a, best_score = encode(c), c.copy(), sc

    def con(v):
        guard()
        dx = v[I]-v[J]
        dy = v[N+I]-v[N+J]
        return np.r_[np.sqrt(dx*dx+dy*dy)-v[2*N+I]-v[2*N+J],bb-B@v]

    def jac(v):
        guard()
        dx = v[I]-v[J]
        dy = v[N+I]-v[N+J]
        dist = np.maximum(np.sqrt(dx*dx+dy*dy),1e-15)
        out = np.zeros_like(A)
        out[P,I] = dx/dist
        out[P,J] = -dx/dist
        out[P,N+I] = dy/dist
        out[P,N+J] = -dy/dist
        out[P,2*N+I] = -1
        out[P,2*N+J] = -1
        out[M:] = -B
        return out

    def fun(v):
        guard()
        return float(obj@v)

    def run(a, steps):
        z = pack(a)
        consider(z)
        lp_stop = min(stop, time.monotonic()+0.12*budget_s)
        for it in range(steps):
            guard()
            if time.monotonic() >= lp_stop:
                break
            dx = z[I]-z[J]
            dy = z[N+I]-z[N+J]
            dist = np.maximum(np.hypot(dx,dy),1e-15)
            nx,ny = dx/dist,dy/dist
            A[P,I],A[P,J] = -nx,nx
            A[P,N+I],A[P,N+J] = -ny,ny
            bounds = np.empty((3*N+1,2))
            bounds[:2*N,0] = z[:2*N]-float(params['trust'])
            bounds[:2*N,1] = z[:2*N]+float(params['trust'])
            bounds[2*N:3*N] = [1e-7,0.5]
            bounds[-1] = [0.05,1.95]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res = linprog(obj,A_ub=A,b_ub=rhs,bounds=bounds,method='highs',options={'threads':1,'time_limit':max(0.001,lp_stop-time.monotonic()),'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
            guard()
            if not res.success:
                break
            z = res.x
            consider(z)
        guard()
        res = minimize(fun,z,jac=lambda v:obj,method='SLSQP',bounds=[(None,None)]*(2*N)+[(1e-7,0.5)]*N+[(0.05,1.95)],constraints={'type':'ineq','fun':con,'jac':jac},callback=consider,options={'maxiter':180,'ftol':1e-12,'disp':False})
        consider(res.x)

    # A second family has three axis circles and nine reflected pairs.
    counts = np.array([5,4,3,4,5])
    row = np.repeat(np.arange(5),counts)
    offsets = np.repeat(np.r_[0,np.cumsum(counts)[:-1]],counts)
    xx = np.arange(N)-offsets-(counts[row]-1)/2
    yy = (row-2)*np.sqrt(3)/2
    lattice = np.column_stack((xx,yy,np.full(N,0.48)))
    deadlines = start+(end-start)*np.array([0.36,0.68,0.85,1.0])
    for stage in range(4):
        if time.monotonic() >= end:
            break
        stop = float(deadlines[stage])
        if time.monotonic() >= stop-0.015:
            continue
        if stage == 0:
            a = raw
            steps = int(params['lp_steps'])
        elif stage == 1:
            a = lattice
            steps = int(params['lp_steps'])
        elif stage == 2:
            a = best_a.copy()
            a[:,:2] += rng.normal(0,float(params['perturbation']),size=(N,2))
            steps = 4
        else:
            a = best_a.copy()
            steps = 0
        try:
            run(a,steps)
        except Deadline:
            pass
        except (ValueError,FloatingPointError):
            pass
    return best_obj
