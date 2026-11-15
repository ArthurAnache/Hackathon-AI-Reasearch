import time
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize

PARAMS = {
    'trust': (0.18, 0.03, 0.35),
    'screen_fraction': (0.32, 0.15, 0.45),
    'screen_steps': (12, 5, 25),
    'curvature_handoff': (0.00002, 0.000001, 0.001, 'log'),
    'gain_handoff': (0.0001, 0.000001, 0.01, 'log'),
    'polish_iterations': (85, 20, 150)
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
K = np.arange(N)
P = np.arange(M)

class Deadline(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    r = a[:, 2]
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < r[I]+r[J]):
        return -np.inf
    x = list(map(Fraction, map(float, a[:, 0])))
    y = list(map(Fraction, map(float, a[:, 1])))
    q = list(map(Fraction, map(float, r)))
    perimeter = (max(u+v for u,v in zip(x,q))-min(u-v for u,v in zip(x,q))
                 +max(u+v for u,v in zip(y,q))-min(u-v for u,v in zip(y,q)))
    return float(np.sum(r)) if perimeter <= 2 else -np.inf

def encode(a):
    return {'coordinate_rows': a.T.tolist()}

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70*budget_s
    screen_end = start + float(params['screen_fraction'])*budget_s
    best = None
    best_score = -np.inf
    raw = None
    if init is not None:
        try:
            raw = np.asarray(init['circles'] if 'circles' in init else np.asarray(init['coordinate_rows']).T, dtype=float)
            score = exact_score(raw)
            if score > best_score:
                best, best_score = init, score
        except (KeyError, ValueError, TypeError, OverflowError):
            raw = None
    fallback = np.column_stack(((K%5+0.5)*0.2, (K//5+0.5)*0.2, np.full(N, 0.1*(1-1e-12))))
    score = exact_score(fallback)
    if score > best_score:
        best, best_score = encode(fallback), score
    if time.monotonic() >= end:
        return best

    def guard():
        if time.monotonic() >= end:
            raise Deadline()

    def normalize(a):
        a = a.copy()
        a[:,2] = np.maximum(np.abs(a[:,2]), 1e-7)
        lo = np.min(a[:,:2]-a[:,2,None], axis=0)
        hi = np.max(a[:,:2]+a[:,2,None], axis=0)
        sides = hi-lo
        scale = 2.0/max(float(sides.sum()), 1e-12)
        a[:,:2] = (a[:,:2]-lo)*scale
        a[:,2] *= scale
        return np.r_[a[:,0], a[:,1], a[:,2], sides[0]*scale]

    def consider(v):
        nonlocal best, best_score
        guard()
        c = np.column_stack((v[:N], v[N:2*N], np.maximum(v[2*N:3*N], 1e-12)))
        d = np.sqrt((c[I,0]-c[J,0])**2+(c[I,1]-c[J,1])**2)
        f = min(1.0, float(np.min(d/(c[I,2]+c[J,2]))))
        if f <= 0 or not np.isfinite(f):
            return -np.inf
        c[:,2] *= f*(1-2e-12)
        lo = np.min(c[:,:2]-c[:,2,None], axis=0)
        hi = np.max(c[:,:2]+c[:,2,None], axis=0)
        c[:,:2] -= (lo+hi)*0.5
        c *= (2.0/float((hi-lo).sum()))*(1-2e-12)
        estimate = float(c[:,2].sum())
        if estimate > best_score:
            sc = exact_score(c)
            if sc > best_score:
                best, best_score = encode(c), sc
        return estimate

    objective = np.r_[np.zeros(2*N), -np.ones(N), 0.0]
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

    def lp_step(z, limit):
        guard()
        dx, dy = z[I]-z[J], z[N+I]-z[N+J]
        d = np.maximum(np.hypot(dx,dy),1e-15)
        nx, ny = dx/d, dy/d
        A[P,I], A[P,J] = -nx,nx
        A[P,N+I], A[P,N+J] = -ny,ny
        trust = float(params['trust'])
        bounds = np.empty((3*N+1,2))
        bounds[:2*N,0] = z[:2*N]-trust
        bounds[:2*N,1] = z[:2*N]+trust
        bounds[2*N:3*N] = [1e-7,0.5]
        bounds[-1] = [0.05,1.95]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = linprog(objective,A_ub=A,b_ub=rhs,bounds=bounds,method='highs',
                options={'threads':1,'time_limit':max(0.001,limit-time.monotonic()),
                         'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
        guard()
        if not result.success:
            return z, True
        v = result.x
        consider(v)
        gain = float(np.sum(v[2*N:3*N]-z[2*N:3*N]))
        dx,dy = v[I]-v[J],v[N+I]-v[N+J]
        projection = nx*dx+ny*dy
        active = projection-v[2*N+I]-v[2*N+J] < 1e-7
        curved = np.maximum(0.0,np.hypot(dx,dy)-projection)
        remaining = float(np.max(curved[active])) if np.any(active) else np.inf
        ready = (gain >= -1e-8 and gain < float(params['gain_handoff']) and
                 remaining < float(params['curvature_handoff']))
        return v, ready

    seeds = []
    if raw is not None and raw.shape == (N,3) and np.isfinite(raw).all():
        seeds.append(normalize(raw))
    layouts = [(4,5,4,5,3), (5,4,5,4,3), (4,3,4,3,4,3)]
    try:
        for rows in layouts:
            guard()
            blocks = []
            for j,n in enumerate(rows):
                guard()
                x = 2*np.arange(n,dtype=float)-(n-1)
                blocks.append(np.column_stack((x,np.full(n,j*np.sqrt(3.0)),np.ones(n))))
            seeds.append(normalize(np.vstack(blocks)))
        states = []
        for index,z in enumerate(seeds):
            guard()
            if time.monotonic() >= screen_end:
                break
            remaining_count = len(seeds)-index
            local_end = time.monotonic()+(screen_end-time.monotonic())/remaining_count
            consider(z)
            for it in range(int(params['screen_steps'])):
                guard()
                if time.monotonic() >= local_end:
                    break
                z,ready = lp_step(z,local_end)
                if ready and it >= 3:
                    break
            states.append((float(np.sum(z[2*N:3*N])),z.copy()))
        states.sort(key=lambda item:item[0],reverse=True)

        def constraints(v):
            guard()
            return np.r_[np.hypot(v[I]-v[J],v[N+I]-v[N+J])-v[2*N+I]-v[2*N+J],bb-B@v]

        def jacobian(v):
            guard()
            dx,dy = v[I]-v[J],v[N+I]-v[N+J]
            d = np.maximum(np.hypot(dx,dy),1e-15)
            out = np.zeros_like(A)
            out[P,I],out[P,J] = dx/d,-dx/d
            out[P,N+I],out[P,N+J] = dy/d,-dy/d
            out[P,2*N+I],out[P,2*N+J] = -1,-1
            out[M:] = -B
            return out

        def fun(v):
            guard()
            return float(objective@v)

        for rank,(_,z) in enumerate(states):
            guard()
            if time.monotonic() >= end-0.04:
                break
            if rank >= 2:
                break
            # A few additional inner-approximation steps settle contact directions.
            lp_limit = min(end-0.10,time.monotonic()+0.055*budget_s)
            for it in range(10):
                guard()
                if time.monotonic() >= lp_limit:
                    break
                z,ready = lp_step(z,lp_limit)
                if ready:
                    break
            if time.monotonic() >= end-0.03:
                break
            result = minimize(fun,z,jac=lambda v: objective,method='SLSQP',
                bounds=[(None,None)]*(2*N)+[(1e-7,0.5)]*N+[(0.05,1.95)],
                constraints={'type':'ineq','fun':constraints,'jac':jacobian},
                callback=consider,
                options={'maxiter':int(params['polish_iterations']),'ftol':2e-12,'disp':False})
            consider(result.x)
    except Deadline:
        pass
    return best
