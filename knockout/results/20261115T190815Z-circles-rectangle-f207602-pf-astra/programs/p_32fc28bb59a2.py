import time
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize

PARAMS = {
    'trust': (0.22, 0.03, 0.45),
    'lp_fraction': (0.46, 0.25, 0.55),
    'trajectory_steps': (25, 10, 60),
    'perturbation': (0.025, 0.001, 0.10),
    'polish_steps': (100, 20, 200)
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
        return -float('inf')
    r = a[:, 2]
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < r[I]+r[J]):
        return -float('inf')
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    q = [Fraction(float(v)) for v in r]
    w = max(u+v for u,v in zip(x,q))-min(u-v for u,v in zip(x,q))
    h = max(u+v for u,v in zip(y,q))-min(u-v for u,v in zip(y,q))
    return float(np.sum(r)) if w+h <= 2 else -float('inf')

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    end = started + 0.70*budget_s
    lp_end = started + float(params['lp_fraction'])*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -float('inf')
    best_array = None
    raw = None
    if init is not None:
        try:
            raw = np.asarray(init['circles'] if 'circles' in init else np.asarray(init['coordinate_rows']).T, dtype=float)
            sc = exact_score(raw)
            if np.isfinite(sc):
                best_score, best_array = sc, raw.copy()
        except (ValueError, TypeError, KeyError, OverflowError):
            raw = None
    fallback = np.column_stack(((K%5+0.5)*0.2, (K//5+0.5)*0.2, np.full(N, 0.1*(1-1e-12))))
    sc = exact_score(fallback)
    if sc > best_score:
        best_score, best_array = sc, fallback.copy()
        best = {'coordinate_rows': fallback.T.tolist()}

    def guard():
        if time.monotonic() >= end:
            raise Deadline()

    def pack(a):
        a = np.asarray(a, dtype=float).copy()
        a[:, 2] = np.maximum(np.abs(a[:, 2]), 1e-7)
        low = np.min(a[:, :2]-a[:, 2,None], axis=0)
        high = np.max(a[:, :2]+a[:, 2,None], axis=0)
        sides = high-low
        scale = 2/max(float(sides.sum()), 1e-12)
        a[:, :2] = (a[:, :2]-low)*scale
        a[:, 2] *= scale
        return np.r_[a[:, 0], a[:, 1], a[:, 2], sides[0]*scale]

    def consider(v):
        nonlocal best, best_score, best_array
        guard()
        c = np.column_stack((v[:N], v[N:2*N], np.maximum(v[2*N:3*N], 1e-10)))
        d = np.sqrt((c[I,0]-c[J,0])**2 + (c[I,1]-c[J,1])**2)
        f = min(1.0, float(np.min(d/(c[I,2]+c[J,2]))))
        if not np.isfinite(f) or f <= 0:
            return
        c[:,2] *= f*(1-3e-12)
        lo = np.min(c[:,:2]-c[:,2,None], axis=0)
        hi = np.max(c[:,:2]+c[:,2,None], axis=0)
        c[:,:2] -= (lo+hi)*0.5
        c *= (2/float((hi-lo).sum()))*(1-3e-12)
        if float(np.sum(c[:,2])) <= best_score:
            return
        score = exact_score(c)
        if score > best_score:
            best_score, best_array = score, c.copy()
            best = {'coordinate_rows': c.T.tolist()}

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
    trust = float(params['trust'])

    def structured(counts):
        pieces = []
        for row, count in enumerate(counts):
            guard()
            xs = 2*np.arange(count, dtype=float)-(count-1)
            pieces.append(np.column_stack((xs, np.full(count, row*np.sqrt(3.0)), np.ones(count))))
        a = np.vstack(pieces)
        a[:,:2] += rng.normal(0, 0.012, (N,2))
        return pack(a)

    try:
        if raw is None or raw.shape != (N,3) or not np.isfinite(raw).all():
            raw = fallback
        starts = [pack(raw), structured([4,5,3,5,4]), structured([4,4,5,4,4])]
        trial = 0
        while time.monotonic() < lp_end-0.015:
            guard()
            if trial < len(starts):
                z = starts[trial].copy()
            else:
                z = pack(best_array)
                amp = float(params['perturbation'])*(1+0.5*((trial-3)%3))
                z[:2*N] += rng.normal(0, amp, 2*N)
            trial += 1
            previous = -float('inf')
            stale = 0
            trajectory_end = min(lp_end, time.monotonic()+0.18*budget_s)
            for iteration in range(int(params['trajectory_steps'])):
                guard()
                now = time.monotonic()
                if now >= trajectory_end-0.004:
                    break
                dx = z[I]-z[J]
                dy = z[N+I]-z[N+J]
                dist = np.maximum(np.hypot(dx,dy),1e-15)
                nx, ny = dx/dist, dy/dist
                A[P,I], A[P,J] = -nx, nx
                A[P,N+I], A[P,N+J] = -ny, ny
                bounds = np.empty((3*N+1,2))
                bounds[:2*N,0] = z[:2*N]-trust
                bounds[:2*N,1] = z[:2*N]+trust
                bounds[2*N:3*N,0] = 1e-7
                bounds[2*N:3*N,1] = 0.5
                bounds[-1] = [0.05,1.95]
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    result = linprog(objective, A_ub=A, b_ub=rhs, bounds=bounds, method='highs', options={'threads':1, 'time_limit':max(0.001, min(0.09, trajectory_end-now)), 'primal_feasibility_tolerance':1e-9, 'dual_feasibility_tolerance':1e-9})
                guard()
                if not result.success:
                    break
                z = result.x
                consider(z)
                score = float(np.sum(z[2*N:3*N]))
                stale = stale+1 if score-previous < 2e-9 else 0
                previous = score
                if stale >= 2:
                    break

        def constraints(v):
            guard()
            return np.r_[np.hypot(v[I]-v[J],v[N+I]-v[N+J])-v[2*N+I]-v[2*N+J], bb-B@v]

        def jacobian(v):
            guard()
            dx,dy = v[I]-v[J],v[N+I]-v[N+J]
            dist = np.maximum(np.hypot(dx,dy),1e-15)
            out = np.zeros_like(A)
            out[P,I],out[P,J] = dx/dist,-dx/dist
            out[P,N+I],out[P,N+J] = dy/dist,-dy/dist
            out[P,2*N+I],out[P,2*N+J] = -1,-1
            out[M:] = -B
            return out

        def fun(v):
            guard()
            return float(objective@v)

        if time.monotonic() < end-0.025:
            z = pack(best_array)
            result = minimize(fun,z,jac=lambda v: objective,method='SLSQP',bounds=[(None,None)]*(2*N)+[(1e-7,0.5)]*N+[(0.05,1.95)],constraints={'type':'ineq','fun':constraints,'jac':jacobian},callback=consider,options={'maxiter':int(params['polish_steps']),'ftol':3e-13,'disp':False})
            consider(result.x)
    except Deadline:
        pass
    return best
