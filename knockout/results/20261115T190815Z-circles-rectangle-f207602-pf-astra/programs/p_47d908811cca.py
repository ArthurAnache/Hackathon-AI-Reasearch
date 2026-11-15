import time
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize

PARAMS = {
    'trust': (0.10, 0.01, 0.30),
    'lp_fraction': (0.49, 0.20, 0.58),
    'iterations': (100, 10, 250),
    'polish_iterations': (100, 10, 250),
    'curvature_threshold': (0.0003, 0.00001, 0.003, 'log')
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
K = np.arange(N)
P = np.arange(M)

class Deadline(Exception):
    pass

def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    r = a[:, 2]
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < r[I] + r[J]):
        return -np.inf
    x = list(map(Fraction, map(float, a[:, 0])))
    y = list(map(Fraction, map(float, a[:, 1])))
    q = list(map(Fraction, map(float, r)))
    perimeter = (max(a+b for a,b in zip(x,q)) - min(a-b for a,b in zip(x,q))
                 + max(a+b for a,b in zip(y,q)) - min(a-b for a,b in zip(y,q)))
    return float(np.sum(r)) if perimeter <= 2 else -np.inf

def encode(a):
    return {'coordinate_rows': a.T.tolist()}

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70 * budget_s
    lp_end = start + float(params['lp_fraction']) * budget_s
    best = None
    best_score = -np.inf
    raw = None
    if init is not None:
        try:
            raw = np.asarray(init['circles'] if 'circles' in init else np.asarray(init['coordinate_rows']).T, dtype=float)
            sc = exact_score(raw)
            if sc > best_score:
                best, best_score = init, sc
        except (KeyError, ValueError, TypeError, OverflowError):
            raw = None
    fallback = np.column_stack(((K % 5 + 0.5)*0.2, (K // 5 + 0.5)*0.2, np.full(N, 0.1*(1-1e-12))))
    fs = exact_score(fallback)
    if fs > best_score:
        best, best_score = encode(fallback), fs
    if raw is None or raw.shape != (N, 3) or not np.isfinite(raw).all():
        raw = fallback
    a = raw.copy()
    a[:, 2] = np.maximum(np.abs(a[:, 2]), 1e-7)
    lo = np.min(a[:, :2]-a[:, 2,None], axis=0)
    hi = np.max(a[:, :2]+a[:, 2,None], axis=0)
    sides = hi-lo
    scale = 2.0/max(float(sides.sum()), 1e-12)
    a[:, :2] = (a[:, :2]-lo)*scale
    a[:, 2] *= scale
    z = np.r_[a[:, 0], a[:, 1], a[:, 2], sides[0]*scale]
    objective = np.r_[np.zeros(2*N), -np.ones(N), 0.0]
    B = np.zeros((4*N, 3*N+1))
    B[K, K] = -1
    B[K, 2*N+K] = 1
    B[N+K, K] = 1
    B[N+K, 2*N+K] = 1
    B[N+K, -1] = -1
    B[2*N+K, N+K] = -1
    B[2*N+K, 2*N+K] = 1
    B[3*N+K, N+K] = 1
    B[3*N+K, 2*N+K] = 1
    B[3*N+K, -1] = 1
    bb = np.r_[np.zeros(3*N), np.full(N, 2.0)]

    def guard():
        if time.monotonic() >= end:
            raise Deadline()

    def consider(v):
        nonlocal best, best_score
        guard()
        c = np.column_stack((v[:N], v[N:2*N], np.maximum(v[2*N:3*N], 1e-12)))
        d = np.sqrt((c[I,0]-c[J,0])**2 + (c[I,1]-c[J,1])**2)
        factor = min(1.0, float(np.min(d/(c[I,2]+c[J,2]))))
        if factor <= 0 or not np.isfinite(factor):
            return
        c[:,2] *= factor*(1-2e-12)
        low = np.min(c[:,:2]-c[:,2,None], axis=0)
        high = np.max(c[:,:2]+c[:,2,None], axis=0)
        c[:,:2] -= (low+high)*0.5
        c *= min(1.0, 2.0/float((high-low).sum()))*(1-2e-12)
        if float(c[:,2].sum()) <= best_score:
            return
        score = exact_score(c)
        if score > best_score:
            best_score, best = score, encode(c)

    A = np.zeros((M+4*N, 3*N+1))
    A[M:] = B
    A[P, 2*N+I] = 1
    A[P, 2*N+J] = 1
    rhs = np.r_[np.zeros(M), bb]
    trust = float(params['trust'])
    stale = 0
    previous_contacts = None
    settled = 0
    try:
        consider(z)
        for iteration in range(int(params['iterations'])):
            guard()
            if time.monotonic() >= lp_end:
                break
            dx = z[I]-z[J]
            dy = z[N+I]-z[N+J]
            dist = np.maximum(np.hypot(dx,dy), 1e-15)
            nx, ny = dx/dist, dy/dist
            A[P,I] = -nx
            A[P,J] = nx
            A[P,N+I] = -ny
            A[P,N+J] = ny
            bounds = np.empty((3*N+1,2))
            bounds[:2*N,0] = z[:2*N]-trust
            bounds[:2*N,1] = z[:2*N]+trust
            bounds[2*N:3*N,0] = 1e-7
            bounds[2*N:3*N,1] = 0.5
            bounds[-1] = [0.05,1.95]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result = linprog(objective, A_ub=A, b_ub=rhs, bounds=bounds,
                    method='highs', options={'threads':1, 'time_limit':max(0.001, lp_end-time.monotonic()),
                    'primal_feasibility_tolerance':1e-9, 'dual_feasibility_tolerance':1e-9})
            guard()
            if not result.success:
                break
            oldsum = float(z[2*N:3*N].sum())
            z = result.x
            consider(z)
            gain = float(z[2*N:3*N].sum())-oldsum
            stale = stale+1 if gain < 1e-9 else 0

            # A settled LP contact graph indicates that nonlinear polishing
            # can work on the correct basin without more direction refreshes.
            contacts = np.asarray(result.ineqlin.residual) < 2e-7
            if previous_contacts is not None:
                changes = np.count_nonzero(contacts != previous_contacts)
                settled = settled+1 if changes <= 2 else 0
            previous_contacts = contacts.copy()
            new_dx = z[I]-z[J]
            new_dy = z[N+I]-z[N+J]
            curvature = np.maximum(0.0, np.hypot(new_dx,new_dy)-nx*new_dx-ny*new_dy)
            weights = np.maximum(0.0, -np.asarray(result.ineqlin.marginals[:M]))
            curvature_value = float(weights @ curvature)
            if iteration >= 5 and settled >= 2 and curvature_value <= float(params['curvature_threshold']):
                break
            if stale >= 3:
                break

        def constraints(v):
            guard()
            dx = v[I]-v[J]
            dy = v[N+I]-v[N+J]
            return np.r_[np.hypot(dx,dy)-v[2*N+I]-v[2*N+J], bb-B@v]

        def jacobian(v):
            guard()
            dx = v[I]-v[J]
            dy = v[N+I]-v[N+J]
            dist = np.maximum(np.hypot(dx,dy),1e-15)
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
            return float(objective@v)

        if time.monotonic() < end-0.03:
            result = minimize(fun, z, jac=lambda v: objective, method='SLSQP',
                bounds=[(None,None)]*(2*N)+[(1e-7,0.5)]*N+[(0.05,1.95)],
                constraints={'type':'ineq','fun':constraints,'jac':jacobian},
                callback=consider,
                options={'maxiter':int(params['polish_iterations']), 'ftol':2e-12, 'disp':False})
            consider(result.x)
    except Deadline:
        pass
    return best
