import time
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize

PARAMS = {
    'trust': (0.16, 0.04, 0.35),
    'screen_steps': (7, 3, 15),
    'screen_fraction': (0.29, 0.15, 0.40),
    'lp_fraction': (0.47, 0.35, 0.55),
    'refine_steps': (24, 8, 60),
    'polish_steps': (100, 30, 180)
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
    lp_end = start + float(params['lp_fraction'])*budget_s
    best = None
    best_score = -np.inf
    raw = None
    if init is not None:
        try:
            raw = np.asarray(init['circles'] if 'circles' in init else np.asarray(init['coordinate_rows']).T, dtype=float)
            score = exact_score(raw)
            if score > best_score:
                best, best_score = init, score
        except (KeyError, TypeError, ValueError, OverflowError):
            raw = None
    k = np.arange(N)
    fallback = np.column_stack(((k%5+0.5)*0.2, (k//5+0.5)*0.2, np.full(N, 0.1*(1-1e-12))))
    score = exact_score(fallback)
    if score > best_score:
        best, best_score = encode(fallback), score
    if time.monotonic() >= end:
        return best

    def guard():
        if time.monotonic() >= end:
            raise Deadline()

    def normalize(a):
        a = np.array(a, dtype=float, copy=True)
        a[:, 2] = np.maximum(np.abs(a[:, 2]), 1e-6)
        lo = np.min(a[:, :2]-a[:, 2,None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2,None], axis=0)
        sides = hi-lo
        scale = 2.0/max(float(sides.sum()), 1e-12)
        a[:, :2] = (a[:, :2]-lo)*scale
        a[:, 2] *= scale
        return np.r_[a[:, 0], a[:, 1], a[:, 2], sides[0]*scale]

    def consider(v):
        nonlocal best, best_score
        guard()
        a = np.column_stack((v[:N], v[N:2*N], np.maximum(v[2*N:3*N], 1e-12)))
        if not np.isfinite(a).all():
            return -np.inf
        d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
        shrink = min(1.0, float(np.min(d/(a[I,2]+a[J,2]))))
        if shrink <= 0:
            return -np.inf
        a[:,2] *= shrink*(1-3e-12)
        lo = np.min(a[:,:2]-a[:,2,None], axis=0)
        hi = np.max(a[:,:2]+a[:,2,None], axis=0)
        a[:,:2] -= (lo+hi)*0.5
        a *= (2.0/float((hi-lo).sum()))*(1-3e-12)
        value = float(a[:,2].sum())
        if value > best_score:
            value = exact_score(a)
            if value > best_score:
                best_score, best = value, encode(a)
        return value

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
    bb = np.r_[np.zeros(3*N), np.full(N, 2.0)]
    A = np.zeros((M+4*N,3*N+1))
    A[M:] = B
    A[P,2*N+I] = 1
    A[P,2*N+J] = 1
    rhs = np.r_[np.zeros(M),bb]
    trust = float(params['trust'])

    def step(state, stop):
        guard()
        if time.monotonic() >= stop:
            return False
        z = state['z']
        dx, dy = z[I]-z[J], z[N+I]-z[N+J]
        dist = np.maximum(np.hypot(dx,dy),1e-15)
        nx, ny = dx/dist, dy/dist
        A[P,I], A[P,J] = -nx, nx
        A[P,N+I], A[P,N+J] = -ny, ny
        bounds = np.empty((3*N+1,2))
        bounds[:2*N,0] = z[:2*N]-trust
        bounds[:2*N,1] = z[:2*N]+trust
        bounds[2*N:3*N,0] = 1e-6
        bounds[2*N:3*N,1] = 0.5
        bounds[-1] = [0.08,1.92]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = linprog(objective, A_ub=A, b_ub=rhs, bounds=bounds, method='highs',
                options={'threads':1, 'time_limit':max(0.001,stop-time.monotonic()),
                         'primal_feasibility_tolerance':1e-9,
                         'dual_feasibility_tolerance':1e-9})
        guard()
        if not result.success:
            state['failed'] = True
            return False
        previous = float(z[2*N:3*N].sum())
        state['z'] = result.x
        state['score'] = consider(result.x)
        state['gain'] = float(result.x[2*N:3*N].sum())-previous
        state['steps'] += 1
        return True

    states = []
    try:
        if raw is not None and raw.shape == (N,3) and np.isfinite(raw).all():
            z = normalize(raw)
            states.append({'z':z,'score':consider(z),'gain':1.0,'steps':0,'failed':False})
        for counts in ((4,3,4,3,4,3), (4,5,3,5,4), (5,4,5,4,3)):
            guard()
            points = []
            for row, count in enumerate(counts):
                guard()
                for col in range(count):
                    guard()
                    points.append((2.0*col-(count-1), np.sqrt(3.0)*row, 1.0))
            z = normalize(np.asarray(points))
            states.append({'z':z,'score':consider(z),'gain':1.0,'steps':0,'failed':False})

        # Equal initial investment avoids letting a slowly improving poor start
        # consume the complete LP budget.
        for iteration in range(int(params['screen_steps'])):
            guard()
            if time.monotonic() >= screen_end:
                break
            for state in states:
                guard()
                if time.monotonic() >= screen_end:
                    break
                if not state['failed']:
                    step(state, screen_end)

        states.sort(key=lambda s:s['score'], reverse=True)
        finalists = states[:2]
        for iteration in range(int(params['refine_steps'])):
            guard()
            if time.monotonic() >= lp_end:
                break
            active = False
            for state in finalists:
                guard()
                if time.monotonic() >= lp_end:
                    break
                if state['failed'] or (state['steps'] >= 12 and state['gain'] < 1e-7):
                    continue
                if step(state,lp_end):
                    active = True
            if not active:
                break
        states.sort(key=lambda s:s['score'], reverse=True)

        def constraints(v):
            guard()
            dx, dy = v[I]-v[J], v[N+I]-v[N+J]
            return np.r_[np.hypot(dx,dy)-v[2*N+I]-v[2*N+J],bb-B@v]

        def jacobian(v):
            guard()
            dx, dy = v[I]-v[J], v[N+I]-v[N+J]
            dist = np.maximum(np.hypot(dx,dy),1e-15)
            out = np.zeros_like(A)
            out[P,I], out[P,J] = dx/dist, -dx/dist
            out[P,N+I], out[P,N+J] = dy/dist, -dy/dist
            out[P,2*N+I], out[P,2*N+J] = -1,-1
            out[M:] = -B
            return out

        def fun(v):
            guard()
            return float(objective@v)

        # A converged winner releases its unused polishing time to the runner-up.
        for state in states[:2]:
            guard()
            if time.monotonic() >= end-0.035:
                break
            result = minimize(fun,state['z'],jac=lambda v:objective,method='SLSQP',
                bounds=[(None,None)]*(2*N)+[(1e-6,0.5)]*N+[(0.08,1.92)],
                constraints={'type':'ineq','fun':constraints,'jac':jacobian},
                callback=consider,
                options={'maxiter':int(params['polish_steps']),'ftol':1e-12,'disp':False})
            consider(result.x)
    except Deadline:
        pass
    return best
