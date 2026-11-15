import time
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize

PARAMS = {
    'trust': (0.22, 0.05, 0.45),
    'lp_fraction': (0.46, 0.25, 0.55),
    'race_rounds': (9, 3, 20),
    'radius_floor': (0.015, 0.000001, 0.04),
    'polish_iterations': (100, 20, 200)
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
    if np.any(d < r[I]+r[J]):
        return -np.inf
    x = list(map(Fraction, map(float, a[:, 0])))
    y = list(map(Fraction, map(float, a[:, 1])))
    q = list(map(Fraction, map(float, r)))
    w = max(a+b for a,b in zip(x,q))-min(a-b for a,b in zip(x,q))
    h = max(a+b for a,b in zip(y,q))-min(a-b for a,b in zip(y,q))
    return float(np.sum(r)) if w+h <= 2 else -np.inf

def encode(a):
    return {'coordinate_rows': a.T.tolist()}

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70*budget_s
    lp_end = start + float(params['lp_fraction'])*budget_s
    rng = np.random.default_rng(seed)
    best = None
    best_a = None
    best_score = -np.inf

    def guard():
        if time.monotonic() >= end:
            raise Deadline()

    raw = None
    if init is not None:
        try:
            raw = np.asarray(init['circles'] if 'circles' in init else np.asarray(init['coordinate_rows']).T, dtype=float)
            s = exact_score(raw)
            if s > best_score:
                best, best_score, best_a = init, s, raw.copy()
        except (KeyError, ValueError, TypeError, OverflowError):
            raw = None

    fallback = np.column_stack(((K%5+0.5)*0.2, (K//5+0.5)*0.2, np.full(N, 0.1*(1-1e-12))))
    s = exact_score(fallback)
    if s > best_score:
        best, best_score, best_a = encode(fallback), s, fallback.copy()

    def to_vector(a):
        a = a.copy()
        a[:, 2] = np.maximum(np.abs(a[:, 2]), 1e-7)
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        scale = 2/max(float(np.sum(hi-lo)), 1e-12)
        a[:, :2] = (a[:, :2]-lo)*scale
        a[:, 2] *= scale
        return np.r_[a[:, 0], a[:, 1], a[:, 2], (hi[0]-lo[0])*scale]

    def repair(v):
        c = np.column_stack((v[:N], v[N:2*N], np.maximum(v[2*N:3*N], 1e-12)))
        d = np.sqrt((c[I,0]-c[J,0])**2+(c[I,1]-c[J,1])**2)
        f = min(1.0, float(np.min(d/(c[I,2]+c[J,2]))))
        if f <= 0 or not np.isfinite(f):
            return None
        c[:,2] *= f*(1-3e-12)
        lo = np.min(c[:,:2]-c[:,2,None], axis=0)
        hi = np.max(c[:,:2]+c[:,2,None], axis=0)
        c[:,:2] -= (lo+hi)*0.5
        c *= min(1.0, 2/float(np.sum(hi-lo)))*(1-3e-12)
        return c

    def consider(v):
        nonlocal best, best_a, best_score
        guard()
        c = repair(v)
        if c is None:
            return -np.inf
        score = float(np.sum(c[:,2]))
        if score > best_score:
            score = exact_score(c)
            if score > best_score:
                best, best_score, best_a = encode(c), score, c
        return score

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

    def step(track, floor):
        guard()
        if time.monotonic() >= lp_end:
            return False
        z = track['z']
        dx, dy = z[I]-z[J], z[N+I]-z[N+J]
        d = np.maximum(np.hypot(dx,dy),1e-15)
        nx, ny = dx/d, dy/d
        A[P,I], A[P,J] = -nx, nx
        A[P,N+I], A[P,N+J] = -ny, ny
        bounds = np.empty((3*N+1,2))
        bounds[:2*N,0] = z[:2*N]-trust
        bounds[:2*N,1] = z[:2*N]+trust
        bounds[2*N:3*N,0] = floor
        bounds[2*N:3*N,1] = 0.5
        bounds[-1] = [0.05,1.95]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = linprog(objective,A_ub=A,b_ub=rhs,bounds=bounds,method='highs',
                options={'threads':1,'time_limit':max(0.001,lp_end-time.monotonic()),
                         'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
        guard()
        if not result.success:
            track['stale'] += 1
            return False
        old = float(np.sum(z[2*N:3*N]))
        track['z'] = result.x
        track['score'] = consider(result.x)
        gain = float(np.sum(result.x[2*N:3*N]))-old
        track['stale'] = track['stale']+1 if abs(gain)<2e-10 else 0
        return True

    try:
        if raw is None or raw.shape != (N,3) or not np.isfinite(raw).all():
            raw = fallback
        tracks = [{'z':to_vector(raw),'score':-np.inf,'stale':0}]
        for counts in ([4,5,4,5,3], [5,4,3,4,5]):
            guard()
            rows = []
            for row, count in enumerate(counts):
                guard()
                xx = 2*(np.arange(count)-(count-1)/2)
                yy = np.full(count,row*np.sqrt(3.0))
                rows.append(np.column_stack((xx,yy,np.ones(count))))
            a = np.vstack(rows)
            a[:,:2] += rng.normal(0,0.012,(N,2))
            tracks.append({'z':to_vector(a),'score':-np.inf,'stale':0})

        floor = float(params['radius_floor'])
        for iteration in range(int(params['race_rounds'])):
            guard()
            if time.monotonic() >= lp_end:
                break
            for track in tracks:
                guard()
                if time.monotonic() >= lp_end:
                    break
                step(track,floor)

        tracks.sort(key=lambda t:t['score'],reverse=True)
        for iteration in range(180):
            guard()
            if time.monotonic() >= lp_end:
                break
            # The leader receives three steps for each challenger step.
            idx = 1 if iteration%4 == 3 and len(tracks)>1 and tracks[1]['stale']<3 else 0
            track = tracks[idx]
            floor_now = floor if iteration < 8 else 1e-7
            if track['stale'] >= 4 and floor_now == 1e-7:
                if all(t['stale']>=4 for t in tracks[:2]):
                    break
                track = tracks[1-idx]
            step(track,floor_now)
            tracks.sort(key=lambda t:t['score'],reverse=True)

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

        if best_a is not None and time.monotonic()<end-0.025:
            z = to_vector(best_a)
            result = minimize(fun,z,jac=lambda v:objective,method='SLSQP',
                bounds=[(None,None)]*(2*N)+[(1e-7,0.5)]*N+[(0.05,1.95)],
                constraints={'type':'ineq','fun':constraints,'jac':jacobian},
                callback=consider,
                options={'maxiter':int(params['polish_iterations']),'ftol':8e-13,'disp':False})
            consider(result.x)
    except Deadline:
        pass
    return best
