import time
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize

PARAMS = {
    'trust': (0.18, 0.04, 0.4),
    'screen_steps': (16, 6, 35),
    'refine_steps': (45, 15, 100),
    'lp_fraction': (0.48, 0.3, 0.58),
    'polish_steps': (90, 20, 180)
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
K = np.arange(N)

class Deadline(Exception):
    pass

def score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return -np.inf
    x = list(map(Fraction, map(float, a[:, 0])))
    y = list(map(Fraction, map(float, a[:, 1])))
    r = list(map(Fraction, map(float, a[:, 2])))
    p = (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
         +max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)))
    return float(np.sum(a[:, 2])) if p <= 2 else -np.inf

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70*budget_s
    lp_end = start + float(params['lp_fraction'])*budget_s
    rng = np.random.default_rng(seed)
    best = None
    best_score = -np.inf
    raw = None

    def guard():
        if time.monotonic() >= end:
            raise Deadline()

    def encode(a):
        return {'coordinate_rows': a.T.tolist()}

    if init is not None:
        try:
            raw = np.asarray(init['circles'] if 'circles' in init else np.asarray(init['coordinate_rows']).T, dtype=float)
            s = score(raw)
            if s > best_score:
                best, best_score = init, s
        except (ValueError, KeyError, TypeError, OverflowError):
            raw = None
    ids = np.arange(N)
    fallback = np.column_stack(((ids%5+0.5)*0.2, (ids//5+0.5)*0.2, np.full(N, 0.1*(1-1e-12))))
    s = score(fallback)
    if s > best_score:
        best, best_score = encode(fallback), s

    B = np.zeros((4*N, 3*N+1))
    B[K,K] = -1
    B[K,2*N+K] = 1
    B[N+K,K] = 1
    B[N+K,2*N+K] = 1
    B[N+K,-1] = -1
    B[2*N+K,N+K] = -1
    B[2*N+K,2*N+K] = 1
    B[2*N+K,-1] = 0.5
    B[3*N+K,N+K] = 1
    B[3*N+K,2*N+K] = 1
    B[3*N+K,-1] = 0.5
    bb = np.r_[np.zeros(2*N), np.ones(2*N)]
    rhs = np.r_[np.zeros(M), bb]

    def consider(full):
        nonlocal best, best_score
        guard()
        a = np.column_stack((full[:N], full[N:2*N], np.maximum(full[2*N:3*N], 1e-12)))
        d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
        f = min(1.0, float(np.min(d/(a[I,2]+a[J,2]))))
        if f <= 0 or not np.isfinite(f):
            return
        a[:,2] *= f*(1-3e-12)
        lo = np.min(a[:,:2]-a[:,2,None], axis=0)
        hi = np.max(a[:,:2]+a[:,2,None], axis=0)
        span = float(np.sum(hi-lo))
        if span <= 0:
            return
        a[:,:2] -= (lo+hi)*0.5
        a *= (2.0/span)*(1-3e-12)
        if float(np.sum(a[:,2])) > best_score:
            s = score(a)
            if s > best_score:
                best_score, best = s, encode(a)

    def model(T, v, positions, radius_indices):
        return {'T':T, 'v':v, 'dx':T[I]-T[J], 'dy':T[N+I]-T[N+J],
                'rr':T[2*N+I]+T[2*N+J], 'B':B@T,
                'obj':-np.sum(T[2*N:3*N], axis=0),
                'positions':positions, 'radii':radius_indices,
                'stale':0, 'value':-np.inf}

    def unrestricted(a):
        a = a.copy()
        a[:,2] = np.maximum(np.abs(a[:,2]), 1e-6)
        lo = np.min(a[:,:2]-a[:,2,None], axis=0)
        hi = np.max(a[:,:2]+a[:,2,None], axis=0)
        sides = hi-lo
        scale = 2.0/max(float(np.sum(sides)), 1e-12)
        a[:,0] = (a[:,0]-lo[0])*scale
        a[:,1] = (a[:,1]-(lo[1]+hi[1])*0.5)*scale
        a[:,2] *= scale
        v = np.r_[a[:,0], a[:,1], a[:,2], sides[0]*scale]
        return model(np.eye(64), v, np.arange(42), np.arange(42,63))

    def symmetric(counts, jitter=0.0):
        T = np.zeros((64,34))
        xid = np.r_[np.arange(12),np.arange(3,12)]
        rid = np.r_[np.arange(21,33),np.arange(24,33)]
        T[K,xid] = 1
        T[2*N+K,rid] = 1
        T[N+np.arange(3,12),np.arange(12,21)] = 1
        T[N+np.arange(12,21),np.arange(12,21)] = -1
        T[-1,-1] = 1
        v = np.zeros(34)
        v[:3] = [0.17,0.5,0.83]
        off = 3
        for row, count in enumerate(counts):
            guard()
            xs = (np.arange(count)+0.5)/count
            v[off:off+count] = xs
            v[12+off-3:12+off-3+count] = 0.43*(row+1)/len(counts)
            off += count
        if jitter:
            v[:12] += rng.normal(0,jitter,12)
            v[12:21] += rng.normal(0,jitter,9)
        v[21:33] = 0.01
        v[-1] = 1.0
        return model(T,v,np.arange(21),np.arange(21,33))

    def advance(m, steps, stop):
        for iteration in range(steps):
            guard()
            if time.monotonic() >= stop or m['stale'] >= 3:
                break
            v = m['v']
            dx, dy = m['dx']@v, m['dy']@v
            dist = np.maximum(np.hypot(dx,dy),1e-15)
            A = np.vstack((m['rr']-(dx/dist)[:,None]*m['dx']-(dy/dist)[:,None]*m['dy'],m['B']))
            bounds = np.empty((len(v),2))
            bounds[:,0], bounds[:,1] = -2.0, 2.0
            pos = m['positions']
            trust = float(params['trust'])
            bounds[pos,0] = v[pos]-trust
            bounds[pos,1] = v[pos]+trust
            bounds[m['radii'],0] = 1e-7
            bounds[m['radii'],1] = 0.5
            bounds[-1] = [0.08,1.92]
            remaining = stop-time.monotonic()
            if remaining <= 0.004:
                break
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res = linprog(m['obj'], A_ub=A, b_ub=rhs, bounds=bounds, method='highs',
                    options={'threads':1,'time_limit':max(0.001,remaining),
                             'primal_feasibility_tolerance':1e-9,
                             'dual_feasibility_tolerance':1e-9})
            guard()
            if res.x is None or not res.success:
                break
            old = -float(m['obj']@v)
            m['v'] = res.x
            value = -float(res.fun)
            m['value'] = value
            m['stale'] = m['stale']+1 if value-old < 2e-10 else 0
            consider(m['T']@res.x)

    try:
        models = []
        if raw is not None and raw.shape == (N,3) and np.isfinite(raw).all():
            models.append(unrestricted(raw))
        else:
            models.append(unrestricted(fallback))
        models.append(symmetric((5,4)))
        models.append(symmetric((4,5)))
        models.append(symmetric((3,3,3),0.012))
        screening_end = start + 0.29*budget_s
        for j,m in enumerate(models):
            guard()
            now = time.monotonic()
            if now >= screening_end:
                break
            stop = now+(screening_end-now)/(len(models)-j)
            advance(m,int(params['screen_steps']),stop)
        models.sort(key=lambda m:m['value'], reverse=True)
        for j,m in enumerate(models[:2]):
            guard()
            now = time.monotonic()
            if now >= lp_end:
                break
            stop = lp_end if j else now+0.65*(lp_end-now)
            advance(m,int(params['refine_steps']),stop)
        models.sort(key=lambda m:m['value'], reverse=True)
        m = models[0]
        if time.monotonic() < end-0.035:
            # Release symmetry during the final Euclidean polish.
            full = m['T']@m['v']
            m = model(np.eye(64),full,np.arange(42),np.arange(42,63))
            def fun(v):
                guard()
                return float(m['obj']@v)
            def con(v):
                guard()
                dx,dy = m['dx']@v,m['dy']@v
                return np.r_[np.hypot(dx,dy)-m['rr']@v,bb-m['B']@v]
            def jac(v):
                guard()
                dx,dy = m['dx']@v,m['dy']@v
                d = np.maximum(np.hypot(dx,dy),1e-15)
                return np.vstack(((dx/d)[:,None]*m['dx']+(dy/d)[:,None]*m['dy']-m['rr'],-m['B']))
            def callback(v):
                consider(v)
            res = minimize(fun,full,jac=lambda v:m['obj'],method='SLSQP',
                bounds=[(None,None)]*42+[(1e-7,0.5)]*21+[(0.08,1.92)],
                constraints={'type':'ineq','fun':con,'jac':jac},callback=callback,
                options={'ftol':2e-12,'maxiter':int(params['polish_steps']),'disp':False})
            consider(res.x)
    except Deadline:
        pass
    return best
