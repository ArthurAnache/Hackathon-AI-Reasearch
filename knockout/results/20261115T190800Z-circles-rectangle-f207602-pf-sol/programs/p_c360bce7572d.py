import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'ftol': (2e-10, 1e-12, 1e-6, 'log'),
    'restart_noise': (0.018, 0.001, 0.08, 'log'),
    'maxiter': (170, 40, 400),
    'time_fraction': (0.69, 0.4, 0.72),
    'contact_margin': (0.055, 0.015, 0.2, 'log')
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class StopSearch(Exception):
    pass

class ExpandContacts(Exception):
    pass

def exact_score(c):
    try:
        a = np.asarray(c['circles'], dtype=float)
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
    except Exception:
        return -np.inf

def repair(z):
    p = np.column_stack((z[:N], z[N:2*N])).copy()
    r = np.maximum(z[2*N:3*N], 1e-9).copy()
    if not np.isfinite(p).all() or not np.isfinite(r).all():
        return None
    d = np.sqrt(np.sum((p[I]-p[J])**2, axis=1))
    ratio = min(1.0, float(np.min(d/(r[I]+r[J]))))
    if ratio <= 0:
        return None
    r *= ratio*(1-2e-13)
    low = np.min(p-r[:, None], axis=0)
    high = np.max(p+r[:, None], axis=0)
    scale = (2.0/(high-low).sum())*(1-2e-13)
    p = (p-low)*scale
    r *= scale
    return {'circles': np.column_stack((p,r)).tolist()}

def make_start():
    pts = []
    counts = [4,5,3,5,4]
    for row, count in enumerate(counts):
        if count == 5:
            xx = np.arange(5)*2+1
        elif count == 4:
            xx = np.arange(4)*2+2
        else:
            xx = np.array([2.0,5.0,8.0])
        pts.extend((float(x), 1+row*np.sqrt(3)) for x in xx)
    p = np.asarray(pts)
    scale = 2/(10+2+4*np.sqrt(3))
    return np.r_[p[:,0]*scale, p[:,1]*scale, np.full(N,scale), 10*scale]

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + float(params['time_fraction'])*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = exact_score(init) if init is not None else -np.inf
    best_z = None

    def consider(z):
        nonlocal best, best_score, best_z
        if time.monotonic() >= deadline:
            return
        c = repair(z)
        if c is not None:
            score = exact_score(c)
            if score > best_score:
                best, best_score = c, score
                a = np.asarray(c['circles'])
                lo = (a[:,:2]-a[:,2,None]).min(axis=0)
                p = a[:,:2]-lo
                width = float(np.max(p[:,0]+a[:,2]))
                best_z = np.r_[p[:,0],p[:,1],a[:,2],width]

    z = make_start()
    consider(z)
    if init is not None:
        try:
            a = np.asarray(init['circles'], dtype=float)
            if a.shape == (N,3) and np.isfinite(a).all() and np.all(a[:,2] > 0):
                lo = (a[:,:2]-a[:,2,None]).min(axis=0)
                p = a[:,:2]-lo
                z = np.r_[p[:,0],p[:,1],a[:,2],np.max(p[:,0]+a[:,2])]
                consider(z)
        except Exception:
            pass

    walls = np.zeros((4*N, 3*N+1))
    ix = np.arange(N)
    walls[ix,ix] = 1
    walls[ix,2*N+ix] = -1
    walls[N+ix,ix] = -1
    walls[N+ix,2*N+ix] = -1
    walls[N+ix,-1] = 1
    walls[2*N+ix,N+ix] = 1
    walls[2*N+ix,2*N+ix] = -1
    walls[3*N+ix,N+ix] = -1
    walls[3*N+ix,2*N+ix] = -1
    walls[3*N+ix,-1] = -1
    objective_grad = np.r_[np.zeros(2*N),-np.ones(N),0.0]
    cache_z = None
    cache_c = None
    cache_j = None
    last = z.copy()
    active = np.ones(M, dtype=bool)
    ai, aj = I, J
    margin = float(params['contact_margin'])

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def gaps(v):
        return np.hypot(v[I]-v[J], v[N+I]-v[N+J])-v[2*N+I]-v[2*N+J]

    def geometry(v):
        nonlocal cache_z, cache_c, cache_j
        check()
        if cache_z is not None and np.array_equal(v,cache_z):
            return cache_c,cache_j
        x,y,r,w = v[:N],v[N:2*N],v[2*N:3*N],v[-1]
        dx,dy = x[ai]-x[aj],y[ai]-y[aj]
        d = np.sqrt(dx*dx+dy*dy+1e-28)
        k = len(ai)
        c = np.r_[d-r[ai]-r[aj],x-r,w-x-r,y-r,2-w-y-r]
        jac = np.zeros((k+4*N,3*N+1))
        rows = np.arange(k)
        jac[rows,ai] = dx/d
        jac[rows,aj] = -dx/d
        jac[rows,N+ai] = dy/d
        jac[rows,N+aj] = -dy/d
        jac[rows,2*N+ai] = -1
        jac[rows,2*N+aj] = -1
        jac[k:] = walls
        cache_z,cache_c,cache_j = v.copy(),c,jac
        return c,jac

    def fun(v):
        check()
        return -float(np.sum(v[2*N:3*N]))

    calls = 0
    def callback(v):
        nonlocal calls,last,active
        check()
        last = v.copy()
        calls += 1
        entering = (~active) & (gaps(v) < margin*0.25)
        if np.any(entering):
            active |= entering
            consider(v)
            raise ExpandContacts()
        if calls % 8 == 0:
            consider(v)

    bounds = [(0,2)]*(2*N)+[(1e-7,0.49)]*N+[(0.15,1.85)]
    attempts = 0
    while time.monotonic() < deadline-0.04:
        active = gaps(z) < margin
        expansions = 0
        while time.monotonic() < deadline-0.025:
            ai, aj = I[active], J[active]
            cache_z = None
            last = z.copy()
            try:
                result = minimize(fun,z,jac=lambda v: objective_grad,method='SLSQP',
                    bounds=bounds,
                    constraints={'type':'ineq','fun':lambda v: geometry(v)[0],
                                 'jac':lambda v: geometry(v)[1]},
                    callback=callback,
                    options={'maxiter':int(params['maxiter']),'ftol':float(params['ftol']),'disp':False})
                consider(result.x)
                last = result.x.copy()
                entering = (~active) & (gaps(last) < margin*0.25)
                if np.any(entering):
                    active |= entering
                    z = last.copy()
                    continue
                break
            except ExpandContacts:
                expansions += 1
                z = last.copy()
                if expansions >= 8:
                    active[:] = True
                continue
            except StopSearch:
                return best
        attempts += 1
        if time.monotonic() >= deadline-0.12 or attempts >= 4:
            break
        z = best_z.copy() if best_z is not None else last.copy()
        z[:2*N] += rng.normal(0,float(params['restart_noise']),2*N)
        z[2*N:3*N] *= 0.99
    return best
