import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'initial_share': (0.38, 0.15, 0.7),
    'contact_margin': (0.085, 0.025, 0.25),
    'stage_iterations': (22, 8, 60),
    'perturbation': (0.009, 0.001, 0.05, 'log'),
    'safety': (4e-14, 1e-14, 1e-10, 'log')
}
N = 21
I, J = np.triu_indices(N, 1)
Q = np.arange(N)
G = np.r_[np.zeros(42), -np.ones(21), 0.0]

class OutOfTime(Exception):
    pass

def exact(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -float('inf')
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return -float('inf')
    x = [Fraction(float(t)) for t in a[:, 0]]
    y = [Fraction(float(t)) for t in a[:, 1]]
    r = [Fraction(float(t)) for t in a[:, 2]]
    p = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
    p += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
    if p > 2:
        return -float('inf')
    return float(np.sum(a[:, 2]))

def decode(z):
    return np.column_stack((z[:21], z[21:42], z[42:63]))

def encode(a):
    a = a.copy()
    lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
    hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
    a[:, :2] -= lo
    wh = hi-lo
    s = 2.0/float(wh.sum())
    a *= s
    return np.r_[a[:, 0], a[:, 1], a[:, 2], wh[0]*s]

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    best = None
    best_a = None
    best_score = -float('inf')
    safety = float(params['safety'])

    def retain(a, repair=True):
        nonlocal best, best_a, best_score
        if time.monotonic() >= deadline:
            return
        a = np.asarray(a, dtype=float).copy()
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        if repair:
            d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
            f = min(1.0, float(np.min(d/(a[I,2]+a[J,2]))))
            if f <= 0:
                return
            a[:,2] *= f*(1-safety)
            lo = np.min(a[:,:2]-a[:,2,None], axis=0)
            hi = np.max(a[:,:2]+a[:,2,None], axis=0)
            a[:,:2] -= (lo+hi)/2
            a *= (2.0/float(np.sum(hi-lo)))*(1-safety)
        if float(np.sum(a[:,2])) <= best_score:
            return
        s = exact(a)
        if s > best_score:
            best_score, best_a = s, a.copy()
            best = {'circles': a.tolist()}

    initial = None
    if init is not None:
        try:
            a = np.asarray(init['circles'], dtype=float)
            score = exact(a)
            if score > best_score:
                best, best_a, best_score = init, a.copy(), score
            if a.shape == (21,3) and np.isfinite(a).all() and np.all(a[:,2] > 0):
                initial = a.copy()
                retain(a)
        except (ValueError, TypeError, KeyError, OverflowError):
            pass

    grid = np.array([[2*c+1., 2*r+1., 1.] for r in range(4) for c in range(5)]
                    + [[4., 4., np.sqrt(2.)-1.]])/9.0
    retain(grid)
    if initial is None:
        initial = grid.copy()
    bounds = [(None,None)]*42 + [(1e-8,0.95)]*21 + [(0.08,1.92)]

    def polish(z, end):
        last = z.copy()
        rounds = 0
        while time.monotonic() < end-0.008:
            dx = last[I]-last[J]
            dy = last[21+I]-last[21+J]
            gaps = np.sqrt(dx*dx+dy*dy)-last[42+I]-last[42+J]
            # Include all potential imminent contacts, and always recheck all pairs.
            ids = np.flatnonzero(gaps < float(params['contact_margin']))
            if len(ids) < 50:
                ids = np.argsort(gaps)[:65]
            ii, jj = I[ids], J[ids]
            m = len(ii)
            kk = np.arange(m)
            def check():
                if time.monotonic() >= end:
                    raise OutOfTime()
            def fun(v):
                check()
                return -float(v[42:63].sum())
            def con(v):
                check()
                x,y,r,w = v[:21],v[21:42],v[42:63],v[63]
                d = np.sqrt((x[ii]-x[jj])**2+(y[ii]-y[jj])**2+1e-30)
                return np.r_[d-r[ii]-r[jj],x-r,w-x-r,y-r,2-w-y-r]
            def jac(v):
                check()
                dx,dy = v[ii]-v[jj],v[21+ii]-v[21+jj]
                d = np.sqrt(dx*dx+dy*dy+1e-30)
                a = np.zeros((m+84,64))
                a[kk,ii],a[kk,jj] = dx/d,-dx/d
                a[kk,21+ii],a[kk,21+jj] = dy/d,-dy/d
                a[kk,42+ii],a[kk,42+jj] = -1,-1
                a[m+Q,Q] = 1
                a[m+21+Q,Q] = -1
                a[m+42+Q,21+Q] = 1
                a[m+63+Q,21+Q] = -1
                a[m+Q,42+Q] = -1
                a[m+21+Q,42+Q] = -1
                a[m+42+Q,42+Q] = -1
                a[m+63+Q,42+Q] = -1
                a[m+21:m+42,63] = 1
                a[m+63:m+84,63] = -1
                return a
            def callback(v):
                nonlocal last
                check()
                last = v.copy()
                retain(decode(v))
            old = last.copy()
            try:
                result = minimize(fun,last,jac=lambda v:G,method='SLSQP',bounds=bounds,
                    constraints={'type':'ineq','fun':con,'jac':jac},callback=callback,
                    options={'maxiter':int(params['stage_iterations']),'ftol':1e-12})
                last = result.x.copy()
            except OutOfTime:
                break
            retain(decode(last))
            rounds += 1
            if np.max(np.abs(last-old)) < 2e-9:
                break
        retain(decode(last))
        return last

    # Sparse cold starts get a bounded recovery window. Strong supplied solutions
    # keep almost the whole first trajectory instead of being replaced by a lattice.
    strong = best_score > 2.30
    first_fraction = 0.72 if strong else float(params['initial_share'])
    polish(encode(initial), min(deadline, start+0.70*budget_s*first_fraction))
    if time.monotonic() < deadline-0.05:
        target = best_a.copy() if strong else grid.copy()
        end = time.monotonic()+0.78*(deadline-time.monotonic())
        polish(encode(target), end)
    attempt = 0
    while time.monotonic() < deadline-0.035:
        a = best_a.copy() if best_a is not None else grid.copy()
        a[:,:2] += rng.normal(0,float(params['perturbation'])*(1.7**attempt),(21,2))
        polish(encode(a), deadline)
        attempt += 1
        if attempt >= 2:
            break
    if best is not None:
        return best
    grid[:,2] *= 1-1e-10
    grid *= 1-1e-10
    return {'circles':grid.tolist()}
