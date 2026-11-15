import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import lstsq

PARAMS = {
    'time_fraction': (0.69, 0.4, 0.7),
    'first_fraction': (0.32, 0.15, 0.6),
    'jitter': (0.035, 0.002, 0.1, 'log'),
    'stage_iterations': (40, 15, 120),
    'contact_margin': (0.045, 0.01, 0.2)
}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + float(params['time_fraction']) * budget_s
    run_deadline = deadline
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    ids = np.arange(n)
    rr = np.arange(len(ii))
    best = init
    best_score = -float('inf')
    best_a = None

    def check():
        if time.monotonic() >= run_deadline:
            raise StopSearch()

    def exact(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        d = a[ii, :2] - a[jj, :2]
        if np.any(np.sqrt(d[:, 0]**2 + d[:, 1]**2) < a[ii, 2] + a[jj, 2]):
            return -float('inf')
        x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
        extent = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        extent += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        if extent > 2:
            return -float('inf')
        return float(np.sum(a[:, 2]))

    if init is not None:
        try:
            a = np.asarray(init['circles'], dtype=float)
            best_score = exact(a)
            if np.isfinite(best_score):
                best_a = a.copy()
        except (ValueError, TypeError, KeyError, OverflowError):
            pass

    def normalize(a):
        a = a.copy()
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        wh = hi-lo
        scale = 2.0 / float(np.sum(wh))
        a[:, :2] -= lo
        a *= scale
        return a, float(wh[0]*scale)

    def offer(a):
        nonlocal best, best_score, best_a
        if time.monotonic() >= deadline:
            return
        a = np.asarray(a, dtype=float).reshape(21, 3).copy()
        if not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return
        delta = a[ii, :2]-a[jj, :2]
        ratio = float(np.min(np.sqrt(delta[:, 0]**2+delta[:, 1]**2)/(a[ii, 2]+a[jj, 2])))
        if ratio <= 0:
            return
        a[:, 2] *= min(1.0, ratio)*(1-3e-15)
        a, _ = normalize(a)
        a *= 1-3e-15
        if float(np.sum(a[:, 2])) <= best_score:
            return
        value = exact(a)
        if value > best_score:
            best_score, best_a = value, a.copy()
            best = {'circles': a.tolist()}

    grid = np.array([[0.1+0.2*x, 0.1+0.2*y, 0.1] for y in range(5) for x in range(5) if (x,y) not in ((0,0),(0,4),(4,0),(4,4))])
    if best_a is None:
        offer(grid)
    if best_a is None:
        return best

    B = np.zeros((84, 64))
    B[ids, ids] = 1
    B[ids, 42+ids] = -1
    B[21+ids, ids] = -1
    B[21+ids, 42+ids] = -1
    B[21+ids, 63] = 1
    B[42+ids, 21+ids] = 1
    B[42+ids, 42+ids] = -1
    B[63+ids, 21+ids] = -1
    B[63+ids, 42+ids] = -1
    B[63+ids, 63] = -1
    bc = np.zeros(84)
    bc[63:] = 2
    grad = np.zeros(64)
    grad[42:63] = -1
    bounds = [(None,None)]*42+[(1e-8,0.65)]*21+[(0.15,1.85)]

    def pack(a):
        a, w = normalize(a)
        return np.r_[a[:,0], a[:,1], a[:,2], w]

    def unpack(z):
        return np.column_stack((z[:21], z[21:42], z[42:63]))

    def values(z):
        check()
        dx, dy = z[ii]-z[jj], z[21+ii]-z[21+jj]
        return np.r_[B@z+bc, np.sqrt(dx*dx+dy*dy)-z[42+ii]-z[42+jj]]

    def jac(z):
        check()
        dx, dy = z[ii]-z[jj], z[21+ii]-z[21+jj]
        d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-20)
        j = np.zeros((len(ii),64))
        j[rr,ii], j[rr,jj] = dx/d, -dx/d
        j[rr,21+ii], j[rr,21+jj] = dy/d, -dy/d
        j[rr,42+ii], j[rr,42+jj] = -1, -1
        return np.vstack((B,j))

    def polish(a):
        z = pack(a)
        v = values(z)
        active = np.flatnonzero(v < 2e-8)
        if len(active) < 64 or len(active) > 70:
            return
        for iteration in range(7):
            check()
            v = values(z)[active]
            j = jac(z)[active]
            step, _, rank, _ = lstsq(j, -v, cond=1e-12, lapack_driver='gelsy')
            if rank < 64 or np.max(np.abs(step)) > 0.025:
                break
            z += step
            offer(unpack(z))
            if np.max(np.abs(step)) < 2e-15:
                break

    def objective(z):
        check()
        return -float(np.sum(z[42:63]))

    def callback(z):
        check()
        offer(unpack(z))

    def optimize(a, seconds, full=False):
        nonlocal run_deadline
        run_deadline = min(deadline, time.monotonic()+seconds)
        z = pack(a)
        try:
            for stage in range(3):
                check()
                if full or stage == 2:
                    selected = np.arange(294)
                else:
                    v = values(z)
                    mask = v[84:] < float(params['contact_margin'])
                    for k in range(21):
                        check()
                        edges = np.flatnonzero((ii == k)|(jj == k))
                        mask[edges[np.argsort(v[84:][edges])[:5]]] = True
                    selected = np.r_[np.arange(84), 84+np.flatnonzero(mask)]
                result = minimize(objective, z, jac=lambda z: grad, method='SLSQP', bounds=bounds,
                    constraints={'type':'ineq', 'fun':lambda z: values(z)[selected], 'jac':lambda z: jac(z)[selected]},
                    callback=callback, options={'maxiter':int(params['stage_iterations']), 'ftol':8e-15, 'disp':False})
                z = result.x
                offer(unpack(z))
                if full:
                    break
            polish(unpack(z))
        except (StopSearch, ValueError, FloatingPointError, np.linalg.LinAlgError):
            pass
        run_deadline = deadline

    try:
        polish(best_a)
    except (StopSearch, ValueError, np.linalg.LinAlgError):
        pass
    optimize(best_a, float(params['first_fraction'])*budget_s, True)
    attempt = 0
    while time.monotonic() < deadline-0.035:
        a = best_a.copy()
        amp = float(params['jitter'])*(1+0.5*(attempt % 3))
        a[:, :2] += rng.normal(0, amp, (21,2))
        a[:, 2] *= rng.uniform(0.97, 1.03, 21)
        optimize(a, min(0.28*budget_s, deadline-time.monotonic()), False)
        attempt += 1
    return best
