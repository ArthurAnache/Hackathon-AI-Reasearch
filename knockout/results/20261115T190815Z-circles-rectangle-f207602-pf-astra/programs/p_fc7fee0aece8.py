import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    "first_share": (0.43, 0.25, 0.55),
    "jitter": (0.004, 0.0, 0.025),
    "maxiter": (220, 40, 500),
    "ftol": (2e-10, 1e-12, 1e-6, "log")
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
K = np.arange(N)

class StopSearch(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0] - a[J, 0])**2 + (a[I, 1] - a[J, 1])**2)
    if np.any(d < a[I, 2] + a[J, 2]):
        return -np.inf
    x = list(map(lambda v: Fraction(float(v)), a[:, 0]))
    y = list(map(lambda v: Fraction(float(v)), a[:, 1]))
    r = list(map(lambda v: Fraction(float(v)), a[:, 2]))
    w = max(u+v for u,v in zip(x,r)) - min(u-v for u,v in zip(x,r))
    h = max(u+v for u,v in zip(y,r)) - min(u-v for u,v in zip(y,r))
    if w+h > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))

def repair(z):
    p = np.column_stack((z[:N], z[N:2*N]))
    r = np.maximum(z[2*N:3*N], 1e-12).copy()
    if not np.isfinite(p).all() or not np.isfinite(r).all():
        return None
    d = np.sqrt((p[I,0]-p[J,0])**2 + (p[I,1]-p[J,1])**2)
    ratio = min(1.0, float(np.min(d/(r[I]+r[J]))))
    if ratio <= 0:
        return None
    r *= ratio * (1.0-3e-14)
    lo = np.min(p-r[:,None], axis=0)
    hi = np.max(p+r[:,None], axis=0)
    scale = (2.0-2e-13)/float(np.sum(hi-lo))
    return np.column_stack(((p-lo)*scale, r*scale))

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + 0.70*budget_s
    rng = np.random.default_rng(seed)
    best = None
    best_score = -np.inf
    original = None
    if init is not None:
        try:
            original = np.asarray(init['circles'], dtype=float)
            score = exact_score(original)
            if score > best_score:
                best, best_score = init, score
        except (ValueError, TypeError, KeyError, OverflowError):
            original = None

    # Twenty equal grid circles plus a smaller circle in an interior void.
    gx, gy = np.meshgrid(np.arange(5), np.arange(4))
    p = np.column_stack(((2*gx.ravel()+1)/9, (2*gy.ravel()+1)/9))
    p = np.vstack((p, [4/9, 4/9]))
    radii = np.r_[np.full(20, 1/9), (np.sqrt(2)-1)/9]
    zgrid = np.r_[p[:,0], p[:,1], radii, 10/9]

    def retain(z):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        a = repair(z)
        if a is not None and float(a[:,2].sum()) > best_score:
            score = exact_score(a)
            if score > best_score:
                best_score = score
                best = {'circles': a.tolist()}

    retain(zgrid)
    starts = [zgrid.copy()]
    if original is not None and original.shape == (21,3) and np.isfinite(original).all():
        rr = np.maximum(np.abs(original[:,2]), 1e-6)
        lo = np.min(original[:,:2]-rr[:,None], axis=0)
        side = np.max(original[:,:2]+rr[:,None], axis=0)-lo
        if side.sum() > 0:
            scale = 2/side.sum()
            pp = (original[:,:2]-lo)*scale
            if best_score < 1.0 or np.sum(rr)*scale < 1.0:
                rr = np.full(N, 0.085)
            else:
                rr = rr*scale
            starts.append(np.r_[pp[:,0], pp[:,1], rr, side[0]*scale])
    if len(starts) == 1:
        zz = zgrid.copy()
        zz[:2*N] += rng.normal(0, 0.035, 2*N)
        starts.append(zz)

    objective_grad = np.r_[np.zeros(2*N), -np.ones(N), 0.0]
    bounds = [(None,None)]*(2*N) + [(1e-8,0.49)]*N + [(0.15,1.85)]
    base = np.zeros((M+4*N, 3*N+1))
    base[np.arange(M),2*N+I] = -1
    base[np.arange(M),2*N+J] = -1
    base[M+K,K] = 1
    base[M+K,2*N+K] = -1
    base[M+N+K,K] = -1
    base[M+N+K,2*N+K] = -1
    base[M+N+K,-1] = 1
    base[M+2*N+K,N+K] = 1
    base[M+2*N+K,2*N+K] = -1
    base[M+3*N+K,N+K] = -1
    base[M+3*N+K,2*N+K] = -1
    base[M+3*N+K,-1] = -1

    for attempt, z in enumerate(starts):
        if time.monotonic() >= deadline:
            break
        local_end = min(deadline, started+float(params['first_share'])*budget_s) if attempt == 0 else deadline
        if local_end-time.monotonic() < 0.025:
            continue
        z = z.copy()
        z[:2*N] += rng.normal(0, float(params['jitter']), 2*N)
        cache_z = None
        cache_c = None
        cache_j = None

        def check():
            if time.monotonic() >= local_end:
                raise StopSearch()

        def objective(v):
            check()
            return -float(np.sum(v[2*N:3*N])), objective_grad

        def calculate(v):
            nonlocal cache_z, cache_c, cache_j
            check()
            if cache_z is not None and np.array_equal(v,cache_z):
                return
            x,y,r,w = v[:N],v[N:2*N],v[2*N:3*N],v[-1]
            dx,dy = x[I]-x[J],y[I]-y[J]
            d = np.sqrt(dx*dx+dy*dy)
            safe = np.maximum(d,1e-15)
            cache_c = np.r_[d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r]
            jac = base.copy()
            q = np.arange(M)
            jac[q,I] = dx/safe
            jac[q,J] = -dx/safe
            jac[q,N+I] = dy/safe
            jac[q,N+J] = -dy/safe
            cache_z,cache_j = v.copy(),jac

        def constraints(v):
            calculate(v)
            return cache_c

        def jacobian(v):
            calculate(v)
            return cache_j

        def callback(v):
            check()
            retain(v)

        try:
            result = minimize(objective,z,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':constraints,'jac':jacobian},
                callback=callback,options={'maxiter':int(params['maxiter']),
                'ftol':float(params['ftol']),'disp':False})
            retain(result.x)
        except StopSearch:
            pass
    if best is None:
        # The grid is independently feasible, even if the input was malformed.
        a = repair(zgrid)
        if exact_score(a) > -np.inf:
            best = {'circles':a.tolist()}
        else:
            a[:,2] *= 1-1e-10
            best = {'circles':a.tolist()}
    return best
