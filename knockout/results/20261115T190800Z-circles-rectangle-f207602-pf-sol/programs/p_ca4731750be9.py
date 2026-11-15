import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'maxiter': (180, 30, 400),
    'jitter': (0.012, 0.0, 0.06),
    'ftol': (2e-10, 1e-12, 1e-6, 'log'),
    'restart_noise': (0.025, 0.0, 0.1)
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
ROWS = np.arange(M)

class Deadline(Exception):
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
        wh = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        wh += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        if wh > 2:
            return -np.inf
        return float(np.sum(a[:, 2]))
    except Exception:
        return -np.inf

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = exact_score(init) if init is not None else -np.inf

    def consider(p, r):
        nonlocal best, best_score
        if time.monotonic() > end + 0.04*budget_s:
            return
        p = np.asarray(p, dtype=float).copy()
        r = np.maximum(np.asarray(r, dtype=float), 1e-9).copy()
        if not np.isfinite(p).all() or not np.isfinite(r).all():
            return
        d = np.sqrt((p[I, 0]-p[J, 0])**2+(p[I, 1]-p[J, 1])**2)
        ratio = float(np.min(d/(r[I]+r[J])))
        if ratio <= 0:
            return
        r *= min(1.0, ratio)*(1-3e-13)
        lo = np.min(p-r[:, None], axis=0)
        hi = np.max(p+r[:, None], axis=0)
        scale = (2.0/(hi-lo).sum())*(1-3e-13)
        p = (p-lo)*scale
        r *= scale
        cand = {'circles': np.column_stack((p,r)).tolist()}
        score = exact_score(cand)
        if score > best_score:
            best, best_score = cand, score

    # A deterministic, immediately valid fallback with substantial total radius.
    pts = []
    for row in range(5):
        if time.monotonic() > end:
            break
        for col in range(5):
            if time.monotonic() > end:
                break
            if row in (0,4) and col in (0,4):
                continue
            pts.append(((col+.5)/5, (row+.5)/5))
    if len(pts) == N:
        consider(np.array(pts), np.full(N, .1))

    wall_jac = np.zeros((4*N, 3*N+1))
    k = np.arange(N)
    wall_jac[k,k] = 1
    wall_jac[k,2*N+k] = -1
    wall_jac[N+k,k] = -1
    wall_jac[N+k,2*N+k] = -1
    wall_jac[N+k,-1] = 1
    wall_jac[2*N+k,N+k] = 1
    wall_jac[2*N+k,2*N+k] = -1
    wall_jac[3*N+k,N+k] = -1
    wall_jac[3*N+k,2*N+k] = -1
    wall_jac[3*N+k,-1] = -1
    objective_jac = np.r_[np.zeros(2*N), -np.ones(N), 0.0]

    def check():
        if time.monotonic() >= end:
            raise Deadline()

    def fun(z):
        check()
        return -float(z[2*N:3*N].sum())

    def con(z):
        check()
        x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
        dx,dy = x[I]-x[J],y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        return np.r_[d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r]

    def jac(z):
        check()
        x,y = z[:N], z[N:2*N]
        dx,dy = x[I]-x[J],y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        a = np.zeros((M+4*N, 3*N+1))
        a[ROWS,I] = dx/d
        a[ROWS,J] = -dx/d
        a[ROWS,N+I] = dy/d
        a[ROWS,N+J] = -dy/d
        a[ROWS,2*N+I] = -1
        a[ROWS,2*N+J] = -1
        a[M:] = wall_jac
        return a

    bounds = [(-1,2)]*(2*N)+[(1e-7,.6)]*N+[(.15,1.85)]
    attempt = 0
    while time.monotonic() < end:
        if attempt == 0 and best_score > 2.2:
            a = np.asarray(best['circles'])
            p,r = a[:,:2].copy(),a[:,2].copy()
        elif attempt < 3:
            counts = ([4,5,4,5,3], [5,4,5,4,3], [5,6,5,5])[attempt]
            pp = []
            nr = len(counts)
            for row, count in enumerate(counts):
                if time.monotonic() >= end:
                    break
                for col in range(count):
                    if time.monotonic() >= end:
                        break
                    pp.append(((col+.5+(0.25 if row%2 else 0))/max(counts), (row+.5)/nr))
            if len(pp) != N:
                break
            p = np.array(pp)
            p += rng.normal(0, float(params['jitter']), p.shape)
            r = np.full(N, .085)
        else:
            a = np.asarray(best['circles'])
            p,r = a[:,:2].copy(),a[:,2].copy()
            p += rng.normal(0,float(params['restart_noise']),p.shape)
        lo = (p-r[:,None]).min(axis=0)
        side = (p+r[:,None]).max(axis=0)-lo
        scale = 2/side.sum()
        p = (p-lo)*scale
        r *= scale
        z = np.r_[p[:,0],p[:,1],r,side[0]*scale]
        last = [z.copy()]
        counter = [0]
        def callback(v):
            check()
            last[0] = v.copy()
            counter[0] += 1
            if counter[0] % 12 == 0:
                consider(np.column_stack((v[:N],v[N:2*N])),v[2*N:3*N])
        try:
            res = minimize(fun,z,jac=lambda v: objective_jac,method='SLSQP',bounds=bounds,
                           constraints={'type':'ineq','fun':con,'jac':jac},callback=callback,
                           options={'maxiter':int(params['maxiter']), 'ftol':float(params['ftol']), 'disp':False})
            last[0] = res.x
        except Deadline:
            pass
        v = last[0]
        consider(np.column_stack((v[:N],v[N:2*N])),v[2*N:3*N])
        attempt += 1
    return best
