import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'iterations': (150, 40, 400),
    'jitter': (0.012, 0.0, 0.08),
    'restart_jitter': (0.025, 0.0, 0.1),
    'time_fraction': (0.70, 0.4, 0.75)
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
AR = np.arange(N)

class StopSearch(Exception):
    pass

def exact_score(c):
    try:
        a = np.asarray(c['circles'], dtype=float)
        if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -np.inf
        p, r = a[:, :2], a[:, 2]
        d = np.sqrt((p[I, 0]-p[J, 0])**2 + (p[I, 1]-p[J, 1])**2)
        if np.any(d < r[I]+r[J]):
            return -np.inf
        x = [Fraction(float(v)) for v in p[:, 0]]
        y = [Fraction(float(v)) for v in p[:, 1]]
        q = [Fraction(float(v)) for v in r]
        perimeter = (max(u+v for u,v in zip(x,q))-min(u-v for u,v in zip(x,q))
                     + max(u+v for u,v in zip(y,q))-min(u-v for u,v in zip(y,q)))
        if perimeter > 2:
            return -np.inf
        return float(np.sum(r))
    except (ValueError, KeyError, TypeError, OverflowError):
        return -np.inf

def repaired(z):
    p = np.column_stack((z[:N], z[N:2*N])).copy()
    r = np.maximum(z[2*N:3*N], 1e-9).copy()
    if not np.isfinite(p).all() or not np.isfinite(r).all():
        return None
    d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
    ratio = float(np.min(d/(r[I]+r[J])))
    if ratio <= 0:
        return None
    r *= min(1.0, ratio)*(1-2e-12)
    lo = np.min(p-r[:,None],axis=0)
    hi = np.max(p+r[:,None],axis=0)
    factor = (2.0/(hi-lo).sum())*(1-2e-12)
    p = (p-lo)*factor
    r *= factor
    return {'circles': np.column_stack((p,r)).tolist()}

def vector(c):
    a = np.array(c['circles'], dtype=float)
    p, r = a[:,:2], a[:,2]
    lo = (p-r[:,None]).min(axis=0)
    side = (p+r[:,None]).max(axis=0)-lo
    s = 2/side.sum()
    p = (p-lo)*s
    return np.r_[p[:,0],p[:,1],r*s,side[0]*s]

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + float(params['time_fraction'])*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = exact_score(init)
    if not np.isfinite(best_score):
        best = None
    def consider(z):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        c = repaired(z)
        if c is not None:
            score = exact_score(c)
            if score > best_score:
                best, best_score = c, score

    # A valid deterministic baseline, independent of the cold input.
    p = np.array([[(i+.5)/5, (j+.5)/5] for j in range(5) for i in range(5)])[:N]
    z = np.r_[p[:,0],p[:,1],np.full(N,.1),1.0]
    consider(z)
    bounds = [(0.0,2.0)]*(2*N)+[(1e-7,.7)]*N+[(.15,1.85)]
    objective_jac = np.r_[np.zeros(2*N),-np.ones(N),0.0]

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch()
    def objective(z):
        check()
        return -float(z[2*N:3*N].sum()), objective_jac
    def constraints(z):
        check()
        x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
        dx,dy = x[I]-x[J],y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        return np.r_[d-r[I]-r[J],x-r,w-x-r,y-r,2-w-y-r]
    def jacobian(z):
        check()
        x,y = z[:N],z[N:2*N]
        dx,dy = x[I]-x[J],y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        a = np.zeros((M+4*N,3*N+1))
        rows = np.arange(M)
        a[rows,I],a[rows,J] = dx/d,-dx/d
        a[rows,N+I],a[rows,N+J] = dy/d,-dy/d
        a[rows,2*N+I],a[rows,2*N+J] = -1,-1
        a[M+AR,AR] = 1
        a[M+N+AR,AR] = -1
        a[M+2*N+AR,N+AR] = 1
        a[M+3*N+AR,N+AR] = -1
        for k in range(4):
            if time.monotonic() >= deadline:
                raise StopSearch()
            a[M+k*N+AR,2*N+AR] = -1
        a[M+N+AR,-1] = 1
        a[M+3*N+AR,-1] = -1
        return a

    attempt = 0
    while time.monotonic() < deadline:
        if attempt == 0 and init is not None and exact_score(init) > 1.5:
            z = vector(init)
        elif attempt < 2:
            counts = [4,5,4,4,4] if attempt == 0 else [4,4,5,4,4]
            pts = []
            for row,count in enumerate(counts):
                if time.monotonic() >= deadline:
                    break
                for col in range(count):
                    if time.monotonic() >= deadline:
                        break
                    pts.append((2*col+1+(5-count),1+np.sqrt(3)*row))
            if len(pts) != N:
                break
            p = np.asarray(pts,dtype=float)
            r = np.ones(N)
            zz = np.r_[p[:,0],p[:,1],r,10.0]
            c = repaired(zz)
            z = vector(c)
            z[:2*N] += rng.normal(0,float(params['jitter']),2*N)
        else:
            z = vector(best)
            z[:2*N] += rng.normal(0,float(params['restart_jitter']),2*N)
        last = [z.copy()]
        counter = [0]
        def callback(q):
            check()
            last[0] = q.copy()
            counter[0] += 1
            if counter[0] % 8 == 0:
                consider(q)
        consider(z)
        try:
            result = minimize(objective,z,jac=True,method='SLSQP',bounds=bounds,
                              constraints={'type':'ineq','fun':constraints,'jac':jacobian},
                              callback=callback,
                              options={'maxiter':int(params['iterations']),'ftol':2e-10,'disp':False})
            consider(result.x)
        except StopSearch:
            break
        consider(last[0])
        attempt += 1
    return best
