import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'contact_band': (0.075, 0.02, 0.2),
    'restarts': (5, 1, 12),
    'tilt': (0.34, 0.02, 0.9),
    'jitter': (0.025, 0.001, 0.12),
    'iterations': (100, 20, 250),
}

I, J = np.triu_indices(21, 1)

class Timeout(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -float('inf')
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return -float('inf')
    x = list(map(lambda v: Fraction(float(v)), a[:, 0]))
    y = list(map(lambda v: Fraction(float(v)), a[:, 1]))
    r = list(map(lambda v: Fraction(float(v)), a[:, 2]))
    width = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
    height = max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
    if width+height > 2:
        return -float('inf')
    return float(np.sum(a[:, 2]))

def construct(params, seed, budget_s, init=None):
    begin = time.monotonic()
    end = begin + 0.69*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -float('inf')
    if init is not None:
        base = np.asarray(init['circles'], dtype=float)
        best_score = exact_score(base)
    else:
        base = np.array([[(i+.5)*.2,(j+.5)*.2,.1] for j in range(7) for i in range(3)])

    def offer(a):
        nonlocal best, best_score
        if time.monotonic() >= end:
            return
        a = np.asarray(a, dtype=float).copy()
        if not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        a[:,:2] -= np.mean(a[:,:2], axis=0)
        d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
        factor = min(1.0, float(np.min(d/(a[I,2]+a[J,2]))))
        if factor <= 0:
            return
        a[:,2] *= factor*(1-3e-15)
        lo = np.min(a[:,:2]-a[:,2,None], axis=0)
        hi = np.max(a[:,:2]+a[:,2,None], axis=0)
        a *= (2.0-1.5e-14)/float(np.sum(hi-lo))
        score = exact_score(a)
        if score > best_score:
            best_score = score
            best = {'circles':a.tolist()}

    offer(base)
    boundary = np.zeros((84,64))
    k = np.arange(21)
    boundary[k,k] = 1
    boundary[k,42+k] = -1
    boundary[21+k,k] = -1
    boundary[21+k,42+k] = -1
    boundary[21+k,63] = 1
    boundary[42+k,21+k] = 1
    boundary[42+k,42+k] = -1
    boundary[63+k,21+k] = -1
    boundary[63+k,42+k] = -1
    boundary[63+k,63] = -1
    offset = np.r_[np.zeros(63),np.full(21,2.0)]
    objective = np.r_[np.zeros(42),-np.ones(21),0.0]
    bounds = [(None,None)]*42 + [(1e-8,.6)]*21 + [(.15,1.85)]

    def encode(a):
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        p = a[:,:2]-lo
        return np.r_[p[:,0],p[:,1],a[:,2],hi[0]-lo[0]]

    def decode(z):
        return np.column_stack((z[:21],z[21:42],z[42:63]))

    def solve(z, deadline, obj, maxiter):
        if time.monotonic() >= deadline:
            return z
        dx = z[I]-z[J]
        dy = z[21+I]-z[21+J]
        gaps = np.hypot(dx,dy)-z[42+I]-z[42+J]
        selected = gaps < float(params['contact_band'])
        pi,pj = I[selected],J[selected]
        m = len(pi)
        rr = np.arange(m)
        cache = [None,None,None]
        last = [z.copy()]
        def cj(v):
            if time.monotonic() >= deadline:
                raise Timeout()
            if cache[0] is not None and np.array_equal(v,cache[0]):
                return cache[1],cache[2]
            dx = v[pi]-v[pj]
            dy = v[21+pi]-v[21+pj]
            dist = np.maximum(np.hypot(dx,dy),1e-25)
            jac = np.zeros((m+84,64))
            jac[rr,pi] = dx/dist
            jac[rr,pj] = -dx/dist
            jac[rr,21+pi] = dy/dist
            jac[rr,21+pj] = -dy/dist
            jac[rr,42+pi] = -1
            jac[rr,42+pj] = -1
            jac[m:] = boundary
            c = np.r_[dist-v[42+pi]-v[42+pj],boundary@v+offset]
            cache[:] = [v.copy(),c,jac]
            return c,jac
        def fun(v):
            if time.monotonic() >= deadline:
                raise Timeout()
            return float(obj@v),obj
        def callback(v):
            if time.monotonic() >= deadline:
                raise Timeout()
            last[0] = v.copy()
        try:
            res = minimize(fun,z,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda v:cj(v)[0], 'jac':lambda v:cj(v)[1]},
                callback=callback,options={'maxiter':maxiter,'ftol':8e-14,'disp':False})
            return res.x
        except Timeout:
            return last[0]

    z0 = encode(base)
    first_end = min(end-.035,begin+.23*budget_s)
    if time.monotonic() < first_end:
        z = solve(z0,first_end,objective,int(params['iterations']))
        offer(decode(z))
    if best is not None:
        z0 = encode(np.asarray(best['circles'],dtype=float))
    total = int(params['restarts'])
    for attempt in range(total):
        now = time.monotonic()
        if now >= end-.045:
            break
        deadline = min(end-.012,now+(end-.02-now)/(total-attempt))
        z = z0.copy()
        # A local cluster is moved together; the tilted radius objective changes
        # several contacts simultaneously rather than releasing one edge.
        a = decode(z)
        pivot = int(rng.integers(21))
        dd = np.sum((a[:,:2]-a[pivot,:2])**2,axis=1)
        ids = np.argsort(dd)[:int(rng.integers(4,9))]
        amplitude = float(params['jitter'])*(1+.4*attempt)
        displacement = rng.normal(0,amplitude,2)
        z[ids] += displacement[0] + rng.normal(0,amplitude*.3,len(ids))
        z[21+ids] += displacement[1] + rng.normal(0,amplitude*.3,len(ids))
        tilt = objective.copy()
        weights = rng.normal(0,1,21)
        weights -= weights.mean()
        tilt[42:63] -= float(params['tilt'])*weights
        tilted_end = min(deadline-.012,time.monotonic()+max(.005,(deadline-time.monotonic())*.32))
        if time.monotonic() < tilted_end:
            z = solve(z,tilted_end,tilt,18)
        if time.monotonic() < deadline-.008:
            z = solve(z,deadline,objective,int(params['iterations']))
            offer(decode(z))
        if best is not None:
            z0 = encode(np.asarray(best['circles'],dtype=float))
    return best
