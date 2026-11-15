import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {'stage_iterations': (38, 15, 90), 'contact_margin': (0.11, 0.03, 0.25), 'jitter': (0.013, 0.002, 0.06), 'time_fraction': (0.70, 0.55, 0.72)}
N = 21
II, JJ = np.triu_indices(N, 1)
K = np.arange(N)

class Timeout(Exception):
    pass

def exact_score(c):
    try:
        a = np.asarray(c['circles'], dtype=np.float64)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        d = np.sqrt((a[II, 0]-a[JJ, 0])**2 + (a[II, 1]-a[JJ, 1])**2)
        if np.any(d < a[II, 2]+a[JJ, 2]):
            return -float('inf')
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        p = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        p += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        return float(np.sum(a[:, 2])) if p <= 2 else -float('inf')
    except (TypeError, ValueError, KeyError, OverflowError):
        return -float('inf')

def repair(z):
    a = np.asarray(z[:63], dtype=float).reshape(N, 3).copy()
    if not np.isfinite(a).all():
        return None
    a[:, 2] = np.maximum(a[:, 2], 1e-9)
    d = np.sqrt((a[II, 0]-a[JJ, 0])**2 + (a[II, 1]-a[JJ, 1])**2)
    f = min(1., float(np.min(d/(a[II, 2]+a[JJ, 2]))))
    if f <= 0:
        return None
    a[:, 2] *= f*(1-8e-14)
    lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
    hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
    a[:, :2] -= lo
    a *= (2/float(np.sum(hi-lo)))*(1-8e-14)
    return {'circles': a.tolist()}

def vector(c):
    a = np.asarray(c['circles'], dtype=float).copy()
    a[:, 2] = np.maximum(a[:, 2], 1e-7)
    lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
    hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
    scale = 2/float(np.sum(hi-lo))
    a[:, :2] -= lo
    a *= scale
    return np.r_[a.ravel(), (hi[0]-lo[0])*scale]

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + float(params['time_fraction'])*budget_s
    rng = np.random.default_rng(seed)
    best = init
    score = exact_score(init)
    if not np.isfinite(score):
        best = None
    def consider(z):
        nonlocal best, score
        c = repair(z)
        if c is not None:
            s = exact_score(c)
            if s > score:
                best, score = c, s
    a = np.array([[(k % 5 + .5)*.2, (k//5 + .5)*.2, .1] for k in range(N)])
    consider(np.r_[a.ravel(), 1.])
    warm = None
    try:
        if init is not None:
            aa = np.asarray(init['circles'], dtype=float)
            if aa.shape == (21,3) and np.isfinite(aa).all():
                warm = vector(init)
                consider(warm)
    except (TypeError, ValueError, KeyError):
        pass
    if warm is None:
        warm = vector(best)
    objective = np.zeros(64)
    objective[2:63:3] = -1
    base = np.zeros((84,64))
    base[K,3*K] = 1
    base[K,3*K+2] = -1
    base[21+K,3*K] = -1
    base[21+K,3*K+2] = -1
    base[21+K,63] = 1
    base[42+K,3*K+1] = 1
    base[42+K,3*K+2] = -1
    base[63+K,3*K+1] = -1
    base[63+K,3*K+2] = -1
    base[63+K,63] = -1
    bounds = [(None,None)]*64
    bounds[63] = (.08,1.92)
    for k in range(N):
        if time.monotonic() >= deadline:
            return best
        bounds[3*k+2] = (1e-7,.7)

    def stage(z, end, full=False):
        end = min(end, deadline)
        def check():
            if time.monotonic() >= end:
                raise Timeout()
        ar = z[:63].reshape(N,3)
        dx = ar[II,0]-ar[JJ,0]
        dy = ar[II,1]-ar[JJ,1]
        gaps = np.sqrt(dx*dx+dy*dy)-ar[II,2]-ar[JJ,2]
        mask = np.ones(len(II), dtype=bool) if full else gaps < float(params['contact_margin'])
        i, j = II[mask], JJ[mask]
        q = 84+np.arange(len(i))
        template = np.zeros((84+len(i),64))
        template[:84] = base
        template[q,3*i+2] = -1
        template[q,3*j+2] = -1
        def con(v):
            check()
            x,y,r = v[:63].reshape(N,3).T
            d = np.sqrt((x[i]-x[j])**2+(y[i]-y[j])**2+1e-30)
            return np.r_[x-r,v[63]-x-r,y-r,2-v[63]-y-r,d-r[i]-r[j]]
        def jac(v):
            check()
            x,y,r = v[:63].reshape(N,3).T
            dx,dy = x[i]-x[j],y[i]-y[j]
            d = np.sqrt(dx*dx+dy*dy+1e-30)
            g = template.copy()
            g[q,3*i] = dx/d
            g[q,3*j] = -dx/d
            g[q,3*i+1] = dy/d
            g[q,3*j+1] = -dy/d
            return g
        last = [z.copy()]
        count = [0]
        def callback(v):
            check()
            last[0] = v.copy()
            count[0] += 1
            if count[0] % 5 == 0:
                consider(v)
        def fun(v):
            check()
            return float(objective@v), objective
        try:
            res = minimize(fun,z,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':con,'jac':jac},callback=callback,
                options={'maxiter':int(params['stage_iterations']),'ftol':5e-13,'disp':False})
            last[0] = res.x.copy()
        except Timeout:
            pass
        consider(last[0])
        return last[0], count[0]

    # Follow the original warm-start trajectory across working-set refreshes.
    z = warm
    initial_end = start + .45*budget_s
    for iteration in range(5):
        if time.monotonic() >= min(initial_end,deadline)-.015:
            break
        z, steps = stage(z, initial_end, full=(iteration == 0))
        if steps < 5:
            break
    if time.monotonic() < deadline-.04:
        z, _ = stage(vector(best), min(deadline, start+.57*budget_s), full=True)
    attempt = 0
    while time.monotonic() < deadline-.045:
        z = vector(best)
        a = z[:63].reshape(N,3)
        amp = float(params['jitter'])*(1.5**min(attempt,4))
        if attempt % 3 == 0:
            chosen = rng.choice(N, size=5, replace=False)
            a[chosen,:2] += rng.normal(0,amp,(5,2))
        else:
            a[:,:2] += rng.normal(0,amp,(N,2))
        local_end = min(deadline,time.monotonic()+.19*budget_s)
        z, steps = stage(z,local_end)
        if time.monotonic() < local_end-.02 and steps >= 10:
            stage(z,local_end,full=True)
        attempt += 1
    return best
