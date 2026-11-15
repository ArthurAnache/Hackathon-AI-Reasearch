import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {'iterations': (160, 40, 300), 'time_fraction': (0.70, 0.5, 0.75), 'jitter': (0.008, 0.001, 0.04)}
N = 21
I, J = np.triu_indices(N, 1)
K = np.arange(N)
Q = np.arange(len(I)) + 84

class StopSearch(Exception):
    pass

def exact_score(c):
    try:
        a = np.asarray(c['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        r = a[:, 2]
        d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
        if np.any(d < r[I]+r[J]):
            return -float('inf')
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in r]
        p = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        p += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        return float(np.sum(a[:, 2])) if p <= 2 else -float('inf')
    except (TypeError, ValueError, KeyError, OverflowError):
        return -float('inf')

def repair(z):
    a = np.asarray(z[:63]).reshape(21, 3).copy()
    if not np.isfinite(a).all():
        return None
    a[:, 2] = np.maximum(a[:, 2], 1e-10)
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    f = min(1.0, float(np.min(d/(a[I, 2]+a[J, 2]))))
    if f <= 0:
        return None
    a[:, 2] *= f*(1-2e-13)
    lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
    hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
    a[:, :2] -= lo
    a *= (2/float(np.sum(hi-lo)))*(1-2e-13)
    return {'circles': a.tolist()}

def vector(c):
    a = np.array(c['circles'], dtype=float)
    lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
    hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
    a[:, :2] -= lo
    s = 2/float(np.sum(hi-lo))
    a *= s
    return np.r_[a.ravel(), (hi[0]-lo[0])*s]

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + float(params['time_fraction'])*budget_s
    phase_end = deadline
    rng = np.random.default_rng(seed)
    best = init
    bestscore = exact_score(init)
    if not np.isfinite(bestscore):
        best = None
    def consider(z):
        nonlocal best, bestscore
        c = repair(z)
        if c is not None:
            score = exact_score(c)
            if score > bestscore:
                best, bestscore = c, score
    # This small baseline also handles invalid or absent warm starts.
    a = np.array([[(k%5+.5)*.2, (k//5+.5)*.2, .099] for k in range(21)])
    consider(np.r_[a.ravel(), 1.0])

    transform = np.zeros((64, 34))
    offset = np.zeros(64)
    for k in range(3):
        if time.monotonic() >= deadline:
            return best
        transform[3*k, 2*k] = 1
        transform[3*k+2, 2*k+1] = 1
        offset[3*k+1] = 1
        transform[3*k+1, 33] = -.5
    for k in range(9):
        if time.monotonic() >= deadline:
            return best
        for side in range(2):
            if time.monotonic() >= deadline:
                return best
            n = 3+2*k+side
            transform[3*n, 6+3*k] = 1
            transform[3*n+2, 8+3*k] = 1
            offset[3*n+1] = 1
            transform[3*n+1, 33] = -.5
            transform[3*n+1, 7+3*k] = 1 if side == 0 else -1
    transform[63, 33] = 1
    objective = np.zeros(64)
    objective[2:63:3] = -1

    def check():
        if time.monotonic() >= phase_end:
            raise StopSearch()

    def constraints(z):
        check()
        x,y,r = z[:63].reshape(21,3).T
        w = z[63]
        d = np.sqrt((x[I]-x[J])**2+(y[I]-y[J])**2+1e-30)
        return np.r_[x-r, w-x-r, y-r, 2-w-y-r, d-r[I]-r[J]]

    def jacobian(z):
        check()
        x,y,r = z[:63].reshape(21,3).T
        g = np.zeros((294,64))
        g[K,3*K] = 1
        g[K,3*K+2] = -1
        g[21+K,3*K] = -1
        g[21+K,3*K+2] = -1
        g[21+K,63] = 1
        g[42+K,3*K+1] = 1
        g[42+K,3*K+2] = -1
        g[63+K,3*K+1] = -1
        g[63+K,3*K+2] = -1
        g[63+K,63] = -1
        dx,dy = x[I]-x[J],y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        g[Q,3*I] = dx/d
        g[Q,3*J] = -dx/d
        g[Q,3*I+1] = dy/d
        g[Q,3*J+1] = -dy/d
        g[Q,3*I+2] = -1
        g[Q,3*J+2] = -1
        return g

    def run(v, symmetric, end):
        nonlocal phase_end
        phase_end = min(end, deadline)
        last = [None]
        count = [0]
        grad = transform.T@objective if symmetric else objective
        def expand(u):
            return offset+transform@u if symmetric else u
        def fun(u):
            check()
            return float(grad@u), grad
        def con(u):
            return constraints(expand(u))
        def jac(u):
            g = jacobian(expand(u))
            return g@transform if symmetric else g
        def callback(u):
            check()
            last[0] = expand(u).copy()
            count[0] += 1
            if count[0] % 6 == 0:
                consider(last[0])
        bounds = [(None,None)]*len(v)
        bounds[-1] = (.12,1.88)
        indices = [1,3,5]+list(range(8,33,3)) if symmetric else range(2,63,3)
        for k in indices:
            if time.monotonic() >= phase_end:
                return
            bounds[k] = (1e-8,.7)
        try:
            result = minimize(fun, v, jac=True, method='SLSQP', bounds=bounds,
                constraints={'type':'ineq','fun':con,'jac':jac}, callback=callback,
                options={'maxiter':int(params['iterations']), 'ftol':2e-12, 'disp':False})
            consider(expand(result.x))
        except StopSearch:
            pass
        if last[0] is not None:
            consider(last[0])

    if init is not None and np.isfinite(exact_score(init)):
        run(vector(init), False, start+.24*budget_s)

    # A different basin: five staggered rows, constrained to reflection symmetry.
    for attempt in range(2):
        if time.monotonic() >= start+.53*budget_s:
            break
        radius = 2/(12+4*np.sqrt(3))
        v = np.zeros(34)
        v[33] = 10*radius
        axis = [1,3,7] if attempt == 0 else [1,5,7]
        for k in range(3):
            if time.monotonic() >= deadline:
                return best
            v[2*k] = (axis[k]+1)*radius
            v[2*k+1] = radius*.98
        for k in range(9):
            if time.monotonic() >= deadline:
                return best
            xx = 2*k+1 if k < 5 else 2*(k-5)+2
            yy = np.sqrt(3) if k < 5 else 2*np.sqrt(3)
            v[6+3*k] = xx*radius
            v[7+3*k] = yy*radius
            v[8+3*k] = radius*.98
        run(v, True, start+.55*budget_s)

    if time.monotonic() < deadline-.04:
        run(vector(best), False, deadline)
    attempt = 0
    while time.monotonic() < deadline-.07:
        z = vector(best)
        noise = rng.normal(0, float(params['jitter'])*(1+attempt*.5), (21,2))
        z[:63].reshape(21,3)[:, :2] += noise
        run(z, False, deadline)
        attempt += 1
    return best
