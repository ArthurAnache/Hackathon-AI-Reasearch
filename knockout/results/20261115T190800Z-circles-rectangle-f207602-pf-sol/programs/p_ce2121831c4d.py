import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {'iterations': (190, 60, 350), 'contact_margin': (0.105, 0.03, 0.3), 'time_fraction': (0.70, 0.5, 0.72), 'jitter': (0.014, 0.002, 0.06)}
I, J = np.triu_indices(21, 1)
K = np.arange(21)

class Finished(Exception):
    pass

def exact_score(c):
    try:
        a = np.asarray(c['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
        if np.any(d < a[I, 2]+a[J, 2]):
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
    a = np.asarray(z[:63], dtype=float).reshape(21, 3).copy()
    if not np.isfinite(a).all():
        return None
    a[:, 2] = np.maximum(a[:, 2], 1e-11)
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    f = min(1.0, float(np.min(d/(a[I, 2]+a[J, 2]))))
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
    lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
    hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
    a[:, :2] -= lo
    s = 2/float(np.sum(hi-lo))
    a *= s
    return np.r_[a.ravel(), (hi[0]-lo[0])*s]

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + float(params['time_fraction'])*budget_s
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
    baseline = np.array([[(k%5+.5)*.2, (k//5+.5)*.2, .099] for k in range(21)])
    consider(np.r_[baseline.ravel(), 1.0])
    objective = np.zeros(64)
    objective[2:63:3] = -1
    T = np.zeros((64,34))
    offset = np.zeros(64)
    for k in range(3):
        if time.monotonic() >= deadline:
            return best
        T[3*k,2*k] = 1
        T[3*k+2,2*k+1] = 1
        offset[3*k+1] = 1
        T[3*k+1,33] = -.5
    for k in range(9):
        if time.monotonic() >= deadline:
            return best
        for side in range(2):
            if time.monotonic() >= deadline:
                return best
            n = 3+2*k+side
            T[3*n,6+3*k] = 1
            T[3*n+2,8+3*k] = 1
            offset[3*n+1] = 1
            T[3*n+1,33] = -.5
            T[3*n+1,7+3*k] = 1 if side == 0 else -1
    T[63,33] = 1
    basejac = np.zeros((84,64))
    basejac[K,3*K] = 1
    basejac[K,3*K+2] = -1
    basejac[21+K,3*K] = -1
    basejac[21+K,3*K+2] = -1
    basejac[21+K,63] = 1
    basejac[42+K,3*K+1] = 1
    basejac[42+K,3*K+2] = -1
    basejac[63+K,3*K+1] = -1
    basejac[63+K,3*K+2] = -1
    basejac[63+K,63] = -1

    def run(v, symmetric, end, maxiter=None):
        end = min(end, deadline)
        def check():
            if time.monotonic() >= end:
                raise Finished()
        def expand(u):
            return offset+T@u if symmetric else u
        z0 = expand(v)
        a = z0[:63].reshape(21,3)
        gaps = np.hypot(a[I,0]-a[J,0], a[I,1]-a[J,1])-a[I,2]-a[J,2]
        active = gaps < float(params['contact_margin'])
        last = [v.copy()]
        grad = T.T@objective if symmetric else objective
        bounds = [(None,None)]*len(v)
        bounds[-1] = (.15,1.85)
        ridx = [1,3,5]+list(range(8,33,3)) if symmetric else range(2,63,3)
        for k in ridx:
            if time.monotonic() >= end:
                return
            bounds[k] = (1e-9,.65)
        for stage in range(5):
            if time.monotonic() >= end-.005:
                break
            ii,jj = I[active],J[active]
            rows = 84+np.arange(len(ii))
            calls = [0]
            def fun(u):
                check()
                return float(grad@u), grad
            def con(u):
                check()
                z = expand(u)
                x,y,r = z[:63].reshape(21,3).T
                d = np.sqrt((x[ii]-x[jj])**2+(y[ii]-y[jj])**2+1e-30)
                return np.r_[x-r,z[63]-x-r,y-r,2-z[63]-y-r,d-r[ii]-r[jj]]
            def jac(u):
                check()
                z = expand(u)
                x,y,r = z[:63].reshape(21,3).T
                dx,dy = x[ii]-x[jj],y[ii]-y[jj]
                d = np.sqrt(dx*dx+dy*dy+1e-30)
                g = np.zeros((84+len(ii),64))
                g[:84] = basejac
                g[rows,3*ii] = dx/d
                g[rows,3*jj] = -dx/d
                g[rows,3*ii+1] = dy/d
                g[rows,3*jj+1] = -dy/d
                g[rows,3*ii+2] = -1
                g[rows,3*jj+2] = -1
                return g@T if symmetric else g
            def callback(u):
                check()
                last[0] = u.copy()
                calls[0] += 1
                if calls[0] % 9 == 0:
                    consider(expand(u))
            try:
                res = minimize(fun,last[0],jac=True,method='SLSQP',bounds=bounds,
                    constraints={'type':'ineq','fun':con,'jac':jac},callback=callback,
                    options={'maxiter':int(maxiter or params['iterations']),'ftol':7e-12,'disp':False})
                last[0] = res.x.copy()
            except Finished:
                consider(expand(last[0]))
                return
            z = expand(last[0])
            consider(z)
            a = z[:63].reshape(21,3)
            gaps = np.hypot(a[I,0]-a[J,0],a[I,1]-a[J,1])-a[I,2]-a[J,2]
            missing = (~active) & (gaps < .018)
            if not np.any(missing):
                break
            active |= missing
        consider(expand(last[0]))

    # Disturbed strong packings receive a long uninterrupted restoration.
    if init is not None:
        try:
            z = vector(init)
            if np.isfinite(z).all():
                run(z, False, start+.47*budget_s)
        except (ValueError, TypeError, KeyError, FloatingPointError):
            pass

    # Use the reflection quotient only if restoration has not reached the elite basin.
    if bestscore < 2.3657 and time.monotonic() < start+.54*budget_s:
        radius = 2/(12+4*np.sqrt(3))
        v = np.zeros(34)
        v[33] = 10*radius
        for k,xx in enumerate((2,4,8)):
            if time.monotonic() >= deadline:
                return best
            v[2*k] = xx*radius
            v[2*k+1] = radius*.98
        for k in range(9):
            if time.monotonic() >= deadline:
                return best
            v[6+3*k] = (2*k+1 if k<5 else 2*(k-5)+2)*radius
            v[7+3*k] = (np.sqrt(3) if k<5 else 2*np.sqrt(3))*radius
            v[8+3*k] = radius*.98
        run(v, True, start+.60*budget_s)

    if time.monotonic() < deadline-.015:
        run(vector(best), False, min(deadline,start+.655*budget_s))
    attempt = 0
    while time.monotonic() < deadline-.035:
        z = vector(best)
        a = z[:63].reshape(21,3)
        chosen = rng.choice(21, size=5, replace=False)
        a[chosen,:2] += rng.normal(0,float(params['jitter'])*(1+.4*attempt),(5,2))
        run(z,False,min(deadline,time.monotonic()+.12*budget_s),75)
        attempt += 1
    return best
