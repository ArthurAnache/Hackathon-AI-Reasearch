import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {'iterations': (240, 80, 400), 'contact_margin': (0.12, 0.04, 0.3), 'time_fraction': (0.70, 0.5, 0.72), 'jitter': (0.022, 0.002, 0.09)}
I, J = np.triu_indices(21, 1)
K = np.arange(21)

class Finished(Exception):
    pass

def exact_score(c):
    try:
        a = np.asarray(c['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        d = np.sqrt((a[I, 0]-a[J, 0])**2+(a[I, 1]-a[J, 1])**2)
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
    a[:, 2] = np.maximum(a[:, 2], 1e-12)
    d = np.sqrt((a[I, 0]-a[J, 0])**2+(a[I, 1]-a[J, 1])**2)
    f = min(1.0, float(np.min(d/(a[I, 2]+a[J, 2]))))
    if f <= 0:
        return None
    a[:, 2] *= f*(1-1e-13)
    lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
    hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
    perimeter = float(np.sum(hi-lo))
    if not np.isfinite(perimeter) or perimeter <= 0:
        return None
    a[:, :2] -= lo
    a *= (2/perimeter)*(1-1e-13)
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
    deadline = start+float(params['time_fraction'])*budget_s
    rng = np.random.default_rng(seed)
    bestscore = exact_score(init)
    initialscore = bestscore
    best = init if np.isfinite(bestscore) else None

    def consider(z):
        nonlocal best, bestscore
        c = repair(z)
        if c is not None:
            score = exact_score(c)
            if score > bestscore:
                best, bestscore = c, score

    baseline = np.array([[(k%5+.5)*.2, (k//5+.5)*.2, .099] for k in range(21)])
    consider(np.r_[baseline.ravel(), 1.0])
    if time.monotonic() >= deadline:
        return best

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
    B = np.zeros((84,64))
    B[K,3*K] = 1
    B[K,3*K+2] = -1
    B[21+K,3*K] = -1
    B[21+K,3*K+2] = -1
    B[21+K,63] = 1
    B[42+K,3*K+1] = 1
    B[42+K,3*K+2] = -1
    B[63+K,3*K+1] = -1
    B[63+K,3*K+2] = -1
    B[63+K,63] = -1
    BT = B@T

    def run(v, symmetric, end, iterations=None):
        end = min(end, deadline)
        if time.monotonic() >= end-.003:
            return v
        transform = T if symmetric else np.eye(64)
        base = BT if symmetric else B
        grad = transform.T@objective
        def check():
            if time.monotonic() >= end:
                raise Finished()
        def expand(u):
            return offset+T@u if symmetric else u
        a = expand(v)[:63].reshape(21,3)
        gaps = np.hypot(a[I,0]-a[J,0],a[I,1]-a[J,1])-a[I,2]-a[J,2]
        active = gaps < float(params['contact_margin'])
        last = [v.copy()]
        bounds = [(None,None)]*len(v)
        bounds[-1] = (.15,1.85)
        ridx = [1,3,5]+list(range(8,33,3)) if symmetric else range(2,63,3)
        for k in ridx:
            if time.monotonic() >= end:
                return last[0]
            bounds[k] = (1e-9,.65)
        for stage in range(5):
            if time.monotonic() >= end-.004:
                break
            ii,jj = I[active],J[active]
            dxrow = transform[3*ii]-transform[3*jj]
            dyrow = transform[3*ii+1]-transform[3*jj+1]
            rr = -transform[3*ii+2]-transform[3*jj+2]
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
                return np.vstack((base,dxrow*(dx/d)[:,None]+dyrow*(dy/d)[:,None]+rr))
            def callback(u):
                check()
                last[0] = u.copy()
                calls[0] += 1
                if calls[0] % 10 == 0:
                    consider(expand(u))
            try:
                res = minimize(fun,last[0],jac=True,method='SLSQP',bounds=bounds,
                    constraints={'type':'ineq','fun':con,'jac':jac},callback=callback,
                    options={'maxiter':int(iterations or params['iterations']),'ftol':3e-13,'disp':False})
                last[0] = res.x.copy()
            except Finished:
                consider(expand(last[0]))
                return last[0]
            z = expand(last[0])
            consider(z)
            a = z[:63].reshape(21,3)
            gaps = np.hypot(a[I,0]-a[J,0],a[I,1]-a[J,1])-a[I,2]-a[J,2]
            missing = (~active)&(gaps < .015)
            if not np.any(missing):
                break
            active |= missing
        consider(expand(last[0]))
        return last[0]

    # A useful supplied packing keeps priority; tiny-radius starts do not.
    if initialscore > 2.15 and time.monotonic() < deadline:
        try:
            run(vector(init),False,start+.43*budget_s)
        except (ValueError,TypeError,KeyError,FloatingPointError):
            pass

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
    if bestscore < 2.3658 and time.monotonic() < deadline-.025:
        v = run(v,True,start+.49*budget_s)
    if time.monotonic() < deadline-.02:
        run(vector(best),False,min(deadline,start+.59*budget_s),150)

    attempt = 0
    while time.monotonic() < deadline-.04:
        if initialscore < 2.15 and attempt % 2 == 0:
            trial = v.copy()
            trial[:33] += rng.normal(0,float(params['jitter']),33)
            trial[[1,3,5]+list(range(8,33,3))] = np.maximum(.025,trial[[1,3,5]+list(range(8,33,3))])
            trial[7:33:3] = np.abs(trial[7:33:3])
            run(trial,True,min(deadline,time.monotonic()+.08*budget_s),90)
        else:
            z = vector(best)
            a = z[:63].reshape(21,3)
            chosen = rng.choice(21,size=5,replace=False)
            a[chosen,:2] += rng.normal(0,float(params['jitter']),(5,2))
            run(z,False,min(deadline,time.monotonic()+.08*budget_s),65)
        attempt += 1
    return best
