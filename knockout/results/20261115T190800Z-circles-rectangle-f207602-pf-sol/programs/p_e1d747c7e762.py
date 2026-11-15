import time
import math
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"polish_fraction": (0.24, 0.1, 0.5), "jitter": (0.022, 0.002, 0.08), "restart_fraction": (0.34, 0.15, 0.6)}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + 0.69 * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    rows = np.arange(m)
    ids = np.arange(n)
    best = init
    best_score = -float('inf')
    best_array = None

    def exact(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        dx = a[ii, 0] - a[jj, 0]
        dy = a[ii, 1] - a[jj, 1]
        if np.any(np.sqrt(dx*dx + dy*dy) < a[ii, 2] + a[jj, 2]):
            return -float('inf')
        xs = [Fraction(float(v)) for v in a[:, 0]]
        ys = [Fraction(float(v)) for v in a[:, 1]]
        rs = [Fraction(float(v)) for v in a[:, 2]]
        w = max(x+r for x,r in zip(xs,rs)) - min(x-r for x,r in zip(xs,rs))
        h = max(y+r for y,r in zip(ys,rs)) - min(y-r for y,r in zip(ys,rs))
        if w+h > 2:
            return -float('inf')
        return float(np.sum(a[:, 2]))

    if init is not None:
        try:
            a = np.asarray(init['circles'], dtype=float)
            best_score = exact(a)
            if np.isfinite(best_score):
                best_array = a.copy()
        except Exception:
            pass

    def consider(z):
        nonlocal best, best_score, best_array
        if time.monotonic() >= deadline:
            return
        a = np.column_stack((z[:21], z[21:42], np.maximum(z[42:63], 1e-10)))
        if not np.isfinite(a).all():
            return
        d = np.hypot(a[ii,0]-a[jj,0], a[ii,1]-a[jj,1])
        shrink = min(1.0, float(np.min(d/(a[ii,2]+a[jj,2]))))
        if shrink <= 0:
            return
        a[:,2] *= shrink * (1.0-3e-14)
        a[:,:2] -= np.min(a[:,:2]-a[:,2,None], axis=0)
        span = np.max(a[:,:2]+a[:,2,None],axis=0)-np.min(a[:,:2]-a[:,2,None],axis=0)
        a *= (2.0 / float(span.sum())) * (1.0-3e-14)
        if float(a[:,2].sum()) > best_score:
            score = exact(a)
            if score > best_score:
                best_score = score
                best_array = a.copy()
                best = {'circles': a.tolist()}

    def vector(a):
        a = a.copy()
        a[:,:2] -= np.min(a[:,:2]-a[:,2,None],axis=0)
        span = np.max(a[:,:2]+a[:,2,None],axis=0)
        a *= 2.0 / span.sum()
        return np.r_[a[:,0],a[:,1],a[:,2],2.0*span[0]/span.sum()]

    if best_array is None:
        pts = [(2.0*c, 2.0*r, 1.0) for r in range(5) for c in range(4)]
        pts.append((3.0,3.0,math.sqrt(2.0)-1.0))
        base = np.array(pts)
        z = vector(base)
        consider(z)
    else:
        z = vector(best_array)

    grad = np.zeros(64)
    grad[42:63] = -1.0
    bounds = [(None,None)]*42 + [(1e-6,0.7)]*21 + [(0.1,1.9)]
    template = np.zeros((m+84,64))
    template[rows,42+ii] = -1.0
    template[rows,42+jj] = -1.0
    q = m+ids
    template[q,ids] = 1.0
    template[q,42+ids] = -1.0
    q = m+21+ids
    template[q,ids] = -1.0
    template[q,42+ids] = -1.0
    template[q,63] = 1.0
    q = m+42+ids
    template[q,21+ids] = 1.0
    template[q,42+ids] = -1.0
    q = m+63+ids
    template[q,21+ids] = -1.0
    template[q,42+ids] = -1.0
    template[q,63] = -1.0

    run = 0
    while time.monotonic() < deadline - 0.015:
        now = time.monotonic()
        if run == 0:
            local_end = min(deadline-0.008, now + params['polish_fraction']*budget_s)
        else:
            local_end = min(deadline-0.008, now + params['restart_fraction']*budget_s)
            if best_array is not None:
                z = vector(best_array)
            amplitude = params['jitter'] * (1.0 if run % 3 == 1 else (2.1 if run % 3 == 2 else 0.4))
            # Displace centres coherently as well as independently, so several contacts can change together.
            displacement = rng.normal(size=(21,2))
            if run % 2 == 0:
                pivot = rng.uniform(0.25,0.75,size=2)
                coords = np.column_stack((z[:21],z[21:42]))
                direction = rng.normal(size=2)
                direction /= max(np.linalg.norm(direction),1e-12)
                side = np.tanh(12.0*((coords-pivot) @ direction))
                displacement += side[:,None]*rng.normal(size=(1,2))
            z[:21] += amplitude*displacement[:,0]
            z[21:42] += amplitude*displacement[:,1]
            z[42:63] *= 0.98
        last = [z.copy()]

        def check():
            if time.monotonic() >= local_end:
                raise StopSearch()

        def fun(v):
            check()
            return -float(np.sum(v[42:63]))

        def cons(v):
            check()
            x,y,r,w = v[:21],v[21:42],v[42:63],v[63]
            d = np.hypot(x[ii]-x[jj],y[ii]-y[jj])
            return np.r_[d-r[ii]-r[jj],x-r,w-x-r,y-r,2.0-w-y-r]

        def jac(v):
            check()
            dx = v[ii]-v[jj]
            dy = v[21+ii]-v[21+jj]
            d = np.maximum(np.hypot(dx,dy),1e-15)
            j = template.copy()
            j[rows,ii] = dx/d
            j[rows,jj] = -dx/d
            j[rows,21+ii] = dy/d
            j[rows,21+jj] = -dy/d
            return j

        def callback(v):
            last[0] = v.copy()
            consider(v)
            check()

        try:
            result = minimize(fun,z,jac=lambda v:grad,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':cons,'jac':jac},callback=callback,
                options={'maxiter':180,'ftol':2e-12,'disp':False})
            consider(result.x)
        except StopSearch:
            consider(last[0])
        except (ValueError,FloatingPointError):
            consider(last[0])
        run += 1
    return best
