import time
import math
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"first_fraction": (0.48, 0.25, 0.7), "jitter": (0.012, 0.0, 0.04)}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.69 * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    ids = np.arange(n)
    best = init
    best_score = -float('inf')

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
            best_score = exact(np.asarray(init['circles'], dtype=float))
        except Exception:
            pass

    def consider(z):
        nonlocal best, best_score
        if time.monotonic() >= end:
            return
        a = np.column_stack((z[:n], z[n:2*n], np.maximum(z[2*n:3*n], 1e-9)))
        if not np.isfinite(a).all():
            return
        dx = a[ii,0]-a[jj,0]
        dy = a[ii,1]-a[jj,1]
        d = np.sqrt(dx*dx+dy*dy)
        f = min(1.0, float(np.min(d/(a[ii,2]+a[jj,2]))))
        if f <= 0:
            return
        a[:,2] *= f * (1.0-2e-13)
        a[:,:2] -= np.min(a[:,:2]-a[:,2,None], axis=0)
        wh = np.max(a[:,:2]+a[:,2,None],axis=0)-np.min(a[:,:2]-a[:,2,None],axis=0)
        a *= (2.0/(wh.sum()))*(1.0-2e-13)
        if float(a[:,2].sum()) > best_score:
            sc = exact(a)
            if sc > best_score:
                best_score = sc
                best = {'circles': a.tolist()}

    starts = []
    # Twenty large circles with one circle in an interior square gap.
    pts = [(2.0*c, 2.0*r, 1.0) for r in range(5) for c in range(4)]
    pts.append((3.0, 3.0, math.sqrt(2.0)-1.0))
    starts.append(np.array(pts))
    # A staggered lattice with an entirely different contact graph.
    pts = []
    for row in range(6):
        if time.monotonic() >= end:
            break
        count = 4 if row % 2 == 0 else 3
        for col in range(count):
            if time.monotonic() >= end:
                break
            pts.append((2.0*col + (row % 2), math.sqrt(3.0)*row, 1.0))
    if len(pts) == 21:
        starts.append(np.array(pts))

    grad = np.zeros(64)
    grad[42:63] = -1.0
    bounds = [(None,None)]*42 + [(1e-7,0.7)]*21 + [(0.1,1.9)]

    for run, a in enumerate(starts):
        if time.monotonic() >= end:
            break
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        a[:,:2] -= lo
        wh = np.max(a[:,:2]+a[:,2,None],axis=0)
        a *= 2.0/wh.sum()
        w = 2.0*wh[0]/wh.sum()
        z = np.r_[a[:,0],a[:,1],a[:,2],w]
        consider(z)
        z[:42] += rng.normal(0.0, params['jitter'],42)
        local_end = start + (end-start)*params['first_fraction'] if run == 0 else end
        counter = [0]
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
            dx,dy = x[ii]-x[jj], y[ii]-y[jj]
            return np.r_[np.sqrt(dx*dx+dy*dy)-r[ii]-r[jj], x-r, w-x-r, y-r, 2.0-w-y-r]

        def jac(v):
            check()
            x,y,r = v[:21],v[21:42],v[42:63]
            dx,dy = x[ii]-x[jj], y[ii]-y[jj]
            d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-15)
            j = np.zeros((m+84,64))
            rows = np.arange(m)
            j[rows,ii],j[rows,jj] = dx/d,-dx/d
            j[rows,21+ii],j[rows,21+jj] = dy/d,-dy/d
            j[rows,42+ii],j[rows,42+jj] = -1.,-1.
            q=m+ids
            j[q,ids]=1.; j[q,42+ids]=-1.
            q=m+21+ids
            j[q,ids]=-1.; j[q,42+ids]=-1.; j[q,63]=1.
            q=m+42+ids
            j[q,21+ids]=1.; j[q,42+ids]=-1.
            q=m+63+ids
            j[q,21+ids]=-1.; j[q,42+ids]=-1.; j[q,63]=-1.
            return j

        def callback(v):
            last[0] = v.copy()
            counter[0] += 1
            if counter[0] % 4 == 0:
                consider(v)
            check()

        try:
            result = minimize(fun,z,jac=lambda v:grad,method='SLSQP',bounds=bounds,
                              constraints={'type':'ineq','fun':cons,'jac':jac},
                              callback=callback,options={'maxiter':180,'ftol':2e-11,'disp':False})
            consider(result.x)
        except StopSearch:
            consider(last[0])
        except (ValueError, FloatingPointError):
            consider(last[0])
    return best
