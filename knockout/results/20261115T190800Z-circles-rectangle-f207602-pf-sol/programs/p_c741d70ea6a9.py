import time
import math
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"iterations": (105, 30, 200), "time_fraction": (0.70, 0.4, 0.8), "perturbation": (0.025, 0.001, 0.1)}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + float(params['time_fraction']) * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    best = init
    best_score = -float('inf')

    def exact_score(obj):
        try:
            if 'circles' in obj:
                a = np.asarray(obj['circles'], dtype=float)
            else:
                a = np.column_stack((obj['centers'], obj['radii']))
            if a.shape != (21, 3) or not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
                return -float('inf')
            d = a[ii, :2] - a[jj, :2]
            if np.any(np.sqrt(d[:, 0]**2 + d[:, 1]**2) < a[ii, 2] + a[jj, 2]):
                return -float('inf')
            xs = [Fraction(float(v)) for v in a[:, 0]]
            ys = [Fraction(float(v)) for v in a[:, 1]]
            rs = [Fraction(float(v)) for v in a[:, 2]]
            perimeter = max(x+r for x,r in zip(xs,rs)) - min(x-r for x,r in zip(xs,rs))
            perimeter += max(y+r for y,r in zip(ys,rs)) - min(y-r for y,r in zip(ys,rs))
            if perimeter > 2:
                return -float('inf')
            return float(np.sum(a[:, 2]))
        except Exception:
            return -float('inf')

    if init is not None:
        best_score = exact_score(init)

    def consider(a):
        nonlocal best, best_score
        if not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return
        a = a.copy()
        d = a[ii, :2] - a[jj, :2]
        distance = np.sqrt(d[:, 0]**2 + d[:, 1]**2)
        contraction = min(1.0, float(np.min(distance / (a[ii, 2] + a[jj, 2]))))
        if contraction <= 0:
            return
        a[:, 2] *= contraction * (1.0 - 4e-14)
        lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
        hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
        extent = float(np.sum(hi-lo))
        if extent <= 0:
            return
        a[:, :2] -= (hi+lo)/2
        a *= (2.0 / extent) * (1.0-4e-14)
        score = float(np.sum(a[:, 2]))
        if score <= best_score:
            return
        obj = {'centers': a[:, :2].tolist(), 'radii': a[:, 2].tolist()}
        value = exact_score(obj)
        if value > best_score:
            best_score, best = value, obj

    fallback = np.array([[2.0*x, 2.0*y, 1.0] for y in range(7) for x in range(3)], dtype=float)
    consider(fallback)
    saved = []

    def run(rows, perturb=None, limit=None):
        if time.monotonic() >= deadline:
            raise StopSearch()
        m = rows[0]
        p = sum(rows[1:])
        g = m+p
        nvar = 2*g+p+1
        ridx = g+p
        X = np.zeros((21,nvar))
        Y = np.zeros_like(X)
        R = np.zeros_like(X)
        for k in range(21):
            if time.monotonic() >= deadline:
                raise StopSearch()
            if k < g:
                q = k
                sign = 1
            else:
                q = m+k-g
                sign = -1
            X[k,q] = 1
            R[k,ridx+q] = 1
            if q >= m:
                Y[k,g+q-m] = sign
        # Each reflected constraint appears only once.
        edges = []
        for i in range(g):
            if time.monotonic() >= deadline:
                raise StopSearch()
            for j in range(i+1,g):
                if time.monotonic() >= deadline:
                    raise StopSearch()
                edges.append((i,j))
        for u in range(p):
            if time.monotonic() >= deadline:
                raise StopSearch()
            for v in range(u,p):
                if time.monotonic() >= deadline:
                    raise StopSearch()
                edges.append((m+u,g+v))
        ei = np.array([e[0] for e in edges])
        ej = np.array([e[1] for e in edges])
        DX = X[ei]-X[ej]
        DY = Y[ei]-Y[ej]
        SR = R[ei]+R[ej]
        L = np.zeros((3*g,nvar))
        offset = np.zeros(3*g)
        L[:g] = X[:g]-R[:g]
        L[g:2*g] = -X[:g]-R[:g]
        L[g:2*g,-1] = 1
        L[2*g:] = -Y[:g]-R[:g]
        L[2*g:,-1] = -.5
        offset[2*g:] = 1
        objective = -np.sum(R,axis=0)
        z = np.zeros(nvar)
        pos = 0
        for level,count in enumerate(rows):
            if time.monotonic() >= deadline:
                raise StopSearch()
            shift = 1.0 if (count+level-m)%2 else 0.0
            z[pos:pos+count] = 2*np.arange(count)-(count-1)+shift
            if level:
                z[g+pos-m:g+pos-m+count] = level*math.sqrt(3)
            z[ridx+pos:ridx+pos+count] = 1
            pos += count
        xmin = np.min(z[:g])-1
        xmax = np.max(z[:g])+1
        z[:g] -= xmin
        width = xmax-xmin
        height = 2*(np.max(z[g:g+p])+1)
        scale = 2/(width+height)
        z[:-1] *= scale
        z[-1] = width*scale
        if perturb is not None:
            z = perturb.copy()
            z[:g] += rng.normal(0, float(params['perturbation']),g)
            z[g:g+p] += rng.normal(0,float(params['perturbation']),p)
        bounds = [(0,2)]*g + [(0,1)]*p + [(1e-7,1)]*g + [(.04,1.96)]
        for k,(lo,hi) in enumerate(bounds):
            if time.monotonic() >= deadline:
                raise StopSearch()
            z[k] = np.clip(z[k],lo+1e-9,hi-1e-9)

        def unpack(v):
            return np.column_stack((X@v,Y@v,R@v))
        cache_x = None
        cache_c = None
        cache_j = None
        def calculate(v):
            nonlocal cache_x,cache_c,cache_j
            if time.monotonic() >= deadline:
                raise StopSearch()
            if cache_x is None or not np.array_equal(cache_x,v):
                dx = DX@v
                dy = DY@v
                rr = SR@v
                cache_c = np.concatenate((L@v+offset,dx*dx+dy*dy-rr*rr))
                cache_j = np.vstack((L,2*(dx[:,None]*DX+dy[:,None]*DY-rr[:,None]*SR)))
                cache_x = v.copy()
            return cache_c
        def jac(v):
            calculate(v)
            return cache_j
        count_iter = 0
        last = z.copy()
        def callback(v):
            nonlocal count_iter,last
            if time.monotonic() >= deadline:
                raise StopSearch()
            last = v.copy()
            count_iter += 1
            if count_iter % 9 == 0:
                consider(unpack(v))
        consider(unpack(z))
        try:
            res = minimize(lambda v: float(objective@v),z,jac=lambda v:objective,
                           method='SLSQP',bounds=bounds,
                           constraints={'type':'ineq','fun':calculate,'jac':jac},
                           callback=callback,
                           options={'maxiter':int(limit or params['iterations']),'ftol':2e-11,'disp':False})
            last = res.x
        except StopSearch:
            consider(unpack(last))
            raise
        consider(unpack(last))
        saved.append((float(-objective@last),tuple(rows),last.copy()))

    try:
        families = [(3,4,5),(5,4,4),(3,3,3,3),(1,5,5),(5,3,5),(7,7)]
        for family in families:
            if time.monotonic() >= deadline:
                break
            run(family)
        saved.sort(key=lambda a:a[0],reverse=True)
        for k in range(8):
            if time.monotonic() >= deadline or not saved:
                break
            _,rows,z = saved[k % min(3,len(saved))]
            run(rows,z,70)
    except StopSearch:
        pass
    return best
