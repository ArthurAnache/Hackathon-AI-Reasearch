import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"search_fraction": (0.70, 0.50, 0.78), "iterations": (150, 30, 300)}


def exact_score(data):
    try:
        c = np.asarray(data['circles'], dtype=float)
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return -float('inf')
        x = [Fraction(float(v)) for v in c[:, 0]]
        y = [Fraction(float(v)) for v in c[:, 1]]
        r = [Fraction(float(v)) for v in c[:, 2]]
        w = max(a+b for a,b in zip(x,r))-min(a-b for a,b in zip(x,r))
        h = max(a+b for a,b in zip(y,r))-min(a-b for a,b in zip(y,r))
        if w+h > 2:
            return -float('inf')
        i,j = np.triu_indices(21,1)
        d = np.sqrt((c[i,0]-c[j,0])**2+(c[i,1]-c[j,1])**2)
        if np.any(d < c[i,2]+c[j,2]):
            return -float('inf')
        return float(np.sum(c[:,2]))
    except Exception:
        return -float('inf')


class StopSearch(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + float(params['search_fraction'])*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = exact_score(init) if init is not None else -float('inf')
    archive = []

    def retain(c):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        c = np.asarray(c, dtype=float).copy()
        if c.shape != (21,3) or not np.isfinite(c).all():
            return
        c[:,2] = np.maximum(c[:,2], 1e-12)
        c[:,:2] -= np.min(c[:,:2],axis=0)
        ii,jj = np.triu_indices(21,1)
        ds = np.sqrt(np.sum((c[ii,:2]-c[jj,:2])**2,axis=1))
        factor = min(1.0, float(np.min(ds/(c[ii,2]+c[jj,2]))))
        if factor <= 0:
            return
        c[:,2] *= factor*(1-5e-14)
        extent = np.max(c[:,:2]+c[:,2,None],axis=0)-np.min(c[:,:2]-c[:,2,None],axis=0)
        c *= (2.0/float(np.sum(extent)))*(1-5e-14)
        data = {'circles':c.tolist()}
        score = exact_score(data)
        if score > best_score:
            best, best_score = data, score

    # Always have a feasible fallback, independent of the supplied candidate.
    fallback = np.array([[float(k%7),float(k//7),0.49] for k in range(21)])
    retain(fallback)

    def solve(a, rowcounts=None, initial=None, allowance=0.2):
        if time.monotonic() >= deadline:
            return
        p = (21-a)//2
        k = a+p
        n = 2*a+3*p+1
        ix = np.empty(k,dtype=int)
        iy = np.full(k,-1,dtype=int)
        ir = np.empty(k,dtype=int)
        q = 0
        for t in range(k):
            if time.monotonic() >= deadline:
                return
            ix[t] = q
            q += 1
            if t >= a:
                iy[t] = q
                q += 1
            ir[t] = q
            q += 1
        ex = np.zeros((k,n)); ey = np.zeros((k,n)); er = np.zeros((k,n))
        ex[np.arange(k),ix] = 1
        er[np.arange(k),ir] = 1
        ey[np.arange(a,k),iy[a:]] = 1
        ew = np.zeros(n); ew[-1] = 1
        weights = np.ones(k); weights[a:] = 2
        objective_gradient = -(weights @ er)
        ii,jj = np.triu_indices(k,1)
        dxmap = ex[ii]-ex[jj]
        dymap = ey[ii]-ey[jj]
        rrmap = er[ii]+er[jj]
        linear = np.vstack((ex-er, ew-ex-er, -0.5*ew-ey-er, ey[a:]-er[a:]))
        offset = np.zeros(linear.shape[0]); offset[2*k:3*k] = 1
        if initial is None:
            z = np.zeros(n)
            coords = [(2*t-(a-1),0.0) for t in range(a)]
            for row,count in enumerate(rowcounts):
                if time.monotonic() >= deadline:
                    return
                coords.extend((2*t-(count-1),(row+1)*np.sqrt(3)) for t in range(count))
            coords = np.asarray(coords)
            width = np.ptp(coords[:,0])+2
            height = 2*(np.max(coords[:,1])+1)
            scale = 2/(width+height)
            z[ix] = (coords[:,0]+width/2)*scale
            z[ir] = scale
            z[iy[a:]] = coords[a:,1]*scale
            z[-1] = width*scale
        else:
            z = initial.copy()
        local_end = min(deadline,time.monotonic()+allowance)
        last = z.copy()
        cached_z = None
        cached_c = None
        cached_j = None

        def check():
            if time.monotonic() >= local_end:
                raise StopSearch()

        def calculate(v):
            nonlocal cached_z,cached_c,cached_j
            check()
            if cached_z is not None and np.array_equal(v,cached_z):
                return
            dx = dxmap @ v
            dy = dymap @ v
            distance = np.sqrt(dx*dx+dy*dy+1e-30)
            cached_c = np.concatenate((linear@v+offset, distance-rrmap@v))
            pairjac = (dx/distance)[:,None]*dxmap+(dy/distance)[:,None]*dymap-rrmap
            cached_j = np.vstack((linear,pairjac))
            cached_z = v.copy()

        def con(v):
            calculate(v)
            return cached_c

        def jac(v):
            calculate(v)
            return cached_j

        def callback(v):
            nonlocal last
            last = v.copy()
            check()

        bounds = [(None,None)]*n
        for t in range(k):
            if time.monotonic() >= deadline:
                return
            bounds[ir[t]] = (1e-8,0.6)
        bounds[-1] = (0.25,1.75)
        try:
            result = minimize(lambda v: float(objective_gradient@v),z,jac=lambda v:objective_gradient,
                constraints={'type':'ineq','fun':con,'jac':jac},bounds=bounds,
                method='SLSQP',callback=callback,
                options={'maxiter':int(params['iterations']),'ftol':2e-12,'disp':False})
            last = result.x
        except StopSearch:
            pass
        except (ValueError, FloatingPointError):
            pass
        upper = np.column_stack((ex@last,ey@last,er@last))
        reflected = upper[a:].copy(); reflected[:,1] *= -1
        circles = np.vstack((upper,reflected))
        retain(circles)
        # Archive by repaired geometric score rather than a possibly infeasible objective.
        ii2,jj2 = np.triu_indices(21,1)
        dist = np.sqrt(np.sum((circles[ii2,:2]-circles[jj2,:2])**2,axis=1))
        shrink = min(1.0,float(np.min(dist/np.maximum(circles[ii2,2]+circles[jj2,2],1e-30))))
        radii = np.maximum(circles[:,2],1e-12)*max(shrink,0)
        span = np.sum(np.max(circles[:,:2]+radii[:,None],axis=0)-np.min(circles[:,:2]-radii[:,None],axis=0))
        quality = 2*np.sum(radii)/max(span,1e-30)
        archive.append((quality,a,last))

    if best_score > 2.0 and init is not None:
        retain(np.asarray(init['circles'],dtype=float))
    families = [(3,[4,5]),(5,[4,4]),(3,[5,4]),(1,[5,5])]
    for a,rows in families:
        if time.monotonic() >= deadline-0.08:
            break
        solve(a,rows,allowance=min(0.245*budget_s,0.235*(deadline-started)))
    while time.monotonic() < deadline-0.06 and archive:
        archive.sort(key=lambda item:item[0],reverse=True)
        _,a,z = archive[0]
        remaining = deadline-time.monotonic()
        solve(a,initial=z,allowance=min(remaining-0.015,0.22*budget_s))
        if time.monotonic() >= deadline-0.06:
            break
        if archive[-1][0] <= archive[0][0]+1e-10:
            z = archive[0][2].copy()
            z[:-1] += rng.normal(0,0.006,len(z)-1)
            solve(a,initial=z,allowance=min(deadline-time.monotonic()-0.01,0.16*budget_s))
    return best
