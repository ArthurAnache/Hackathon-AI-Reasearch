import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"iterations": (110, 40, 200), "slice": (0.255, 0.12, 0.5), "jitter": (0.025, 0.0, 0.12)}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    best = init
    best_score = -float('inf')

    def exact(c):
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return -float('inf')
        d = c[ii, :2] - c[jj, :2]
        if np.any(np.sqrt(d[:, 0]**2 + d[:, 1]**2) < c[ii, 2] + c[jj, 2]):
            return -float('inf')
        xx, yy, rr = [], [], []
        for x, y, r in c:
            if time.monotonic() >= end:
                raise StopSearch
            xx.append(Fraction(float(x)))
            yy.append(Fraction(float(y)))
            rr.append(Fraction(float(r)))
        w = max(x+r for x,r in zip(xx,rr)) - min(x-r for x,r in zip(xx,rr))
        h = max(y+r for y,r in zip(yy,rr)) - min(y-r for y,r in zip(yy,rr))
        if w+h > 2:
            return -float('inf')
        return float(np.sum(c[:, 2]))

    def retain(c):
        nonlocal best, best_score
        if time.monotonic() >= end:
            raise StopSearch
        c = np.array(c, dtype=float, copy=True)
        if c.shape != (21,3) or not np.isfinite(c).all():
            return
        c[:, 2] = np.maximum(c[:, 2], 1e-10)
        d = c[ii, :2] - c[jj, :2]
        ds = np.sqrt(np.sum(d*d, axis=1))
        shrink = min(1.0, float(np.min(ds / (c[ii,2]+c[jj,2]))))
        if shrink <= 0:
            return
        c[:,2] *= shrink * (1-2e-13)
        lower = np.min(c[:,:2] - c[:,2,None], axis=0)
        upper = np.max(c[:,:2] + c[:,2,None], axis=0)
        perimeter = float(np.sum(upper-lower))
        c[:,:2] -= lower
        c *= (2.0/perimeter)*(1-3e-13)
        score = float(np.sum(c[:,2]))
        if score > best_score:
            value = exact(c)
            if value > best_score:
                best_score = value
                best = {"circles": c.tolist()}

    try:
        if init is not None:
            try:
                best_score = exact(np.asarray(init['circles'],dtype=float))
            except (ValueError, KeyError, TypeError):
                pass
        # Each construction lists the central row and the rows in the upper half.
        root3 = np.sqrt(3.)
        families = [
            ([1.,4.,7.], [(root3,[0.,2.,4.,6.,8.]),(2*root3,[1.,3.,5.,7.])]),
            ([0.,2.,4.,6.,8.], [(root3,[1.,3.,5.,7.]),(2*root3,[1.,3.,5.,7.])]),
            ([1.,3.,7.], [(root3,[0.,2.,4.,6.,8.]),(2*root3,[1.,3.,5.,7.])]),
            ([0.,2.,4.], [(root3,[1.,3.,5.]),(2*root3,[0.,2.,4.]),(3*root3,[1.,3.,5.])]),
            ([0.,4.,8.], [(root3,[1.,3.,5.,7.]),(2*root3,[0.,2.,4.,6.,8.])])
        ]
        saved = []

        def run(axis, rows, initial=None, limit=None):
            if time.monotonic() >= end:
                raise StopSearch
            k = len(axis)
            upper = [(x,y) for y,xs in rows for x in xs]
            m = len(upper)
            q = k+m
            nv = 2*q+m+1
            wi = nv-1
            X = np.zeros((21,nv)); Y = X.copy(); R = X.copy()
            ids = np.r_[np.arange(k),np.arange(k,q),np.arange(k,q)]
            X[np.arange(21),ids] = 1
            R[np.arange(21),q+m+ids] = 1
            Y[np.arange(k,k+m),q+np.arange(m)] = 1
            Y[np.arange(k+m,21),q+np.arange(m)] = -1
            mirror = np.r_[np.arange(k),np.arange(k+m,21),np.arange(k,k+m)]
            pairs = []
            for a,b in zip(ii,jj):
                if time.monotonic() >= end:
                    raise StopSearch
                other = tuple(sorted((int(mirror[a]),int(mirror[b]))))
                if (int(a),int(b)) <= other:
                    pairs.append((a,b))
            a,b = np.asarray(pairs).T
            DX = X[a]-X[b]; DY = Y[a]-Y[b]; RS = R[a]+R[b]
            # Boundary inequalities need only one representative of each pair.
            bx = X[:q]; by = Y[:q]; br = R[:q]
            B = np.vstack((bx-br,-bx-br,-by-br))
            B[q:2*q,wi] = 1
            B[2*q:,wi] = -.5
            const = np.r_[np.zeros(2*q),np.ones(q)]
            weights = np.sum(R,axis=0)
            if initial is None:
                z = np.zeros(nv)
                z[:q] = np.r_[axis,[p[0] for p in upper]]
                z[q:q+m] = [p[1] for p in upper]
                z[q+m:q+m+q] = 1
                z[:q] += 1
                width = max(z[:q])+1
                height = 2*(max(z[q:q+m])+1)
                scale = 2/(width+height)
                z *= scale
                z[wi] = width*scale
                z[:q] += rng.normal(0,params['jitter']*scale,q)
            else:
                z = initial.copy()
            last = [z.copy()]
            cache = [None,None,None]
            deadline = min(end, time.monotonic() + (params['slice'] if limit is None else limit))
            def calculate(v):
                if time.monotonic() >= deadline:
                    raise StopSearch
                if cache[0] is None or not np.array_equal(v,cache[0]):
                    dx = DX@v; dy = DY@v
                    dist = np.maximum(np.sqrt(dx*dx+dy*dy),1e-15)
                    val = np.r_[dist-RS@v,B@v+const]
                    jac = np.vstack(((dx[:,None]*DX+dy[:,None]*DY)/dist[:,None]-RS,B))
                    cache[:] = [v.copy(),val,jac]
                return cache[1],cache[2]
            count = [0]
            def callback(v):
                if time.monotonic() >= deadline:
                    raise StopSearch
                last[0] = v.copy()
                count[0] += 1
                if count[0] % 10 == 0:
                    retain(np.column_stack((X@v,Y@v,R@v)))
            bounds = [(None,None)]*nv
            for t in range(q+m,q+m+q):
                if time.monotonic() >= end:
                    raise StopSearch
                bounds[t] = (1e-7,1.)
            bounds[wi] = (.05,1.95)
            try:
                result = minimize(lambda v:-float(weights@v),z,jac=lambda v:-weights,
                    method='SLSQP',bounds=bounds,
                    constraints={'type':'ineq','fun':lambda v:calculate(v)[0],'jac':lambda v:calculate(v)[1]},
                    callback=callback,options={'maxiter':params['iterations'],'ftol':2e-11,'disp':False})
                last[0] = result.x
            except StopSearch:
                pass
            if time.monotonic() < end:
                v = last[0]
                retain(np.column_stack((X@v,Y@v,R@v)))
                saved.append((float(weights@v),axis,rows,v.copy()))

        for axis,rows in families:
            if time.monotonic() >= end-.08:
                break
            run(axis,rows)
        if saved and time.monotonic() < end-.04:
            _,axis,rows,z = max(saved,key=lambda a:a[0])
            run(axis,rows,z,limit=max(.01,end-time.monotonic()-.01))
    except StopSearch:
        pass
    if best is None:
        # Guaranteed non-overlapping fallback with ample perimeter slack.
        c = [[.08*(i%7),.08*(i//7),.039] for i in range(21)]
        return {'circles':c}
    return best
