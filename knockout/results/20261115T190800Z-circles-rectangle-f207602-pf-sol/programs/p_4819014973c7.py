import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"iterations": (85, 30, 150), "search_fraction": (0.69, 0.4, 0.72)}


def exact_score(data):
    try:
        a = np.asarray(data['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        w = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        h = max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        if w+h > 2:
            return -float('inf')
        for i in range(21):
            for j in range(i):
                if np.sqrt((a[i,0]-a[j,0])**2+(a[i,1]-a[j,1])**2) < a[i,2]+a[j,2]:
                    return -float('inf')
        return float(np.sum(a[:,2]))
    except Exception:
        return -float('inf')


class StopSearch(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + float(params['search_fraction'])*budget_s
    best = init
    best_score = exact_score(init) if init is not None else -float('inf')
    ii, jj = np.triu_indices(21, 1)

    def consider(a):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        a = np.array(a, dtype=float, copy=True)
        if not np.isfinite(a).all() or np.min(a[:,2]) <= 0:
            return
        d = np.sqrt(np.sum((a[ii,:2]-a[jj,:2])**2, axis=1))
        ratio = min(1.0, float(np.min(d/(a[ii,2]+a[jj,2]))))
        if ratio <= 0:
            return
        a[:,2] *= ratio*(1-3e-14)
        lo = np.min(a[:,:2]-a[:,2,None], axis=0)
        hi = np.max(a[:,:2]+a[:,2,None], axis=0)
        a[:,:2] -= (lo+hi)/2
        a *= (2/(np.sum(hi-lo)))*(1-3e-14)
        score = float(np.sum(a[:,2]))
        if score > best_score:
            candidate = {'circles': a.tolist()}
            value = exact_score(candidate)
            if value > best_score:
                best, best_score = candidate, value

    def family(rows):
        # Each noncentral circle is paired with its reflection in x=0.
        specs = []
        nvar = 1
        for row, count in enumerate(rows):
            if time.monotonic() >= deadline:
                raise StopSearch
            if count % 2:
                specs.append((row, 0.0, None, nvar, nvar+1))
                nvar += 2
            for k in range(count//2):
                xpos = k+1 if count % 2 else k+0.5
                specs.append((row, xpos, nvar, nvar+1, nvar+2))
                nvar += 3
        X = np.zeros((21,nvar)); Y = X.copy(); R = X.copy()
        q = np.zeros(nvar)
        bounds = [(0.2,1.8)] + [(-1.0,1.0)]*(nvar-1)
        representatives = []
        mirror = []
        index = 0
        rad = 1/(max(rows)+1+(len(rows)-1)*np.sqrt(3)/2)
        for row, xpos, xv, yv, rv in specs:
            if time.monotonic() >= deadline:
                raise StopSearch
            representatives.append(index)
            q[yv] = (row-(len(rows)-1)/2)*np.sqrt(3)*rad
            q[rv] = rad*0.98
            bounds[yv] = (-0.99,0.99)
            bounds[rv] = (0.0001,0.5)
            Y[index,yv] = R[index,rv] = 1
            if xv is None:
                mirror.append(index)
                index += 1
            else:
                q[xv] = 2*rad*xpos
                bounds[xv] = (0.0001,0.99)
                X[index,xv] = 1
                X[index+1,xv] = -1
                Y[index+1,yv] = R[index+1,rv] = 1
                mirror.extend([index+1,index])
                index += 2
        q[0] = 2*max(rows)*rad
        pairs = []
        for i,j in zip(ii,jj):
            if time.monotonic() >= deadline:
                raise StopSearch
            reflected = tuple(sorted((mirror[i],mirror[j])))
            if (int(i),int(j)) <= reflected:
                pairs.append((i,j))
        pi = np.array([p[0] for p in pairs]); pj = np.array([p[1] for p in pairs])
        DX = X[pi]-X[pj]
        DY = Y[pi]-Y[pj]
        RR = R[pi]+R[pj]
        linear = []
        constants = []
        for k in representatives:
            if time.monotonic() >= deadline:
                raise StopSearch
            v = -X[k]-R[k]; v[0] += 0.5
            linear.append(v); constants.append(0.)
            v = -Y[k]-R[k]; v[0] -= 0.5
            linear.append(v); constants.append(1.)
            v = Y[k]-R[k]; v[0] -= 0.5
            linear.append(v); constants.append(1.)
        L = np.array(linear); C = np.array(constants)
        obj = -np.sum(R,axis=0)
        cache_q = None
        cache_f = None
        cache_j = None

        def compute(z):
            nonlocal cache_q, cache_f, cache_j
            if time.monotonic() >= deadline:
                raise StopSearch
            if cache_q is None or not np.array_equal(z,cache_q):
                dx = DX@z; dy = DY@z
                dist = np.sqrt(dx*dx+dy*dy)
                safe = np.maximum(dist,1e-15)
                cache_f = np.concatenate((dist-RR@z,L@z+C))
                cache_j = np.vstack(((dx[:,None]*DX+dy[:,None]*DY)/safe[:,None]-RR,L))
                cache_q = z.copy()
            return cache_f,cache_j

        last = q.copy()
        calls = 0
        def callback(z):
            nonlocal last,calls
            if time.monotonic() >= deadline:
                raise StopSearch
            last = z.copy()
            calls += 1
            if calls % 5 == 0:
                consider(np.column_stack((X@z,Y@z,R@z)))
        consider(np.column_stack((X@q,Y@q,R@q)))
        try:
            result = minimize(lambda z: float(obj@z),q,jac=lambda z: obj,
                method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda z:compute(z)[0], 'jac':lambda z:compute(z)[1]},
                callback=callback,
                options={'maxiter':int(params['iterations']),'ftol':2e-12,'disp':False})
            last = result.x
        finally:
            if time.monotonic() < deadline:
                consider(np.column_stack((X@last,Y@last,R@last)))

    # Different orders alter the adjacency graph even with identical row populations.
    families = [(4,5,4,5,3),(4,5,3,5,4),(5,4,3,4,5),
                (5,4,5,4,3),(3,5,5,5,3),(4,4,5,4,4)]
    try:
        for rows in families:
            if time.monotonic() >= deadline:
                break
            family(rows)
    except StopSearch:
        pass
    if best is None:
        # An explicitly separated grid, used only if there was no valid incumbent.
        a = np.array([[0.18*(i%5),0.18*(i//5),0.08] for i in range(21)])
        best = {'circles':a.tolist()}
    return best
