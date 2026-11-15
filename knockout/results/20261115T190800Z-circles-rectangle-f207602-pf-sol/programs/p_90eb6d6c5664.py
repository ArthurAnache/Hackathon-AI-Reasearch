import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {'fraction': (0.70, 0.5, 0.75), 'iterations': (170, 50, 300)}


def exact_score(c):
    try:
        a = np.asarray(c['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        w = max(u + v for u, v in zip(x, r)) - min(u - v for u, v in zip(x, r))
        h = max(u + v for u, v in zip(y, r)) - min(u - v for u, v in zip(y, r))
        if w + h > 2:
            return -float('inf')
        for i, j in itertools.combinations(range(21), 2):
            if np.sqrt((a[i, 0]-a[j, 0])**2 + (a[i, 1]-a[j, 1])**2) < a[i, 2]+a[j, 2]:
                return -float('inf')
        return float(np.sum(a[:, 2]))
    except Exception:
        return -float('inf')


class StopSearch(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + params['fraction'] * budget_s
    best = init if init is not None and 'circles' in init else None
    if init is not None and 'centers' in init:
        best = {'circles': [[float(p[0]), float(p[1]), float(r)] for p, r in zip(init['centers'], init['radii'])]}
    score = exact_score(best) if best is not None else -float('inf')
    fallback = {'circles': [[(i % 7 + .5)/5, (i // 7 + .5)/5, .09999999999999] for i in range(21)]}
    fs = exact_score(fallback)
    if fs > score:
        best, score = fallback, fs

    def retain(a):
        nonlocal best, score
        if time.monotonic() >= deadline:
            return
        a = np.array(a, dtype=float, copy=True)
        if not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return
        ii, jj = np.triu_indices(21, 1)
        d = np.sqrt(np.sum((a[ii, :2]-a[jj, :2])**2, axis=1))
        factor = min(1., float(np.min(d/(a[ii, 2]+a[jj, 2]))))
        a[:, 2] *= factor * (1-2e-13)
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        a[:, :2] -= (lo+hi)/2
        a *= (2-8e-13)/float(np.sum(hi-lo))
        cand = {'circles': a.tolist()}
        s = exact_score(cand)
        if s > score:
            best, score = cand, s

    def solve(axis, upper):
        if time.monotonic() >= deadline:
            return
        na, nu = len(axis), len(upper)
        k = na+nu
        nvar = 3*k+1
        X = np.zeros((21, nvar)); Y = X.copy(); R = X.copy()
        for i in range(k):
            X[i, 3*i] = 1
            R[i, 3*i+2] = 1
            if i >= na:
                Y[i, 3*i+1] = 1
        for j in range(nu):
            i = na+j
            X[k+j] = X[i]
            Y[k+j] = -Y[i]
            R[k+j] = R[i]
        reps = list(axis)+list(upper)
        z = np.zeros(nvar)
        for i, row in enumerate(reps):
            z[3*i:3*i+3] = row
        full = np.column_stack((X@z, Y@z, R@z))
        lo = np.min(full[:, :2]-full[:, 2,None], axis=0)
        hi = np.max(full[:, :2]+full[:, 2,None], axis=0)
        scale = 2/np.sum(hi-lo)
        for i in range(k):
            z[3*i] = (z[3*i]-lo[0])*scale
            z[3*i+1] *= scale
            z[3*i+2] *= scale
        z[-1] = (hi[0]-lo[0])*scale
        pairs = []
        for i in range(na):
            for j in range(i+1,k):
                if time.monotonic() >= deadline: return
                pairs.append((i,j))
        for i in range(na,k):
            for j in range(i+1,k):
                if time.monotonic() >= deadline: return
                pairs.append((i,j))
            for j in range(i,k):
                if time.monotonic() >= deadline: return
                pairs.append((i,k+j-na))
        p = np.array([u for u,v in pairs]); q = np.array([v for u,v in pairs])
        DX = X[p]-X[q]; DY = Y[p]-Y[q]; SR = R[p]+R[q]
        walls = []
        constants = []
        for i in range(k):
            if time.monotonic() >= deadline: return
            v = X[i]-R[i]; walls.append(v); constants.append(0.)
            v = -X[i]-R[i]; v[-1] += 1; walls.append(v); constants.append(0.)
            v = -Y[i]-R[i]; v[-1] -= .5; walls.append(v); constants.append(1.)
        W = np.array(walls); C = np.array(constants)
        obj = -np.sum(R, axis=0)
        cache_z = None
        cache_f = None
        cache_j = None
        def compute(v):
            nonlocal cache_z, cache_f, cache_j
            if time.monotonic() >= deadline:
                raise StopSearch()
            if cache_z is not None and np.array_equal(v, cache_z):
                return cache_f, cache_j
            dx = DX@v; dy = DY@v
            d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-16)
            cache_f = np.concatenate((d-SR@v, W@v+C))
            cache_j = np.vstack(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-SR, W))
            cache_z = v.copy()
            return cache_f, cache_j
        bounds = []
        for i in range(k):
            if time.monotonic() >= deadline: return
            bounds.extend([(0,2), (0,0) if i < na else (0,1), (1e-7,1)])
        bounds.append((.1,1.9))
        last = z.copy()
        def callback(v):
            nonlocal last
            if time.monotonic() >= deadline: raise StopSearch()
            last = v.copy()
        try:
            res = minimize(lambda v: float(obj@v), z, jac=lambda v: obj,
                bounds=bounds, constraints={'type':'ineq','fun':lambda v: compute(v)[0], 'jac':lambda v: compute(v)[1]},
                method='SLSQP', callback=callback,
                options={'maxiter':params['iterations'], 'ftol':2e-12, 'disp':False})
            last = res.x
        except StopSearch:
            pass
        if time.monotonic() < deadline:
            retain(np.column_stack((X@last,Y@last,R@last)))

    # Sixteen large circles with five interstitial circles, arranged as a cross.
    axis = [(2.,0.,.41421356),(4.,0.,.41421356),(6.,0.,.41421356)]
    upper = [(float(x),float(y),1.) for y in (1,3) for x in (1,3,5,7)]
    upper.append((4.,2.,.41421356))
    solve(axis, upper)
    if time.monotonic() < deadline:
        # Three rows of five large circles, with six triangular-gap inserts.
        axis = [(float(x),0.,1.) for x in (1,3,5,7,9)]
        upper = [(float(x),2.,1.) for x in (1,3,5,7,9)]
        upper += [(float(x),1.,.41421356) for x in (2,4,6)]
        solve(axis, upper)
    layouts = [(3,[4,5]), (3,[3,3,3]), (5,[4,4]), (3,[4,3,2])]
    for a, rows in layouts:
        if time.monotonic() >= deadline: break
        axis = [(2.*j-(a-1),0.,1.) for j in range(a)]
        upper = []
        for level, count in enumerate(rows,1):
            if time.monotonic() >= deadline: break
            upper.extend([(2.*j-(count-1),level*np.sqrt(3.),1.) for j in range(count)])
        if len(axis)+2*len(upper) == 21:
            solve(axis,upper)
    return best
