import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {"primary_fraction": (0.47, 0.25, 0.58), "jitter": (0.004, 0.001, 0.02)}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best = None
    best_score = -float('inf')
    I, J = np.triu_indices(21, 1)

    def score(obj):
        try:
            a = np.asarray(obj['circles'], dtype=float)
            if a.shape != (21, 3) or not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
                return -float('inf')
            dx = a[I, 0] - a[J, 0]
            dy = a[I, 1] - a[J, 1]
            if np.any(np.sqrt(dx**2 + dy**2) < a[I, 2] + a[J, 2]):
                return -float('inf')
            x = [Fraction(float(v)) for v in a[:, 0]]
            y = [Fraction(float(v)) for v in a[:, 1]]
            r = [Fraction(float(v)) for v in a[:, 2]]
            extent = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
            extent += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
            if extent > 2:
                return -float('inf')
            return float(np.sum(a[:, 2]))
        except Exception:
            return -float('inf')

    if init is not None:
        if 'circles' in init:
            initial = init
        else:
            circles = []
            for x,y,r,m in init['orbits']:
                if time.monotonic() >= deadline:
                    break
                circles.append([x,y,r])
                if m == 2:
                    circles.append([x,-y,r])
            initial = {'circles': circles}
        s = score(initial)
        if s > best_score:
            best, best_score = initial, s
    else:
        initial = None

    def consider(a):
        nonlocal best, best_score
        if time.monotonic() >= deadline + 0.03:
            return
        a = np.asarray(a, dtype=float).copy()
        if a.shape != (21,3) or not np.isfinite(a).all() or np.min(a[:,2]) <= 0:
            return
        d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
        ratio = np.min(d/(a[I,2]+a[J,2]))
        a[:,2] *= min(1.0, ratio)*(1.0-3e-14)
        w = np.max(a[:,0]+a[:,2])-np.min(a[:,0]-a[:,2])
        h = np.max(a[:,1]+a[:,2])-np.min(a[:,1]-a[:,2])
        a *= min(1.0, 2.0/(w+h))*(1.0-3e-14)
        obj = {'circles': a.tolist()}
        s = score(obj)
        if s > best_score:
            best, best_score = obj, s

    fallback = np.array([[0.2*i,0.2*j,0.1*(1-1e-12)] for i in range(7) for j in range(3)])
    consider(fallback)

    p, q = np.triu_indices(12, 1)
    rows = np.arange(len(p))
    Y = np.zeros((12,34))
    Y[np.arange(3,12),np.arange(12,21)] = 1
    DX = np.zeros((len(p),34))
    DX[rows,p] = 1
    DX[rows,q] = -1
    DY = Y[p]-Y[q]
    RS = np.zeros((len(p),34))
    RS[rows,21+p] = 1
    RS[rows,21+q] = 1
    L = np.zeros((45,34))
    z = np.arange(12)
    L[z,z] = 1
    L[z,21+z] = -1
    L[12+z,z] = -1
    L[12+z,21+z] = -1
    L[12+z,33] = 1
    L[24:36] = -Y
    L[24+z,21+z] = -1
    L[24+z,33] = -0.5
    L[36:45] = Y[3:]
    L[np.arange(36,45),np.arange(24,33)] = -1
    offset = np.zeros(45)
    offset[24:36] = 1
    objective = np.zeros(34)
    objective[21:24] = -1
    objective[24:33] = -2
    bounds = [(None,None)]*12+[(0.000001,1.0)]*9+[(0.000001,0.6)]*12+[(0.2,1.8)]

    def expand(v):
        out = []
        for k in range(12):
            if time.monotonic() >= deadline + 0.025:
                return None
            yy = 0.0 if k < 3 else v[12+k-3]
            out.append([v[k], yy, v[21+k]])
            if k >= 3:
                out.append([v[k], -yy, v[21+k]])
        return np.asarray(out)

    def projection(a, swap):
        a = a.copy()
        if swap:
            a[:,[0,1]] = a[:,[1,0]]
        mid = (np.max(a[:,1]+a[:,2])+np.min(a[:,1]-a[:,2]))/2
        a[:,1] -= mid
        order = np.argsort(np.abs(a[:,1]))
        singles = order[:3]
        rem = order[3:]
        rem = rem[np.argsort(a[rem,1])]
        lo, hi = rem[:9], rem[9:]
        cost = (a[hi,None,0]-a[lo,0])**2+(a[hi,None,1]+a[lo,1])**2
        u,v = linear_sum_assignment(cost)
        hi,lo = hi[u],lo[v]
        v0 = np.zeros(34)
        v0[:3] = a[singles,0]
        v0[3:12] = (a[hi,0]+a[lo,0])/2
        v0[12:21] = (a[hi,1]-a[lo,1])/2
        v0[21:24] = a[singles,2]
        v0[24:33] = (a[hi,2]+a[lo,2])/2
        left = np.min(v0[:12]-v0[21:33])
        v0[:12] -= left
        v0[33] = np.clip(np.max(v0[:12]+v0[21:33]),0.3,1.7)
        residual = float(np.sum(cost[u,v])+np.sum(a[singles,1]**2)*2)
        return residual,v0

    candidates = []
    if initial is not None:
        try:
            a = np.asarray(initial['circles'],dtype=float)
            if a.shape == (21,3) and np.isfinite(a).all():
                consider(a)
                candidates = [projection(a,False),projection(a,True)]
                candidates.sort(key=lambda t:t[0])
        except Exception:
            pass
    if not candidates:
        v = np.zeros(34)
        v[:3] = [0.2,0.5,0.8]
        v[3:12] = np.tile([0.2,0.5,0.8],3)
        v[12:21] = np.repeat([0.15,0.30,0.45],3)
        v[21:33] = 0.07
        v[33] = 1.0
        candidates = [(0,v)]

    endpoints = []
    def solve(v0, until, maxiter=160):
        cache_x = None
        cache_c = None
        cache_j = None
        last = v0.copy()
        def calc(v):
            nonlocal cache_x,cache_c,cache_j,last
            if time.monotonic() >= until:
                raise StopSearch()
            last = v.copy()
            if cache_x is None or not np.array_equal(v,cache_x):
                dx = DX@v
                dy = DY@v
                rr = RS@v
                cache_c = np.concatenate((dx*dx+dy*dy-rr*rr,L@v+offset))
                cache_j = np.vstack((2*(dx[:,None]*DX+dy[:,None]*DY-rr[:,None]*RS),L))
                cache_x = v.copy()
            return cache_c,cache_j
        try:
            result = minimize(lambda v: float(objective@v),v0,jac=lambda v: objective,
                method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda v:calc(v)[0],'jac':lambda v:calc(v)[1]},
                options={'ftol':2e-13,'maxiter':maxiter,'disp':False})
            last = result.x
        except StopSearch:
            pass
        a = expand(last)
        if a is not None:
            consider(a)
            endpoints.append((float(-objective@last),last.copy()))
        return last

    first_end = min(deadline-0.08,start+params['primary_fraction']*budget_s)
    v = solve(candidates[0][1],first_end)
    if time.monotonic() < deadline-0.07:
        if best_score < 2.36 and len(candidates)>1:
            v = solve(candidates[1][1],min(deadline-0.04,time.monotonic()+0.22*budget_s))
        elif endpoints:
            v = solve(max(endpoints,key=lambda t:t[0])[1],min(deadline-0.04,time.monotonic()+0.13*budget_s),90)
    while time.monotonic() < deadline-0.08:
        if not endpoints:
            break
        base = max(endpoints,key=lambda t:t[0])[1].copy()
        base[:21] += rng.normal(0,params['jitter'],21)
        base[12:21] = np.maximum(base[12:21],0.001)
        solve(base,deadline-0.025,100)
    return best
