import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {"stage_iterations": (85, 30, 150), "release": (0.012, 0.002, 0.04), "time_fraction": (0.70, 0.5, 0.75)}


def canonical(obj):
    if obj is None:
        return None
    if 'circles' in obj:
        return obj
    rows = []
    for x, y, r, m in obj['orbits']:
        rows.append([x, y, r])
        if m == 2:
            rows.append([x, -y, r])
    return {'circles': rows}


def score(obj):
    try:
        a = np.asarray(canonical(obj)['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        x = list(map(lambda v: Fraction(float(v)), a[:, 0]))
        y = list(map(lambda v: Fraction(float(v)), a[:, 1]))
        r = list(map(lambda v: Fraction(float(v)), a[:, 2]))
        w = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        h = max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        if w+h > 2:
            return -float('inf')
        for i in range(21):
            for j in range(i):
                dx = a[i,0]-a[j,0]
                dy = a[i,1]-a[j,1]
                if np.sqrt(dx*dx+dy*dy) < a[i,2]+a[j,2]:
                    return -float('inf')
        return float(np.sum(a[:,2]))
    except Exception:
        return -float('inf')


class StopSearch(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + float(params['time_fraction'])*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = score(init)

    def tick():
        if time.monotonic() >= deadline:
            raise StopSearch()

    # A valid fallback independent of the input representation.
    fallback = {'circles': [[(i%7)*0.2, (i//7)*0.2, 0.09999999999999] for i in range(21)]}
    fs = score(fallback)
    if fs > best_score:
        best, best_score = fallback, fs
    if time.monotonic() >= deadline:
        return best

    groups = np.array([0,1,2]+[g for g in range(3,12) for _ in range(2)])
    signs = np.array([0.,0.,0.]+[s for _ in range(9) for s in (1.,-1.)])
    X = np.zeros((21,34)); Y = np.zeros((21,34)); R = np.zeros((21,34))
    for k,g in enumerate(groups):
        tick()
        X[k,g] = 1
        R[k,21+g] = 1
        if g >= 3:
            Y[k,9+g] = signs[k]
    pairs = []
    seen = set()
    for i in range(21):
        tick()
        for j in range(i):
            tick()
            key = (min(groups[i],groups[j]),max(groups[i],groups[j]),int(signs[i]*signs[j]))
            if key not in seen:
                seen.add(key)
                pairs.append((i,j))
    pi,pj = np.array(pairs).T
    DX = X[pi]-X[pj]; DY = Y[pi]-Y[pj]; RR = R[pi]+R[pj]
    walls = np.zeros((36,34)); offset = np.zeros(36)
    for g in range(12):
        tick()
        walls[g,g] = 1; walls[g,21+g] = -1; walls[g,33] = .5
        walls[12+g,g] = -1; walls[12+g,21+g] = -1; walls[12+g,33] = .5
        walls[24+g,21+g] = -1; walls[24+g,33] = -.5; offset[24+g] = 1
        if g >= 3:
            walls[24+g,9+g] = -1
    weights = np.array([1.]*3+[2.]*9)
    grad = np.zeros(34); grad[21:33] = -weights
    bounds = [(-1.,1.)]*12+[(0.,1.)]*9+[(1e-7,.6)]*12+[(.15,1.85)]
    ii,jj = np.triu_indices(21,1)
    release = np.zeros(len(pi))
    best_vector = None
    cache_v = None
    cache_c = None
    cache_j = None

    def retain(v):
        nonlocal best, best_score, best_vector
        tick()
        if not np.isfinite(v).all() or np.min(v[21:33]) <= 0:
            return
        a = np.column_stack((X@v,Y@v,R@v))
        dx = a[ii,0]-a[jj,0]; dy = a[ii,1]-a[jj,1]
        shrink = min(1.,float(np.min(np.sqrt(dx*dx+dy*dy)/(a[ii,2]+a[jj,2]))))
        if shrink <= 0:
            return
        r = v[21:33]*shrink*(1.-2e-14)
        x = v[:12]; y = np.r_[np.zeros(3),v[12:21]]
        w = np.max(x+r)-np.min(x-r)
        h = 2*np.max(y+r)
        scale = min(1.,2./(w+h))*(1.-2e-14)
        estimate = float(weights@r)*scale
        if estimate <= best_score:
            return
        obj = {'orbits': [[float(x[g]*scale),float(y[g]*scale),float(r[g]*scale),1 if g<3 else 2] for g in range(12)]}
        val = score(obj)
        if val > best_score:
            best = obj; best_score = val
            best_vector = v.copy()

    def calculate(v):
        nonlocal cache_v,cache_c,cache_j
        tick()
        if cache_v is None or not np.array_equal(v,cache_v):
            dx = DX@v; dy = DY@v
            dist = np.maximum(np.sqrt(dx*dx+dy*dy),1e-24)
            cache_c = np.r_[dist-RR@v-release,walls@v+offset]
            cache_j = np.vstack(((dx/dist)[:,None]*DX+(dy/dist)[:,None]*DY-RR,walls))
            cache_v = v.copy()
        return cache_c,cache_j

    def objective(v):
        tick()
        return float(grad@v)

    def objective_jac(v):
        tick()
        return grad

    def solve(v, n):
        nonlocal cache_v
        tick()
        cache_v = None
        ans = minimize(objective,v,jac=objective_jac,method='SLSQP',bounds=bounds,
            constraints={'type':'ineq','fun':lambda q:calculate(q)[0], 'jac':lambda q:calculate(q)[1]},
            callback=retain, options={'maxiter':int(n),'ftol':2e-13,'disp':False})
        retain(ans.x)
        return ans.x

    def structured(counts, variant=0):
        tick()
        z = np.zeros(34)
        z[:3] = [-.28,0,.28]
        z[21:24] = .14
        k = 3
        for layer,count in enumerate(counts):
            tick()
            spacing = .2 if len(counts)==2 else .205
            yy = (.225+.18*layer) if len(counts)==2 else (.18*(layer+1))
            for j in range(count):
                tick()
                z[k] = (j-(count-1)/2)*spacing
                if variant and layer%2 == 0:
                    z[k] += .025
                z[9+k] = yy
                z[21+k] = .09 if len(counts)==2 else .1
                k += 1
        if len(counts)==3:
            z[:3] = [-.205,0,.205]
            z[21:24] = .1
        z[33] = max(.5,2*float(np.max(np.abs(z[:12])+z[21:33])))
        return z

    try:
        seeds = []
        if init is not None and score(init) > 2.0:
            a = np.asarray(canonical(init)['circles'],float).copy()
            lo = np.min(a[:,:2]-a[:,2,None],axis=0)
            hi = np.max(a[:,:2]+a[:,2,None],axis=0)
            a[:,:2] -= (lo+hi)/2
            projections = []
            for swap in (False,True):
                tick()
                b = a.copy()
                if swap:
                    b[:,[0,1]] = b[:,[1,0]]
                order = np.argsort(b[:,1]); lower=order[:9]; singles=order[9:12]; upper=order[12:]
                ref = b[lower].copy(); ref[:,1] *= -1
                cost = np.sum((b[upper,None,:]-ref[None,:,:])**2,axis=2)
                u,v = linear_sum_assignment(cost)
                g = np.vstack((np.column_stack((b[singles,0],np.zeros(3),b[singles,2])),(b[upper[u]]+ref[v])/2))
                error = float(cost[u,v].sum()+np.sum(b[singles,1]**2))
                z = np.zeros(34); z[:12]=g[:,0]; z[12:21]=np.abs(g[3:,1]); z[21:33]=g[:,2]
                z[33] = float(np.max(g[:,0]+g[:,2])-np.min(g[:,0]-g[:,2]))
                projections.append((error,z))
            seeds.append(min(projections,key=lambda t:t[0])[1])
        seeds.extend([structured([5,4]),structured([4,5]),structured([3,3,3]),structured([4,3,2],1)])
        for z in seeds:
            tick()
            solve(z,int(params['stage_iterations']))
        if best_vector is not None:
            anchor = solve(best_vector.copy(),120)
            attempt = 0
            while time.monotonic() < deadline:
                tick()
                release[:] = 0
                dx=DX@anchor; dy=DY@anchor
                gap=np.sqrt(dx*dx+dy*dy)-RR@anchor
                near=np.flatnonzero(gap < .002)
                if len(near)<3:
                    break
                chosen=rng.choice(near,min(4,len(near)),replace=False)
                release[chosen]=float(params['release'])*rng.uniform(.7,1.3,len(chosen))
                displaced=solve(anchor.copy(),20)
                release[:]=0
                recovered=solve(displaced,65)
                if float(grad@recovered)<float(grad@anchor) and np.min(calculate(recovered)[0])>-1e-7:
                    anchor=recovered
                attempt += 1
    except (StopSearch,ValueError,FloatingPointError):
        pass
    return best
