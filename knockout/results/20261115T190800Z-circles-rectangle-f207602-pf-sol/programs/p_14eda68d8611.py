import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {'probe_iterations': (48, 20, 100), 'polish_iterations': (95, 30, 160), 'jitter': (0.009, 0.001, 0.04)}

class StageEnd(Exception):
    pass

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
            return -np.inf
        i, j = np.triu_indices(21, 1)
        d = a[i, :2] - a[j, :2]
        if np.any(np.sqrt(d[:, 0]**2 + d[:, 1]**2) < a[i, 2] + a[j, 2]):
            return -np.inf
        x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
        perimeter = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        perimeter += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        return float(np.sum(a[:, 2])) if perimeter <= 2 else -np.inf
    except Exception:
        return -np.inf

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + .69*budget_s
    stage_deadline = deadline
    rng = np.random.default_rng(seed)
    best = init
    best_score = score(init)
    grid = np.array([[2*(i%7), 2*(i//7), 1.] for i in range(21)])/10.000000000001
    fallback = {'circles': grid.tolist()}
    fs = score(fallback)
    if fs > best_score:
        best, best_score = fallback, fs

    def tick():
        if time.monotonic() >= min(deadline, stage_deadline):
            raise StageEnd

    groups = np.r_[np.arange(3), np.repeat(np.arange(3,12),2)]
    signs = np.r_[np.zeros(3), np.tile([1.,-1.],9)]
    X = np.zeros((21,34)); Y = X.copy(); R = X.copy()
    X[np.arange(21),groups] = 1
    R[np.arange(21),21+groups] = 1
    mask = groups >= 3
    Y[np.flatnonzero(mask),9+groups[mask]] = signs[mask]
    pairs = []; seen = set()
    for i in range(21):
        if time.monotonic() >= deadline:
            return best
        for j in range(i):
            if time.monotonic() >= deadline:
                return best
            key = (min(groups[i],groups[j]),max(groups[i],groups[j]),signs[i]*signs[j])
            if key not in seen:
                seen.add(key); pairs.append((i,j))
    pi,pj = np.asarray(pairs).T
    DX = X[pi]-X[pj]; DY = Y[pi]-Y[pj]; RS = R[pi]+R[pj]
    walls = np.zeros((36,34)); offset = np.zeros(36)
    for g in range(12):
        if time.monotonic() >= deadline:
            return best
        walls[g,g] = 1; walls[g,21+g] = -1; walls[g,33] = .5
        walls[12+g,g] = -1; walls[12+g,21+g] = -1; walls[12+g,33] = .5
        walls[24+g,21+g] = -1; walls[24+g,33] = -.5; offset[24+g] = 1
        if g >= 3:
            walls[24+g,9+g] = -1
    grad = np.zeros(34); grad[21:33] = -np.r_[np.ones(3),np.full(9,2.)]
    bounds = [(-1.,1.)]*12 + [(0.,1.)]*9 + [(1e-6,.5)]*12 + [(.12,1.88)]
    ii,jj = np.triu_indices(21,1)
    best_z = None
    cache_v = None; cache_c = None; cache_j = None

    def retain(v):
        nonlocal best, best_score, best_z
        if time.monotonic() >= deadline or not np.isfinite(v).all() or np.any(v[21:33] <= 0):
            return
        a = np.column_stack((X@v,Y@v,R@v))
        d = a[ii,:2]-a[jj,:2]
        ratio = np.min(np.sqrt(d[:,0]**2+d[:,1]**2)/(a[ii,2]+a[jj,2]))
        a[:,2] *= max(0.,min(1.,float(ratio)))*(1.-3e-14)
        if np.any(a[:,2] <= 0):
            return
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        a[:,:2] -= (lo+hi)/2
        a *= min(1.,2./float(np.sum(hi-lo)))*(1.-3e-14)
        if float(np.sum(a[:,2])) <= best_score:
            return
        obj = {'circles':a.tolist()}
        s = score(obj)
        if s > best_score:
            best, best_score, best_z = obj, s, v.copy()

    def calculate(v):
        nonlocal cache_v, cache_c, cache_j
        tick()
        if cache_v is not None and np.array_equal(v,cache_v):
            return
        dx = DX@v; dy = DY@v
        dist = np.maximum(np.sqrt(dx*dx+dy*dy),1e-25)
        cache_c = np.r_[dist-RS@v,walls@v+offset]
        cache_j = np.vstack(((dx/dist)[:,None]*DX+(dy/dist)[:,None]*DY-RS,walls))
        cache_v = v.copy()

    def con(v):
        calculate(v); return cache_c
    def cjac(v):
        calculate(v); return cache_j
    def fun(v):
        tick(); return float(grad@v)
    def jac(v):
        tick(); return grad

    def solve(z, seconds, iterations):
        nonlocal stage_deadline, cache_v
        stage_deadline = min(deadline,time.monotonic()+max(.001,seconds))
        cache_v = None
        last = z.copy()
        calls = 0
        def cb(v):
            nonlocal last,calls
            last = v.copy(); calls += 1
            tick()
            if calls % 3 == 0:
                retain(v)
        try:
            res = minimize(fun,z,jac=jac,method='SLSQP',bounds=bounds,
                           constraints={'type':'ineq','fun':con,'jac':cjac},callback=cb,
                           options={'maxiter':iterations,'ftol':3e-13,'disp':False})
            last = res.x.copy()
        except (StageEnd, ValueError, FloatingPointError):
            pass
        stage_deadline = deadline
        retain(last)
        return last

    def row_seed(counts):
        z = np.zeros(34)
        z[:3] = [-.28,0.,.28]; z[21:24] = .145
        k = 3
        for row,count in enumerate(counts):
            if time.monotonic() >= deadline:
                return z
            spacing = .235 if count >= 4 else .27
            for x in (np.arange(count)-(count-1)/2)*spacing:
                if time.monotonic() >= deadline:
                    return z
                z[k] = x; z[9+k] = (row+1)*(.205 if len(counts)==2 else .215)
                z[21+k] = .105 if count >= 5 else .117
                k += 1
        w = 2*np.max(np.abs(z[:12])+z[21:33])
        h = 2*np.max(np.r_[np.zeros(3),z[12:21]]+z[21:33])
        z[:33] *= 2/(w+h); z[33] = 2*w/(w+h)
        return z

    projections = []
    try:
        if init is not None:
            a = np.asarray(canonical(init)['circles'],float).copy()
            if a.shape == (21,3) and np.isfinite(a).all():
                lo = np.min(a[:,:2]-a[:,2,None],axis=0)
                hi = np.max(a[:,:2]+a[:,2,None],axis=0)
                a[:,:2] -= (lo+hi)/2
                for swap in (False,True):
                    tick()
                    b = a[:,[1,0,2]].copy() if swap else a.copy()
                    order = np.argsort(b[:,1]); low=order[:9]; singles=order[9:12]; up=order[12:]
                    reflected = b[low].copy(); reflected[:,1] *= -1
                    cost = np.sum((b[up,None,:]-reflected[None,:,:])**2,axis=2)
                    p,q = linear_sum_assignment(cost)
                    g = np.vstack((b[singles],(b[up[p]]+reflected[q])/2))
                    error = np.sum(cost[p,q])+np.sum(g[:3,1]**2)
                    z = np.zeros(34); z[:12]=g[:,0]; z[12:21]=np.abs(g[3:,1]); z[21:33]=np.maximum(g[:,2],1e-6)
                    z[33]=np.clip(hi[int(swap)]-lo[int(swap)],.12,1.88)
                    projections.append((error,z))
        projections.sort(key=lambda v:v[0])
        duration = max(.01,deadline-time.monotonic())
        anchors = []
        if projections:
            anchors.append(solve(projections[0][1],.32*duration,int(params['polish_iterations'])))
        for counts in ((5,4),(4,5)):
            tick()
            anchors.append(solve(row_seed(counts),.20*duration,int(params['probe_iterations'])))
        tick()
        if best_z is not None:
            anchor = best_z.copy()
        else:
            anchor = min(anchors,key=lambda z:float(grad@z)).copy()
        anchor = solve(anchor,.18*duration,int(params['polish_iterations']))
        while time.monotonic() < deadline:
            tick()
            base = best_z.copy() if best_z is not None else anchor.copy()
            base[:21] += rng.normal(0,float(params['jitter']),21)
            base[12:21] = np.maximum(base[12:21],0)
            anchor = solve(base,min(.12*duration,deadline-time.monotonic()),35)
    except (StageEnd,ValueError,FloatingPointError):
        pass
    return best
