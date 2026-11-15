import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {'iterations': (120, 40, 220), 'probe': (0.23, 0.10, 0.45), 'jitter': (0.022, 0.003, 0.09)}

class Halt(Exception):
    pass

class SliceEnd(Exception):
    pass

def canonical(obj):
    if obj is None or 'circles' in obj:
        return obj
    rows = []
    for x, y, r, m in obj['orbits']:
        rows.append([x, y, r])
        if m == 2:
            rows.append([x, -y, r])
    return {'circles': rows}

def exact_score(obj):
    try:
        a = np.asarray(canonical(obj)['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        perimeter = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        perimeter += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        if perimeter > 2:
            return -float('inf')
        i,j = np.triu_indices(21,1)
        d = np.sqrt((a[i,0]-a[j,0])**2+(a[i,1]-a[j,1])**2)
        if np.any(d < a[i,2]+a[j,2]):
            return -float('inf')
        return float(np.sum(a[:,2]))
    except (ValueError, TypeError, KeyError, OverflowError):
        return -float('inf')

def construct(params, seed, budget_s, init=None):
    began = time.monotonic()
    deadline = began + .70*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = exact_score(init)
    fallback = {'circles': [[.2*(k%7), .2*(k//7), .09999999999998] for k in range(21)]}
    val = exact_score(fallback)
    if val > best_score:
        best, best_score = fallback, val
    if time.monotonic() >= deadline:
        return best

    def tick():
        if time.monotonic() >= deadline:
            raise Halt()

    ids = np.array([0,1,2]+[g for g in range(3,12) for _ in (0,1)])
    signs = np.array([0.,0.,0.]+[s for _ in range(9) for s in (1.,-1.)])
    X = np.zeros((21,34)); Y = X.copy(); R = X.copy()
    for k in range(21):
        if time.monotonic() >= deadline:
            return best
        g = ids[k]
        X[k,g] = 1.; R[k,21+g] = 1.
        if g >= 3:
            Y[k,9+g] = signs[k]
    pairs = []; seen = set()
    for i in range(21):
        if time.monotonic() >= deadline:
            return best
        for j in range(i):
            if time.monotonic() >= deadline:
                return best
            key = (min(ids[i],ids[j]),max(ids[i],ids[j]),signs[i]*signs[j])
            if key not in seen:
                seen.add(key); pairs.append((i,j))
    pi,pj = np.array(pairs).T
    DX = X[pi]-X[pj]; DY = Y[pi]-Y[pj]; RS = R[pi]+R[pj]
    W = np.zeros((36,34)); offset = np.zeros(36)
    for g in range(12):
        if time.monotonic() >= deadline:
            return best
        W[g,g] = 1.; W[g,21+g] = -1.; W[g,33] = .5
        W[12+g,g] = -1.; W[12+g,21+g] = -1.; W[12+g,33] = .5
        W[24+g,21+g] = -1.; W[24+g,33] = -.5; offset[24+g] = 1.
        if g >= 3:
            W[24+g,9+g] = -1.
    objective = np.zeros(34); objective[21:24] = -1.; objective[24:33] = -2.
    bounds = [(-.99,.99)]*12+[(0.,.99)]*9+[(1e-7,.49)]*12+[(.12,1.88)]
    ii,jj = np.triu_indices(21,1)
    best_z = None

    def retain(z):
        nonlocal best, best_score, best_z
        if not np.isfinite(z).all() or np.any(z[21:33] <= 0):
            return -float('inf')
        a = np.column_stack((X@z,Y@z,R@z))
        d = np.sqrt((a[ii,0]-a[jj,0])**2+(a[ii,1]-a[jj,1])**2)
        shrink = min(1.,float(np.min(d/(a[ii,2]+a[jj,2]))))*(1.-4e-14)
        if shrink <= 0:
            return -float('inf')
        a[:,2] *= shrink
        span = np.max(a[:,:2]+a[:,2,None],axis=0)-np.min(a[:,:2]-a[:,2,None],axis=0)
        scale = 2./float(np.sum(span))*(1.-4e-14)
        a *= scale
        score = float(np.sum(a[:,2]))
        if score > best_score:
            obj = {'circles': a.tolist()}
            value = exact_score(obj)
            if value > best_score:
                best, best_score = obj, value
                best_z = z.copy(); best_z[:33] *= scale
                best_z[21:33] *= shrink; best_z[33] = float(span[0]*scale)
        return score

    def make_z(groups):
        z = np.zeros(34)
        z[:12] = groups[:,0]; z[12:21] = np.abs(groups[3:,1]); z[21:33] = groups[:,2]
        width = np.max(groups[:,0]+groups[:,2])-np.min(groups[:,0]-groups[:,2])
        height = 2*np.max(np.abs(groups[:,1])+groups[:,2])
        z[:33] *= 2/(width+height); z[33] = 2*width/(width+height)
        return z

    starts = []
    try:
        if init is not None:
            a = np.asarray(canonical(init)['circles'],float).copy()
            if a.shape == (21,3) and np.isfinite(a).all() and np.all(a[:,2]>0):
                lo = np.min(a[:,:2]-a[:,2,None],axis=0)
                hi = np.max(a[:,:2]+a[:,2,None],axis=0)
                a[:,:2] -= (lo+hi)/2
                projections = []
                for swap in (False,True):
                    tick()
                    b = a.copy()
                    if swap:
                        b[:,[0,1]] = b[:,[1,0]]
                    order = np.argsort(b[:,1]); lower = order[:9]; singles = order[9:12]; upper = order[12:]
                    reflected = b[lower].copy(); reflected[:,1] *= -1
                    cost = np.sum((b[upper,None,:]-reflected[None,:,:])**2,axis=2)
                    u,v = linear_sum_assignment(cost)
                    groups = np.vstack((np.column_stack((b[singles,0],np.zeros(3),b[singles,2])),(b[upper[u]]+reflected[v])/2))
                    projections.append((float(np.sum(cost[u,v])+np.sum(b[singles,1]**2)),make_z(groups)))
                projections.sort(key=lambda p:p[0])
                starts.append(projections[0][1])
                if projections[1][0] < .003:
                    starts.append(projections[1][1])
        for counts in ((4,5),(5,4),(3,3,3)):
            tick()
            groups = [[-.2,0,.095],[0,0,.095],[.2,0,.095]]
            for row,n in enumerate(counts):
                tick()
                for k in range(n):
                    tick()
                    groups.append([.2*(k-(n-1)/2), (row+1)*np.sqrt(3)*.1,.095])
            starts.append(make_z(np.asarray(groups)))
    except Halt:
        return best
    except (ValueError,TypeError,KeyError):
        pass
    if not starts:
        return best

    slice_end = deadline
    cache_z = None; cache_g = None; cache_j = None
    def check():
        tick()
        if time.monotonic() >= slice_end:
            raise SliceEnd()
    def calc(z):
        nonlocal cache_z,cache_g,cache_j
        check()
        if cache_z is None or not np.array_equal(z,cache_z):
            dx,dy = DX@z,DY@z
            d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-25)
            cache_z = z.copy()
            cache_g = np.r_[d-RS@z,W@z+offset]
            cache_j = np.vstack(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RS,W))
        return cache_g,cache_j
    def fun(z):
        check(); return float(objective@z)
    def jac(z):
        check(); return objective
    last = None
    pool = []
    attempt = 0
    try:
        while time.monotonic() < deadline-.015:
            tick()
            if attempt < len(starts):
                z = starts[attempt].copy()
                duration = float(params['probe'])
            elif best_z is not None:
                z = best_z.copy()
                duration = .32
                if attempt > len(starts):
                    z[:21] += rng.normal(0,float(params['jitter']),21)
                    z[12:21] = np.maximum(z[12:21],.001)
            elif pool:
                z = max(pool,key=lambda p:p[0])[1].copy(); duration = .30
            else:
                z = starts[0].copy(); duration = .30
            last = z.copy()
            slice_end = min(deadline-.006,time.monotonic()+duration)
            calls = 0
            def callback(q):
                nonlocal last,calls
                last = q.copy(); calls += 1
                check()
                if calls%10 == 0:
                    retain(q)
            try:
                result = minimize(fun,z,jac=jac,method='SLSQP',bounds=bounds,
                    constraints={'type':'ineq','fun':lambda q:calc(q)[0],'jac':lambda q:calc(q)[1]},
                    callback=callback,options={'maxiter':int(params['iterations']),'ftol':8e-13,'disp':False})
                last = result.x
            except SliceEnd:
                pass
            score = retain(last)
            pool.append((score,last.copy()))
            attempt += 1
    except (Halt,ValueError,FloatingPointError):
        if last is not None and time.monotonic() < began+.77*budget_s:
            retain(last)
    return best
