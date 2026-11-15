import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {'iterations': (95, 40, 180), 'jitter': (0.045, 0.005, 0.15), 'slice': (0.29, 0.12, 0.6)}

class StopSearch(Exception):
    pass

class EndSlice(Exception):
    pass

def canonical(obj):
    if 'circles' in obj:
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
        x = [Fraction(float(t)) for t in a[:, 0]]
        y = [Fraction(float(t)) for t in a[:, 1]]
        r = [Fraction(float(t)) for t in a[:, 2]]
        p = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        p += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        if p > 2:
            return -float('inf')
        i,j = np.triu_indices(21,1)
        d = np.sqrt((a[i,0]-a[j,0])**2+(a[i,1]-a[j,1])**2)
        if np.any(d < a[i,2]+a[j,2]):
            return -float('inf')
        return float(np.sum(a[:,2]))
    except (ValueError, TypeError, KeyError, OverflowError):
        return -float('inf')

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = exact_score(init) if init is not None else -float('inf')
    best_z = None

    def tick():
        if time.monotonic() >= deadline:
            raise StopSearch()

    # Every representation expands into three singles and nine pairs.
    ids = np.array([0,1,2]+[g for g in range(3,12) for _ in (0,1)])
    signs = np.array([0.,0.,0.]+[s for _ in range(9) for s in (1.,-1.)])
    X = np.zeros((21,34)); Y = X.copy(); R = X.copy()
    for k in range(21):
        tick()
        g = ids[k]
        X[k,g] = 1.
        R[k,21+g] = 1.
        if g >= 3:
            Y[k,9+g] = signs[k]
    pairs = []
    seen = set()
    for i in range(21):
        tick()
        for j in range(i):
            tick()
            key = (min(ids[i],ids[j]),max(ids[i],ids[j]),signs[i]*signs[j])
            if key not in seen:
                seen.add(key)
                pairs.append((i,j))
    pi,pj = np.array(pairs).T
    DX = X[pi]-X[pj]; DY = Y[pi]-Y[pj]; RS = R[pi]+R[pj]
    W = np.zeros((36,34)); offset = np.zeros(36)
    for g in range(12):
        tick()
        W[g,g] = 1.; W[g,21+g] = -1.; W[g,33] = .5
        W[12+g,g] = -1.; W[12+g,21+g] = -1.; W[12+g,33] = .5
        W[24+g,21+g] = -1.; W[24+g,33] = -.5
        offset[24+g] = 1.
        if g >= 3:
            W[24+g,9+g] = -1.
    weights = np.r_[np.ones(3),np.full(9,2.)]
    objective = np.zeros(34); objective[21:33] = -weights
    bounds = [(-.99,.99)]*12+[(0.,.99)]*9+[(1e-7,.49)]*12+[(.12,1.88)]
    ii,jj = np.triu_indices(21,1)

    def retain(z):
        nonlocal best,best_score,best_z
        if not np.isfinite(z).all() or np.any(z[21:33] <= 0):
            return
        a = np.column_stack((X@z,Y@z,R@z))
        d = np.sqrt((a[ii,0]-a[jj,0])**2+(a[ii,1]-a[jj,1])**2)
        shrink = min(1.,float(np.min(d/(a[ii,2]+a[jj,2]))))*(1.-3e-14)
        if shrink <= 0:
            return
        a[:,2] *= shrink
        span = np.max(a[:,:2]+a[:,2,None],axis=0)-np.min(a[:,:2]-a[:,2,None],axis=0)
        scale = 2./float(np.sum(span))*(1.-3e-14)
        a *= scale
        if float(np.sum(a[:,2])) <= best_score:
            return
        obj = {'circles': a.tolist()}
        value = exact_score(obj)
        if value > best_score:
            best,best_score = obj,value
            best_z = z.copy()
            best_z[:33] *= scale
            best_z[21:33] *= shrink
            best_z[33] = float(span[0]*scale)

    def make_z(groups):
        z = np.zeros(34)
        z[:12] = groups[:,0]
        z[12:21] = np.abs(groups[3:,1])
        z[21:33] = groups[:,2]
        width = np.max(groups[:,0]+groups[:,2])-np.min(groups[:,0]-groups[:,2])
        height = 2.*np.max(np.abs(groups[:,1])+groups[:,2])
        z[:33] *= 2./(width+height)
        z[33] = 2.*width/(width+height)
        return z

    starts = []
    try:
        if init is not None:
            a = np.asarray(canonical(init)['circles'],float).copy()
            if a.shape == (21,3) and np.isfinite(a).all():
                lo = np.min(a[:,:2]-a[:,2,None],axis=0)
                hi = np.max(a[:,:2]+a[:,2,None],axis=0)
                a[:,:2] -= (lo+hi)/2.
                projections = []
                for swap in (False,True):
                    tick()
                    b = a.copy()
                    if swap:
                        b[:,[0,1]] = b[:,[1,0]]
                    order = np.argsort(b[:,1])
                    lower,singles,upper = order[:9],order[9:12],order[12:]
                    reflected = b[lower].copy(); reflected[:,1] *= -1
                    cost = np.sum((b[upper,None,:]-reflected[None,:,:])**2,axis=2)
                    u,v = linear_sum_assignment(cost)
                    groups = np.vstack((np.column_stack((b[singles,0],np.zeros(3),b[singles,2])),(b[upper[u]]+reflected[v])/2.))
                    groups[3:,1] = np.abs(groups[3:,1])
                    projections.append((float(np.sum(cost[u,v])+np.sum(b[singles,1]**2)),make_z(groups)))
                starts.append(min(projections,key=lambda p:p[0])[1])
    except (ValueError,KeyError,TypeError):
        pass

    # Different row topologies, not repeated perturbations of one contact graph.
    for counts in ((4,5),(5,4),(3,3,3)):
        tick()
        groups = [[-.24,0.,.10],[0.,0.,.10],[.24,0.,.10]]
        for row,n in enumerate(counts):
            tick()
            spacing = .215 if len(counts)==2 else .23
            for k in range(n):
                tick()
                groups.append([(k-(n-1)/2)*spacing,(row+1)*.20,.095])
        starts.append(make_z(np.asarray(groups)))
    if starts:
        retain(starts[-1])
    # An unconditional valid fallback.
    fallback = np.array([[2*(i%7),2*(i//7),1.] for i in range(21)],float)/10.000000000001
    obj = {'circles':fallback.tolist()}
    val = exact_score(obj)
    if val > best_score:
        best,best_score = obj,val

    slice_end = deadline
    calls = 0
    last = None
    def check():
        tick()
        if time.monotonic() >= slice_end:
            raise EndSlice()
    def fun(z):
        check()
        return float(objective@z)
    def jac(z):
        check()
        return objective
    def con(z):
        check()
        dx,dy = DX@z,DY@z
        return np.r_[np.sqrt(dx*dx+dy*dy)-RS@z,W@z+offset]
    def cjac(z):
        check()
        dx,dy = DX@z,DY@z
        d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-25)
        return np.vstack(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RS,W))
    def callback(z):
        nonlocal calls,last
        check()
        last = z.copy()
        calls += 1
        if calls%6 == 0:
            retain(z)

    try:
        attempt = 0
        while time.monotonic() < deadline:
            tick()
            if attempt < len(starts):
                z = starts[attempt].copy()
            elif best_z is not None:
                z = best_z.copy()
                strength = float(params['jitter'])*(1.+.5*((attempt-len(starts))%3))
                z[:21] += rng.normal(0.,strength,21)
                z[12:21] = np.maximum(z[12:21],.005)
            else:
                z = starts[attempt%len(starts)].copy()
            last = z.copy(); calls = 0
            remaining = deadline-time.monotonic()
            slice_end = min(deadline,time.monotonic()+max(.08,min(float(params['slice']),remaining)))
            try:
                result = minimize(fun,z,jac=jac,method='SLSQP',bounds=bounds,
                    constraints={'type':'ineq','fun':con,'jac':cjac},callback=callback,
                    options={'maxiter':int(params['iterations']),'ftol':2e-12,'disp':False})
                retain(result.x)
            except EndSlice:
                if last is not None:
                    retain(last)
            attempt += 1
    except (StopSearch,ValueError,FloatingPointError):
        if last is not None and time.monotonic() < start+.80*budget_s:
            retain(last)
    return best
