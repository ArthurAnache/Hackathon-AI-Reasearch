import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {
    'projection_share': (0.22, 0.10, 0.35),
    'lattice_share': (0.14, 0.08, 0.22),
    'release': (0.012, 0.003, 0.04),
    'contacts': (4, 3, 6),
    'jitter': (0.012, 0.002, 0.04)
}

class StopSearch(Exception):
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

def score(obj):
    try:
        a = np.asarray(canonical(obj)['circles'], dtype=np.float64)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        wh = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        wh += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        if wh > 2:
            return -float('inf')
        i,j = np.triu_indices(21, 1)
        d = np.sqrt((a[i,0]-a[j,0])**2+(a[i,1]-a[j,1])**2)
        if np.any(d < a[i,2]+a[j,2]):
            return -float('inf')
        return float(np.sum(a[:,2]))
    except Exception:
        return -float('inf')

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + 0.69*budget_s
    local_end = deadline
    rng = np.random.default_rng(seed)
    best = init
    best_value = score(init) if init is not None else -float('inf')
    fallback = {'circles': [[(i%7)*.2, (i//7)*.2, .09999999999998] for i in range(21)]}
    fallback_value = score(fallback)
    if fallback_value > best_value:
        best, best_value = fallback, fallback_value
    def tick():
        if time.monotonic() >= local_end:
            raise StopSearch()
    if time.monotonic() >= deadline:
        return best

    group = np.r_[np.arange(3),np.repeat(np.arange(3,12),2)]
    sign = np.r_[np.zeros(3),np.tile([1.,-1.],9)]
    X = np.zeros((21,34)); Y = X.copy(); R = X.copy()
    X[np.arange(21),group] = 1
    R[np.arange(21),21+group] = 1
    mask = group >= 3
    Y[np.flatnonzero(mask),group[mask]+9] = sign[mask]
    ai,aj = np.triu_indices(21,1)
    keys = np.column_stack((np.minimum(group[ai],group[aj]),np.maximum(group[ai],group[aj]),sign[ai]*sign[aj]))
    _,keep = np.unique(keys,axis=0,return_index=True)
    pi,pj = ai[keep],aj[keep]
    DX,DY,RS = X[pi]-X[pj],Y[pi]-Y[pj],R[pi]+R[pj]
    W = np.zeros((36,34)); off = np.zeros(36)
    q = np.arange(12)
    W[q,q] = 1; W[q,21+q] = -1; W[q,33] = .5
    W[12+q,q] = -1; W[12+q,21+q] = -1; W[12+q,33] = .5
    W[24+q,21+q] = -1; W[24+q,33] = -.5
    W[27+np.arange(9),12+np.arange(9)] = -1
    off[24:] = 1
    objective = np.zeros(34)
    objective[21:33] = -np.r_[np.ones(3),np.full(9,2.)]
    bounds = [(-1.,1.)]*12+[(0.,1.)]*9+[(1e-7,.6)]*12+[(.12,1.88)]
    release = np.zeros(len(pi)+36)
    cached_v = None; cached_c = None; cached_j = None
    last = None
    best_anchor = None
    best_anchor_value = -float('inf')
    callbacks = 0

    def geometry(v):
        nonlocal cached_v,cached_c,cached_j
        tick()
        if cached_v is None or not np.array_equal(v,cached_v):
            dx,dy = DX@v,DY@v
            d = np.sqrt(dx*dx+dy*dy)
            safe = np.maximum(d,1e-25)
            cached_c = np.r_[d-RS@v,W@v+off]
            cached_j = np.vstack(((dx/safe)[:,None]*DX+(dy/safe)[:,None]*DY-RS,W))
            cached_v = v.copy()
        return cached_c,cached_j

    def repaired(v):
        if v is None or not np.isfinite(v).all():
            return None
        a = np.column_stack((X@v,Y@v,R@v))
        if np.any(a[:,2] <= 0):
            return None
        d = np.sqrt((a[ai,0]-a[aj,0])**2+(a[ai,1]-a[aj,1])**2)
        f = min(1.,float(np.min(d/(a[ai,2]+a[aj,2]))))*(1.-8e-15)
        if f <= 0:
            return None
        a[:,2] *= f
        span = np.max(a[:,:2]+a[:,2,None],axis=0)-np.min(a[:,:2]-a[:,2,None],axis=0)
        scale = 2./float(np.sum(span))*(1.-8e-15)
        a *= scale
        return a

    def retain(v):
        nonlocal best,best_value,best_anchor,best_anchor_value
        if time.monotonic() >= deadline:
            return -float('inf')
        a = repaired(v)
        if a is None:
            return -float('inf')
        estimate = float(np.sum(a[:,2]))
        if estimate <= min(best_value,best_anchor_value):
            return estimate
        obj = {'circles': a.tolist()}
        value = score(obj)
        if value > best_anchor_value:
            best_anchor_value = value
            best_anchor = v.copy()
        if value > best_value:
            best,best_value = obj,value
        return value

    def fun(v):
        tick()
        return float(objective@v)
    def jac(v):
        tick()
        return objective
    def con(v):
        return geometry(v)[0]-release
    def cjac(v):
        return geometry(v)[1]
    def callback(v):
        nonlocal last,callbacks
        tick()
        last = v.copy()
        callbacks += 1
        if callbacks % 6 == 0:
            retain(v)

    def solve(v, seconds, iterations=100):
        nonlocal local_end,last
        local_end = min(deadline,time.monotonic()+max(0.,seconds))
        last = v.copy()
        try:
            result = minimize(fun,v,jac=jac,bounds=bounds,method='SLSQP',constraints={'type':'ineq','fun':con,'jac':cjac},callback=callback,options={'maxiter':int(iterations),'ftol':3e-14,'disp':False})
            last = result.x.copy()
        except (StopSearch,ValueError,FloatingPointError,np.linalg.LinAlgError):
            pass
        retain(last)
        local_end = deadline
        return last.copy()

    def pack(g):
        z = np.zeros(34)
        z[:12] = g[:,0]
        z[12:21] = np.abs(g[3:,1])
        z[21:33] = np.maximum(g[:,2],1e-6)
        z[33] = np.clip(np.max(g[:,0]+g[:,2])-np.min(g[:,0]-g[:,2]),.13,1.87)
        return z

    projections = []
    try:
        a = np.asarray(canonical(init)['circles'],float).copy()
        if a.shape == (21,3) and np.isfinite(a).all():
            lo = np.min(a[:,:2]-a[:,2,None],axis=0)
            hi = np.max(a[:,:2]+a[:,2,None],axis=0)
            a[:,:2] -= (lo+hi)/2
            for swap in (False,True):
                tick()
                b = a[:,[1,0,2]].copy() if swap else a.copy()
                order = np.argsort(b[:,1])
                lower,axis,upper = order[:9],order[9:12],order[12:]
                reflected = b[lower].copy(); reflected[:,1] *= -1
                costs = np.sum((b[upper,None,:]-reflected[None,:,:])**2,axis=2)
                u,v = linear_sum_assignment(costs)
                g = np.vstack((b[axis],(b[upper[u]]+reflected[v])/2))
                g[:3,1] = 0
                error = float(np.sum(costs[u,v])+np.sum(b[axis,1]**2))
                projections.append((error,pack(g)))
    except (StopSearch,ValueError,TypeError,KeyError):
        pass
    endpoints = []
    if projections and time.monotonic() < deadline:
        projections.sort(key=lambda p:p[0])
        z = solve(projections[0][1],float(params['projection_share'])*budget_s,100)
        endpoints.append(z)

    for inner,outer in ((4,5),(5,4)):
        if time.monotonic() >= deadline-.015:
            break
        g = [[2.*(k-1),0.,1.] for k in range(3)]
        for count,y in ((inner,np.sqrt(3.)),(outer,2*np.sqrt(3.))):
            if time.monotonic() >= deadline:
                break
            for k in range(count):
                if time.monotonic() >= deadline:
                    break
                g.append([2.*k-count+1,y,1.])
        if len(g) != 12:
            break
        g = np.asarray(g,float)
        g *= 2./(np.ptp(g[:,0])+2+2*np.max(g[:,1]+g[:,2]))
        z = solve(pack(g),float(params['lattice_share'])*budget_s,100)
        endpoints.append(z)

    attempt = 0
    while time.monotonic() < deadline-.025:
        if best_anchor is None:
            break
        base = best_anchor.copy()
        release[:] = 0
        if attempt == 0:
            base = solve(base,min(.16,deadline-time.monotonic()),100)
            try:
                for _ in range(4):
                    tick()
                    c,J = geometry(base)
                    active = np.flatnonzero(c < 1e-7)
                    if len(active) < 33:
                        break
                    delta = np.linalg.lstsq(J[active],-c[active],rcond=1e-11)[0]
                    if np.max(np.abs(delta)) > .002:
                        break
                    trial = base+delta
                    if np.min(geometry(trial)[0]) < -1e-9:
                        break
                    base = trial
                    retain(base)
            except (StopSearch,ValueError,np.linalg.LinAlgError):
                pass
        elif attempt % 3 == 2:
            trial = base.copy()
            trial[:21] += rng.normal(0.,float(params['jitter']),21)
            trial[12:21] = np.maximum(trial[12:21],0.)
            trial[21:33] *= .99
            solve(trial,min(.18,deadline-time.monotonic()),65)
        else:
            try:
                c,J = geometry(base)
                active = np.flatnonzero(c < 2e-5)
                count = min(int(params['contacts']),len(active))
                if count < 3:
                    solve(base,min(.15,deadline-time.monotonic()),60)
                else:
                    multipliers = np.linalg.lstsq(J[active].T,objective,rcond=1e-10)[0]
                    weight = 1./(.04+np.abs(multipliers))
                    first = int(rng.choice(active,p=weight/weight.sum()))
                    similarity = np.abs(J[active]@J[first])
                    probability = weight*(.1+similarity)
                    probability[active == first] = 0
                    probability /= probability.sum()
                    chosen = np.r_[first,rng.choice(active,count-1,replace=False,p=probability)]
                    release[chosen] = float(params['release'])*rng.uniform(.7,1.3,count)
                    trial = solve(base,min(.075,deadline-time.monotonic()),16)
                    release[:] = 0
                    solve(trial,min(.16,deadline-time.monotonic()),60)
            except (StopSearch,ValueError,FloatingPointError,np.linalg.LinAlgError):
                release[:] = 0
                break
        attempt += 1
    return best
