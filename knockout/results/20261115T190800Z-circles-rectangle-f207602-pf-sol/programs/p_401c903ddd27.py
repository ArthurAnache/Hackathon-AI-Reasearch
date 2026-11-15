import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {'iterations': (100, 40, 160), 'stage_seconds': (0.28, 0.12, 0.5), 'release': (0.012, 0.002, 0.04)}


def canonical(obj):
    if obj is None:
        return None
    if 'circles' in obj:
        return obj
    a = []
    for x, y, r, m in obj['orbits']:
        a.append([x, y, r])
        if m == 2:
            a.append([x, -y, r])
    return {'circles': a}


def score(obj):
    try:
        a = np.asarray(canonical(obj)['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        x = [Fraction(float(t)) for t in a[:, 0]]
        y = [Fraction(float(t)) for t in a[:, 1]]
        r = [Fraction(float(t)) for t in a[:, 2]]
        if max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))+max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)) > 2:
            return -float('inf')
        i,j = np.triu_indices(21, 1)
        d = np.sqrt((a[i,0]-a[j,0])**2+(a[i,1]-a[j,1])**2)
        if np.any(d < a[i,2]+a[j,2]):
            return -float('inf')
        return float(np.sum(a[:,2]))
    except Exception:
        return -float('inf')


class EndStage(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    begin = time.monotonic()
    deadline = begin + .70*budget_s
    stage_end = deadline
    rng = np.random.default_rng(seed)
    best = init
    best_score = score(init)
    fallback = {'circles': [[(k%7)*.2, (k//7)*.2, .09999999999998] for k in range(21)]}
    value = score(fallback)
    if value > best_score:
        best, best_score = fallback, value

    def tick():
        if time.monotonic() >= min(deadline, stage_end):
            raise EndStage()

    groups = np.r_[np.arange(3), np.repeat(np.arange(3,12),2)]
    signs = np.r_[np.zeros(3), np.tile([1.,-1.],9)]
    X = np.zeros((21,34)); Y = X.copy(); R = X.copy()
    X[np.arange(21),groups] = 1
    R[np.arange(21),21+groups] = 1
    mask = groups >= 3
    Y[np.flatnonzero(mask),9+groups[mask]] = signs[mask]
    ii,jj = np.triu_indices(21,1)
    keys = np.column_stack((np.minimum(groups[ii],groups[jj]), np.maximum(groups[ii],groups[jj]), signs[ii]*signs[jj]))
    _, keep = np.unique(keys,axis=0,return_index=True)
    pi,pj = ii[keep],jj[keep]
    DX,DY,RR = X[pi]-X[pj],Y[pi]-Y[pj],R[pi]+R[pj]
    W = np.zeros((36,34)); offset = np.zeros(36)
    q = np.arange(12)
    W[q,q] = 1; W[q,21+q] = -1; W[q,33] = .5
    W[12+q,q] = -1; W[12+q,21+q] = -1; W[12+q,33] = .5
    W[24+q,21+q] = -1; W[24+q,33] = -.5
    W[27+np.arange(9),12+np.arange(9)] = -1
    offset[24:] = 1
    weights = np.r_[np.ones(3),np.full(9,2.)]
    grad = np.zeros(34); grad[21:33] = -weights
    bounds = [(-1.,1.)]*12+[(0.,1.)]*9+[(1e-7,.6)]*12+[(.12,1.88)]
    release = np.zeros(len(pi))
    cache_v = None
    cache_c = None
    cache_j = None
    best_z = None
    last = None
    calls = 0

    def retain(v):
        nonlocal best, best_score, best_z
        if v is None or not np.isfinite(v).all() or np.min(v[21:33]) <= 0:
            return
        a = np.column_stack((X@v,Y@v,R@v))
        d = np.sqrt((a[ii,0]-a[jj,0])**2+(a[ii,1]-a[jj,1])**2)
        shrink = min(1.,float(np.min(d/(a[ii,2]+a[jj,2]))))*(1.-3e-14)
        if shrink <= 0:
            return
        a[:,2] *= shrink
        extent = np.max(a[:,:2]+a[:,2,None],axis=0)-np.min(a[:,:2]-a[:,2,None],axis=0)
        scale = 2./float(np.sum(extent))*(1.-3e-14)
        a *= scale
        if float(np.sum(a[:,2])) <= best_score:
            return
        obj = {'circles': a.tolist()}
        val = score(obj)
        if val > best_score:
            best, best_score = obj, val
            best_z = v.copy()
            best_z[:33] *= scale
            best_z[21:33] *= shrink
            lo = np.min(a[:,0]-a[:,2]); hi = np.max(a[:,0]+a[:,2])
            best_z[:12] -= (lo+hi)/2
            best_z[33] = hi-lo

    def calculate(v):
        nonlocal cache_v, cache_c, cache_j
        tick()
        if cache_v is None or not np.array_equal(v,cache_v):
            dx,dy = DX@v,DY@v
            d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-24)
            cache_c = np.r_[d-RR@v-release,W@v+offset]
            cache_j = np.vstack(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RR,W))
            cache_v = v.copy()
        return cache_c,cache_j

    def fun(v):
        tick()
        return float(grad@v)

    def jac(v):
        tick()
        return grad

    def callback(v):
        nonlocal last,calls
        last = v.copy()
        calls += 1
        if calls % 4 == 0:
            retain(v)
        tick()

    def solve(v, n, seconds):
        nonlocal stage_end, cache_v, last
        stage_end = min(deadline,time.monotonic()+seconds)
        cache_v = None
        last = v.copy()
        try:
            tick()
            ans = minimize(fun,v,jac=jac,bounds=bounds,method='SLSQP',constraints={'type':'ineq','fun':lambda z:calculate(z)[0],'jac':lambda z:calculate(z)[1]},callback=callback,options={'maxiter':int(n),'ftol':3e-13,'disp':False})
            last = ans.x.copy()
        except (EndStage,ValueError,FloatingPointError):
            pass
        retain(last)
        return last.copy()

    def lattice(counts):
        v = np.zeros(34)
        v[:3] = [-2.,0.,2.]
        k = 3
        for layer,count in enumerate(counts):
            if time.monotonic() >= deadline:
                return v
            v[k:k+count] = 2*np.arange(count)-(count-1)
            v[9+k:9+k+count] = (layer+1)*np.sqrt(3.)
            k += count
        v[21:33] = 1.
        width = np.ptp(v[:12])+2
        height = 2*(np.max(v[12:21])+1)
        s = 2./(width+height)
        v[:33] *= s
        v[33] = width*s
        return v

    starts = []
    try:
        a = np.asarray(canonical(init)['circles'],float).copy()
        if a.shape == (21,3) and np.isfinite(a).all():
            low = np.min(a[:,:2]-a[:,2,None],axis=0)
            high = np.max(a[:,:2]+a[:,2,None],axis=0)
            a[:,:2] -= (low+high)/2
            projected = []
            for swap in (False,True):
                if time.monotonic() >= deadline:
                    break
                b = a[:,[1,0,2]].copy() if swap else a.copy()
                order = np.argsort(b[:,1])
                lower,axis,upper = order[:9],order[9:12],order[12:]
                ref = b[lower].copy(); ref[:,1] *= -1
                cost = np.sum((b[upper,None,:]-ref[None,:,:])**2,axis=2)
                u,v = linear_sum_assignment(cost)
                g = np.vstack((b[axis],(b[upper[u]]+ref[v])/2))
                z = np.zeros(34)
                z[:12] = g[:,0]; z[12:21] = np.abs(g[3:,1]); z[21:33] = np.maximum(g[:,2],1e-5)
                z[33] = np.max(g[:,0]+g[:,2])-np.min(g[:,0]-g[:,2])
                err = float(np.sum(cost[u,v])+np.sum(b[axis,1]**2))
                projected.append((err,z))
            projected.sort(key=lambda p:p[0])
            starts.extend([p[1] for p in projected[:1]])
    except (TypeError,KeyError,ValueError):
        pass
    starts.extend([lattice([4,5]),lattice([5,4]),lattice([3,3,3])])
    for z in starts:
        if time.monotonic() >= deadline:
            break
        retain(z)
        solve(z,params['iterations'],params['stage_seconds'])
    attempt = 0
    while time.monotonic() < deadline:
        anchor = best_z.copy() if best_z is not None else starts[0].copy()
        if attempt == 0:
            solve(anchor,140,max(.01,deadline-time.monotonic()))
        else:
            release[:] = 0
            gap = np.sqrt((DX@anchor)**2+(DY@anchor)**2)-RR@anchor
            near = np.flatnonzero(gap < .003)
            if len(near) < 3:
                break
            chosen = rng.choice(near,min(4,len(near)),replace=False)
            release[chosen] = params['release']*rng.uniform(.7,1.3,len(chosen))
            displaced = solve(anchor,20,.09)
            release[:] = 0
            solve(displaced,80,.19)
        attempt += 1
    return best
