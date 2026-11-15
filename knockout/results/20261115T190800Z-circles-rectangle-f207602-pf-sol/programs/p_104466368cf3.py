import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {'first_slice': (0.48, 0.25, 0.8), 'restart_slice': (0.24, 0.1, 0.4), 'release': (0.012, 0.002, 0.04), 'contacts': (3, 3, 5)}


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
        xs = [Fraction(float(v)) for v in a[:, 0]]
        ys = [Fraction(float(v)) for v in a[:, 1]]
        rs = [Fraction(float(v)) for v in a[:, 2]]
        wh = max(x+r for x,r in zip(xs,rs))-min(x-r for x,r in zip(xs,rs))
        wh += max(y+r for y,r in zip(ys,rs))-min(y-r for y,r in zip(ys,rs))
        if wh > 2:
            return -float('inf')
        for i in range(21):
            for j in range(i):
                if np.sqrt((a[i,0]-a[j,0])**2+(a[i,1]-a[j,1])**2) < a[i,2]+a[j,2]:
                    return -float('inf')
        return float(np.sum(a[:, 2]))
    except (TypeError, ValueError, KeyError, OverflowError):
        return -float('inf')


class SliceEnd(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.69*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = score(init) if init is not None else -float('inf')
    fallback = {'circles': [[.2*(i%7), .2*(i//7), .09999999999998] for i in range(21)]}
    fs = score(fallback)
    if fs > best_score:
        best, best_score = fallback, fs

    def alive():
        return time.monotonic() < deadline

    groups = np.r_[np.arange(3), np.repeat(np.arange(3,12),2)]
    signs = np.r_[np.zeros(3), np.tile([1.,-1.],9)]
    X = np.zeros((21,34)); Y = X.copy(); R = X.copy()
    for k,g in enumerate(groups):
        if not alive():
            return best
        X[k,g] = 1
        R[k,21+g] = 1
        if g >= 3:
            Y[k,9+g] = signs[k]
    ii,jj = np.triu_indices(21,1)
    pairs = []; seen = set()
    for i,j in zip(ii,jj):
        if not alive():
            return best
        g,h = sorted((int(groups[i]),int(groups[j])))
        key = (g,h,int(signs[i]*signs[j]))
        if key not in seen:
            seen.add(key)
            pairs.append((i,j))
    pi,pj = np.asarray(pairs).T
    DX = X[pi]-X[pj]; DY = Y[pi]-Y[pj]; RS = R[pi]+R[pj]
    walls = np.zeros((36,34)); off = np.zeros(36)
    for g in range(12):
        if not alive():
            return best
        walls[g,g] = 1; walls[g,21+g] = -1; walls[g,33] = .5
        walls[12+g,g] = -1; walls[12+g,21+g] = -1; walls[12+g,33] = .5
        walls[24+g,21+g] = -1; walls[24+g,33] = -.5; off[24+g] = 1
        if g >= 3:
            walls[24+g,9+g] = -1
    weights = np.r_[np.ones(3),np.full(9,2.)]
    grad = np.zeros(34); grad[21:33] = -weights
    bounds = [(-1.,1.)]*12+[(0.,1.)]*9+[(1e-7,.6)]*12+[(.15,1.85)]
    release = np.zeros(len(pi))

    def pack(g):
        z = np.zeros(34)
        z[:12] = g[:,0]
        z[12:21] = np.abs(g[3:,1])
        z[21:33] = np.maximum(g[:,2],1e-6)
        z[33] = np.clip(np.ptp(g[:,0])+2*np.max(g[:,2]),.2,1.8)
        return z

    def retain(v):
        nonlocal best,best_score
        if not alive() or not np.isfinite(v).all():
            return -float('inf')
        r = v[21:33].copy()
        if np.any(r <= 0):
            return -float('inf')
        a = np.column_stack((X@v,Y@v,R@v))
        dx = a[ii,0]-a[jj,0]; dy = a[ii,1]-a[jj,1]
        ratio = np.min(np.sqrt(dx*dx+dy*dy)/(a[ii,2]+a[jj,2]))
        r *= min(1.,float(ratio))*(1.-1.2e-14)
        xx = v[:12]; yy = np.r_[np.zeros(3),np.abs(v[12:21])]
        wh = np.max(xx+r)-np.min(xx-r)+2*np.max(yy+r)
        scale = min(1.,2./max(wh,1e-30))*(1.-1.2e-14)
        estimate = float(weights@r)*scale
        if estimate <= best_score:
            return estimate
        obj = {'orbits': [[float(xx[g]*scale),float(yy[g]*scale),float(r[g]*scale),1 if g<3 else 2] for g in range(12)]}
        s = score(obj)
        if s > best_score:
            best,best_score = obj,s
        return s

    local_end = deadline
    cache_v = None; cache_c = None; cache_j = None
    latest = None
    def tick():
        if time.monotonic() >= local_end:
            raise SliceEnd()
    def update(v):
        nonlocal cache_v,cache_c,cache_j
        tick()
        if cache_v is not None and np.array_equal(v,cache_v):
            return
        dx,dy = DX@v,DY@v
        d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-25)
        cache_v = v.copy()
        cache_c = np.r_[d-RS@v,walls@v+off]
        cache_j = np.vstack(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RS,walls))
    def fun(v):
        tick()
        return float(grad@v)
    def jac(v):
        tick()
        return grad
    def con(v):
        update(v)
        c = cache_c.copy()
        c[:len(release)] -= release
        return c
    def cjac(v):
        update(v)
        return cache_j
    def callback(v):
        nonlocal latest
        tick()
        latest = v.copy()
        retain(v)
    def solve(z, seconds, iterations=120):
        nonlocal latest,local_end,cache_v
        local_end = min(deadline,time.monotonic()+max(0.,seconds))
        latest = z.copy(); cache_v = None
        try:
            res = minimize(fun,z,jac=jac,bounds=bounds,method='SLSQP',constraints={'type':'ineq','fun':con,'jac':cjac},callback=callback,options={'maxiter':iterations,'ftol':4e-14,'disp':False})
            latest = res.x.copy()
        except (SliceEnd,ValueError,FloatingPointError):
            pass
        retain(latest)
        return latest.copy()

    projections = []
    if init is not None and alive():
        try:
            a = np.asarray(canonical(init)['circles'],float).copy()
            if a.shape == (21,3) and np.isfinite(a).all():
                lo = np.min(a[:,:2]-a[:,2,None],axis=0)
                hi = np.max(a[:,:2]+a[:,2,None],axis=0)
                a[:,:2] -= (lo+hi)/2
                for swap in (False,True):
                    if not alive():
                        return best
                    b = a[:,[1,0,2]].copy() if swap else a.copy()
                    order = np.argsort(b[:,1]); low,axis,up = order[:9],order[9:12],order[12:]
                    refl = b[low].copy(); refl[:,1] *= -1
                    cost = np.sum((b[up,None,:]-refl[None,:,:])**2,axis=2)
                    u,v = linear_sum_assignment(cost)
                    g = np.zeros((12,3)); g[:3] = b[axis]; g[:3,1] = 0
                    g[3:] = (b[up[u]]+refl[v])/2
                    err = float(np.sum(cost[u,v])+np.sum(b[axis,1]**2))
                    projections.append((err,pack(g)))
        except (ValueError,KeyError,TypeError):
            pass
    seeds = []
    if projections:
        projections.sort(key=lambda t:t[0])
        seeds.append(projections[0][1])
    for inner,outer in ((4,5),(5,4)):
        if not alive():
            return best
        g = [[2.*(k-1),0.,1.] for k in range(3)]
        for count,y in ((inner,np.sqrt(3.)),(outer,2*np.sqrt(3.))):
            if not alive():
                return best
            for k in range(count):
                if not alive():
                    return best
                g.append([2*k-count+1,y,1.])
        g = np.asarray(g,float)
        g *= 2/(np.ptp(g[:,0])+2+2*np.max(g[:,1]+g[:,2]))
        seeds.append(pack(g))
    endpoints = []
    for k,z in enumerate(seeds):
        if not alive():
            break
        z = solve(z,float(params['first_slice'] if k == 0 else params['restart_slice']))
        endpoints.append((retain(z),z))
    attempt = 0
    while time.monotonic() < deadline-.025:
        if not endpoints:
            break
        endpoints.sort(key=lambda t:t[0],reverse=True)
        anchor = endpoints[0][1].copy()
        release[:] = 0
        if attempt == 0:
            z = solve(anchor,min(.16,deadline-time.monotonic()))
        else:
            dx,dy = DX@anchor,DY@anchor
            gaps = np.sqrt(dx*dx+dy*dy)-RS@anchor
            near = np.flatnonzero(gaps < .001)
            if len(near) < 3:
                break
            count = min(int(params['contacts']),len(near))
            chosen = rng.choice(near,count,replace=False)
            release[chosen] = float(params['release'])*rng.uniform(.7,1.3,count)
            z = solve(anchor,min(.075,max(0.,(deadline-time.monotonic())*.35)),24)
            release[:] = 0
            z = solve(z,min(.19,deadline-time.monotonic()),90)
        endpoints.append((retain(z),z))
        attempt += 1
    return best
