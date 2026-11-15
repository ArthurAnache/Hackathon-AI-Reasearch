import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {"first_slice": (0.44, 0.2, 0.7), "restart_slice": (0.27, 0.12, 0.5), "jitter": (0.018, 0.002, 0.06)}


def canonical(obj):
    if 'circles' in obj:
        return obj
    out = []
    for x,y,r,m in obj['orbits']:
        out.append([x,y,r])
        if m == 2:
            out.append([x,-y,r])
    return {'circles':out}


def score(obj):
    try:
        a = np.asarray(canonical(obj)['circles'], dtype=float)
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return -float('inf')
        xs = [Fraction(float(x)) for x in a[:,0]]
        ys = [Fraction(float(x)) for x in a[:,1]]
        rs = [Fraction(float(x)) for x in a[:,2]]
        wh = max(x+r for x,r in zip(xs,rs))-min(x-r for x,r in zip(xs,rs))
        wh += max(y+r for y,r in zip(ys,rs))-min(y-r for y,r in zip(ys,rs))
        if wh > 2:
            return -float('inf')
        for i in range(21):
            for j in range(i):
                if np.sqrt((a[i,0]-a[j,0])**2+(a[i,1]-a[j,1])**2) < a[i,2]+a[j,2]:
                    return -float('inf')
        return float(np.sum(a[:,2]))
    except (ValueError,TypeError,KeyError,OverflowError):
        return -float('inf')


class StopSlice(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + .69*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = score(init) if init is not None else -float('inf')
    fallback = {'circles': [[(i%7)*.2,(i//7)*.2,.09999999999998] for i in range(21)]}
    fs = score(fallback)
    if fs > best_score:
        best,best_score = fallback,fs
    if time.monotonic() >= deadline:
        return best

    # Variables: twelve x coordinates, nine positive y coordinates,
    # twelve orbit radii, and rectangle width.
    groups = np.r_[np.arange(3),np.repeat(np.arange(3,12),2)]
    signs = np.r_[np.zeros(3),np.tile([1.,-1.],9)]
    X = np.zeros((21,34)); Y = X.copy(); R = X.copy()
    for k,g in enumerate(groups):
        if time.monotonic() >= deadline:
            return best
        X[k,g] = 1
        R[k,21+g] = 1
        if g >= 3:
            Y[k,9+g] = signs[k]
    ii,jj = np.triu_indices(21,1)
    pairs = []; seen = set()
    for i,j in zip(ii,jj):
        if time.monotonic() >= deadline:
            return best
        g,h = sorted((int(groups[i]),int(groups[j])))
        key = (g,h,int(signs[i]*signs[j]))
        if key not in seen:
            seen.add(key); pairs.append((i,j))
    pi,pj = np.array(pairs).T
    DX = X[pi]-X[pj]; DY = Y[pi]-Y[pj]; RS = R[pi]+R[pj]
    walls = np.zeros((36,34)); off = np.zeros(36)
    for g in range(12):
        if time.monotonic() >= deadline:
            return best
        walls[g,g] = 1; walls[g,21+g] = -1; walls[g,33] = .5
        walls[12+g,g] = -1; walls[12+g,21+g] = -1; walls[12+g,33] = .5
        walls[24+g,21+g] = -1; walls[24+g,33] = -.5; off[24+g] = 1
        if g >= 3:
            walls[24+g,9+g] = -1
    weights = np.r_[np.ones(3),np.full(9,2.)]
    grad = np.zeros(34); grad[21:33] = -weights
    bounds = [(-1.,1.)]*12+[(0.,1.)]*9+[(1e-7,.6)]*12+[(.15,1.85)]

    def pack(g):
        z = np.zeros(34)
        z[:12] = g[:,0]
        z[12:21] = np.abs(g[3:,1])
        z[21:33] = np.maximum(g[:,2],1e-5)
        z[33] = np.clip(np.max(g[:,0]+g[:,2])-np.min(g[:,0]-g[:,2]),.2,1.8)
        return z

    def retain(v):
        nonlocal best,best_score
        if time.monotonic() >= deadline or not np.isfinite(v).all():
            return
        r = v[21:33].copy()
        if np.any(r <= 0):
            return
        a = np.column_stack((X@v,Y@v,R@v))
        dist = np.hypot(a[ii,0]-a[jj,0],a[ii,1]-a[jj,1])
        shrink = min(1.,float(np.min(dist/(a[ii,2]+a[jj,2]))))*(1-8e-15)
        r *= shrink
        xx = v[:12]; yy = np.r_[np.zeros(3),np.abs(v[12:21])]
        wh = np.max(xx+r)-np.min(xx-r)+2*np.max(yy+r)
        scale = min(1.,2./max(wh,1e-20))*(1-8e-15)
        if float(weights@r)*scale <= best_score:
            return
        obj = {'orbits': [[float(xx[g]*scale),float(yy[g]*scale),float(r[g]*scale),1 if g<3 else 2] for g in range(12)]}
        s = score(obj)
        if s > best_score:
            best,best_score = obj,s

    local_end = deadline
    latest = None
    def tick():
        if time.monotonic() >= local_end:
            raise StopSlice()
    def fun(v):
        tick()
        return float(grad@v)
    def jac(v):
        tick()
        return grad
    def con(v):
        tick()
        dx,dy = DX@v,DY@v
        return np.r_[np.hypot(dx,dy)-RS@v,walls@v+off]
    def cjac(v):
        tick()
        dx,dy = DX@v,DY@v
        d = np.maximum(np.hypot(dx,dy),1e-25)
        return np.vstack(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RS,walls))
    def callback(v):
        nonlocal latest
        tick()
        latest = v.copy()
        retain(v)
    def solve(z, seconds, iterations=140):
        nonlocal local_end,latest
        local_end = min(deadline,time.monotonic()+seconds)
        latest = z.copy()
        try:
            res = minimize(fun,z,jac=jac,bounds=bounds,method='SLSQP',constraints={'type':'ineq','fun':con,'jac':cjac},callback=callback,options={'maxiter':iterations,'ftol':2e-13,'disp':False})
            latest = res.x.copy()
        except (StopSlice,ValueError,FloatingPointError):
            pass
        retain(latest)
        return latest.copy()

    projections = []
    if init is not None:
        try:
            a = np.asarray(canonical(init)['circles'],float).copy()
            if a.shape == (21,3) and np.isfinite(a).all():
                lo = np.min(a[:,:2]-a[:,2,None],axis=0)
                hi = np.max(a[:,:2]+a[:,2,None],axis=0)
                a[:,:2] -= (lo+hi)/2
                for swap in (False,True):
                    if time.monotonic() >= deadline:
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
    # Five-row families, with the central row fixed to three circles.
    for inner,outer in ((4,5),(5,4)):
        if time.monotonic() >= deadline:
            return best
        g = []
        for k in range(3):
            if time.monotonic() >= deadline:
                return best
            g.append([2.*(k-1),0.,1.])
        for count,y in ((inner,np.sqrt(3.)),(outer,2*np.sqrt(3.))):
            if time.monotonic() >= deadline:
                return best
            for k in range(count):
                if time.monotonic() >= deadline:
                    return best
                g.append([2*k-count+1,y,1.])
        g = np.array(g,float)
        width = np.ptp(g[:,0])+2
        height = 2*np.max(g[:,1]+g[:,2])
        g *= 2/(width+height)
        seeds.append(pack(g))
    endpoints = []
    for k,z in enumerate(seeds):
        if time.monotonic() >= deadline:
            break
        seconds = float(params['first_slice']) if k == 0 else float(params['restart_slice'])
        endpoints.append(solve(z,seconds))
    # Spend the remaining budget on the strongest relaxed endpoint and
    # small coordinated orbit perturbations rather than forced releases.
    attempt = 0
    while time.monotonic() < deadline-.015:
        if not endpoints:
            break
        anchor = min(endpoints,key=lambda v:float(grad@v)).copy()
        if attempt > 0:
            anchor[:21] += rng.normal(0,float(params['jitter']),21)
            anchor[12:21] = np.maximum(anchor[12:21],0)
            anchor[21:33] *= .985
        z = solve(anchor,min(.22,deadline-time.monotonic()),100)
        endpoints.append(z)
        attempt += 1
    return best
