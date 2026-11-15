import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {'iterations': (120, 60, 200), 'warm_slice': (0.36, 0.15, 0.65), 'screen_slice': (0.16, 0.08, 0.30), 'release': (0.006, 0.001, 0.025)}

class Deadline(Exception):
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
        a = np.asarray(canonical(obj)['circles'], dtype=np.float64)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        x, y, r = [[Fraction(float(t)) for t in a[:, k]] for k in range(3)]
        perimeter = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        perimeter += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        if perimeter > 2:
            return -float('inf')
        i, j = np.triu_indices(21, 1)
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
    fallback = {'circles': [[.2*(k%7-3), .2*(k//7-1), .09999999999997] for k in range(21)]}
    fs = exact_score(fallback)
    if fs > best_score:
        best, best_score = fallback, fs

    def tick():
        if time.monotonic() >= deadline:
            raise Deadline()

    if time.monotonic() >= deadline-.02:
        return best
    group = np.array([0,1,2]+[g for g in range(3,12) for _ in range(2)])
    signs = np.array([0.,0.,0.]+[s for _ in range(9) for s in (1.,-1.)])
    X = np.zeros((21,34)); Y = np.zeros_like(X); R = np.zeros_like(X)
    try:
        for k in range(21):
            tick()
            g = group[k]
            X[k,g] = 1.; R[k,21+g] = 1.
            if g >= 3:
                Y[k,9+g] = signs[k]
        pairs = []; seen = set()
        for i in range(21):
            tick()
            for j in range(i):
                tick()
                key = (min(group[i],group[j]),max(group[i],group[j]),int(signs[i]*signs[j]))
                if key not in seen:
                    seen.add(key); pairs.append((i,j))
        pi,pj = np.asarray(pairs).T
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
    except Deadline:
        return best

    objective = np.zeros(34)
    objective[21:24] = -1.; objective[24:33] = -2.
    bounds = [(-1.,1.)]*12+[(0.,1.)]*9+[(1e-7,.55)]*12+[(.10,1.90)]
    ii,jj = np.triu_indices(21,1)
    release = np.zeros(len(pi))
    champion = None
    champion_score = -float('inf')

    def retain(z):
        nonlocal best, best_score, champion, champion_score
        tick()
        if not np.isfinite(z).all() or np.any(z[21:33] <= 0):
            return -float('inf')
        a = np.column_stack((X@z,Y@z,R@z))
        distances = np.sqrt((a[ii,0]-a[jj,0])**2+(a[ii,1]-a[jj,1])**2)
        shrink = min(1.,float(np.min(distances/(a[ii,2]+a[jj,2]))))*(1.-5e-14)
        if shrink <= 0:
            return -float('inf')
        a[:,2] *= shrink
        low = np.min(a[:,:2]-a[:,2,None],axis=0)
        high = np.max(a[:,:2]+a[:,2,None],axis=0)
        span = high-low
        scale = 2./float(np.sum(span))*(1.-5e-14)
        a *= scale
        value = float(np.sum(a[:,2]))
        if value > champion_score:
            champion_score = value
            champion = z.copy()
            champion[:33] *= scale
            champion[21:33] *= shrink
            champion[33] = span[0]*scale
            champion[:12] -= .5*(low[0]+high[0])*scale
        if value > best_score:
            obj = {'circles': a.tolist()}
            verified = exact_score(obj)
            if verified > best_score:
                best, best_score = obj, verified
        return value

    def make_z(g):
        g = np.asarray(g,dtype=float).copy()
        lo = np.min(g[:,0]-g[:,2]); hi = np.max(g[:,0]+g[:,2])
        g[:,0] -= (lo+hi)/2
        width = hi-lo
        height = 2*np.max(np.abs(g[:,1])+g[:,2])
        scale = 2./(width+height)
        z = np.zeros(34)
        z[:12] = g[:,0]*scale
        z[12:21] = np.abs(g[3:,1])*scale
        z[21:33] = g[:,2]*scale
        z[33] = width*scale
        return z

    starts = []
    try:
        if init is not None:
            a = np.asarray(canonical(init)['circles'],dtype=float).copy()
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
                    order = np.argsort(b[:,1])
                    lower,singles,upper = order[:9],order[9:12],order[12:]
                    reflected = b[lower].copy(); reflected[:,1] *= -1
                    cost = np.sum((b[upper,None,:]-reflected[None,:,:])**2,axis=2)
                    u,v = linear_sum_assignment(cost)
                    singles_data = np.column_stack((b[singles,0],np.zeros(3),b[singles,2]))
                    g = np.vstack((singles_data,(b[upper[u]]+reflected[v])/2))
                    residual = float(np.sum(cost[u,v])+np.sum(b[singles,1]**2))
                    projections.append((residual,make_z(g)))
                projections.sort(key=lambda p:p[0])
                starts.append((projections[0][1],float(params['warm_slice'])))
                alternate = projections[1] if projections[1][0] < .025 else None
            else:
                alternate = None
        else:
            alternate = None
        for rows in ((4,3,2),(5,4),(4,5),(3,3,3)):
            tick()
            g = [[-.2,0.,.095],[0.,0.,.095],[.2,0.,.095]]
            for row,n in enumerate(rows):
                tick()
                for k in range(n):
                    tick()
                    g.append([.2*(k-(n-1)/2),(row+1)*np.sqrt(3.)*.1,.095])
            starts.append((make_z(g),float(params['screen_slice'])))
        if alternate is not None:
            starts.insert(2,(alternate[1],float(params['screen_slice'])))
    except Deadline:
        return best
    except (ValueError,TypeError,KeyError):
        pass
    if not starts:
        return best

    slice_end = deadline
    cache_z = None; cache_g = None; cache_j = None
    last = None
    def check():
        tick()
        if time.monotonic() >= slice_end:
            raise SliceEnd()
    def calc(z):
        nonlocal cache_z,cache_g,cache_j
        check()
        if cache_z is None or not np.array_equal(cache_z,z):
            dx,dy = DX@z,DY@z
            d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-25)
            cache_z = z.copy()
            cache_g = np.r_[d-RS@z,W@z+offset]
            cache_j = np.vstack(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RS,W))
        return cache_g,cache_j
    def fun(z):
        check()
        return float(objective@z)
    def jac(z):
        check()
        return objective
    def con(z):
        g = calc(z)[0].copy()
        g[:len(pi)] -= release
        return g
    def solve(z,duration,iterations):
        nonlocal slice_end,last
        tick()
        slice_end = min(deadline-.008,time.monotonic()+duration)
        last = z.copy()
        count = 0
        def callback(q):
            nonlocal last,count
            last = q.copy(); count += 1
            check()
            if count%7 == 0:
                retain(q)
        try:
            result = minimize(fun,z,jac=jac,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':con,'jac':lambda q:calc(q)[1]},
                callback=callback,options={'maxiter':iterations,'ftol':5e-13,'disp':False})
            last = result.x
        except SliceEnd:
            pass
        value = retain(last)
        return last.copy(),value

    pool = []
    try:
        for k,(z,duration) in enumerate(starts):
            tick()
            if time.monotonic() > began+.47*budget_s and pool:
                break
            if k > 1 and champion_score > 2.3658:
                break
            q,value = solve(z,duration,int(params['iterations']))
            pool.append((value,q))
        if champion is not None and time.monotonic() < deadline-.05:
            release[:] = 0
            q,value = solve(champion.copy(),min(.28,max(.04,deadline-time.monotonic()-.02)),int(params['iterations']))
            pool.append((value,q))
        attempt = 0
        while time.monotonic() < deadline-.06:
            tick()
            if champion is None:
                break
            anchor = champion.copy()
            if champion_score < 2.36580 and attempt < 2:
                if attempt == 0 and pool:
                    z = max(pool,key=lambda p:p[0])[1].copy()
                else:
                    z = anchor.copy()
                    z[:21] += rng.normal(0.,.009,21)
                    z[12:21] = np.maximum(.001,z[12:21])
                release[:] = 0
                solve(z,.18,int(params['iterations']))
            else:
                gaps = calc(anchor)[0][:len(pi)]
                near = np.flatnonzero(gaps < .003)
                if len(near) < 3:
                    break
                chosen = rng.choice(near,min(3,len(near)),replace=False)
                release[:] = 0
                release[chosen] = float(params['release'])
                q,_ = solve(anchor,.065,14)
                release[:] = 0
                solve(q,.18,75)
            attempt += 1
    except (Deadline,SliceEnd,ValueError,FloatingPointError):
        pass
    return best
