import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {'first_fraction': (0.38, 0.25, 0.55), 'race_iterations': (65, 35, 100), 'release': (0.012, 0.003, 0.04)}

class Stop(Exception):
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
        a = np.asarray(canonical(obj)['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        for i in range(21):
            for j in range(i):
                dx, dy = a[i, :2] - a[j, :2]
                if np.sqrt(dx*dx + dy*dy) < a[i, 2] + a[j, 2]:
                    return -float('inf')
        x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
        wh = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        wh += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        if wh > 2:
            return -float('inf')
        return float(np.sum(a[:, 2]))
    except (ValueError, TypeError, KeyError, OverflowError):
        return -float('inf')


def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + 0.70*budget_s
    stage_end = deadline
    rng = np.random.default_rng(seed)
    best = init
    best_score = score(init) if init is not None else -float('inf')
    if best_score == -float('inf'):
        best = {'circles': [[(i%7-3)*0.17, (i//7-1)*0.17, 0.08] for i in range(21)]}
        best_score = score(best)
    raw = np.asarray(canonical(init if init is not None else best)['circles'], float)
    if raw.shape != (21,3) or not np.isfinite(raw).all():
        raw = np.asarray(best['circles'], float)

    def tick():
        if time.monotonic() >= min(stage_end, deadline):
            raise Stop()

    oi = np.r_[np.arange(12), np.arange(3,12)]
    signs = np.r_[np.ones(12), -np.ones(9)]
    X = np.zeros((21,34)); Y = X.copy(); R = X.copy()
    for i,g in enumerate(oi):
        tick()
        X[i,g] = 1
        R[i,21+g] = 1
        if g >= 3:
            Y[i,9+g] = signs[i]
    pairs = []
    seen = set()
    for i in range(21):
        tick()
        for j in range(i):
            tick()
            key = (min(oi[i],oi[j]), max(oi[i],oi[j]), int(signs[i]*signs[j]) if min(oi[i],oi[j]) >= 3 else 0)
            if key not in seen:
                seen.add(key); pairs.append((i,j))
    p,q = np.asarray(pairs).T
    DX = X[p]-X[q]; DY = Y[p]-Y[q]; RR = R[p]+R[q]
    L = np.zeros((36,34)); off = np.zeros(36)
    for g in range(12):
        tick()
        L[g,g] = 1; L[g,21+g] = -1; L[g,33] = 1
        L[12+g,g] = -1; L[12+g,21+g] = -1; L[12+g,33] = 1
        L[24+g,21+g] = -1; L[24+g,33] = -1; off[24+g] = 1
        if g >= 3:
            L[24+g,9+g] = -1
    grad = np.zeros(34); grad[21:33] = -np.r_[np.ones(3),np.full(9,2.)]
    bounds = [(-1.,1.)]*12 + [(0.,1.)]*9 + [(1e-6,.5)]*12 + [(.06,.94)]
    ii,jj = np.triu_indices(21,1)
    release = np.zeros(len(p))
    cache_z = None; cache_c = None; cache_j = None
    last = None
    endpoints = []

    def retain(z):
        nonlocal best,best_score
        tick()
        if not np.isfinite(z).all() or np.min(z[21:33]) <= 0:
            return -float('inf')
        a = np.column_stack((X@z,Y@z,R@z))
        d = a[ii,:2]-a[jj,:2]
        ratio = np.min(np.sqrt(np.sum(d*d,axis=1))/(a[ii,2]+a[jj,2]))
        if ratio <= 0:
            return -float('inf')
        a[:,2] *= min(1.,float(ratio))*(1.-1.5e-14)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        a[:,:2] -= (lo+hi)/2
        a *= 2./np.sum(hi-lo)*(1.-1.5e-14)
        estimate = float(np.sum(a[:,2]))
        if estimate > best_score:
            obj = {'circles':a.tolist()}
            exact = score(obj)
            if exact > best_score:
                best,best_score = obj,exact
        return estimate

    def con(z):
        nonlocal cache_z,cache_c,cache_j
        tick()
        if cache_z is None or not np.array_equal(cache_z,z):
            dx=DX@z; dy=DY@z
            dist=np.sqrt(dx*dx+dy*dy)
            safe=np.maximum(dist,1e-25)
            cache_c=np.r_[dist-RR@z,L@z+off]
            cache_j=np.vstack(((dx/safe)[:,None]*DX+(dy/safe)[:,None]*DY-RR,L))
            cache_z=z.copy()
        out=cache_c.copy(); out[:len(p)]-=release
        return out

    def jac(z):
        con(z)
        return cache_j

    steps=0
    def callback(z):
        nonlocal last,steps
        tick(); last=z.copy(); steps+=1
        if steps%8==0:
            retain(z)

    def fun(z):
        tick()
        return float(grad@z)

    def solve(z, iterations, seconds):
        nonlocal stage_end,last
        stage_end=min(deadline,time.monotonic()+max(0.,seconds))
        last=z.copy()
        try:
            res=minimize(fun,z,jac=lambda z:grad,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':con,'jac':jac},callback=callback,
                options={'maxiter':iterations,'ftol':3e-13,'disp':False})
            last=res.x.copy()
        except Stop:
            pass
        finally:
            stage_end=deadline
        if time.monotonic()<deadline:
            value=retain(last)
            endpoints.append((value,last.copy()))
        return last.copy()

    def encode(g):
        z=np.zeros(34)
        z[:12]=g[:,0]; z[12:21]=np.abs(g[3:,1]); z[21:33]=np.maximum(g[:,2],1e-5)
        z[33]=np.clip((np.max(g[:,0]+g[:,2])-np.min(g[:,0]-g[:,2]))/2,.06,.94)
        return z

    projections=[]
    for swap in (False,True):
        tick()
        b=raw.copy()
        if swap:
            b[:,[0,1]]=b[:,[1,0]]
        lo=np.min(b[:,:2]-b[:,2,None],axis=0); hi=np.max(b[:,:2]+b[:,2,None],axis=0)
        b[:,:2]-=(lo+hi)/2
        order=np.argsort(b[:,1]); low=order[:9]; axis=order[9:12]; high=order[12:]
        reflected=b[low].copy(); reflected[:,1]*=-1
        cost=np.sum((b[high,None,:]-reflected[None,:,:])**2,axis=2)
        u,v=linear_sum_assignment(cost)
        g=np.vstack((np.column_stack((b[axis,0],np.zeros(3),b[axis,2])),(b[high[u]]+reflected[v])/2))
        projections.append((float(np.sum(cost[u,v])+np.sum(b[axis,1]**2)),encode(g)))
    projections.sort(key=lambda item:item[0])

    seeds=[]
    for inner,outer in ((5,4),(4,5)):
        tick()
        g=[[x,0.,.097] for x in (-.22,0.,.22)]
        g += [[(k-(inner-1)/2)*.205,.1775,.097] for k in range(inner)]
        g += [[(k-(outer-1)/2)*.205,.355,.097] for k in range(outer)]
        seeds.append(encode(np.asarray(g)))

    try:
        solve(projections[0][1],110,budget_s*float(params['first_fraction']))
        if time.monotonic()<deadline:
            solve(seeds[0],int(params['race_iterations']),budget_s*.16)
        if time.monotonic()<deadline:
            if projections[1][0]<projections[0][0]*3+1e-5:
                alternate=projections[1][1]
            else:
                alternate=seeds[1]
            solve(alternate,int(params['race_iterations']),budget_s*.10)
        if endpoints and time.monotonic()<deadline:
            endpoints.sort(key=lambda item:item[0],reverse=True)
            anchor=solve(endpoints[0][1],150,max(0.,deadline-time.monotonic())*.85)
            while time.monotonic()<deadline-.025:
                tick()
                release[:]=0
                gaps=con(anchor)[:len(p)]
                near=np.flatnonzero(gaps<.002)
                if len(near)<3:
                    break
                chosen=rng.choice(near,3,replace=False)
                release[chosen]=float(params['release'])
                moved=solve(anchor,10,min(.045,(deadline-time.monotonic())*.3))
                release[:]=0
                recovered=solve(moved,45,deadline-time.monotonic())
                if float(grad@recovered)<float(grad@anchor) and np.min(con(recovered))>-1e-7:
                    anchor=recovered
    except (Stop,ValueError,FloatingPointError):
        pass
    return best
