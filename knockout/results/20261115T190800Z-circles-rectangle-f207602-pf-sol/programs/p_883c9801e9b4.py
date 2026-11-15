import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {'iterations': (140, 40, 300), 'restart_noise': (0.025, 0.0, 0.1), 'search_fraction': (0.69, 0.4, 0.75)}
N = 21
II, JJ = np.triu_indices(N, 1)

class StopSearch(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[II, 0]-a[JJ, 0])**2 + (a[II, 1]-a[JJ, 1])**2)
    if np.any(d < a[II, 2]+a[JJ, 2]):
        return -np.inf
    x = list(map(Fraction, map(float, a[:, 0])))
    y = list(map(Fraction, map(float, a[:, 1])))
    r = list(map(Fraction, map(float, a[:, 2])))
    half = max(map(sum, zip(x,r))) - min(u-v for u,v in zip(x,r))
    half += max(map(sum, zip(y,r))) - min(u-v for u,v in zip(y,r))
    return float(a[:,2].sum()) if half <= 2 else -np.inf

def repair(a):
    a = a.copy()
    a[:, :2] -= a[:, :2].mean(axis=0)
    r = np.maximum(a[:,2], 1e-10)
    d = np.sqrt(((a[II,:2]-a[JJ,:2])**2).sum(axis=1))
    factor = min(1.0, float(np.min(d/(r[II]+r[JJ]))))
    r *= factor * (1-3e-13)
    a[:,2] = r
    half = np.ptp(a[:,0]) + np.ptp(a[:,1])
    half = float(np.sum(np.max(a[:,:2]+r[:,None],axis=0)-np.min(a[:,:2]-r[:,None],axis=0)))
    a *= (2.0/half)*(1-3e-13)
    return a

def model(symmetric, end):
    if symmetric:
        # Three axis circles followed by nine upper/lower pairs.
        T = np.zeros((64,34))
        T[np.arange(12),np.arange(12)] = 1
        T[12:21,:12] = T[3:12,:12]
        T[21+np.arange(3,12),12+np.arange(9)] = 1
        T[21+np.arange(12,21),12+np.arange(9)] = -1
        T[42+np.arange(12),21+np.arange(12)] = 1
        T[54:63,21:33] = T[45:54,21:33]
        T[63,33] = 1
        mirror = np.r_[np.arange(3),np.arange(12,21),np.arange(3,12)]
        keep = []
        for k,(i,j) in enumerate(zip(II,JJ)):
            if time.monotonic() > end: raise StopSearch()
            u,v = sorted((int(mirror[i]),int(mirror[j])))
            if (int(i),int(j)) <= (u,v): keep.append(k)
        ii,jj = II[keep],JJ[keep]
        inds = np.arange(12)
    else:
        T = np.eye(64)
        ii,jj = II,JJ
        inds = np.arange(21)
    X,Y,R = T[:21],T[21:42],T[42:63]
    W = T[63]
    dx,dy,rs = X[ii]-X[jj],Y[ii]-Y[jj],R[ii]+R[jj]
    B = np.vstack((W/2-X[inds]-R[inds],W/2+X[inds]-R[inds],-W/2-Y[inds]-R[inds],-W/2+Y[inds]-R[inds]))
    b = np.r_[np.zeros(2*len(inds)),np.ones(2*len(inds))]
    # For symmetric circles the omitted lower boundaries follow from reflection.
    objective = -R.sum(axis=0)
    def fun(q):
        if time.monotonic() > end: raise StopSearch()
        return float(objective@q), objective
    def con(q):
        if time.monotonic() > end: raise StopSearch()
        vx,vy = dx@q,dy@q
        dist = np.sqrt(vx*vx+vy*vy+1e-28)
        return np.r_[dist-rs@q,B@q+b]
    def jac(q):
        if time.monotonic() > end: raise StopSearch()
        vx,vy = dx@q,dy@q
        dist = np.sqrt(vx*vx+vy*vy+1e-28)
        return np.vstack((vx[:,None]/dist[:,None]*dx+vy[:,None]/dist[:,None]*dy-rs,B))
    return T,fun,con,jac

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + float(params['search_fraction'])*budget_s
    rng = np.random.default_rng(seed)
    best = None
    bestscore = -np.inf
    if init is not None:
        try:
            initial = np.asarray(init['circles'],dtype=float)
            bestscore = exact_score(initial)
            if np.isfinite(bestscore): best = init
        except Exception:
            initial = None
    else:
        initial = None
    # Retain inexpensive repaired iterates, with exact scoring at each improvement.
    def consider(z):
        nonlocal best,bestscore
        if time.monotonic() > end: raise StopSearch()
        a = np.column_stack((z[:21],z[21:42],z[42:63]))
        if not np.isfinite(a).all(): return
        a = repair(a)
        if float(a[:,2].sum()) <= bestscore: return
        score = exact_score(a)
        if score > bestscore:
            bestscore = score
            best = {'circles': a.tolist()}
    try:
        if initial is not None and bestscore > 2.2:
            T,fun,con,jac = model(False,end)
            a = initial.copy()
            lo = np.min(a[:,:2]-a[:,2,None],axis=0)
            hi = np.max(a[:,:2]+a[:,2,None],axis=0)
            a[:,:2] -= (lo+hi)/2
            q = np.r_[a[:,0],a[:,1],a[:,2],hi[0]-lo[0]]
            bounds = [(None,None)]*42+[(1e-7,None)]*21+[(0.1,1.9)]
            result = minimize(fun,q,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':con,'jac':jac},
                callback=consider,options={'maxiter':int(params['iterations']),'ftol':2e-12})
            consider(result.x)
        T,fun,con,jac = model(True,end)
        bounds = [(None,None)]*12+[(0.00001,None)]*9+[(1e-7,None)]*12+[(0.2,1.8)]
        topologies = [(4,5),(5,4),(3,6),(6,3)]
        bestq = None
        bestobjective = -np.inf
        attempt = 0
        while time.monotonic() < end:
            outer,inner = topologies[attempt%4]
            if attempt < 4:
                w = 1.02 if attempt%2 == 0 else 1.16
                h = 2-w
                q = np.zeros(34)
                q[:3] = np.linspace(-0.33*w,0.33*w,3)
                q[3:3+inner] = np.linspace(-0.40*w,0.40*w,inner)
                q[3+inner:12] = np.linspace(-0.39*w,0.39*w,outer)
                q[12:12+inner] = 0.22*h
                q[12+inner:21] = 0.40*h
                q[21:33] = 0.085
                q[33] = w
            else:
                q = bestq.copy() if bestq is not None else q.copy()
                q[:21] += rng.normal(0,float(params['restart_noise']),21)
                q[12:21] = np.maximum(q[12:21],0.015)
            def callback(v):
                nonlocal bestq,bestobjective
                if time.monotonic() > end: raise StopSearch()
                consider(T@v)
                value = float(-fun(v)[0])
                if np.min(con(v)) > -1e-5 and value > bestobjective:
                    bestobjective = value
                    bestq = v.copy()
            result = minimize(fun,q,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':con,'jac':jac}, callback=callback,
                options={'maxiter':int(params['iterations']),'ftol':3e-11})
            callback(result.x)
            attempt += 1
    except StopSearch:
        pass
    if best is None:
        # Separated, exactly safe emergency construction.
        a = np.zeros((21,3))
        a[:,0] = np.arange(21)/12.0
        a[:,2] = 0.02
        best = {'circles':a.tolist()}
    return best
