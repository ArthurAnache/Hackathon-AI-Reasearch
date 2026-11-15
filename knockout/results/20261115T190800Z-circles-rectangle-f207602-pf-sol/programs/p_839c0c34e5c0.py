import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment
from scipy.linalg import lstsq

PARAMS = {
    'restarts': (12, 2, 30),
    'amplitude': (0.075, 0.005, 0.22),
    'iterations': (110, 30, 250),
    'safety': (1.5e-14, 3e-15, 1e-11, 'log'),
}
I, J = np.triu_indices(21, 1)

class Halt(Exception):
    pass

def valid(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return False
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return False
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    return (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r)) +
            max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))) <= 2

def repair(a, eps):
    a = np.array(a, dtype=float, copy=True)
    if a.shape != (21,3) or not np.isfinite(a).all() or np.min(a[:,2]) <= 0:
        return None
    d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
    q = min(1., float(np.min(d/(a[I,2]+a[J,2]))))
    if q <= 0:
        return None
    a[:,2] *= q*(1-eps)
    lo = np.min(a[:,:2]-a[:,2,None], axis=0)
    hi = np.max(a[:,:2]+a[:,2,None], axis=0)
    s = (2-4*eps)/np.sum(hi-lo)
    a[:,:2] = (a[:,:2]-(hi+lo)*.5)*s
    a[:,2] *= s
    return a if valid(a) else None

def model(sym):
    if sym:
        n = 34
        ids = np.r_[np.arange(12), np.arange(3,12)]
        signs = np.r_[np.ones(12), -np.ones(9)]
        X = np.zeros((21,n)); Y = X.copy(); R = X.copy()
        X[np.arange(21),ids] = 1
        mask = ids >= 3
        Y[np.arange(21)[mask],12+ids[mask]-3] = signs[mask]
        R[np.arange(21),21+ids] = 1
        pairs = [(i,j) for i in range(12) for j in range(i+1,12)]
        pairs += [(i,12+j-3) for i in range(3,12) for j in range(i,12)]
        pi = np.array([p[0] for p in pairs]); pj = np.array([p[1] for p in pairs])
        b = np.arange(12)
    else:
        n = 64
        X = np.zeros((21,n)); Y = X.copy(); R = X.copy()
        X[np.arange(21),np.arange(21)] = 1
        Y[np.arange(21),21+np.arange(21)] = 1
        R[np.arange(21),42+np.arange(21)] = 1
        pi,pj = I,J
        b = np.arange(21)
    w = np.zeros(n); w[-1] = 1
    B = np.concatenate((X[b]-R[b],w-X[b]-R[b],-.5*w-Y[b]-R[b],-.5*w+Y[b]-R[b]))
    off = np.r_[np.zeros(2*len(b)),np.ones(2*len(b))]
    return X,Y,R,X[pi]-X[pj],Y[pi]-Y[pj],R[pi]+R[pj],B,off,-R.sum(0)

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + .70*budget_s
    rng = np.random.default_rng(seed)
    best = init
    bestscore = -np.inf
    if init is not None:
        try:
            a = np.asarray(init['circles'],float)
            if valid(a):
                bestscore = float(np.sum(a[:,2]))
        except (ValueError, KeyError, TypeError):
            pass
    eps = float(params['safety'])
    def offer(a):
        nonlocal best,bestscore
        if time.monotonic() >= end:
            return
        b = repair(a,eps)
        if b is not None:
            s = float(np.sum(b[:,2]))
            if s > bestscore:
                bestscore = s
                best = {'circles':b.tolist()}
    if bestscore == -np.inf:
        offer(np.array([[(i+.5)*.2,(j+.5)*.2,.099999999999] for j in range(7) for i in range(3)]))
    if best is None:
        return {'circles': [[0.,0.,.01] for _ in range(21)]}
    full = model(False)
    sym = model(True)

    def unpack(z,m):
        return np.column_stack((m[0]@z,m[1]@z,m[2]@z))

    def optimize(z,m,deadline,steps=None):
        X,Y,R,DX,DY,RS,B,off,obj = m
        cached = [None,None,None]
        last = [z.copy()]
        def calc(v):
            if time.monotonic() >= deadline:
                raise Halt()
            if cached[0] is not None and np.array_equal(v,cached[0]):
                return cached[1],cached[2]
            dx = DX@v; dy = DY@v
            d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-18)
            c = np.r_[d-RS@v,B@v+off]
            jac = np.concatenate(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RS,B))
            cached[:] = [v.copy(),c,jac]
            last[0] = v.copy()
            return c,jac
        def fun(v):
            if time.monotonic() >= deadline:
                raise Halt()
            return float(obj@v),obj
        bounds = [(None,None)]*len(z)
        for k in (range(21,33) if len(z)==34 else range(42,63)):
            bounds[k] = (1e-6,.5)
        bounds[-1] = (.25,1.75)
        try:
            res = minimize(fun,z,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda v:calc(v)[0],'jac':lambda v:calc(v)[1]},
                options={'ftol':3e-14,'maxiter':int(steps or params['iterations']),'disp':False})
            z = res.x
        except Halt:
            z = last[0]
        # Polish a nearly square active contact system without changing its topology.
        for _ in range(4):
            if time.monotonic() >= deadline-.004:
                break
            try:
                c,jac = calc(z)
            except Halt:
                break
            active = np.flatnonzero(np.abs(c)<2e-7)
            if len(active)<len(z):
                break
            delta,_,rank,_ = lstsq(jac[active],-c[active],lapack_driver='gelsy',check_finite=False)
            if rank<len(z) or np.max(np.abs(delta))>1e-4:
                break
            z = z+delta
        return z

    def full_start(a):
        lo = np.min(a[:,:2]-a[:,2,None],0)
        hi = np.max(a[:,:2]+a[:,2,None],0)
        p = a[:,:2]-lo
        p[:,1] -= (hi[1]-lo[1])*.5
        return np.r_[p[:,0],p[:,1],a[:,2],hi[0]-lo[0]]

    z = full_start(np.asarray(best['circles'],float))
    z = optimize(z,full,min(end-.02,start+.16*budget_s),50)
    offer(unpack(z,full))
    a = np.asarray(best['circles'],float).copy()
    # Choose the reflection axis from the actual incumbent rather than assuming orientation.
    choices = []
    for swap in (False,True):
        if time.monotonic() >= end-.02:
            return best
        b = a.copy()
        if swap:
            b[:,:2] = b[:,[1,0]]
        lo = np.min(b[:,:2]-b[:,2,None],0)
        hi = np.max(b[:,:2]+b[:,2,None],0)
        b[:,0] -= lo[0]
        b[:,1] -= .5*(lo[1]+hi[1])
        reflected = b.copy(); reflected[:,1] *= -1
        cost = np.sum((b[:,None,:]-reflected[None,:,:])**2,axis=2)
        rows,cols = linear_sum_assignment(cost)
        choices.append((float(np.sum(cost[rows,cols])),b,hi[0]-lo[0]))
    _,a,w = min(choices,key=lambda t:t[0])
    singles = np.argsort(np.abs(a[:,1]))[:3]
    remaining = np.array([i for i in range(21) if i not in singles])
    pos = remaining[np.argsort(a[remaining,1])[-9:]]
    neg = remaining[np.argsort(a[remaining,1])[:9]]
    cost = (a[pos,None,0]-a[None,neg,0])**2+(a[pos,None,1]+a[None,neg,1])**2+(a[pos,None,2]-a[None,neg,2])**2
    rows,cols = linear_sum_assignment(cost)
    neg = neg[cols]
    xx = np.r_[a[singles,0],.5*(a[pos,0]+a[neg,0])]
    yy = .5*(a[pos,1]-a[neg,1])
    rr = np.r_[a[singles,2],.5*(a[pos,2]+a[neg,2])]
    base = np.r_[xx,yy,rr,w]
    z = optimize(base,sym,min(end-.025,time.monotonic()+.16*budget_s),80)
    offer(unpack(z,sym))
    if np.min(unpack(z,sym)[:,2])>0 and np.linalg.norm(z-base)<.2:
        base = z.copy()
    bestlocal = float(np.sum(unpack(base,sym)[:,2]))
    for attempt in range(int(params['restarts'])):
        now = time.monotonic()
        if now >= end-.035:
            break
        z = base.copy()
        amplitude = float(params['amplitude'])*[.35,.7,1.,1.5,2.,.6][attempt%6]
        # Move a connected spatial cluster of quotient circles together.
        points = np.column_stack((base[:12],np.r_[np.zeros(3),base[12:21]]))
        anchor = int(rng.integers(12))
        order = np.argsort(np.sum((points-points[anchor])**2,axis=1))
        selected = order[:int(rng.integers(3,7))]
        shift = rng.normal(0,amplitude,2)
        for k in selected:
            if time.monotonic() >= end-.02:
                break
            z[k] += shift[0]+rng.normal(0,amplitude*.3)
            if k>=3:
                z[12+k-3] = abs(z[12+k-3]+shift[1]+rng.normal(0,amplitude*.3))
            z[21+k] *= rng.uniform(.85,1.08)
        z[-1] += rng.normal(0,amplitude*.3)
        deadline = min(end-.018,now+max(.055,(end-now-.025)/min(4,int(params['restarts'])-attempt)))
        z = optimize(z,sym,deadline)
        candidate = unpack(z,sym)
        offer(candidate)
        if time.monotonic()<end-.02:
            b = repair(candidate,eps)
            if b is not None:
                s = float(np.sum(b[:,2]))
                if s > bestlocal+1e-11:
                    base = z.copy(); bestlocal = s
    return best
