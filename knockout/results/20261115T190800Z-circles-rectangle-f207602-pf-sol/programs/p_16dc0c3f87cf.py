import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment
from scipy.linalg import lstsq

PARAMS = {
    'iterations': (180, 50, 400),
    'safety': (4e-15, 1e-15, 1e-11, 'log'),
    'contact_threshold': (8e-7, 1e-9, 1e-4, 'log'),
    'restart_size': (0.025, 0.002, 0.09)
}
I, J = np.triu_indices(21, 1)

class Stop(Exception):
    pass

def valid(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return False
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return False
    x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
    return (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r)) +
            max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))) <= 2

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    eps = float(params['safety'])
    best = init
    score = -np.inf
    best_a = None
    if init is not None:
        try:
            a = (np.asarray(init['circles'], dtype=float) if 'circles' in init else
                 np.asarray(init['state'], dtype=float).reshape(3,21).T)
            if valid(a):
                best_a = a.copy()
                score = float(np.sum(a[:,2]))
        except (ValueError, TypeError, KeyError):
            pass

    def offer(a):
        nonlocal best, best_a, score
        if time.monotonic() >= end:
            return
        a = np.array(a, dtype=float, copy=True)
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
        q = min(1., float(np.min(d/(a[I,2]+a[J,2]))))
        if q <= 0:
            return
        a[:,2] *= q*(1-eps)
        lo = np.min(a[:,:2]-a[:,2,None], axis=0)
        hi = np.max(a[:,:2]+a[:,2,None], axis=0)
        scale = (2-8*eps)/float(np.sum(hi-lo))
        a[:,:2] = (a[:,:2]-(hi+lo)/2)*scale
        a[:,2] *= scale
        s = float(np.sum(a[:,2]))
        if s > score and valid(a):
            score, best_a = s, a.copy()
            best = {'state': a.T.ravel().tolist()}

    if best_a is None:
        offer(np.array([[k*np.sqrt(3.),2*j-(n-1),1.]
                        for k,n in enumerate((3,4,5,4,5)) for j in range(n)]))
    if best_a is None:
        return {'circles': [[.2*(i%3),.2*(i//3),.09] for i in range(21)]}
    offer(best_a)

    E = np.eye(64)
    X,Y,R,W = E[:21],E[21:42],E[42:63],E[63]
    DX,DY,RS = X[I]-X[J],Y[I]-Y[J],R[I]+R[J]
    B = np.concatenate((X-R,W-X-R,-.5*W-Y-R,-.5*W+Y-R))
    off = np.r_[np.zeros(42),np.ones(42)]
    objective = -R.sum(axis=0)

    def state(a):
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        return np.r_[a[:,0]-lo[0],a[:,1]-(hi[1]+lo[1])/2,a[:,2],hi[0]-lo[0]]

    def solve(z,T,stop,polish_only=False):
        n = len(z)
        dxm,dym,rsm,b = DX@T,DY@T,RS@T,B@T
        obj = objective@T
        weights = 1/np.maximum(2*(rsm@z),.06)
        cache = [None,None,None]
        last = [z.copy()]
        def calc(v):
            if time.monotonic() >= stop:
                raise Stop
            if cache[0] is not None and np.array_equal(v,cache[0]):
                return cache[1],cache[2]
            dx,dy,rr = dxm@v,dym@v,rsm@v
            c = np.r_[(dx*dx+dy*dy-rr*rr)*weights,b@v+off]
            jac = np.concatenate((2*weights[:,None]*(dx[:,None]*dxm+dy[:,None]*dym-rr[:,None]*rsm),b))
            cache[:] = [v.copy(),c,jac]
            last[0] = v.copy()
            return c,jac
        def emit(v):
            offer((T@v)[:63].reshape(3,21).T)
        def polish(v):
            for step in range(7):
                if time.monotonic() >= stop-.006:
                    break
                c,jac = calc(v)
                active = np.flatnonzero(np.abs(c)<float(params['contact_threshold']))
                if len(active)<n-1:
                    break
                delta,_,rank,_ = lstsq(jac[active],-c[active],lapack_driver='gelsy',check_finite=False)
                if rank<n-1 or np.max(np.abs(delta))>1e-3:
                    break
                v = v+delta
                emit(v)
                if np.max(np.abs(delta))<2e-16:
                    break
            return v
        try:
            z = polish(z)
            if not polish_only and time.monotonic()<stop-.012:
                bounds = [(None,None)]*n
                bounds[42:63] = [(1e-8,.6)]*21 if n==64 else bounds[42:63]
                if n==34:
                    bounds[21:33] = [(1e-8,.6)]*12
                    bounds[12:21] = [(0,None)]*9
                bounds[-1] = (.15,1.85)
                count = [0]
                def fun(v):
                    if time.monotonic()>=stop:
                        raise Stop
                    return float(obj@v),obj
                def callback(v):
                    if time.monotonic()>=stop:
                        raise Stop
                    count[0] += 1
                    if count[0]%10==0:
                        emit(v)
                res = minimize(fun,z,jac=True,method='SLSQP',bounds=bounds,
                    constraints={'type':'ineq','fun':lambda v:calc(v)[0], 'jac':lambda v:calc(v)[1]},
                    callback=callback, options={'maxiter':int(params['iterations']),'ftol':1e-14,'disp':False})
                z = res.x
                emit(z)
                z = polish(z)
        except Stop:
            z = last[0]
        emit(z)
        return z

    def quotient(a):
        choices = []
        for swap in (False,True):
            if time.monotonic()>=end-.02:
                break
            b = a.copy()
            if swap:
                b[:,:2] = b[:,[1,0]]
            lo = np.min(b[:,:2]-b[:,2,None],axis=0)
            hi = np.max(b[:,:2]+b[:,2,None],axis=0)
            b[:,0] -= lo[0]
            b[:,1] -= (lo[1]+hi[1])/2
            singles = np.argsort(abs(b[:,1]))[:3]
            remaining = np.setdiff1d(np.arange(21),singles)
            order = remaining[np.argsort(b[remaining,1])]
            neg,pos = order[:9],order[9:]
            cost = ((b[pos,None,0]-b[None,neg,0])**2 +
                    (b[pos,None,1]+b[None,neg,1])**2 +
                    (b[pos,None,2]-b[None,neg,2])**2)
            rr,cc = linear_sum_assignment(cost)
            neg = neg[cc]
            error = float(cost[rr,cc].sum()+np.sum(b[singles,1]**2))
            z = np.r_[b[singles,0],(b[pos,0]+b[neg,0])/2,
                      (b[pos,1]-b[neg,1])/2,b[singles,2],
                      (b[pos,2]+b[neg,2])/2,hi[0]-lo[0]]
            choices.append((error,z))
        return min(choices,key=lambda t:t[0])[1] if choices else None

    T = np.zeros((64,34))
    ids = np.r_[np.arange(12),np.arange(3,12)]
    signs = np.r_[np.ones(12),-np.ones(9)]
    T[np.arange(21),ids] = 1
    mask = ids>=3
    T[21+np.arange(21)[mask],12+ids[mask]-3] = signs[mask]
    T[42+np.arange(21),21+ids] = 1
    T[63,33] = 1
    if time.monotonic()<end-.04:
        solve(state(best_a),E,min(end-.02,start+.24*budget_s))
    z = quotient(best_a)
    if z is not None and time.monotonic()<end-.04:
        solve(z,T,min(end-.02,start+.46*budget_s))
    if time.monotonic()<end-.04:
        solve(state(best_a),E,min(end-.02,start+.55*budget_s))
    for attempt in range(5):
        if time.monotonic()>=end-.075:
            break
        z = quotient(best_a)
        if z is None:
            break
        s = float(params['restart_size'])*(1+.35*attempt)
        z[:12] += rng.normal(0,s,12)
        z[12:21] = np.maximum(0,z[12:21]+rng.normal(0,s,9))
        solve(z,T,end-.015)
    return best
