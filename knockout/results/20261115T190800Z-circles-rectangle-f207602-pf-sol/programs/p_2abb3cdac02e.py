import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment
from scipy.linalg import lstsq

PARAMS = {
    'recovery_fraction': (0.82, 0.55, 0.95),
    'symmetry_threshold': (0.035, 0.002, 0.15),
    'safety': (1e-14, 3e-15, 1e-10, 'log'),
    'iterations': (300, 80, 500)
}
I, J = np.triu_indices(21, 1)

class Timeout(Exception):
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
    end = start + .70*budget_s
    eps = params['safety']
    best = None
    bestscore = -np.inf
    original = None
    try:
        if init is not None:
            original = np.array(init['circles'], dtype=float) if 'circles' in init else np.array(init['state'], dtype=float).reshape(3,21).T
            if valid(original):
                best, bestscore = init, float(np.sum(original[:,2]))
    except (ValueError, TypeError, KeyError):
        original = None

    def offer(a):
        nonlocal best, bestscore
        if time.monotonic() >= end:
            return
        a = np.array(a, dtype=float, copy=True)
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        d = np.sqrt((a[I,0]-a[J,0])**2 + (a[I,1]-a[J,1])**2)
        q = min(1., float(np.min(d/(a[I,2]+a[J,2]))))
        if q <= 0:
            return
        a[:,2] *= q*(1-eps)
        lo = np.min(a[:,:2]-a[:,2,None], axis=0)
        hi = np.max(a[:,:2]+a[:,2,None], axis=0)
        scale = (2-8*eps)/np.sum(hi-lo)
        a[:,:2] = (a[:,:2]-(hi+lo)/2)*scale
        a[:,2] *= scale
        score = float(np.sum(a[:,2]))
        if score > bestscore and valid(a):
            bestscore = score
            best = {'state': a.T.ravel().tolist()}

    if original is not None:
        offer(original)
    lattice = np.array([[2*c+(1 if row%2 == 0 else 0), np.sqrt(3.)*row, 1.]
                        for row,count in enumerate((4,5,4,5,3)) for c in range(count)])
    offer(lattice)
    if best is None:
        best = {'circles': [[(i%3)*.2,(i//3)*.2,.09] for i in range(21)]}
        bestscore = 1.89

    def array_best():
        return np.array(best['circles']) if 'circles' in best else np.array(best['state']).reshape(3,21).T

    def state(a):
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        return np.r_[a[:,0]-lo[0], a[:,1]-(lo[1]+hi[1])/2, a[:,2], hi[0]-lo[0]]

    eye = np.eye(64)
    X,Y,R = eye[:21], eye[21:42], eye[42:63]
    W = eye[-1]
    wall = np.concatenate((X-R, W-X-R, -.5*W-Y-R, -.5*W+Y-R))
    offset = np.r_[np.zeros(42),np.ones(42)]
    DX,DY,RS = X[I]-X[J],Y[I]-Y[J],R[I]+R[J]
    objective = -R.sum(axis=0)

    def solve(z, transform, stop):
        n = transform.shape[1]
        dxm,dym,rsm = DX@transform,DY@transform,RS@transform
        walls = wall@transform
        obj = objective@transform
        weights = 1/np.maximum(2*(rsm@z),.08)
        cache = [None,None,None]
        last = [z.copy()]
        count = [0]
        def calc(v):
            if time.monotonic() >= stop:
                raise Timeout
            if cache[0] is not None and np.array_equal(cache[0],v):
                return cache[1],cache[2]
            dx,dy,rs = dxm@v,dym@v,rsm@v
            c = np.r_[(dx*dx+dy*dy-rs*rs)*weights, walls@v+offset]
            jac = np.concatenate((2*weights[:,None]*(dx[:,None]*dxm+dy[:,None]*dym-rs[:,None]*rsm), walls))
            cache[:] = [v.copy(),c,jac]
            last[0] = v.copy()
            return c,jac
        def fun(v):
            if time.monotonic() >= stop:
                raise Timeout
            return float(obj@v),obj
        def callback(v):
            if time.monotonic() >= stop:
                raise Timeout
            count[0] += 1
            if count[0]%15 == 0:
                offer((transform@v)[:63].reshape(3,21).T)
        bounds = [(None,None)]*n
        if n == 64:
            bounds[42:63] = [(1e-7,.5)]*21
        else:
            bounds[21:33] = [(1e-7,.5)]*12
        bounds[-1] = (.15,1.85)
        try:
            res = minimize(fun,z,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda v:calc(v)[0],'jac':lambda v:calc(v)[1]},
                callback=callback,options={'maxiter':params['iterations'],'ftol':8e-14})
            z = res.x
        except Timeout:
            z = last[0]
        offer((transform@z)[:63].reshape(3,21).T)
        for _ in range(4):
            if time.monotonic() >= stop-.006:
                break
            try:
                c,jac = calc(z)
            except Timeout:
                break
            active = np.flatnonzero(np.abs(c)<2e-7)
            if len(active)<n-1:
                break
            delta,_,rank,_ = lstsq(jac[active],-c[active],lapack_driver='gelsy',check_finite=False)
            if rank<n-1 or np.max(np.abs(delta))>2e-5:
                break
            z += delta
            offer((transform@z)[:63].reshape(3,21).T)
        return z

    a = original
    if a is None or a.shape!=(21,3) or not np.isfinite(a).all() or np.any(a[:,2]<=0):
        a = array_best()
    firststop = start + .70*budget_s*params['recovery_fraction']
    solve(state(a),eye,firststop)
    if time.monotonic() >= end-.025:
        return best
    a = array_best()
    choices = []
    for swap in (False,True):
        if time.monotonic() >= end-.02:
            return best
        b = a.copy()
        if swap:
            b[:,:2] = b[:,[1,0]]
        lo = np.min(b[:,:2]-b[:,2,None],axis=0)
        hi = np.max(b[:,:2]+b[:,2,None],axis=0)
        b[:,0] -= lo[0]
        b[:,1] -= (lo[1]+hi[1])/2
        ref = b.copy(); ref[:,1] *= -1
        cost = np.sum((b[:,None,:]-ref[None,:,:])**2,axis=2)
        rows,cols = linear_sum_assignment(cost)
        choices.append((float(cost[rows,cols].sum()),b,hi[0]-lo[0]))
    error,b,width = min(choices,key=lambda q:q[0])
    if error > params['symmetry_threshold']:
        solve(state(a),eye,end-.012)
        return best
    singles = np.argsort(np.abs(b[:,1]))[:3]
    remaining = np.array([k for k in range(21) if k not in singles])
    ordered = remaining[np.argsort(b[remaining,1])]
    neg,pos = ordered[:9],ordered[9:]
    cost = ((b[pos,None,0]-b[None,neg,0])**2 +
            (b[pos,None,1]+b[None,neg,1])**2 +
            (b[pos,None,2]-b[None,neg,2])**2)
    _,cols = linear_sum_assignment(cost)
    neg = neg[cols]
    z = np.r_[b[singles,0],(b[pos,0]+b[neg,0])/2,
              (b[pos,1]-b[neg,1])/2,b[singles,2],
              (b[pos,2]+b[neg,2])/2,width]
    T = np.zeros((64,34))
    ids = np.r_[np.arange(12),np.arange(3,12)]
    signs = np.r_[np.ones(12),-np.ones(9)]
    T[np.arange(21),ids] = 1
    mask = ids>=3
    T[21+np.arange(21)[mask],12+ids[mask]-3] = signs[mask]
    T[42+np.arange(21),21+ids] = 1
    T[63,33] = 1
    solve(z,T,end-.012)
    return best
