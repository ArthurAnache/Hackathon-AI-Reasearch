import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'first_fraction': (0.53, 0.35, 0.75),
    'second_fraction': (0.80, 0.76, 0.90),
    'iterations': (220, 80, 400),
    'safety': (2e-14, 5e-15, 1e-10, 'log')
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
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    return (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r)) +
            max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))) <= 2

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70*budget_s
    eps = params['safety']
    best = None
    best_a = None
    bestscore = -np.inf
    original = None
    try:
        if init is not None:
            original = (np.array(init['circles'], dtype=float) if 'circles' in init
                        else np.array(init['state'], dtype=float).reshape(3,21).T)
            if valid(original):
                best, best_a = init, original.copy()
                bestscore = float(np.sum(original[:,2]))
    except (ValueError, TypeError, KeyError):
        original = None

    def offer(a):
        nonlocal best, best_a, bestscore
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
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        scale = (2-12*eps)/float(np.sum(hi-lo))
        a[:,:2] = (a[:,:2]-(hi+lo)/2)*scale
        a[:,2] *= scale
        score = float(np.sum(a[:,2]))
        if score > bestscore and valid(a):
            bestscore = score
            best_a = a
            best = {'state': a.T.ravel().tolist()}

    def lattice(reverse=False):
        a = np.zeros((21,3))
        a[:,2] = 1
        a[:3,0] = [-2,0,2]
        counts = (4,5) if reverse else (5,4)
        k = 3
        for row, count in enumerate(counts,1):
            if time.monotonic() >= end:
                break
            a[k:k+count,0] = 2*np.arange(count)-(count-1)
            a[k:k+count,1] = np.sqrt(3.)*row
            k += count
        a[12:,0] = a[3:12,0]
        a[12:,1] = -a[3:12,1]
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        return a*(2/np.sum(hi-lo))

    a0 = lattice(False)
    offer(a0)
    if original is not None:
        offer(original)
    if best is None:
        a = np.array([[(i%7)*.25,(i//7)*.25,.1] for i in range(21)])
        best = {'circles': a.tolist()}
        best_a = a
        bestscore = float(np.sum(a[:,2]))

    E = np.eye(64)
    X,Y,R,W = E[:21],E[21:42],E[42:63],E[63]
    DX,DY,RS = X[I]-X[J],Y[I]-Y[J],R[I]+R[J]
    walls = np.concatenate((.5*W-X-R,.5*W+X-R,-.5*W-Y-R,-.5*W+Y-R))
    offset = np.r_[np.zeros(42),np.ones(42)]
    objective = -R.sum(axis=0)
    T = np.zeros((64,34))
    ids = np.r_[np.arange(12),np.arange(3,12)]
    T[np.arange(21),ids] = 1
    T[21+np.arange(3,12),12+np.arange(9)] = 1
    T[21+np.arange(12,21),12+np.arange(9)] = -1
    T[42+np.arange(21),21+ids] = 1
    T[63,33] = 1

    def state(a):
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        b = a.copy()
        b[:,:2] -= (lo+hi)/2
        return np.r_[b.T.ravel(),hi[0]-lo[0]]

    def quotient(a):
        v = state(a)
        return np.r_[v[:12],v[24:33],v[42:54],v[63]]

    def solve(z, transform, stop):
        if time.monotonic() >= stop-.008:
            return
        n = len(z)
        dxm,dym,rsm = DX@transform,DY@transform,RS@transform
        wm = walls@transform
        off = offset.copy()
        if n == 34:
            # Quotient out constraints duplicated by reflection.
            signatures = np.concatenate((np.abs(dxm),np.abs(dym),rsm),axis=1)
            _, keep = np.unique(signatures,axis=0,return_index=True)
            dxm,dym,rsm = dxm[keep],dym[keep],rsm[keep]
            _, keep = np.unique(np.column_stack((wm,off)),axis=0,return_index=True)
            wm,off = wm[keep],off[keep]
        obj = objective@transform
        weights = 1/np.maximum(2*(rsm@z),.08)
        cache = [None,None,None]
        last = [z.copy()]
        count = [0]
        def calc(v):
            if time.monotonic() >= stop:
                raise Timeout
            if cache[0] is not None and np.array_equal(v,cache[0]):
                return cache[1],cache[2]
            dx,dy,rs = dxm@v,dym@v,rsm@v
            c = np.r_[(dx*dx+dy*dy-rs*rs)*weights,wm@v+off]
            jac = np.concatenate((2*weights[:,None]*(dx[:,None]*dxm+dy[:,None]*dym-rs[:,None]*rsm),wm))
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
            last[0] = v.copy()
            count[0] += 1
            if count[0]%10 == 0:
                offer((transform@v)[:63].reshape(3,21).T)
        bounds = [(None,None)]*n
        if n == 34:
            bounds[21:33] = [(1e-6,.49)]*12
            bounds[12:21] = [(0,.95)]*9
        else:
            bounds[42:63] = [(1e-6,.49)]*21
        bounds[-1] = (.15,1.85)
        try:
            res = minimize(fun,z,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda v:calc(v)[0],'jac':lambda v:calc(v)[1]},
                callback=callback,options={'maxiter':params['iterations'],'ftol':2e-13})
            last[0] = res.x
        except Timeout:
            pass
        offer((transform@last[0])[:63].reshape(3,21).T)

    if bestscore > 2.30 and original is not None and valid(original):
        solve(state(best_a),E,end-.012)
        return best
    solve(quotient(a0),T,start+.70*budget_s*params['first_fraction'])
    if time.monotonic() < end-.035:
        a1 = lattice(True)
        offer(a1)
        solve(quotient(a1),T,start+.70*budget_s*params['second_fraction'])
    if time.monotonic() < end-.025:
        solve(state(best_a),E,end-.012)
    return best
