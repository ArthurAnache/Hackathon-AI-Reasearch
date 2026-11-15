import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'screen_fraction': (0.19, 0.12, 0.25),
    'refine_fraction': (0.57, 0.45, 0.63),
    'screen_iterations': (42, 20, 90),
    'iterations': (170, 70, 300),
    'safety': (1.0e-14, 4e-15, 1e-11, 'log')
}
I, J = np.triu_indices(21, 1)

class Timeout(Exception):
    pass

def valid(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return False
    d = np.sqrt((a[I, 0]-a[J, 0])**2+(a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return False
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    return (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))+
            max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))) <= 2

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + .70*budget_s
    eps = params['safety']
    best = None
    best_a = None
    score_best = -float('inf')
    original = None
    try:
        if init is not None:
            original = (np.array(init['circles'], dtype=float) if 'circles' in init
                        else np.array(init['state'], dtype=float).reshape(3,21).T)
            if valid(original):
                best = init
                best_a = original.copy()
                score_best = float(np.sum(original[:,2]))
    except (ValueError, TypeError, KeyError):
        original = None

    def repair(a):
        a = np.array(a, dtype=float, copy=True)
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return None
        dist = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
        ratio = min(1., float(np.min(dist/(a[I,2]+a[J,2]))))
        if ratio <= 0:
            return None
        a[:,2] *= ratio*(1-eps)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        scale = (2-16*eps)/float(np.sum(hi-lo))
        a[:,:2] = (a[:,:2]-(hi+lo)/2)*scale
        a[:,2] *= scale
        return a

    def offer(a):
        nonlocal best, best_a, score_best
        if time.monotonic() >= end:
            return -float('inf'), None
        a = repair(a)
        if a is None:
            return -float('inf'), None
        score = float(np.sum(a[:,2]))
        if score > score_best and valid(a):
            score_best = score
            best_a = a.copy()
            best = {'state': a.T.ravel().tolist()}
        return score, a

    def lattice(reverse):
        a = np.zeros((21,3))
        a[:,2] = 1
        a[:3,0] = [-2,0,2]
        k = 3
        for row, count in enumerate((4,5) if reverse else (5,4),1):
            if time.monotonic() >= end:
                break
            a[k:k+count,0] = 2*np.arange(count)-(count-1)
            a[k:k+count,1] = np.sqrt(3.)*row
            k += count
        a[12:,0] = a[3:12,0]
        a[12:,1] = -a[3:12,1]
        return a

    a0 = repair(lattice(True))
    offer(a0)
    if original is not None:
        offer(original)
    if best is None:
        a = np.array([[(i%7)*.25,(i//7)*.25,.09] for i in range(21)])
        best = {'circles': a.tolist()}
        best_a = a
        score_best = float(np.sum(a[:,2]))

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

    models = {}
    for key, transform in ((34,T),(64,E)):
        if time.monotonic() >= end:
            return best
        dxm,dym,rsm = DX@transform,DY@transform,RS@transform
        wm,off = walls@transform,offset.copy()
        if key == 34:
            signatures = np.concatenate((np.abs(dxm),np.abs(dym),rsm),axis=1)
            _,keep = np.unique(signatures,axis=0,return_index=True)
            dxm,dym,rsm = dxm[keep],dym[keep],rsm[keep]
            _,keep = np.unique(np.column_stack((wm,off)),axis=0,return_index=True)
            wm,off = wm[keep],off[keep]
        models[key] = (dxm,dym,rsm,wm,off,objective@transform)

    def solve(z, transform, stop, iterations):
        n = len(z)
        local_score = -float('inf')
        local_a = None
        last = z.copy()
        if time.monotonic() >= stop-.006:
            a = (transform@z)[:63].reshape(3,21).T
            sc,aa = offer(a)
            return z, sc, aa
        dxm,dym,rsm,wm,off,obj = models[n]
        cache = [None,None,None]
        count = 0

        def checkpoint(v):
            nonlocal local_score, local_a
            sc, aa = offer((transform@v)[:63].reshape(3,21).T)
            if sc > local_score:
                local_score, local_a = sc, aa

        def calc(v):
            if time.monotonic() >= stop:
                raise Timeout
            if cache[0] is not None and np.array_equal(v,cache[0]):
                return cache[1],cache[2]
            dx,dy = dxm@v,dym@v
            dist = np.sqrt(dx*dx+dy*dy)
            inv = 1/np.maximum(dist,1e-12)
            c = np.r_[dist-rsm@v,wm@v+off]
            jac = np.concatenate(((dx*inv)[:,None]*dxm+(dy*inv)[:,None]*dym-rsm,wm))
            cache[:] = [v.copy(),c,jac]
            return c,jac

        def fun(v):
            if time.monotonic() >= stop:
                raise Timeout
            return float(obj@v),obj

        def callback(v):
            nonlocal last, count
            if time.monotonic() >= stop:
                raise Timeout
            last = v.copy()
            count += 1
            if count % 9 == 0:
                checkpoint(v)

        bounds = [(None,None)]*n
        if n == 34:
            bounds[21:33] = [(1e-6,.49)]*12
            bounds[12:21] = [(0,.95)]*9
        else:
            bounds[42:63] = [(1e-6,.49)]*21
        bounds[-1] = (.15,1.85)
        checkpoint(z)
        try:
            res = minimize(fun,z,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda v:calc(v)[0], 'jac':lambda v:calc(v)[1]},
                callback=callback,options={'maxiter':iterations,'ftol':5e-13})
            last = res.x
        except Timeout:
            pass
        checkpoint(last)
        return last, local_score, local_a

    if score_best > 2.30 and original is not None and valid(original):
        solve(state(best_a),E,end-.010,params['iterations'])
        return best

    f = params['screen_fraction']
    z0,s0,b0 = solve(quotient(a0),T,start+f*budget_s,params['screen_iterations'])
    if time.monotonic() >= end-.025:
        return best
    a1 = repair(lattice(False))
    z1,s1,b1 = solve(quotient(a1),T,start+2*f*budget_s,params['screen_iterations'])
    if time.monotonic() < end-.025:
        chosen = z0 if s0 >= s1 else z1
        solve(chosen,T,start+params['refine_fraction']*budget_s,params['iterations'])
    if time.monotonic() < end-.025:
        solve(state(best_a),E,end-.012,params['iterations'])
    return best
