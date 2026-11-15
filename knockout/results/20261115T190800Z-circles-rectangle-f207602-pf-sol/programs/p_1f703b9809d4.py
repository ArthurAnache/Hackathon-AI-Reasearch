import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment
from scipy.linalg import lstsq

PARAMS = {
    'full_fraction': (0.81, 0.5, 0.96),
    'iterations': (350, 100, 600),
    'safety': (1.5e-14, 5e-15, 1e-10, 'log')
}
I, J = np.triu_indices(21, 1)
ROWS = np.arange(len(I))

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

def repair(a, eps):
    a = np.asarray(a, dtype=float).copy()
    if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
        return None
    d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
    f = min(1., float(np.min(d/(a[I,2]+a[J,2]))))
    if f <= 0:
        return None
    a[:,2] *= f*(1-eps)
    lo = np.min(a[:,:2]-a[:,2,None], axis=0)
    hi = np.max(a[:,:2]+a[:,2,None], axis=0)
    s = (2-8*eps)/float(np.sum(hi-lo))
    a[:,:2] = (a[:,:2]-(hi+lo)/2)*s
    a[:,2] *= s
    return a if valid(a) else None

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + .70*budget_s
    eps = float(params['safety'])
    best = None
    bestscore = -float('inf')
    original = None
    try:
        if init is not None:
            original = (np.asarray(init['circles'],float) if 'circles' in init else
                        np.asarray(init['state'],float).reshape(3,21).T)
            if valid(original):
                best = init
                bestscore = float(np.sum(original[:,2]))
    except (ValueError, TypeError, KeyError):
        original = None

    def offer(a):
        nonlocal best, bestscore
        b = repair(a, eps)
        if b is not None:
            score = float(np.sum(b[:,2]))
            if score > bestscore:
                bestscore = score
                best = {'state': b.T.reshape(-1).tolist()}
        return b

    if original is not None:
        offer(original)
    fallback = np.array([[2*(k+(.5 if n == 4 else 0)), np.sqrt(3.)*row, 1.]
                         for row,n in enumerate((4,5,4,5,3)) for k in range(n)])
    offer(fallback)
    if best is None:
        return {'circles': [[.2*(i%3), .2*(i//3), .09] for i in range(21)]}

    def best_array():
        return (np.asarray(best['circles'],float).copy() if 'circles' in best else
                np.asarray(best['state'],float).reshape(3,21).T.copy())

    def state(a):
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        b = a.copy()
        b[:,:2] -= (hi+lo)/2
        return np.r_[b.T.reshape(-1), hi[0]-lo[0]]

    # Constant wall Jacobian in centered rectangle coordinates.
    wall = np.zeros((84,64))
    k = np.arange(21)
    wall[k,k] = 1
    wall[21+k,k] = -1
    wall[42+k,21+k] = 1
    wall[63+k,21+k] = -1
    wall[:,42:63] = np.tile(-np.eye(21),(4,1))
    wall[:42,63] = .5
    wall[42:,63] = -.5
    offsets = np.r_[np.zeros(42), np.ones(42)]
    objective = np.r_[np.zeros(42), -np.ones(21), 0.]

    def optimize(v, stop, transform=None):
        if time.monotonic() >= stop-.005:
            return v
        T = transform
        z0 = v if T is None else T@v
        weights = 1/np.maximum(2*(z0[42+I]+z0[42+J]), .08)
        wjac = wall if T is None else wall@T
        obj = objective if T is None else objective@T
        cache = [None,None,None]
        last = [v.copy()]
        count = [0]

        def unpack(q):
            z = q if T is None else T@q
            return z[:63].reshape(3,21).T

        def calc(q):
            if time.monotonic() >= stop:
                raise Timeout()
            if cache[0] is not None and np.array_equal(q,cache[0]):
                return cache[1],cache[2]
            z = q if T is None else T@q
            dx = z[I]-z[J]
            dy = z[21+I]-z[21+J]
            rs = z[42+I]+z[42+J]
            c = np.r_[(dx*dx+dy*dy-rs*rs)*weights, wall@z+offsets]
            jac = np.zeros((210,64))
            vx = 2*weights*dx
            vy = 2*weights*dy
            vr = -2*weights*rs
            jac[ROWS,I] = vx
            jac[ROWS,J] = -vx
            jac[ROWS,21+I] = vy
            jac[ROWS,21+J] = -vy
            jac[ROWS,42+I] = vr
            jac[ROWS,42+J] = vr
            if T is not None:
                jac = jac@T
            jac = np.vstack((jac,wjac))
            cache[:] = [q.copy(),c,jac]
            last[0] = q.copy()
            return c,jac

        def fun(q):
            if time.monotonic() >= stop:
                raise Timeout()
            return float(obj@q),obj

        def callback(q):
            if time.monotonic() >= stop:
                raise Timeout()
            count[0] += 1
            if count[0] % 9 == 0:
                offer(unpack(q))

        bounds = [(None,None)]*len(v)
        ridx = range(42,63) if T is None else range(21,33)
        for idx in ridx:
            if time.monotonic() >= stop:
                return v
            bounds[idx] = (1e-7,.6)
        bounds[-1] = (.12,1.88)
        try:
            result = minimize(fun,v,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda q:calc(q)[0], 'jac':lambda q:calc(q)[1]},
                callback=callback, options={'maxiter':int(params['iterations']), 'ftol':8e-14, 'disp':False})
            v = result.x
        except Timeout:
            v = last[0]
        offer(unpack(v))
        for _ in range(4):
            if time.monotonic() >= stop-.006:
                break
            try:
                c,jac = calc(v)
            except Timeout:
                break
            active = np.flatnonzero(np.abs(c)<2e-7)
            if len(active)<len(v):
                break
            delta,_,rank,_ = lstsq(jac[active],-c[active],lapack_driver='gelsy',check_finite=False)
            if rank<len(v) or np.max(np.abs(delta))>2e-4:
                break
            v = v+delta
            offer(unpack(v))
        return v

    a = original
    if a is None or a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2]<=0):
        a = best_array()
    fullstop = start + .70*budget_s*float(params['full_fraction'])
    optimize(state(a),min(fullstop,end-.02))
    if time.monotonic() >= end-.035:
        return best

    # Infer the better reflection axis only after full-space recovery.
    a = best_array()
    choices = []
    for swap in (False,True):
        if time.monotonic() >= end-.025:
            return best
        b = a.copy()
        if swap:
            b[:,:2] = b[:,[1,0]]
        z = state(b)
        b = z[:63].reshape(3,21).T.copy()
        ref = b.copy()
        ref[:,1] *= -1
        costs = np.sum((b[:,None,:]-ref[None,:,:])**2,axis=2)
        rr,cc = linear_sum_assignment(costs)
        choices.append((float(costs[rr,cc].sum()),b,z[-1]))
    _,a,width = min(choices,key=lambda q:q[0])
    singles = np.argsort(np.abs(a[:,1]))[:3]
    keep = np.ones(21,bool)
    keep[singles] = False
    rem = np.flatnonzero(keep)
    order = rem[np.argsort(a[rem,1])]
    neg,pos = order[:9],order[9:]
    cost = ((a[pos,None,0]-a[None,neg,0])**2 +
            (a[pos,None,1]+a[None,neg,1])**2 +
            (a[pos,None,2]-a[None,neg,2])**2)
    _,cc = linear_sum_assignment(cost)
    neg = neg[cc]
    v = np.r_[a[singles,0],.5*(a[pos,0]+a[neg,0]),
              .5*(a[pos,1]-a[neg,1]),a[singles,2],
              .5*(a[pos,2]+a[neg,2]),width]
    T = np.zeros((64,34))
    ids = np.r_[np.arange(12),np.arange(3,12)]
    signs = np.r_[np.ones(12),-np.ones(9)]
    T[np.arange(21),ids] = 1
    mask = ids>=3
    T[21+np.arange(21)[mask],12+ids[mask]-3] = signs[mask]
    T[42+np.arange(21),21+ids] = 1
    T[63,33] = 1
    optimize(v,end-.012,T)
    return best
