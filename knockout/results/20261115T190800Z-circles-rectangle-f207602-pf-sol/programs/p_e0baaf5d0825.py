import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'quotient_fraction': (0.70, 0.4, 0.9),
    'iterations': (240, 80, 500),
    'safety': (2e-14, 8e-15, 1e-10, 'log')
}
I, J = np.triu_indices(21, 1)
RR = np.arange(210)

class Timeout(Exception):
    pass

def valid(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return False
    d = np.sqrt((a[I,0]-a[J,0])**2 + (a[I,1]-a[J,1])**2)
    if np.any(d < a[I,2]+a[J,2]):
        return False
    xs = [Fraction(float(v)) for v in a[:,0]]
    ys = [Fraction(float(v)) for v in a[:,1]]
    rs = [Fraction(float(v)) for v in a[:,2]]
    return (max(x+r for x,r in zip(xs,rs))-min(x-r for x,r in zip(xs,rs)) +
            max(y+r for y,r in zip(ys,rs))-min(y-r for y,r in zip(ys,rs))) <= 2

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70*budget_s
    eps = float(params['safety'])
    best = None
    bestscore = -float('inf')
    original = None
    if init is not None:
        try:
            original = (np.asarray(init['circles'], float) if 'circles' in init else
                        np.asarray(init['state'], float).reshape(3,21).T)
            if valid(original):
                best = init
                bestscore = float(np.sum(original[:,2]))
        except (ValueError, TypeError, KeyError):
            original = None

    def offer(a):
        nonlocal best, bestscore
        a = np.asarray(a, float).copy()
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return None
        d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
        f = min(1.0, float(np.min(d/(a[I,2]+a[J,2]))))
        if f <= 0:
            return None
        a[:,2] *= f*(1-eps)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        scale = (2-8*eps)/float(np.sum(hi-lo))
        a[:,:2] = (a[:,:2]-(lo+hi)*.5)*scale
        a[:,2] *= scale
        score = float(np.sum(a[:,2]))
        if score > bestscore and valid(a):
            bestscore = score
            best = {'state':a.T.reshape(-1).tolist()}
        return a

    # Three axial singles, then nine upper and nine lower partners.
    singles = [[np.sqrt(3)*k,0.,1.] for k in (0,2,4)]
    upper = []
    for k in range(6):
        if time.monotonic() >= end:
            break
        for y in ((2.,) if k%2 == 0 else (1.,3.)):
            if time.monotonic() >= end:
                break
            upper.append([np.sqrt(3)*k,y,1.])
    if len(upper) != 9:
        return best if best is not None else {'circles':[[.2*(i%3),.2*(i//3),.09] for i in range(21)]}
    lattice = np.array(singles+upper+[[x,-y,r] for x,y,r in upper])
    lattice = offer(lattice)

    wall = np.zeros((84,64))
    k = np.arange(21)
    wall[k,k] = 1
    wall[21+k,k] = -1
    wall[42+k,21+k] = 1
    wall[63+k,21+k] = -1
    wall[:,42:63] = np.tile(-np.eye(21),(4,1))
    wall[:42,63] = .5
    wall[42:,63] = -.5
    offsets = np.r_[np.zeros(42),np.ones(42)]
    objective = np.r_[np.zeros(42),-np.ones(21),0.]
    T = np.zeros((64,34))
    ids = np.r_[np.arange(12),np.arange(3,12)]
    signs = np.r_[np.ones(12),-np.ones(9)]
    T[np.arange(21),ids] = 1
    mask = ids >= 3
    T[21+np.arange(21)[mask],12+ids[mask]-3] = signs[mask]
    T[42+np.arange(21),21+ids] = 1
    T[63,33] = 1

    def state(a):
        a = a.copy()
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        a[:,:2] -= (lo+hi)*.5
        return np.r_[a.T.reshape(-1),hi[0]-lo[0]]

    def solve(v, stop, trans=None):
        if time.monotonic() >= stop-.008:
            return
        reduced = trans is not None
        wj = wall@trans if reduced else wall
        obj = objective@trans if reduced else objective
        last = [v.copy()]
        cache = [None,None,None]
        count = [0]
        def unpack(q):
            return trans@q if reduced else q
        def calc(q):
            if time.monotonic() >= stop:
                raise Timeout()
            if cache[0] is not None and np.array_equal(q,cache[0]):
                return cache[1],cache[2]
            z = unpack(q)
            dx = z[I]-z[J]
            dy = z[21+I]-z[21+J]
            rs = z[42+I]+z[42+J]
            c = np.r_[(dx*dx+dy*dy-rs*rs)*4.,wall@z+offsets]
            jac = np.zeros((210,64))
            jac[RR,I] = 8*dx
            jac[RR,J] = -8*dx
            jac[RR,21+I] = 8*dy
            jac[RR,21+J] = -8*dy
            jac[RR,42+I] = -8*rs
            jac[RR,42+J] = -8*rs
            if reduced:
                jac = jac@trans
            jac = np.vstack((jac,wj))
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
            if count[0]%12 == 0:
                offer(unpack(q)[:63].reshape(3,21).T)
        bounds = [(None,None)]*len(v)
        for j in (range(21,33) if reduced else range(42,63)):
            if time.monotonic() >= stop:
                return
            bounds[j] = (1e-6,.7)
        bounds[-1] = (.1,1.9)
        try:
            result = minimize(fun,v,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda q:calc(q)[0],
                             'jac':lambda q:calc(q)[1]},callback=callback,
                options={'maxiter':int(params['iterations']),'ftol':1e-13,'disp':False})
            last[0] = result.x
        except Timeout:
            pass
        offer(unpack(last[0])[:63].reshape(3,21).T)

    if original is not None and bestscore > 2.30:
        solve(state(original),end-.012)
        return best

    z = state(lattice)
    v = np.r_[z[:12],z[24:33],z[42:54],z[-1]]
    split = start + .70*budget_s*float(params['quotient_fraction'])
    solve(v,min(split,end-.02),T)
    if time.monotonic() < end-.018:
        a = (np.asarray(best['circles'],float) if 'circles' in best else
             np.asarray(best['state'],float).reshape(3,21).T)
        solve(state(a),end-.012)
    return best
