import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'first_fraction': (0.43, 0.25, 0.65),
    'trial_fraction': (0.20, 0.1, 0.3),
    'iterations': (180, 60, 350),
    'safety': (1e-14, 4e-15, 1e-11, 'log')
}
I, J = np.triu_indices(21, 1)

class SearchStop(Exception):
    pass

def valid(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return False
    if np.any(np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2) < a[I, 2]+a[J, 2]):
        return False
    xs = [Fraction(float(v)) for v in a[:, 0]]
    ys = [Fraction(float(v)) for v in a[:, 1]]
    rs = [Fraction(float(v)) for v in a[:, 2]]
    return (max(x+r for x,r in zip(xs,rs))-min(x-r for x,r in zip(xs,rs)) +
            max(y+r for y,r in zip(ys,rs))-min(y-r for y,r in zip(ys,rs))) <= 2

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
    scale = (2-12*eps)/float(np.sum(hi-lo))
    a[:,:2] = (a[:,:2]-(hi+lo)/2)*scale
    a[:,2] *= scale
    return a if valid(a) else None

def model(sym):
    if sym:
        n = 34
        ids = np.r_[np.arange(12), np.arange(3,12)]
        signs = np.r_[np.ones(12), -np.ones(9)]
        x = np.zeros((21,n)); y = x.copy(); r = x.copy()
        x[np.arange(21), ids] = 1
        mask = ids >= 3
        y[np.arange(21)[mask], 12+ids[mask]-3] = signs[mask]
        r[np.arange(21), 21+ids] = 1
        pairs = [(i,j) for i in range(12) for j in range(i+1,12)]
        pairs += [(i,12+j-3) for i in range(3,12) for j in range(i,12)]
        pi,pj = np.array(pairs).T
        edge = np.arange(12)
        ridx = np.arange(21,33)
    else:
        n = 64
        x = np.zeros((21,n)); y = x.copy(); r = x.copy()
        x[np.arange(21),np.arange(21)] = 1
        y[np.arange(21),21+np.arange(21)] = 1
        r[np.arange(21),42+np.arange(21)] = 1
        pi,pj = I,J
        edge = np.arange(21)
        ridx = np.arange(42,63)
    w = np.zeros(n); w[-1] = 1
    walls = np.concatenate((x[edge]-r[edge], w-x[edge]-r[edge],
                            1*-0.5*w-y[edge]-r[edge], -0.5*w+y[edge]-r[edge]))
    off = np.r_[np.zeros(2*len(edge)),np.ones(2*len(edge))]
    bounds = [(None,None)]*n
    for k in ridx:
        bounds[int(k)] = (1e-7, .5)
    bounds[-1] = (.15,1.85)
    return x,y,r,x[pi]-x[pj],y[pi]-y[pj],r[pi]+r[pj],walls,off,-r.sum(axis=0),bounds

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + .70*budget_s
    eps = float(params['safety'])
    best = None
    bestscore = -float('inf')
    best_a = None
    def offer(a):
        nonlocal best,bestscore,best_a
        b = repair(a,eps)
        if b is not None:
            score = float(np.sum(b[:,2]))
            if score > bestscore:
                bestscore = score
                best_a = b.copy()
                best = {'state':b.T.reshape(-1).tolist()}
        return b
    original = None
    try:
        if init is not None:
            original = np.asarray(init['circles'],float) if 'circles' in init else np.asarray(init['state'],float).reshape(3,21).T.copy()
            if valid(original):
                best = init
                best_a = original.copy()
                bestscore = float(np.sum(original[:,2]))
            offer(original)
    except (ValueError, KeyError, TypeError):
        original = None
    sym = model(True)
    full = model(False)
    def unpack(z,m):
        return np.column_stack((m[0]@z,m[1]@z,m[2]@z))
    def fullstate(a):
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        return np.r_[a[:,0]-lo[0],a[:,1]-(lo[1]+hi[1])/2,a[:,2],hi[0]-lo[0]]
    def solve(z,m,stop):
        x,y,r,dxm,dym,rsm,b,off,obj,bounds = m
        weights = 1/np.maximum(2*(rsm@z),.05)
        cache = [None,None,None]
        accepted = [z.copy()]
        last = [z.copy()]
        calls = [0]
        def calc(v):
            if time.monotonic() >= stop:
                raise SearchStop
            if cache[0] is not None and np.array_equal(v,cache[0]):
                return cache[1],cache[2]
            dx = dxm@v; dy = dym@v; rs = rsm@v
            c = np.r_[(dx*dx+dy*dy-rs*rs)*weights,b@v+off]
            jac = np.concatenate((2*weights[:,None]*(dx[:,None]*dxm+dy[:,None]*dym-rs[:,None]*rsm),b))
            cache[:] = [v.copy(),c,jac]
            last[0] = v.copy()
            return c,jac
        def fun(v):
            if time.monotonic() >= stop:
                raise SearchStop
            return float(obj@v),obj
        def callback(v):
            if time.monotonic() >= stop:
                raise SearchStop
            accepted[0] = v.copy()
            calls[0] += 1
            if calls[0] % 15 == 0:
                offer(unpack(v,m))
        try:
            result = minimize(fun,z,jac=True,method='SLSQP',bounds=bounds,
                              constraints={'type':'ineq','fun':lambda v:calc(v)[0], 'jac':lambda v:calc(v)[1]},
                              callback=callback,options={'maxiter':int(params['iterations']),'ftol':3e-13,'disp':False})
            z = result.x
        except SearchStop:
            z = accepted[0]
        offer(unpack(z,m))
        offer(unpack(last[0],m))
        return z
    def strip(counts):
        singles = []; pairs = []
        xx = 0.
        previous = None
        for count in counts:
            if time.monotonic() >= deadline:
                return None
            if previous is not None:
                xx += np.sqrt(3.) if count != previous else 2.
            if count == 3:
                singles.append((xx,0.,1.))
                pairs.append((xx,2.,1.))
            else:
                pairs.extend(((xx,1.,1.),(xx,3.,1.)))
            previous = count
        a = np.array(singles+pairs+[(x,-y,r) for x,y,r in pairs])
        a = repair(a,eps)
        if a is None:
            return None
        offer(a)
        s = fullstate(a)
        return np.r_[s[:12],s[24:33],s[42:54],s[-1]]
    patterns = [(3,4,3,4,3,4),(3,4,4,3,4,3),(4,3,3,4,3,4)]
    for k,pattern in enumerate(patterns):
        if time.monotonic() >= deadline-.12:
            break
        q = strip(pattern)
        if q is None:
            continue
        if k == 0:
            stop = min(deadline-.15,start+float(params['first_fraction'])*.70*budget_s)
        else:
            stop = min(deadline-.18,time.monotonic()+float(params['trial_fraction'])*.70*budget_s)
        if stop > time.monotonic()+.015:
            solve(q,sym,stop)
        if bestscore > 2.3658:
            break
    if best_a is None:
        a = np.array([[.2*(i%3),.2*(i//3),.09] for i in range(21)])
        offer(a)
    if best_a is not None and time.monotonic() < deadline-.025:
        solve(fullstate(best_a),full,deadline-.012)
    return best
