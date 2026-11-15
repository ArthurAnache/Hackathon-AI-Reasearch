import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import lstsq

PARAMS = {
    'first_share': (0.31, 0.2, 0.45),
    'second_share': (0.27, 0.15, 0.4),
    'third_share': (0.20, 0.1, 0.3),
    'iterations': (180, 80, 300),
    'safety': (3e-14, 1e-14, 1e-10, 'log')
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
    x, y, r = [[Fraction(float(t)) for t in a[:, k]] for k in range(3)]
    return (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r)) +
            max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))) <= 2

def repaired(a, eps):
    a = np.array(a, dtype=float, copy=True)
    if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
        return None
    d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
    f = min(1., float(np.min(d/(a[I,2]+a[J,2]))))
    if f <= 0:
        return None
    a[:,2] *= f*(1-eps)
    lo = np.min(a[:,:2]-a[:,2,None],axis=0)
    hi = np.max(a[:,:2]+a[:,2,None],axis=0)
    scale = (2-12*eps)/float(np.sum(hi-lo))
    a[:,:2] = (a[:,:2]-(hi+lo)/2)*scale
    a[:,2] *= scale
    return a if valid(a) else None

def model(sym):
    n = 34 if sym else 64
    X = np.zeros((21,n)); Y = X.copy(); R = X.copy()
    if sym:
        ids = np.r_[np.arange(12),np.arange(3,12)]
        X[np.arange(21),ids] = 1
        mask = ids >= 3
        Y[np.flatnonzero(mask),12+ids[mask]-3] = np.r_[np.ones(9),-np.ones(9)]
        R[np.arange(21),21+ids] = 1
        pairs = [(i,j) for i in range(12) for j in range(i+1,12)]
        pairs += [(i,12+j-3) for i in range(3,12) for j in range(i,12)]
        pi,pj = np.array(pairs).T
        wall = np.arange(12)
        ridx = np.arange(21,33)
    else:
        X[np.arange(21),np.arange(21)] = 1
        Y[np.arange(21),21+np.arange(21)] = 1
        R[np.arange(21),42+np.arange(21)] = 1
        pi,pj = I,J
        wall = np.arange(21)
        ridx = np.arange(42,63)
    W = np.zeros(n); W[-1] = 1
    B = np.concatenate((X[wall]-R[wall],W-X[wall]-R[wall],
                        1-Y[wall]*0 if False else -.5*W-Y[wall]-R[wall],
                        -.5*W+Y[wall]-R[wall]))
    off = np.r_[np.zeros(2*len(wall)),np.ones(2*len(wall))]
    return X,Y,R,X[pi]-X[pj],Y[pi]-Y[pj],R[pi]+R[pj],B,off,-R.sum(axis=0),ridx

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + .70*budget_s
    eps = float(params['safety'])
    incumbent = None
    score = -float('inf')
    best_a = None
    original = None
    try:
        if init is not None:
            original = np.asarray(init['circles'],float) if 'circles' in init else np.asarray(init['state'],float).reshape(3,21).T
            if valid(original):
                incumbent = init
                score = float(original[:,2].sum())
                best_a = original.copy()
    except (ValueError,TypeError,KeyError):
        original = None

    def offer(a):
        nonlocal incumbent, score, best_a
        b = repaired(a,eps)
        if b is not None:
            s = float(np.sum(b[:,2]))
            if s > score:
                score = s
                best_a = b
                incumbent = {'state':b.T.reshape(-1).tolist()}
        return b

    # A six-column triangular strip has exactly three axis circles and nine pairs.
    singles = []
    upper = []
    for j,count in enumerate((3,4,3,4,3,4)):
        if time.monotonic() >= end:
            break
        xx = np.sqrt(3.)*j
        if count == 3:
            singles.append([xx,0.,1.])
            upper.append([xx,2.,1.])
        else:
            upper.extend([[xx,1.,1.],[xx,3.,1.]])
    first = np.array(singles+upper+[[x,-y,r] for x,y,r in upper])
    a0 = offer(first)
    if original is not None:
        offer(original)
    if incumbent is None:
        return {'circles':[[.22*(i%3),.22*(i//3),.09] for i in range(21)]}
    if time.monotonic() >= end-.02:
        return incumbent
    sm = model(True)
    fm = model(False)

    def unpack(z,m):
        return np.column_stack((m[0]@z,m[1]@z,m[2]@z))

    def state(a,sym):
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        a = a.copy()
        a[:,0] -= lo[0]
        a[:,1] -= (lo[1]+hi[1])/2
        if sym:
            return np.r_[a[:12,0],a[3:12,1],a[:12,2],hi[0]-lo[0]]
        return np.r_[a[:,0],a[:,1],a[:,2],hi[0]-lo[0]]

    def optimize(z,m,stop):
        X,Y,R,D,E,S,B,off,obj,ridx = m
        cache = [None,None,None]
        last = [z.copy()]
        count = [0]
        def calc(v):
            if time.monotonic() >= stop:
                raise Timeout()
            if cache[0] is not None and np.array_equal(v,cache[0]):
                return cache[1],cache[2]
            dx = D@v; dy = E@v; rr = S@v
            dist = np.sqrt(dx*dx+dy*dy)
            inv = 1/np.maximum(dist,1e-14)
            c = np.r_[dist-rr,B@v+off]
            jac = np.concatenate(((dx*inv)[:,None]*D+(dy*inv)[:,None]*E-S,B))
            cache[:] = [v.copy(),c,jac]
            last[0] = v.copy()
            return c,jac
        def fun(v):
            if time.monotonic() >= stop:
                raise Timeout()
            return float(obj@v),obj
        def callback(v):
            if time.monotonic() >= stop:
                raise Timeout()
            count[0] += 1
            if count[0]%15 == 0:
                offer(unpack(v,m))
        bounds = [(None,None)]*len(z)
        for k in ridx:
            if time.monotonic() >= stop:
                return z
            bounds[int(k)] = (1e-6,.6)
        bounds[-1] = (.2,1.8)
        try:
            res = minimize(fun,z,jac=True,method='SLSQP',bounds=bounds,
                           constraints={'type':'ineq','fun':lambda v:calc(v)[0],'jac':lambda v:calc(v)[1]},
                           callback=callback,options={'maxiter':int(params['iterations']),'ftol':3e-13,'disp':False})
            z = res.x
        except Timeout:
            z = last[0]
        offer(unpack(z,m))
        for iteration in range(3):
            if time.monotonic() >= stop-.004:
                break
            try:
                c,jac = calc(z)
            except Timeout:
                break
            active = np.flatnonzero(np.abs(c)<2e-7)
            if len(active)<len(z):
                break
            delta,_,rank,_ = lstsq(jac[active],-c[active],lapack_driver='gelsy',check_finite=False)
            if rank<len(z) or np.max(np.abs(delta))>2e-4:
                break
            z += delta
            offer(unpack(z,m))
        return z

    if original is not None and float(np.sum(original[:,2])) > 2.30:
        optimize(state(original,False),fm,end-.01)
        return incumbent

    total = end-start
    cut1 = start+float(params['first_share'])*total
    if a0 is not None and time.monotonic()<cut1-.01:
        optimize(state(a0,True),sm,cut1)

    # A five-row hexagonal family, with the central three circles on the axis.
    axis = [[2.*k,0.,1.] for k in (-1,0,1)]
    up = [[2.*k+1,np.sqrt(3.),1.] for k in (-2,-1,0,1)]
    up += [[2.*k,2*np.sqrt(3.),1.] for k in (-2,-1,0,1,2)]
    second = np.array(axis+up+[[x,-y,r] for x,y,r in up])
    cut2 = cut1+float(params['second_share'])*total
    if time.monotonic()<cut2-.02:
        b = repaired(second,eps)
        optimize(state(b,True),sm,cut2)

    # A seven-column, three-circle strip reflected across its middle column.
    axis = [[float(k),0.,1.] for k in (-2,0,2)]
    up = []
    for d in (1,2,3):
        if time.monotonic()>=end-.03:
            break
        shift = 1. if d%2 else 0.
        up.extend([[2.*k+shift,np.sqrt(3.)*d,1.] for k in (-1,0,1)])
    cut3 = min(end-.08,cut2+float(params['third_share'])*total)
    if len(up)==9 and time.monotonic()<cut3-.02:
        third = np.array(axis+up+[[x,-y,r] for x,y,r in up])
        b = repaired(third,eps)
        optimize(state(b,True),sm,cut3)
    if time.monotonic()<end-.025:
        optimize(state(best_a,False),fm,end-.008)
    return incumbent
