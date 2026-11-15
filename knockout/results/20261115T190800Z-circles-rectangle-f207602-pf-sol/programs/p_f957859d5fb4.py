import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'first_fraction': (0.46, 0.25, 0.65),
    'second_fraction': (0.77, 0.66, 0.88),
    'iterations': (180, 80, 350),
    'safety': (1e-14, 4e-15, 1e-10, 'log')
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
    x = list(map(lambda v: Fraction(float(v)), a[:, 0]))
    y = list(map(lambda v: Fraction(float(v)), a[:, 1]))
    r = list(map(lambda v: Fraction(float(v)), a[:, 2]))
    return (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r)) +
            max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))) <= 2

def repair(a, eps):
    a = np.asarray(a, dtype=float).copy()
    if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
        return None
    d = np.sqrt((a[I,0]-a[J,0])**2 + (a[I,1]-a[J,1])**2)
    ratio = min(1., float(np.min(d/(a[I,2]+a[J,2]))))
    if ratio <= 0:
        return None
    a[:,2] *= ratio*(1-eps)
    lo = np.min(a[:,:2]-a[:,2,None], axis=0)
    hi = np.max(a[:,:2]+a[:,2,None], axis=0)
    scale = (2-8*eps)/float(np.sum(hi-lo))
    a[:,:2] = (a[:,:2]-(hi+lo)/2)*scale
    a[:,2] *= scale
    return a if valid(a) else None

def model(sym):
    if sym:
        n = 34
        ids = np.r_[np.arange(12), np.arange(3,12)]
        signs = np.r_[np.ones(12), -np.ones(9)]
        x = np.zeros((21,n)); y = x.copy(); r = x.copy()
        x[np.arange(21),ids] = 1
        mask = ids >= 3
        y[np.arange(21)[mask],12+ids[mask]-3] = signs[mask]
        r[np.arange(21),21+ids] = 1
        pairs = [(i,j) for i in range(12) for j in range(i+1,12)]
        pairs += [(i,12+j-3) for i in range(3,12) for j in range(i,12)]
        pi,pj = np.array(pairs).T
        boundary = np.arange(12)
        ridx = np.arange(21,33)
    else:
        n = 64
        x = np.zeros((21,n)); y = x.copy(); r = x.copy()
        x[np.arange(21),np.arange(21)] = 1
        y[np.arange(21),21+np.arange(21)] = 1
        r[np.arange(21),42+np.arange(21)] = 1
        pi,pj = I,J
        boundary = np.arange(21)
        ridx = np.arange(42,63)
    w = np.zeros(n); w[-1] = 1
    b = np.concatenate((x[boundary]-r[boundary],w-x[boundary]-r[boundary],
                        -.5*w-y[boundary]-r[boundary],-.5*w+y[boundary]-r[boundary]))
    off = np.r_[np.zeros(2*len(boundary)),np.ones(2*len(boundary))]
    return x,y,r,x[pi]-x[pj],y[pi]-y[pj],r[pi]+r[pj],b,off,-r.sum(axis=0),ridx

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    span = .70*budget_s
    deadline = start+span
    eps = float(params['safety'])
    best = None
    bestscore = -np.inf
    bestarray = None
    try:
        a = (np.asarray(init['circles'],float) if 'circles' in init else
             np.asarray(init['state'],float).reshape(3,21).T.copy())
        if valid(a):
            best = init
            bestscore = float(np.sum(a[:,2]))
            bestarray = a.copy()
    except (TypeError, ValueError, KeyError):
        pass

    def offer(a):
        nonlocal best,bestscore,bestarray
        a = repair(a,eps)
        if a is not None:
            score = float(np.sum(a[:,2]))
            if score > bestscore:
                bestscore = score
                bestarray = a.copy()
                best = {'state':a.T.reshape(-1).tolist()}
        return a

    # An unconditional feasible fallback, independent of the input geometry.
    offer(np.array([[.2*(i%3), .2*(i//3), .09] for i in range(21)]))
    if time.monotonic() >= deadline:
        return best
    sym = model(True)
    full = model(False)

    def unpack(z,m):
        return np.column_stack((m[0]@z,m[1]@z,m[2]@z))

    def full_state(a):
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        return np.r_[a[:,0]-lo[0], a[:,1]-(lo[1]+hi[1])/2, a[:,2], hi[0]-lo[0]]

    def lattice(counts):
        singles = []
        pairs = []
        for k,count in enumerate(counts):
            if time.monotonic() >= deadline:
                raise Timeout
            xx = np.sqrt(3.)*k
            if count % 2:
                singles.append([xx,0.,1.])
            for yy in range(1 if count%2 == 0 else 2, count, 2):
                if time.monotonic() >= deadline:
                    raise Timeout
                pairs.append([xx,float(yy),1.])
        a = np.array(singles+pairs+[[p[0],-p[1],p[2]] for p in pairs])
        a = repair(a,eps)
        offer(a)
        z = full_state(a)
        return np.r_[z[:12],z[24:33],z[42:54],z[-1]]

    def optimize(z,m,stop):
        if time.monotonic() >= stop:
            return z
        x,y,r,dxm,dym,rsm,b,off,obj,ridx = m
        weights = 1/np.maximum(2*(rsm@z),.06)
        cache = [None,None,None]
        accepted = [z.copy()]
        last = [z.copy()]
        counter = [0]
        def calc(v):
            if time.monotonic() >= stop:
                raise Timeout
            if cache[0] is not None and np.array_equal(cache[0],v):
                return cache[1],cache[2]
            dx = dxm@v; dy = dym@v; rs = rsm@v
            c = np.r_[(dx*dx+dy*dy-rs*rs)*weights,b@v+off]
            jac = np.concatenate((2*weights[:,None]*(dx[:,None]*dxm+dy[:,None]*dym-rs[:,None]*rsm),b))
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
            accepted[0] = v.copy()
            counter[0] += 1
            if counter[0] % 10 == 0:
                offer(unpack(v,m))
        bounds = [(None,None)]*len(z)
        for k in ridx:
            if time.monotonic() >= stop:
                return z
            bounds[int(k)] = (1e-7,.5)
        bounds[-1] = (.12,1.88)
        try:
            result = minimize(fun,z,jac=True,method='SLSQP',bounds=bounds,
                              constraints={'type':'ineq','fun':lambda v:calc(v)[0],
                                           'jac':lambda v:calc(v)[1]},callback=callback,
                              options={'maxiter':int(params['iterations']),'ftol':8e-14,'disp':False})
            z = result.x
        except Timeout:
            z = accepted[0]
        offer(unpack(z,m))
        offer(unpack(last[0],m))
        return z

    if bestscore > 2.35:
        optimize(full_state(bestarray),full,deadline-.012)
        return best
    try:
        q = lattice((3,4,3,4,3,4))
        optimize(q,sym,start+float(params['first_fraction'])*span)
        if time.monotonic() < start+float(params['second_fraction'])*span-.04:
            q = lattice((4,5,4,5,3))
            optimize(q,sym,start+float(params['second_fraction'])*span)
    except Timeout:
        return best
    if time.monotonic() < deadline-.035:
        optimize(full_state(bestarray),full,deadline-.012)
    return best
