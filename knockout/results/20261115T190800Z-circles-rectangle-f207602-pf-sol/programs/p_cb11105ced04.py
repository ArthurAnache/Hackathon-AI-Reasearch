import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'search_fraction': (0.70, 0.60, 0.75),
    'first_fraction': (0.39, 0.25, 0.55),
    'second_fraction': (0.77, 0.60, 0.90),
    'iterations': (180, 60, 350),
    'safety': (1.5e-14, 5e-15, 1e-11, 'log')
}
I, J = np.triu_indices(21, 1)

class Timeout(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return -np.inf
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    p = (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r)) +
         max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)))
    return float(np.sum(a[:, 2])) if p <= 2 else -np.inf

def model(symmetric):
    n = 34 if symmetric else 64
    x = np.zeros((21,n)); y = x.copy(); r = x.copy()
    if symmetric:
        ids = np.r_[np.arange(12), np.arange(3,12)]
        sg = np.r_[np.ones(12), -np.ones(9)]
        x[np.arange(21),ids] = 1
        k = np.flatnonzero(ids >= 3)
        y[k,12+ids[k]-3] = sg[k]
        r[np.arange(21),21+ids] = 1
        pp = [(i,j) for i in range(12) for j in range(i+1,12)]
        pp += [(i,12+j-3) for i in range(3,12) for j in range(i,12)]
        pi,pj = np.array(pp).T
        wall = np.arange(12)
        ridx = np.arange(21,33)
    else:
        x[np.arange(21),np.arange(21)] = 1
        y[np.arange(21),21+np.arange(21)] = 1
        r[np.arange(21),42+np.arange(21)] = 1
        pi,pj = I,J
        wall = np.arange(21)
        ridx = np.arange(42,63)
    w = np.zeros(n); w[-1] = 1
    b = np.concatenate((x[wall]-r[wall],w-x[wall]-r[wall],
                        -.5*w-y[wall]-r[wall],-.5*w+y[wall]-r[wall]))
    off = np.r_[np.zeros(2*len(wall)),np.ones(2*len(wall))]
    return (x,y,r,x[pi]-x[pj],y[pi]-y[pj],r[pi]+r[pj],b,off,-r.sum(axis=0),ridx)

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + float(params['search_fraction'])*budget_s
    eps = float(params['safety'])
    best = None
    best_a = None
    bestscore = -np.inf
    original = None
    try:
        if init is not None:
            original = (np.asarray(init['circles'],float) if 'circles' in init else
                        np.asarray(init['state'],float).reshape(3,21).T.copy())
            sc = exact_score(original)
            if sc > bestscore:
                best, best_a, bestscore = init, original.copy(), sc
    except (ValueError, TypeError, KeyError):
        original = None

    def offer(a):
        nonlocal best,best_a,bestscore
        a = np.asarray(a,dtype=float).copy()
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        ds = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
        ratio = min(1.,float(np.min(ds/(a[I,2]+a[J,2]))))
        if ratio <= 0:
            return
        a[:,2] *= ratio*(1-eps)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        scale = (2-12*eps)/float(np.sum(hi-lo))
        a[:,:2] = (a[:,:2]-(hi+lo)/2)*scale
        a[:,2] *= scale
        sc = exact_score(a)
        if sc > bestscore:
            bestscore,best_a = sc,a
            best = {'state':a.T.reshape(-1).tolist()}

    # Always retain an explicit feasible fallback, independently of the input.
    fallback = np.array([[2.*(i%7),2.*(i//7),1.] for i in range(21)])
    offer(fallback)
    if original is not None:
        offer(original)

    sym = model(True)
    full = model(False)

    def unpack(z,m):
        return np.column_stack((m[0]@z,m[1]@z,m[2]@z))

    def full_state(a):
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        p = a[:,:2].copy()
        p[:,0] -= lo[0]
        p[:,1] -= .5*(lo[1]+hi[1])
        return np.r_[p[:,0],p[:,1],a[:,2],hi[0]-lo[0]]

    def optimize(z,m,stop):
        if time.monotonic() >= stop:
            return z
        x,y,r,dxm,dym,rsm,b,off,obj,ridx = m
        weights = 1/np.maximum(2*(rsm@z),.08)
        cache = [None,None,None]
        accepted = [z.copy()]
        last = [z.copy()]
        count = [0]
        def calc(v):
            if time.monotonic() >= stop:
                raise Timeout
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
                raise Timeout
            return float(obj@v),obj
        def callback(v):
            if time.monotonic() >= stop:
                raise Timeout
            accepted[0] = v.copy()
            count[0] += 1
            if count[0] % 8 == 0:
                offer(unpack(v,m))
        bounds = [(None,None)]*len(z)
        for k in ridx:
            if time.monotonic() >= stop:
                return z
            bounds[int(k)] = (1e-7,.5)
        bounds[-1] = (.15,1.85)
        try:
            out = minimize(fun,z,jac=True,method='SLSQP',bounds=bounds,
                           constraints={'type':'ineq','fun':lambda v:calc(v)[0],
                                        'jac':lambda v:calc(v)[1]},
                           callback=callback,
                           options={'maxiter':int(params['iterations']),'ftol':5e-14,'disp':False})
            z = out.x
        except Timeout:
            z = accepted[0]
        offer(unpack(z,m))
        offer(unpack(last[0],m))
        return z

    def lattice(kind):
        # Three axis circles, with four and five circles on each side.
        if kind == 0:
            px = np.r_[[0.,4.,8.],[1.,3.,5.,7.],[0.,2.,4.,6.,8.]]
            py = np.r_[np.full(4,np.sqrt(3.)),np.full(5,2*np.sqrt(3.))]
        else:
            px = np.r_[[2.,4.,6.],[0.,2.,4.,6.,8.],[1.,3.,5.,7.]]
            py = np.r_[np.full(5,2.),np.full(4,2.+np.sqrt(3.))]
        scale = 2/(10+2*(float(np.max(py))+1))
        return np.r_[(px+1)*scale,py*scale,np.full(12,scale),10*scale]

    cold = original is None or not np.isfinite(original).all() or float(np.sum(original[:,2])) < 1.5
    if cold:
        duration = deadline-start
        q0 = lattice(0)
        offer(unpack(q0,sym))
        optimize(q0,sym,start+float(params['first_fraction'])*duration)
        if time.monotonic() < deadline-.05:
            q1 = lattice(1)
            offer(unpack(q1,sym))
            optimize(q1,sym,start+float(params['second_fraction'])*duration)
        if time.monotonic() < deadline-.035:
            optimize(full_state(best_a),full,deadline-.015)
    else:
        optimize(full_state(best_a),full,deadline-.015)
    return best
