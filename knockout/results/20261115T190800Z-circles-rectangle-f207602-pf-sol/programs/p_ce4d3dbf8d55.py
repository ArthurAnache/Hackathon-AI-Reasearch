import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'first_fraction': (0.34, 0.2, 0.5),
    'second_fraction': (0.32, 0.2, 0.5),
    'safety': (2e-14, 5e-15, 1e-10, 'log'),
    'iterations': (250, 80, 500)
}
I, J = np.triu_indices(21, 1)

class Timeout(Exception):
    pass

def valid(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return False
    d = np.sqrt((a[I,0]-a[J,0])**2 + (a[I,1]-a[J,1])**2)
    if np.any(d < a[I,2]+a[J,2]):
        return False
    x,y,r = [[Fraction(float(v)) for v in a[:,k]] for k in range(3)]
    w = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
    h = max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
    return w+h <= 2

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70*budget_s
    eps = params['safety']
    best = None
    bestscore = -np.inf
    besta = None
    original = None
    try:
        if init is not None:
            original = np.array(init['circles'],dtype=float) if 'circles' in init else np.array(init['state'],dtype=float).reshape(3,21).T
            if valid(original):
                best = init
                bestscore = float(np.sum(original[:,2]))
                besta = original.copy()
    except (KeyError, ValueError, TypeError):
        original = None

    def offer(a):
        nonlocal best,bestscore,besta
        if time.monotonic() >= end:
            return
        a = np.array(a,dtype=float,copy=True)
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        d = np.sqrt((a[I,0]-a[J,0])**2 + (a[I,1]-a[J,1])**2)
        q = min(1.,float(np.min(d/(a[I,2]+a[J,2]))))
        if q <= 0:
            return
        a[:,2] *= q*(1-eps)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        scale = (2-8*eps)/float(np.sum(hi-lo))
        a[:,:2] = (a[:,:2]-(hi+lo)/2)*scale
        a[:,2] *= scale
        score = float(np.sum(a[:,2]))
        if score > bestscore and valid(a):
            besta = a.copy()
            bestscore = score
            best = {'state': a.T.ravel().tolist()}

    def state(a):
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        return np.r_[a[:,0]-lo[0],a[:,1]-(lo[1]+hi[1])/2,a[:,2],hi[0]-lo[0]]

    eye = np.eye(64)
    X,Y,R = eye[:21],eye[21:42],eye[42:63]
    W = eye[-1]
    wall = np.concatenate((X-R,W-X-R,-.5*W-Y-R,-.5*W+Y-R))
    offset = np.r_[np.zeros(42),np.ones(42)]
    DX,DY,RS = X[I]-X[J],Y[I]-Y[J],R[I]+R[J]
    objective = -R.sum(axis=0)
    T = np.zeros((64,34))
    ids = np.r_[np.arange(12),np.arange(3,12)]
    signs = np.r_[np.ones(12),-np.ones(9)]
    T[np.arange(21),ids] = 1
    mask = ids>=3
    T[21+np.arange(21)[mask],12+ids[mask]-3] = signs[mask]
    T[42+np.arange(21),21+ids] = 1
    T[63,33] = 1

    def lattice(inner,outer):
        singles = np.array([[-2.,0.,1.],[0.,0.,1.],[2.,0.,1.]])
        xx1 = 2*np.arange(inner)-(inner-1)
        xx2 = 2*np.arange(outer)-(outer-1)
        upper = np.array([[float(x),np.sqrt(3.),1.] for x in xx1]+[[float(x),2*np.sqrt(3.),1.] for x in xx2])
        lower = upper.copy()
        lower[:,1] *= -1
        a = np.vstack((singles,upper,lower))
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        s = 2/np.sum(hi-lo)
        a[:,:2] = (a[:,:2]-(lo+hi)/2)*s
        a[:,2] *= s
        return a

    def solve(a, transform, stop):
        if time.monotonic() >= stop-.005:
            return
        full = state(a)
        if transform.shape[1] == 34:
            z = np.r_[full[:12],full[24:33],full[42:54],full[63]]
        else:
            z = full
        dxm,dym,rsm = DX@transform,DY@transform,RS@transform
        walls = wall@transform
        obj = objective@transform
        weights = 1/np.maximum(2*(rsm@z),.06)
        cache = [None,None,None]
        last = [z.copy()]
        count = [0]
        def calc(v):
            if time.monotonic() >= stop:
                raise Timeout
            if cache[0] is not None and np.array_equal(cache[0],v):
                return cache[1],cache[2]
            dx,dy,rs = dxm@v,dym@v,rsm@v
            c = np.r_[(dx*dx+dy*dy-rs*rs)*weights,walls@v+offset]
            jac = np.concatenate((2*weights[:,None]*(dx[:,None]*dxm+dy[:,None]*dym-rs[:,None]*rsm),walls))
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
            if count[0]%8 == 0:
                offer((transform@v)[:63].reshape(3,21).T)
        n = transform.shape[1]
        bounds = [(None,None)]*n
        if n == 34:
            bounds[21:33] = [(1e-6,.5)]*12
        else:
            bounds[42:63] = [(1e-6,.5)]*21
        bounds[-1] = (.1,1.9)
        try:
            result = minimize(fun,z,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda v:calc(v)[0],'jac':lambda v:calc(v)[1]},
                callback=callback,options={'maxiter':params['iterations'],'ftol':1e-13})
            z = result.x
        except Timeout:
            z = last[0]
        offer((transform@z)[:63].reshape(3,21).T)

    a1 = lattice(5,4)
    a2 = lattice(4,5)
    offer(a1)
    offer(a2)
    if original is not None:
        offer(original)
    if best is None:
        a = np.array([[.18*(i%3),.18*(i//3),.08] for i in range(21)])
        best = {'circles':a.tolist()}
        besta = a
        bestscore = float(np.sum(a[:,2]))
    window = .70*budget_s
    if bestscore > 2.3 and original is not None:
        solve(besta,eye,end-.012)
        return best
    solve(a1,T,start+window*params['first_fraction'])
    if time.monotonic() < end-.025:
        solve(a2,T,start+window*(params['first_fraction']+params['second_fraction']))
    if time.monotonic() < end-.025:
        solve(besta,eye,end-.012)
    return best
