import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'tolerance': (2e-11, 1e-13, 1e-7, 'log'),
    'safety': (2e-14, 5e-15, 1e-10, 'log'),
    'iterations': (220, 80, 500),
    'first_share': (0.34, 0.2, 0.5)
}
I, J = np.triu_indices(21, 1)

class StopSearch(Exception):
    pass

def valid(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return False
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return False
    x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
    p = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
    p += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
    return p <= 2

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70*budget_s
    eps = float(params['safety'])
    best = None
    best_a = None
    score = -np.inf
    def offer(a, repair=True):
        nonlocal best, best_a, score
        a = np.asarray(a, dtype=float).copy()
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        if repair:
            d = np.sqrt(np.sum((a[I,:2]-a[J,:2])**2,axis=1))
            q = min(1., float(np.min(d/(a[I,2]+a[J,2]))))
            if q <= 0:
                return
            a[:,2] *= q*(1-eps)
            lo = np.min(a[:,:2]-a[:,2,None],axis=0)
            hi = np.max(a[:,:2]+a[:,2,None],axis=0)
            scale = (2-16*eps)/float(np.sum(hi-lo))
            a[:,:2] = (a[:,:2]-(hi+lo)/2)*scale
            a[:,2] *= scale
        s = float(np.sum(a[:,2]))
        if s > score and valid(a):
            score, best_a = s, a
            best = {'state': a.T.reshape(-1).tolist()}
    try:
        if init is not None:
            a = np.asarray(init['circles'],float) if 'circles' in init else np.asarray(init['state'],float).reshape(3,21).T
            if valid(a):
                score, best, best_a = float(np.sum(a[:,2])), init, a.copy()
            offer(a)
    except (KeyError, TypeError, ValueError):
        pass
    offer(np.array([[2*(i%3),2*(i//3),0.999999999] for i in range(21)],float))

    walls = np.zeros((84,64))
    k = np.arange(21)
    walls[k,k] = 1
    walls[21+k,k] = -1
    walls[42+k,21+k] = 1
    walls[63+k,21+k] = -1
    walls[np.arange(84),42+np.tile(k,4)] = -1
    walls[:42,63] = .5
    walls[42:,63] = -.5
    offset = np.r_[np.zeros(42),np.ones(42)]
    objective = np.r_[np.zeros(42),-np.ones(21),0.]
    T = np.zeros((64,34))
    ids = np.r_[np.arange(12),np.arange(3,12)]
    T[np.arange(21),ids] = 1
    T[21+np.arange(3,12),np.arange(12,21)] = 1
    T[21+np.arange(12,21),np.arange(12,21)] = -1
    T[42+np.arange(21),21+ids] = 1
    T[63,33] = 1
    mirror = np.r_[np.arange(3),np.arange(12,21),np.arange(3,12)]
    selected = []
    for q in range(len(I)):
        if time.monotonic() >= end:
            return best
        p = (int(I[q]),int(J[q]))
        other = tuple(sorted((int(mirror[I[q]]),int(mirror[J[q]]))))
        if p <= other:
            selected.append(q)
    selected = np.asarray(selected)

    def state(a):
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        b = a.copy()
        b[:,:2] -= (hi+lo)/2
        return np.r_[b.T.reshape(-1),hi[0]-lo[0]]

    def run(z, symmetric, deadline):
        if time.monotonic() >= deadline-.005:
            return
        transform = T if symmetric else np.eye(64)
        sel = selected if symmetric else np.arange(len(I))
        ii,jj = I[sel],J[sel]
        dxmat = transform[ii]-transform[jj]
        dymat = transform[21+ii]-transform[21+jj]
        rmat = transform[42+ii]+transform[42+jj]
        wsel = np.r_[np.arange(12),21+np.arange(12),42+np.arange(12),63+np.arange(12)] if symmetric else np.arange(84)
        linear = walls[wsel]@transform
        off = offset[wsel]
        obj = objective@transform
        weights = 1/np.maximum(2*(rmat@z),.12)
        cache = [None,None,None]
        last = [z.copy()]
        count = [0]
        def calc(v):
            if time.monotonic() >= deadline:
                raise StopSearch()
            if cache[0] is not None and np.array_equal(cache[0],v):
                return cache[1],cache[2]
            dx,dy,rr = dxmat@v,dymat@v,rmat@v
            values = np.r_[(dx*dx+dy*dy-rr*rr)*weights,linear@v+off]
            jac = np.vstack(((2*weights*dx)[:,None]*dxmat+(2*weights*dy)[:,None]*dymat-(2*weights*rr)[:,None]*rmat,linear))
            cache[:] = [v.copy(),values,jac]
            last[0] = v.copy()
            return values,jac
        def fun(v):
            if time.monotonic() >= deadline:
                raise StopSearch()
            return float(obj@v),obj
        def callback(v):
            if time.monotonic() >= deadline:
                raise StopSearch()
            count[0] += 1
            if count[0]%15 == 0:
                offer((transform@v)[:63].reshape(3,21).T)
        bounds = [(None,None)]*len(z)
        for j in (range(21,33) if symmetric else range(42,63)):
            if time.monotonic() >= deadline:
                return
            bounds[j] = (1e-6,.6)
        bounds[-1] = (.12,1.88)
        try:
            result = minimize(fun,z,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda v:calc(v)[0],'jac':lambda v:calc(v)[1]},
                callback=callback,options={'ftol':float(params['tolerance']),'maxiter':int(params['iterations']),'disp':False})
            last[0] = result.x
        except StopSearch:
            pass
        offer((transform@last[0])[:63].reshape(3,21).T)

    if score > 2.30:
        run(state(best_a),False,end-.012)
        return best

    def layout(rows):
        a = np.zeros((21,3))
        a[:3,0] = [-2.,0.,2.]
        a[:,2] = 1.
        cursor = 3
        for row,(n,height,shift) in enumerate(rows):
            if time.monotonic() >= end:
                return None
            a[cursor:cursor+n,0] = 2*np.arange(n)-(n-1)+shift
            a[cursor:cursor+n,1] = height
            cursor += n
        a[12:21] = a[3:12]
        a[12:21,1] *= -1
        lo = np.min(a[:,:2]-1,axis=0)
        hi = np.max(a[:,:2]+1,axis=0)
        a *= 2/np.sum(hi-lo)
        u = state(a)
        return np.r_[u[:12],u[24:33],u[42:54],u[63]]

    layouts = [
        [(5,2.,0.),(4,2.+np.sqrt(3),0.)],
        [(3,np.sqrt(3),1.),(3,2*np.sqrt(3),0.),(3,3*np.sqrt(3),1.)],
        [(4,np.sqrt(3),0.),(5,2*np.sqrt(3),0.)]
    ]
    window = end-start
    cutoffs = [start+float(params['first_share'])*window,start+.63*window,start+.82*window]
    for rows,cutoff in zip(layouts,cutoffs):
        if time.monotonic() >= end-.04:
            break
        z = layout(rows)
        if z is not None:
            run(z,True,min(cutoff,end-.025))
    if time.monotonic() < end-.035:
        run(state(best_a),False,end-.012)
    return best
