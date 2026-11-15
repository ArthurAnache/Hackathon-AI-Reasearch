import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'iterations': (200, 80, 400),
    'tolerance': (8e-12, 1e-13, 1e-8, 'log'),
    'safety': (3e-14, 1e-14, 1e-10, 'log'),
    'polish_share': (0.20, 0.10, 0.35)
}
I, J = np.triu_indices(21, 1)

class SearchEnd(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    end = started + .70 * budget_s
    eps = float(params['safety'])
    best = None
    best_a = None
    best_score = -np.inf

    def exact_score(a):
        if a is None or a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -np.inf
        d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
        if np.any(d < a[I, 2] + a[J, 2]):
            return -np.inf
        x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
        perimeter = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        perimeter += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        return float(np.sum(a[:, 2])) if perimeter <= 2 else -np.inf

    def repair(a):
        a = np.asarray(a, dtype=float).copy()
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return None
        lo = np.min(a[:,:2]-a[:,2,None], axis=0)
        hi = np.max(a[:,:2]+a[:,2,None], axis=0)
        a[:,:2] -= (lo+hi)/2
        d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
        q = min(1., float(np.min(d/(a[I,2]+a[J,2]))))
        if q <= 0:
            return None
        a[:,2] *= q*(1-eps)
        lo = np.min(a[:,:2]-a[:,2,None], axis=0)
        hi = np.max(a[:,:2]+a[:,2,None], axis=0)
        a[:,:2] -= (lo+hi)/2
        a *= (2-16*eps)/float(np.sum(hi-lo))
        return a

    def offer(a):
        nonlocal best, best_a, best_score
        a = repair(a)
        if a is None or float(np.sum(a[:,2])) <= best_score:
            return
        s = exact_score(a)
        if s > best_score:
            best_score, best_a = s, a.copy()
            best = {'state': a.T.ravel().tolist()}

    try:
        if init is not None:
            a = np.asarray(init['circles'],float) if 'circles' in init else np.asarray(init['state'],float).reshape(3,21).T
            s = exact_score(a)
            if s > best_score:
                best_score, best_a, best = s, a.copy(), init
            offer(a)
    except (ValueError, TypeError, KeyError):
        pass
    offer(np.array([[2*(k%3),2*(k//3),1.] for k in range(21)],float))

    # Quotient coordinates: twelve y's, nine positive x's, twelve radii, width.
    T = np.zeros((64,34))
    ids = np.r_[np.arange(12),np.arange(3,12)]
    T[21+np.arange(21),ids] = 1
    T[3+np.arange(9),12+np.arange(9)] = 1
    T[12+np.arange(9),12+np.arange(9)] = -1
    T[42+np.arange(21),21+ids] = 1
    T[63,33] = 1
    W = np.zeros((84,64))
    k = np.arange(21)
    W[k,k] = -1
    W[21+k,k] = 1
    W[42+k,21+k] = -1
    W[63+k,21+k] = 1
    W[np.arange(84),42+np.tile(k,4)] = -1
    W[:42,63] = .5
    W[42:,63] = -.5
    offset = np.r_[np.zeros(42),np.ones(42)]
    gradient = np.r_[np.zeros(42),-np.ones(21),0.]
    partner = np.r_[np.arange(3),np.arange(12,21),np.arange(3,12)]
    keep = []
    for q in range(len(I)):
        if time.monotonic() >= end:
            return best
        reflected = tuple(sorted((int(partner[I[q]]),int(partner[J[q]]))))
        if (int(I[q]),int(J[q])) <= reflected:
            keep.append(q)
    keep = np.asarray(keep,dtype=int)

    def state(a):
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        b = a.copy()
        b[:,:2] -= (lo+hi)/2
        return np.r_[b.T.ravel(),hi[0]-lo[0]]

    def solve(a, symmetric, until):
        if time.monotonic() >= until-.008:
            return
        v = state(a)
        if symmetric:
            z = np.r_[v[21:33],v[3:12],v[42:54],v[63]]
            transform = T
            pairs = keep
            wall_ids = np.r_[np.arange(12),42+np.arange(12),63+np.arange(12)]
        else:
            z = v
            transform = np.eye(64)
            pairs = np.arange(len(I))
            wall_ids = np.arange(84)
        ii,jj = I[pairs],J[pairs]
        X = transform[ii]-transform[jj]
        Y = transform[21+ii]-transform[21+jj]
        R = transform[42+ii]+transform[42+jj]
        L = W[wall_ids]@transform
        off = offset[wall_ids]
        obj = gradient@transform
        last = [z.copy()]
        cache = [None,None,None]
        ticks = [0]
        def check():
            if time.monotonic() >= until:
                raise SearchEnd()
        def calc(u):
            check()
            if cache[0] is not None and np.array_equal(cache[0],u):
                return cache[1],cache[2]
            dx,dy = X@u,Y@u
            dist = np.maximum(np.sqrt(dx*dx+dy*dy),1e-16)
            c = np.r_[dist-R@u,L@u+off]
            jac = np.vstack(((dx/dist)[:,None]*X+(dy/dist)[:,None]*Y-R,L))
            cache[:] = [u.copy(),c,jac]
            return c,jac
        def fun(u):
            check()
            return float(obj@u),obj
        def callback(u):
            last[0] = u.copy()
            ticks[0] += 1
            if ticks[0] % 12 == 0:
                offer((transform@u)[:63].reshape(3,21).T)
            check()
        bounds = [(None,None)]*len(z)
        if symmetric:
            for p in range(12,21):
                if time.monotonic() >= until:
                    return
                bounds[p] = (1e-7,1.)
            radial = range(21,33)
        else:
            radial = range(42,63)
        for p in radial:
            if time.monotonic() >= until:
                return
            bounds[p] = (1e-7,.7)
        bounds[-1] = (.08,1.92)
        try:
            result = minimize(fun,z,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda u:calc(u)[0],'jac':lambda u:calc(u)[1]},
                callback=callback,options={'maxiter':int(params['iterations']),'ftol':float(params['tolerance']),'disp':False})
            last[0] = result.x
        except (SearchEnd, ValueError, FloatingPointError):
            pass
        offer((transform@last[0])[:63].reshape(3,21).T)

    def lattice(counts):
        singles,pairs = [],[]
        for row,n in enumerate(counts):
            if time.monotonic() >= end:
                return None
            y = (row-(len(counts)-1)/2)*np.sqrt(3.)
            if n%2:
                singles.append([0.,y,1.])
            for j in range(n//2):
                if time.monotonic() >= end:
                    return None
                pairs.append([float(2*j+(1 if n%2 == 0 else 2)),y,1.])
        return repair(np.array(singles+pairs+[[-x,y,r] for x,y,r in pairs]))

    def wide():
        a = np.zeros((21,3))
        a[:,2] = 1
        a[:3,1] = [-2,0,2]
        a[3:8,0] = 2
        a[3:8,1] = 2*np.arange(5)-4
        a[8:12,0] = 2+np.sqrt(3.)
        a[8:12,1] = 2*np.arange(4)-3
        a[12:] = a[3:12]
        a[12:,0] *= -1
        return repair(a)

    if best_score > 2.34:
        solve(best_a,False,end-.008)
        return best
    seeds = [lattice([2,3,4,3,4,3,2]),wide(),lattice([3,4,3,4,3,4])]
    search_end = started+(1-float(params['polish_share']))*(end-started)
    for q,a in enumerate(seeds):
        if time.monotonic() >= search_end-.012:
            break
        if a is None:
            continue
        offer(a)
        remaining = len(seeds)-q
        until = time.monotonic()+(search_end-time.monotonic())/remaining
        solve(a,True,until)
    if time.monotonic() < end-.018:
        solve(best_a,False,end-.008)
    return best
