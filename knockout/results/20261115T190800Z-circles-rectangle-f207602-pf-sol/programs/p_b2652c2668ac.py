import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import lstsq
from scipy.optimize import linear_sum_assignment

PARAMS = {
    'first_fraction': (0.36, 0.20, 0.48),
    'second_fraction': (0.14, 0.06, 0.24),
    'iterations': (180, 70, 350),
    'safety': (1.2e-14, 3e-15, 1e-10, 'log')
}
I, J = np.triu_indices(21, 1)

class Stop(Exception):
    pass

def valid(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return False
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return False
    x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
    return (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r)) +
            max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))) <= 2

def construct(params, seed, budget_s, init=None):
    begin = time.monotonic()
    deadline = begin + 0.70*budget_s
    eps = params['safety']
    best = None
    best_a = None
    best_score = -float('inf')

    def offer(a, repair=True):
        nonlocal best, best_a, best_score
        a = np.array(a, dtype=float, copy=True)
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        if repair:
            d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
            q = min(1.0, float(np.min(d/(a[I,2]+a[J,2]))))
            if q <= 0:
                return
            a[:,2] *= q*(1-eps)
            lo = np.min(a[:,:2]-a[:,2,None], axis=0)
            hi = np.max(a[:,:2]+a[:,2,None], axis=0)
            scale = (2-12*eps)/float(np.sum(hi-lo))
            a[:,:2] = (a[:,:2]-(lo+hi)/2)*scale
            a[:,2] *= scale
        score = float(np.sum(a[:,2]))
        if score > best_score and valid(a):
            best_a = a.copy()
            best_score = score
            best = {'state': a.T.ravel().tolist()}

    if init is not None:
        try:
            a = np.array(init['circles'],float) if 'circles' in init else np.array(init['state'],float).reshape(3,21).T
            if valid(a):
                best, best_a, best_score = init, a.copy(), float(np.sum(a[:,2]))
            offer(a)
        except (ValueError, KeyError, TypeError):
            pass

    def lattice(counts):
        points = []
        for column, count in enumerate(counts):
            if time.monotonic() >= deadline:
                return None
            for k in range(count):
                if time.monotonic() >= deadline:
                    return None
                points.append([column*np.sqrt(3.0), 2*k-count+1, 1.0])
        return np.array(points)

    seeds = []
    for counts in ((3,4,3,4,3,4), (3,4,5,4,5)):
        if time.monotonic() >= deadline:
            break
        a = lattice(counts)
        if a is not None:
            offer(a)
            seeds.append(a)
    if best_a is None:
        a = np.array([[0.2*(i%3),0.2*(i//3),0.08] for i in range(21)])
        offer(a)
    if time.monotonic() >= deadline-.03:
        return best

    def quotient(a):
        choices = []
        for swap in (False, True):
            if time.monotonic() >= deadline:
                break
            b = a.copy()
            if swap:
                b[:,:2] = b[:,[1,0]]
            lo = np.min(b[:,:2]-b[:,2,None],axis=0)
            hi = np.max(b[:,:2]+b[:,2,None],axis=0)
            scale = 2/float(np.sum(hi-lo))
            b[:,:2] = (b[:,:2]-np.array([lo[0],(lo[1]+hi[1])/2]))*scale
            b[:,2] *= scale
            singles = np.argsort(abs(b[:,1]))[:3]
            rem = np.setdiff1d(np.arange(21),singles)
            order = rem[np.argsort(b[rem,1])]
            neg,pos = order[:9],order[9:]
            cost = ((b[pos,None,0]-b[None,neg,0])**2 +
                    (b[pos,None,1]+b[None,neg,1])**2 +
                    (b[pos,None,2]-b[None,neg,2])**2)
            rr,cc = linear_sum_assignment(cost)
            neg = neg[cc]
            error = float(cost[rr,cc].sum()+np.sum(b[singles,1]**2))
            z = np.r_[b[singles,0], (b[pos,0]+b[neg,0])/2,
                      (b[pos,1]-b[neg,1])/2, b[singles,2],
                      (b[pos,2]+b[neg,2])/2, (hi[0]-lo[0])*scale]
            choices.append((error,z))
        return min(choices,key=lambda t:t[0])[1] if choices else None

    def model(sym):
        if sym:
            T = np.zeros((64,34))
            ids = np.r_[np.arange(12),np.arange(3,12)]
            signs = np.r_[np.zeros(3),np.ones(9),-np.ones(9)]
            T[np.arange(21),ids] = 1
            mask = ids >= 3
            T[21+np.arange(21)[mask],12+ids[mask]-3] = signs[mask]
            T[42+np.arange(21),21+ids] = 1
            T[63,33] = 1
            pairs = [(i,j) for i in range(12) for j in range(i+1,12)]
            pairs += [(i,12+j-3) for i in range(3,12) for j in range(i,12)]
            pi,pj = np.array(pairs).T
            wallids = np.arange(12)
            bounds = [(None,None)]*34
            bounds[12:21] = [(0,None)]*9
            bounds[21:33] = [(1e-7,.8)]*12
        else:
            T = np.eye(64)
            pi,pj = I,J
            wallids = np.arange(21)
            bounds = [(None,None)]*64
            bounds[42:63] = [(1e-7,.8)]*21
        bounds[-1] = (.15,1.85)
        X,Y,R,W = T[:21],T[21:42],T[42:63],T[-1]
        dx,dy,rs = X[pi]-X[pj],Y[pi]-Y[pj],R[pi]+R[pj]
        xx,yy,rr = X[wallids],Y[wallids],R[wallids]
        walls = np.concatenate((xx-rr,W-xx-rr,-.5*W-yy-rr,-.5*W+yy-rr))
        off = np.r_[np.zeros(2*len(wallids)),np.ones(2*len(wallids))]
        obj = -R.sum(axis=0)
        def raw(z):
            u,v,r = dx@z,dy@z,rs@z
            c = np.r_[u*u+v*v-r*r,walls@z+off]
            jac = np.concatenate((2*(u[:,None]*dx+v[:,None]*dy-r[:,None]*rs),walls))
            return c,jac
        return T,bounds,obj,raw

    def solve(z, mod, stop):
        if z is None or time.monotonic() >= stop-.005:
            return
        T,bounds,obj,raw = mod
        last = [z.copy()]
        cache = [None,None,None]
        ticks = [0]
        def calc(v):
            if time.monotonic() >= stop:
                raise Stop
            if cache[0] is None or not np.array_equal(cache[0],v):
                c,j = raw(v)
                cache[:] = [v.copy(),c,j]
            return cache[1],cache[2]
        def fun(v):
            if time.monotonic() >= stop:
                raise Stop
            return float(obj@v),obj
        def callback(v):
            if time.monotonic() >= stop:
                raise Stop
            last[0] = v.copy()
            ticks[0] += 1
            if ticks[0] % 8 == 0:
                offer((T@v)[:63].reshape(3,21).T)
        try:
            res = minimize(fun,z,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda v:calc(v)[0], 'jac':lambda v:calc(v)[1]},
                callback=callback,
                options={'maxiter':int(params['iterations']),'ftol':2e-13,'disp':False})
            z = res.x
        except Stop:
            z = last[0]
        offer((T@z)[:63].reshape(3,21).T)
        for k in range(3):
            if time.monotonic() >= stop-.004:
                break
            c,j = raw(z)
            active = np.flatnonzero(abs(c)<2e-7)
            if len(active)<len(z)-1:
                break
            delta,_,rank,_ = lstsq(j[active],-c[active],lapack_driver='gelsy',check_finite=False)
            if rank<len(z)-1 or np.max(abs(delta))>2e-4:
                break
            z += delta
            offer((T@z)[:63].reshape(3,21).T)

    sym = model(True)
    solve(quotient(best_a),sym,min(deadline-.10,begin+params['first_fraction']*budget_s))
    if len(seeds)>1 and time.monotonic() < deadline-.20:
        solve(quotient(seeds[1]),sym,min(deadline-.16,time.monotonic()+params['second_fraction']*budget_s))
    if time.monotonic() < deadline-.08:
        a = best_a.copy()
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        z = np.r_[a[:,0]-lo[0],a[:,1]-(lo[1]+hi[1])/2,a[:,2],hi[0]-lo[0]]
        solve(z,model(False),deadline-.008)
    return best
