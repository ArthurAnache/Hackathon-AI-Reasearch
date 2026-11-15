import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import lstsq

PARAMS = {
    'iterations': (220, 80, 450),
    'first_share': (0.53, 0.35, 0.7),
    'search_share': (0.79, 0.65, 0.9),
    'repair': (1.0e-15, 2.0e-16, 1.0e-12, 'log')
}
I, J = np.triu_indices(21, 1)

class Halt(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    dx = a[I, 0] - a[J, 0]
    dy = a[I, 1] - a[J, 1]
    if np.any(np.sqrt(dx*dx + dy*dy) < a[I, 2] + a[J, 2]):
        return -np.inf
    x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
    p = max(u+v for u,v in zip(x,r)) - min(u-v for u,v in zip(x,r))
    p += max(u+v for u,v in zip(y,r)) - min(u-v for u,v in zip(y,r))
    return float(np.sum(a[:, 2])) if p <= 2 else -np.inf

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    best = None
    best_a = None
    best_score = -np.inf
    try:
        a = (np.asarray(init['circles'], dtype=float) if 'circles' in init else
             np.asarray(init['state'], dtype=float).reshape(3,21).T)
        s = exact_score(a)
        if s > best_score:
            best, best_a, best_score = init, a.copy(), s
    except (TypeError, KeyError, ValueError):
        pass

    def offer(a):
        nonlocal best, best_a, best_score
        a = np.array(a, dtype=float, copy=True)
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        lo = np.min(a[:,:2]-a[:,2,None], axis=0)
        hi = np.max(a[:,:2]+a[:,2,None], axis=0)
        a[:,:2] -= (lo+hi)/2
        dx, dy = a[I,0]-a[J,0], a[I,1]-a[J,1]
        factor = min(1., float(np.min(np.sqrt(dx*dx+dy*dy)/(a[I,2]+a[J,2]))))
        if factor <= 0:
            return
        a[:,2] *= factor
        for slack in (float(params['repair']), 4e-15, 2e-14):
            if time.monotonic() >= deadline:
                break
            c = a.copy()
            c[:,2] *= 1-slack
            lo = np.min(c[:,:2]-c[:,2,None], axis=0)
            hi = np.max(c[:,:2]+c[:,2,None], axis=0)
            c[:,:2] -= (lo+hi)/2
            c *= (2/float(np.sum(hi-lo)))*(1-slack)
            value = float(np.sum(c[:,2]))
            if value <= best_score:
                break
            if exact_score(c) > best_score:
                best_a, best_score = c, value
                best = {'state': c.T.ravel().tolist()}
                break

    def seed_rows(pattern):
        singles, upper = [], []
        for row, count in enumerate(pattern):
            if time.monotonic() >= deadline:
                return None
            x = (row-(len(pattern)-1)/2)*np.sqrt(3.)
            if count % 2:
                singles.append([x,0.,1.])
            for k in range(count//2):
                if time.monotonic() >= deadline:
                    return None
                upper.append([x,float(2*k+(2 if count%2 else 1)),1.])
        if len(singles) != 3 or len(upper) != 9:
            return None
        a = np.array(singles+upper+[[x,-y,r] for x,y,r in upper])
        a *= 2/np.sum(np.ptp(a[:,:2],axis=0)+2)
        return a

    axial = np.column_stack(([-3.,-1.,1.],np.zeros(3),np.ones(3)))
    upper = np.array([[x,np.sqrt(3.),1.] for x in (-4.,-2.,0.,2.,4.)] +
                     [[x,2*np.sqrt(3.),1.] for x in (-3.,-1.,1.,3.)])
    lower = upper.copy(); lower[:,1] *= -1
    first = np.concatenate((axial,upper,lower))
    first *= 2/np.sum(np.ptp(first[:,:2],axis=0)+2)
    offer(first)

    def model(sym):
        n = 34 if sym else 64
        if sym:
            ids = np.r_[np.arange(12),np.arange(3,12)]
            xi = ids
            yi = np.where(ids >= 3,9+ids,0)
            ys = np.r_[np.zeros(3),np.ones(9),-np.ones(9)]
            ri = 21+ids
            reflection = np.r_[np.arange(3),np.arange(12,21),np.arange(3,12)]
            mi = np.minimum(reflection[I],reflection[J])
            mj = np.maximum(reflection[I],reflection[J])
            keep = (I < mi) | ((I == mi) & (J <= mj))
            pi,pj = I[keep],J[keep]
            wall = np.arange(12)
        else:
            xi = np.arange(21); yi = 21+xi; ri = 42+xi
            ys = np.ones(21)
            pi,pj = I,J
            wall = np.arange(21)
        X = np.zeros((21,n)); Y = np.zeros_like(X); R = np.zeros_like(X)
        X[np.arange(21),xi] = 1
        Y[np.arange(21),yi] = ys
        R[np.arange(21),ri] = 1
        W = np.zeros(n); W[-1] = .5
        walls = np.vstack((W-X[wall]-R[wall],W+X[wall]-R[wall],
                           -W-Y[wall]-R[wall],-W+Y[wall]-R[wall]))
        offset = np.r_[np.zeros(2*len(wall)),np.ones(2*len(wall))]
        DX,DY,RR = X[pi]-X[pj],Y[pi]-Y[pj],R[pi]+R[pj]
        obj = -np.sum(R,axis=0)
        bounds = [(None,None)]*n
        for k in np.unique(ri):
            if time.monotonic() >= deadline:
                break
            bounds[int(k)] = (1e-9,.65)
        bounds[-1] = (.12,1.88)
        return X,Y,R,DX,DY,RR,walls,offset,obj,bounds

    def optimize(a, sym, stop):
        if time.monotonic() >= stop-.005:
            return
        X,Y,R,DX,DY,RR,walls,offset,obj,bounds = model(sym)
        a = a.copy()
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        a[:,:2] -= (lo+hi)/2
        if sym:
            z = np.r_[a[:12,0],a[3:12,1],a[:12,2],hi[0]-lo[0]]
        else:
            z = np.r_[a.T.ravel(),hi[0]-lo[0]]
        cache = [None,None,None]
        last = [z.copy()]
        count = [0]
        def unpack(v):
            return np.column_stack((X@v,Y@v,R@v))
        def calc(v):
            if time.monotonic() >= stop:
                raise Halt()
            if cache[0] is None or not np.array_equal(v,cache[0]):
                dx,dy = DX@v,DY@v
                d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-20)
                c = np.r_[d-RR@v,walls@v+offset]
                jac = np.vstack(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RR,walls))
                cache[:] = [v.copy(),c,jac]
            return cache[1],cache[2]
        def fun(v):
            if time.monotonic() >= stop:
                raise Halt()
            return float(obj@v),obj
        def callback(v):
            if time.monotonic() >= stop:
                raise Halt()
            last[0] = v.copy()
            count[0] += 1
            if count[0] % 20 == 0:
                offer(unpack(v))
        try:
            result = minimize(fun,z,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda v:calc(v)[0],
                             'jac':lambda v:calc(v)[1]},callback=callback,
                options={'maxiter':int(params['iterations']),'ftol':1e-13,'disp':False})
            z = result.x
        except (Halt,ValueError,FloatingPointError):
            z = last[0]
        offer(unpack(z))
        for _ in range(4):
            if time.monotonic() >= stop-.006:
                break
            try:
                c,jac = calc(z)
            except Halt:
                break
            active = np.flatnonzero(np.abs(c)<2e-7)
            if len(active)<len(z):
                break
            delta,_,rank,_ = lstsq(jac[active],-c[active],lapack_driver='gelsy',check_finite=False)
            if rank<len(z) or np.max(np.abs(delta))>3e-4:
                break
            z += delta
            offer(unpack(z))

    if best_score > 2.36:
        optimize(best_a,False,deadline-.012)
        return best
    span = deadline-start
    optimize(first,True,start+span*float(params['first_share']))
    search_end = start+span*float(params['search_share'])
    if best_score < 2.36583 and time.monotonic()<search_end-.02:
        second = seed_rows((4,3,4,3,4,3))
        if second is not None:
            offer(second)
            optimize(second,True,search_end)
    if time.monotonic()<deadline-.025 and best_a is not None:
        optimize(best_a,False,deadline-.012)
    return best
