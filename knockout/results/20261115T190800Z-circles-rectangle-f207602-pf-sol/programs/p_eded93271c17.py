import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'screen_fraction': (0.70, 0.45, 0.85),
    'iterations': (220, 60, 400),
    'safety': (1e-14, 4e-15, 1e-10, 'log')
}
N = 21
I, J = np.triu_indices(N, 1)
K = np.arange(N)

class Timeout(Exception):
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
    start = time.monotonic()
    end = start + 0.70*budget_s
    eps = float(params['safety'])
    best = None
    best_a = None
    score = -float('inf')
    original = None
    try:
        if init is not None:
            original = (np.asarray(init['circles'], float) if 'circles' in init else
                        np.asarray(init['state'], float).reshape(3,21).T)
            if valid(original):
                best, best_a = init, original.copy()
                score = float(np.sum(original[:,2]))
    except (ValueError, TypeError, KeyError):
        original = None

    def offer(a):
        nonlocal best, best_a, score
        a = np.array(a, float, copy=True)
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        d = np.sqrt((a[I,0]-a[J,0])**2 + (a[I,1]-a[J,1])**2)
        factor = min(1., float(np.min(d/(a[I,2]+a[J,2]))))
        if factor <= 0:
            return
        a[:,2] *= factor*(1-eps)
        lo = np.min(a[:,:2]-a[:,2,None], axis=0)
        hi = np.max(a[:,:2]+a[:,2,None], axis=0)
        scale = (2-12*eps)/float(np.sum(hi-lo))
        a[:,:2] = (a[:,:2]-(hi+lo)/2)*scale
        a[:,2] *= scale
        value = float(np.sum(a[:,2]))
        if value > score and valid(a):
            score = value
            best_a = a.copy()
            best = {'state': a.T.reshape(-1).tolist()}

    if original is not None:
        offer(original)
    grid = np.column_stack((2*(K%5), 2*(K//5), np.ones(21)))
    offer(grid)

    wall = np.zeros((84,64))
    wall[K,K] = -1
    wall[K,42+K] = -1
    wall[K,63] = .5
    wall[21+K,K] = 1
    wall[21+K,42+K] = -1
    wall[21+K,63] = .5
    wall[42+K,21+K] = -1
    wall[42+K,42+K] = -1
    wall[42+K,63] = -.5
    wall[63+K,21+K] = 1
    wall[63+K,42+K] = -1
    wall[63+K,63] = -.5
    offset = np.r_[np.zeros(42), np.ones(42)]
    obj = np.r_[np.zeros(42), -np.ones(21), 0.]
    T = np.zeros((64,34))
    ids = np.r_[np.arange(12), np.arange(3,12)]
    T[K,ids] = 1
    T[42+K,21+ids] = 1
    T[21+np.arange(3,12),12+np.arange(9)] = 1
    T[21+np.arange(12,21),12+np.arange(9)] = -1
    T[63,33] = 1

    reflection = np.r_[np.arange(3), np.arange(12,21), np.arange(3,12)]
    representatives = []
    for k, (i,j) in enumerate(zip(I,J)):
        if time.monotonic() >= end:
            return best
        other = tuple(sorted((int(reflection[i]), int(reflection[j]))))
        if (int(i),int(j)) <= other:
            representatives.append(k)

    def setup(transform, reduced):
        ii = I[representatives] if reduced else I
        jj = J[representatives] if reduced else J
        ax = transform[ii]-transform[jj]
        ay = transform[21+ii]-transform[21+jj]
        ar = transform[42+ii]+transform[42+jj]
        bw = wall@transform
        bo = offset
        if reduced:
            combined = np.column_stack((bw,bo))
            _, keep = np.unique(combined, axis=0, return_index=True)
            bw, bo = bw[keep], bo[keep]
        return (transform, ax, ay, ar, bw, bo, transform.T@obj)

    reduced_model = setup(T, True)
    full_model = None

    def state(a):
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        xy = a[:,:2]-(hi+lo)/2
        return np.r_[xy[:,0],xy[:,1],a[:,2],hi[0]-lo[0]]

    def optimize(v, stop, model, reduced):
        transform, ax, ay, ar, bw, bo, objective = model
        accepted = v.copy()
        cache = None
        cv = cj = None
        count = 0
        def calc(q):
            nonlocal cache, cv, cj
            if time.monotonic() >= stop:
                raise Timeout()
            if cache is not None and np.array_equal(cache,q):
                return cv,cj
            dx = ax@q
            dy = ay@q
            dist = np.sqrt(dx*dx+dy*dy)
            inv = 1/np.maximum(dist,1e-15)
            cv = np.r_[dist-ar@q, bw@q+bo]
            cj = np.vstack(((dx*inv)[:,None]*ax + (dy*inv)[:,None]*ay-ar, bw))
            cache = q.copy()
            return cv,cj
        def fun(q):
            if time.monotonic() >= stop:
                raise Timeout()
            return float(objective@q), objective
        def callback(q):
            nonlocal accepted, count
            if time.monotonic() >= stop:
                raise Timeout()
            accepted = q.copy()
            count += 1
            if count % 10 == 0:
                z = transform@q
                offer(z[:63].reshape(3,21).T)
        bounds = [(None,None)]*len(v)
        radius_indices = range(21,33) if reduced else range(42,63)
        for k in radius_indices:
            if time.monotonic() >= stop:
                return
            bounds[k] = (1e-7,.65)
        bounds[-1] = (.12,1.88)
        if reduced:
            for k in range(12,21):
                if time.monotonic() >= stop:
                    return
                bounds[k] = (0.,1.)
        try:
            result = minimize(fun,v,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda q:calc(q)[0],
                             'jac':lambda q:calc(q)[1]}, callback=callback,
                options={'maxiter':int(params['iterations']),'ftol':2e-13,'disp':False})
            accepted = result.x
        except Timeout:
            pass
        z = transform@accepted
        offer(z[:63].reshape(3,21).T)

    def row_seed(first, second, spacing):
        sx = np.array([-spacing,0.,spacing])
        ux = np.r_[(np.arange(first)-(first-1)/2)*2,
                   (np.arange(second)-(second-1)/2)*2]
        uy = np.r_[np.full(first,np.sqrt(3)),np.full(second,2*np.sqrt(3))]
        q = np.r_[sx,ux,uy,np.ones(12),0.]
        z = T@q
        a = z[:63].reshape(3,21).T
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        scale = 2/np.sum(hi-lo)
        q[:-1] *= scale
        q[-1] = (hi[0]-lo[0])*scale
        return q

    if score < 2.30:
        screen_end = start + .70*budget_s*float(params['screen_fraction'])
        configurations = [(5,4,2.), (4,5,2.), (5,4,2.65)]
        for index, config in enumerate(configurations):
            now = time.monotonic()
            if now >= screen_end-.015:
                break
            stop = now+(screen_end-now)/(len(configurations)-index)
            q = row_seed(*config)
            offer((T@q)[:63].reshape(3,21).T)
            optimize(q,stop,reduced_model,True)
        # Continue the winning symmetric trajectory before releasing symmetry.
        if time.monotonic() < end-.16:
            z = state(best_a)
            q = (T.T@z)/np.sum(T*T,axis=0)
            residual = np.max(np.abs(T@q-z))
            if residual < 1e-7:
                optimize(q,end-.14,reduced_model,True)
    if time.monotonic() < end-.025:
        full_model = setup(np.eye(64),False)
        optimize(state(best_a),end-.008,full_model,False)
    return best
