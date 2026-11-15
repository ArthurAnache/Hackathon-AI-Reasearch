import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'screen_share': (0.36, 0.2, 0.55),
    'semifinal_share': (0.32, 0.15, 0.45),
    'iterations': (240, 80, 400),
    'safety': (1e-14, 4e-15, 1e-11, 'log')
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
    end = start + .70*budget_s
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
                best = init
                best_a = original.copy()
                score = float(np.sum(original[:,2]))
    except (ValueError, TypeError, KeyError):
        original = None

    def repair(a):
        a = np.array(a, float, copy=True)
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return None
        d = np.sqrt((a[I,0]-a[J,0])**2 + (a[I,1]-a[J,1])**2)
        factor = min(1., float(np.min(d/(a[I,2]+a[J,2]))))
        if factor <= 0:
            return None
        a[:,2] *= factor*(1-eps)
        lo = np.min(a[:,:2]-a[:,2,None], axis=0)
        hi = np.max(a[:,:2]+a[:,2,None], axis=0)
        scale = (2-12*eps)/float(np.sum(hi-lo))
        a[:,:2] = (a[:,:2]-(hi+lo)/2)*scale
        a[:,2] *= scale
        return a

    def offer(a):
        nonlocal best, best_a, score
        a = repair(a)
        if a is None:
            return -float('inf')
        value = float(np.sum(a[:,2]))
        if value > score and valid(a):
            score = value
            best_a = a.copy()
            best = {'state': a.T.reshape(-1).tolist()}
        return value

    offer(np.column_stack((2*(K%5), 2*(K//5), np.ones(21))))
    if original is not None:
        offer(original)
    if time.monotonic() >= end:
        return best

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
    objective = np.r_[np.zeros(42), -np.ones(21), 0.]
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
            _, keep = np.unique(np.column_stack((bw,bo)), axis=0, return_index=True)
            bw, bo = bw[keep], bo[keep]
        return transform, ax, ay, ar, bw, bo, transform.T@objective

    reduced_model = setup(T, True)

    def state(a):
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        xy = a[:,:2]-(hi+lo)/2
        return np.r_[xy[:,0],xy[:,1],a[:,2],hi[0]-lo[0]]

    def optimize(v, stop, model, reduced):
        transform, ax, ay, ar, bw, bo, obj = model
        accepted = v.copy()
        cache = None
        cv = cj = None
        count = 0
        local_value = -float('inf')
        local_q = accepted.copy()
        def save(q):
            nonlocal local_value, local_q
            z = transform@q
            value = offer(z[:63].reshape(3,21).T)
            if value > local_value:
                local_value = value
                local_q = q.copy()
        def calc(q):
            nonlocal cache, cv, cj
            if time.monotonic() >= stop:
                raise Timeout()
            if cache is not None and np.array_equal(cache,q):
                return cv,cj
            dx, dy = ax@q, ay@q
            dist = np.sqrt(dx*dx+dy*dy)
            inv = 1/np.maximum(dist,1e-15)
            cv = np.r_[dist-ar@q, bw@q+bo]
            cj = np.vstack(((dx*inv)[:,None]*ax + (dy*inv)[:,None]*ay-ar, bw))
            cache = q.copy()
            return cv,cj
        def fun(q):
            if time.monotonic() >= stop:
                raise Timeout()
            return float(obj@q), obj
        def callback(q):
            nonlocal accepted, count
            if time.monotonic() >= stop:
                raise Timeout()
            accepted = q.copy()
            count += 1
            if count % 8 == 0:
                save(q)
        bounds = [(None,None)]*len(v)
        for k in (range(21,33) if reduced else range(42,63)):
            if time.monotonic() >= stop:
                return accepted, local_value
            bounds[k] = (1e-7,.65)
        bounds[-1] = (.12,1.88)
        if reduced:
            for k in range(12,21):
                if time.monotonic() >= stop:
                    return accepted, local_value
                bounds[k] = (0.,1.)
        save(v)
        try:
            result = minimize(fun,v,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda q:calc(q)[0],
                             'jac':lambda q:calc(q)[1]}, callback=callback,
                options={'maxiter':int(params['iterations']),'ftol':1e-12,'disp':False})
            accepted = result.x
        except Timeout:
            pass
        save(accepted)
        return local_q, local_value

    def row_seed(first, second, spacing):
        sx = np.array([-spacing,0.,spacing])
        ux = np.r_[(np.arange(first)-(first-1)/2)*2,
                   (np.arange(second)-(second-1)/2)*2]
        uy = np.r_[np.full(first,np.sqrt(3)),np.full(second,2*np.sqrt(3))]
        q = np.r_[sx,ux,uy,np.ones(12),0.]
        a = (T@q)[:63].reshape(3,21).T
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        scale = 2/np.sum(hi-lo)
        q[:-1] *= scale
        q[-1] = (hi[0]-lo[0])*scale
        return q

    if score < 2.30:
        duration = max(0., end-time.monotonic())
        screen_end = time.monotonic()+float(params['screen_share'])*duration
        racers = []
        configurations = [(4,5,2.), (5,4,2.), (5,4,2.65)]
        for index, config in enumerate(configurations):
            now = time.monotonic()
            if now >= screen_end-.006:
                break
            q, value = optimize(row_seed(*config), now+(screen_end-now)/(3-index), reduced_model, True)
            racers.append((value,q))
        racers.sort(key=lambda item:item[0], reverse=True)
        semi_end = min(end-.12, screen_end+float(params['semifinal_share'])*duration)
        finalists = []
        for index, (_,q) in enumerate(racers[:2]):
            now = time.monotonic()
            if now >= semi_end-.006:
                break
            q, value = optimize(q, now+(semi_end-now)/(min(2,len(racers))-index), reduced_model, True)
            finalists.append((value,q))
        pool = racers+finalists
        if pool and time.monotonic() < end-.15:
            _, q = max(pool,key=lambda item:item[0])
            optimize(q,end-.13,reduced_model,True)
    elif time.monotonic() < end-.15:
        z = state(best_a)
        q = (T.T@z)/np.sum(T*T,axis=0)
        if np.max(np.abs(T@q-z)) < 1e-7:
            optimize(q,end-.13,reduced_model,True)
    if time.monotonic() < end-.025:
        full_model = setup(np.eye(64),False)
        optimize(state(best_a),end-.008,full_model,False)
    return best
