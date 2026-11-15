import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'iterations': (180, 60, 400),
    'exploration': (0.79, 0.5, 0.9),
    'safety': (1e-14, 5e-15, 1e-11, 'log')
}
I, J = np.triu_indices(21, 1)
K = np.arange(21)

class Stop(Exception):
    pass

def valid(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return False
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return False
    x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
    return (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
            +max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))) <= 2

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + 0.70*budget_s
    eps = float(params['safety'])
    best = None
    score = -float('inf')
    original = None
    try:
        if init is not None:
            original = np.asarray(init['circles'], dtype=float) if 'circles' in init else np.asarray(init['state'], dtype=float).reshape(3,21).T
            if valid(original):
                best, score = init, float(np.sum(original[:,2]))
    except (ValueError, KeyError, TypeError):
        original = None

    def offer(a):
        nonlocal best, score
        a = np.asarray(a, dtype=float).copy()
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        d = np.sqrt((a[I,0]-a[J,0])**2 + (a[I,1]-a[J,1])**2)
        ratio = min(1.0, float(np.min(d/(a[I,2]+a[J,2]))))
        if ratio <= 0:
            return
        a[:,2] *= ratio*(1-eps)
        lo = np.min(a[:,:2]-a[:,2,None], axis=0)
        hi = np.max(a[:,:2]+a[:,2,None], axis=0)
        extent = float(np.sum(hi-lo))
        if extent <= 0:
            return
        scale = (2-8*eps)/extent
        a[:,:2] = (a[:,:2]-(lo+hi)/2)*scale
        a[:,2] *= scale
        value = float(np.sum(a[:,2]))
        if value > score and valid(a):
            score = value
            best = {'state': a.T.reshape(-1).tolist()}

    def getbest():
        if 'circles' in best:
            return np.asarray(best['circles'], dtype=float).copy()
        return np.asarray(best['state'], dtype=float).reshape(3,21).T.copy()

    def state(a):
        lo = np.min(a[:,:2]-a[:,2,None], axis=0)
        hi = np.max(a[:,:2]+a[:,2,None], axis=0)
        return np.r_[a[:,0]-(hi[0]+lo[0])/2, a[:,1]-(hi[1]+lo[1])/2, a[:,2], hi[0]-lo[0]]

    offer(np.array([[2*(i%3),2*(i//3),1.] for i in range(21)], dtype=float))
    if original is not None:
        offer(original)

    wall = np.zeros((84,64))
    wall[K,K] = -1; wall[K,42+K] = -1; wall[K,63] = .5
    wall[21+K,K] = 1; wall[21+K,42+K] = -1; wall[21+K,63] = .5
    wall[42+K,21+K] = -1; wall[42+K,42+K] = -1; wall[42+K,63] = -.5
    wall[63+K,21+K] = 1; wall[63+K,42+K] = -1; wall[63+K,63] = -.5
    offset = np.r_[np.zeros(42), np.ones(42)]
    objective = np.r_[np.zeros(42), -np.ones(21), 0.]
    T = np.zeros((64,34))
    ids = np.r_[np.arange(12),np.arange(3,12)]
    T[K,ids] = 1
    T[42+K,21+ids] = 1
    T[21+np.arange(3,12),12+np.arange(9)] = 1
    T[21+np.arange(12,21),12+np.arange(9)] = -1
    T[-1,-1] = 1

    def optimize(q, transform, stop, maxiter=None):
        if time.monotonic() >= stop:
            return
        X = transform[:21]
        Y = transform[21:42]
        R = transform[42:63]
        DX = X[I]-X[J]
        DY = Y[I]-Y[J]
        DR = R[I]+R[J]
        W = wall@transform
        obj = objective@transform
        initial_rs = DR@q
        weights = 1/np.maximum(2*initial_rs, .08)
        last = q.copy()
        cached = None
        cv = None
        cj = None
        count = 0
        def calc(v):
            nonlocal cached, cv, cj
            if time.monotonic() >= stop:
                raise Stop()
            if cached is not None and np.array_equal(cached,v):
                return cv,cj
            dx = DX@v; dy = DY@v; rs = DR@v
            pair = weights*(dx*dx+dy*dy-rs*rs)
            jac = (2*weights*dx)[:,None]*DX + (2*weights*dy)[:,None]*DY - (2*weights*rs)[:,None]*DR
            cv = np.r_[pair,W@v+offset]
            cj = np.vstack((jac,W))
            cached = v.copy()
            return cv,cj
        def fun(v):
            if time.monotonic() >= stop:
                raise Stop()
            return float(obj@v), obj
        def callback(v):
            nonlocal last,count
            if time.monotonic() >= stop:
                raise Stop()
            last = v.copy()
            count += 1
            if count % 18 == 0:
                z = transform@v
                offer(z[:63].reshape(3,21).T)
        bounds = [(None,None)]*len(q)
        if len(q) == 34:
            bounds[21:33] = [(1e-7,.6)]*12
        else:
            bounds[42:63] = [(1e-7,.6)]*21
        bounds[-1] = (.15,1.85)
        try:
            res = minimize(fun,q,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda v:calc(v)[0],'jac':lambda v:calc(v)[1]},
                callback=callback,options={'maxiter':int(maxiter or params['iterations']),'ftol':5e-13,'disp':False})
            last = res.x
        except Stop:
            pass
        z = transform@last
        offer(z[:63].reshape(3,21).T)

    def lattice(counts, stretch=1.0):
        singles = []
        pairs = []
        for col,n in enumerate(counts):
            if time.monotonic() >= deadline:
                raise Stop()
            x = col*np.sqrt(3.)*stretch
            if n % 2:
                singles.append(x)
                heights = range(2,n,2)
            else:
                heights = range(1,n,2)
            for h in heights:
                if time.monotonic() >= deadline:
                    raise Stop()
                pairs.append((x,float(h)))
        xs = np.array(singles+[p[0] for p in pairs])
        ys = np.array([p[1] for p in pairs])
        low = float(xs.min()-1)
        high = float(xs.max()+1)
        height = 2*(float(ys.max())+1)
        scale = 2/(high-low+height)
        q = np.r_[(xs-(low+high)/2)*scale,ys*scale,np.full(12,scale),(high-low)*scale]
        return q

    # Cold runs explore distinct contact graphs in the small quotient.
    if score < 2.32:
        explore_end = started + .70*budget_s*float(params['exploration'])
        families = [([3,4,3,4,3,4],1.0), ([5,4,3,4,5],1.0), ([4,5,3,5,4],1.0)]
        for idx,(counts,stretch) in enumerate(families):
            now = time.monotonic()
            if now >= explore_end-.025:
                break
            allowance = (explore_end-now)/(len(families)-idx)
            try:
                q = lattice(counts,stretch)
            except Stop:
                break
            z = T@q
            offer(z[:63].reshape(3,21).T)
            optimize(q,T,min(explore_end,now+allowance))
    if time.monotonic() < deadline-.035:
        optimize(state(getbest()),np.eye(64),deadline-.012,240)
    return best
