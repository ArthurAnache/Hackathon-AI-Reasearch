import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'iterations': (240, 80, 500),
    'exploration': (0.78, 0.55, 0.88),
    'noise': (0.009, 0.001, 0.04),
    'safety': (1e-14, 5e-15, 1e-10, 'log')
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
            + max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))) <= 2

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + .70*budget_s
    eps = float(params['safety'])
    rng = np.random.default_rng(seed)
    best = None
    best_score = -np.inf
    original = None
    try:
        if init is not None:
            original = np.asarray(init['circles'],float) if 'circles' in init else np.asarray(init['state'],float).reshape(3,21).T
            if valid(original):
                best = init
                best_score = float(np.sum(original[:,2]))
    except (ValueError, KeyError, TypeError):
        original = None

    def offer(a):
        nonlocal best, best_score
        a = np.asarray(a, dtype=float).copy()
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
        factor = min(1.,float(np.min(d/(a[I,2]+a[J,2]))))
        if factor <= 0:
            return
        a[:,2] *= factor*(1-eps)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        extent = float(np.sum(hi-lo))
        if extent <= 0:
            return
        scale = (2-12*eps)/extent
        a[:,:2] = (a[:,:2]-(lo+hi)/2)*scale
        a[:,2] *= scale
        value = float(np.sum(a[:,2]))
        if value > best_score and valid(a):
            best_score = value
            best = {'state': a.T.reshape(-1).tolist()}

    offer(np.array([[2*(k%3),2*(k//3),1.] for k in range(21)]))
    if original is not None:
        offer(original)

    def current():
        return np.asarray(best['circles'],float).copy() if 'circles' in best else np.asarray(best['state'],float).reshape(3,21).T.copy()

    def state(a):
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        return np.r_[a[:,0]-(hi[0]+lo[0])/2,a[:,1]-(hi[1]+lo[1])/2,a[:,2],hi[0]-lo[0]]

    walls = np.zeros((84,64))
    walls[K,K] = -1
    walls[21+K,K] = 1
    walls[42+K,21+K] = -1
    walls[63+K,21+K] = 1
    walls[np.arange(84),42+np.tile(K,4)] = -1
    walls[:42,63] = .5
    walls[42:,63] = -.5
    offset = np.r_[np.zeros(42),np.ones(42)]
    objective = np.r_[np.zeros(42),-np.ones(21),0.]
    T = np.zeros((64,34))
    ids = np.r_[np.arange(12),np.arange(3,12)]
    T[K,ids] = 1
    T[42+K,21+ids] = 1
    T[21+np.arange(3,12),12+np.arange(9)] = 1
    T[21+np.arange(12,21),12+np.arange(9)] = -1
    T[63,33] = 1
    mirror = np.r_[np.arange(3),np.arange(12,21),np.arange(3,12)]
    seen = set()
    representatives = []
    for k in range(len(I)):
        if time.monotonic() >= end:
            return best
        p = (int(I[k]),int(J[k]))
        q = tuple(sorted((int(mirror[I[k]]),int(mirror[J[k]]))))
        key = min(p,q)
        if key not in seen:
            seen.add(key)
            representatives.append(k)
    representatives = np.array(representatives)

    def run(q, transform, limit):
        if time.monotonic() >= limit:
            return q
        symmetric = transform.shape[1] == 34
        ix = representatives if symmetric else np.arange(len(I))
        pi,pj = I[ix],J[ix]
        DX = transform[pi]-transform[pj]
        DY = transform[21+pi]-transform[21+pj]
        DR = transform[42+pi]+transform[42+pj]
        wi = np.r_[np.arange(12),21+np.arange(12),42+np.arange(12),63+np.arange(12)] if symmetric else np.arange(84)
        W = walls[wi]@transform
        off = offset[wi]
        obj = objective@transform
        cache = [None,None,None]
        last = q.copy()
        count = 0
        def calc(v):
            if time.monotonic() >= limit:
                raise Stop()
            if cache[0] is not None and np.array_equal(cache[0],v):
                return cache[1],cache[2]
            dx,dy,rs = DX@v,DY@v,DR@v
            dist = np.sqrt(dx*dx+dy*dy)
            denom = np.maximum(dist,1e-15)
            values = np.r_[dist-rs,W@v+off]
            jac = np.vstack(((dx/denom)[:,None]*DX+(dy/denom)[:,None]*DY-DR,W))
            cache[:] = [v.copy(),values,jac]
            return values,jac
        def fun(v):
            if time.monotonic() >= limit:
                raise Stop()
            return float(obj@v),obj
        def callback(v):
            nonlocal last,count
            if time.monotonic() >= limit:
                raise Stop()
            last = v.copy()
            count += 1
            if count % 14 == 0:
                u = transform@v
                offer(u[:63].reshape(3,21).T)
        bounds = [(None,None)]*len(q)
        if symmetric:
            bounds[21:33] = [(1e-7,.6)]*12
        else:
            bounds[42:63] = [(1e-7,.6)]*21
        bounds[-1] = (.15,1.85)
        try:
            result = minimize(fun,q,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda v:calc(v)[0],'jac':lambda v:calc(v)[1]},
                callback=callback,options={'maxiter':int(params['iterations']),'ftol':3e-13,'disp':False})
            last = result.x
        except Stop:
            pass
        u = transform@last
        offer(u[:63].reshape(3,21).T)
        return last

    def lattice(counts,stretch):
        singles,pairs = [],[]
        for col,n in enumerate(counts):
            if time.monotonic() >= end:
                raise Stop()
            x = col*np.sqrt(3.)*stretch
            if n%2:
                singles.append(x)
            for h in range(2 if n%2 else 1,n,2):
                if time.monotonic() >= end:
                    raise Stop()
                pairs.append((x,float(h)))
        xs = np.array(singles+[p[0] for p in pairs])
        ys = np.array([p[1] for p in pairs])
        width = float(xs.max()-xs.min()+2)
        height = 2*(float(ys.max())+1)
        s = 2/(width+height)
        return np.r_[(xs-(xs.max()+xs.min())/2)*s,ys*s,np.full(12,s),width*s]

    if best_score < 2.32:
        explore_end = start+.70*budget_s*float(params['exploration'])
        families = [([5,4,3,4,5],1.),([3,4,3,4,3,4],1.),([4,5,3,5,4],1.)]
        trajectories = []
        for idx,(counts,stretch) in enumerate(families):
            now = time.monotonic()
            if now >= explore_end-.025:
                break
            try:
                q = lattice(counts,stretch)
            except Stop:
                break
            u = T@q
            offer(u[:63].reshape(3,21).T)
            limit = min(explore_end,now+(explore_end-now)/(len(families)-idx))
            q = run(q,T,limit)
            trajectories.append(q)
        if trajectories and time.monotonic() < end-.24:
            q = min(trajectories,key=lambda v:float(objective@(T@v))).copy()
            q[:21] += rng.normal(0,float(params['noise']),21)
            q[12:21] = np.abs(q[12:21])
            run(q,T,end-.21)
    if time.monotonic() < end-.035:
        run(state(current()),np.eye(64),end-.012)
    return best
