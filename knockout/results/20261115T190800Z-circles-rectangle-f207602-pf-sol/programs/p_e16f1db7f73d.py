import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'search_fraction': (0.73, 0.5, 0.85),
    'iterations': (180, 60, 350),
    'safety': (2e-14, 5e-15, 1e-10, 'log')
}
I, J = np.triu_indices(21, 1)
K = np.arange(21)

class Timeout(Exception):
    pass

def valid(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return False
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return False
    x, y, r = [[Fraction(float(t)) for t in a[:, k]] for k in range(3)]
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
            original = (np.asarray(init['circles'], float) if 'circles' in init
                        else np.asarray(init['state'], float).reshape(3,21).T)
            if valid(original):
                best, best_a = init, original.copy()
                score = float(np.sum(original[:,2]))
    except (ValueError, TypeError, KeyError):
        original = None

    def offer(a):
        nonlocal best, best_a, score
        a = np.array(a, dtype=float, copy=True)
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        d = np.sqrt((a[I,0]-a[J,0])**2 + (a[I,1]-a[J,1])**2)
        f = min(1., float(np.min(d/(a[I,2]+a[J,2]))))
        if f <= 0:
            return
        a[:,2] *= f*(1-eps)
        lo = np.min(a[:,:2]-a[:,2,None], axis=0)
        hi = np.max(a[:,:2]+a[:,2,None], axis=0)
        scale = (2-12*eps)/float(np.sum(hi-lo))
        a[:,:2] = (a[:,:2]-(hi+lo)/2)*scale
        a[:,2] *= scale
        value = float(np.sum(a[:,2]))
        if value > score and valid(a):
            score, best_a = value, a.copy()
            best = {'state': a.T.reshape(-1).tolist()}

    if original is not None:
        offer(original)
    offer(np.array([[2*(i%5), 2*(i//5), 1.] for i in range(21)]))

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
    ids = np.r_[np.arange(12),np.arange(3,12)]
    T[K,ids] = 1
    T[42+K,21+ids] = 1
    T[21+np.arange(3,12),12+np.arange(9)] = 1
    T[21+np.arange(12,21),12+np.arange(9)] = -1
    T[63,33] = 1
    mirror = np.r_[np.arange(3),np.arange(12,21),np.arange(3,12)]
    keep = []
    for k,(i,j) in enumerate(zip(I,J)):
        if time.monotonic() >= end:
            return best
        mi,mj = sorted((int(mirror[i]),int(mirror[j])))
        if (int(i),int(j)) <= (mi,mj):
            keep.append(k)
    ri, rj = I[keep], J[keep]
    tw = wall@T
    _, wi = np.unique(np.column_stack((tw,offset)), axis=0, return_index=True)
    rw, ro = tw[wi], offset[wi]

    def state(a):
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        xy = a[:,:2]-(hi+lo)/2
        return np.r_[xy[:,0],xy[:,1],a[:,2],hi[0]-lo[0]]

    def optimize(v, stop, reduced=False):
        if time.monotonic() >= stop:
            return
        obj = T.T@objective if reduced else objective
        bw, bo = (rw,ro) if reduced else (wall,offset)
        ii,jj = (ri,rj) if reduced else (I,J)
        pp = np.arange(len(ii))
        cache = None
        cv = cj = None
        accepted = v.copy()
        count = 0
        def calc(q):
            nonlocal cache,cv,cj
            if time.monotonic() >= stop:
                raise Timeout()
            if cache is not None and np.array_equal(cache,q):
                return cv,cj
            z = T@q if reduced else q
            dx = z[ii]-z[jj]
            dy = z[21+ii]-z[21+jj]
            rs = z[42+ii]+z[42+jj]
            dist = np.sqrt(dx*dx+dy*dy)
            inv = 1/np.maximum(dist,1e-15)
            jac = np.zeros((len(ii),64))
            gx,gy = dx*inv,dy*inv
            jac[pp,ii] = gx
            jac[pp,jj] = -gx
            jac[pp,21+ii] = gy
            jac[pp,21+jj] = -gy
            jac[pp,42+ii] = -1
            jac[pp,42+jj] = -1
            if reduced:
                jac = jac@T
            cv = np.r_[dist-rs,bw@q+bo]
            cj = np.vstack((jac,bw))
            cache = q.copy()
            return cv,cj
        def fun(q):
            if time.monotonic() >= stop:
                raise Timeout()
            return float(obj@q),obj
        def callback(q):
            nonlocal accepted,count
            if time.monotonic() >= stop:
                raise Timeout()
            accepted = q.copy()
            count += 1
            if count % 16 == 0:
                z = T@q if reduced else q
                offer(z[:63].reshape(3,21).T)
        bounds = [(None,None)]*len(v)
        for k in (range(21,33) if reduced else range(42,63)):
            if time.monotonic() >= stop:
                return
            bounds[k] = (1e-7,.65)
        bounds[-1] = (.12,1.88)
        try:
            result = minimize(fun,v,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda q:calc(q)[0],
                             'jac':lambda q:calc(q)[1]},callback=callback,
                options={'maxiter':int(params['iterations']),'ftol':3e-13,'disp':False})
            accepted = result.x
        except Timeout:
            pass
        z = T@accepted if reduced else accepted
        offer(z[:63].reshape(3,21).T)

    def row_seed(first,second,spacing):
        sx = np.array([-spacing,0.,spacing])
        ux = np.r_[(np.arange(first)-(first-1)/2)*2,
                   (np.arange(second)-(second-1)/2)*2]
        uy = np.r_[np.full(first,np.sqrt(3)),np.full(second,2*np.sqrt(3))]
        q = np.r_[sx,ux,uy,np.ones(12),0.]
        a = (T@q)[:63].reshape(3,21).T
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        f = 2/np.sum(hi-lo)
        q[:-1] *= f
        q[-1] = (hi[0]-lo[0])*f
        return q

    if score > 2.30:
        optimize(state(best_a), end-.025)
    else:
        search_end = start+.70*budget_s*float(params['search_fraction'])
        configs = [(4,5,2.),(5,4,2.65),(5,4,2.)]
        for index,config in enumerate(configs):
            now = time.monotonic()
            if now >= search_end-.025:
                break
            stop = now+(search_end-now)/(len(configs)-index)
            q = row_seed(*config)
            offer((T@q)[:63].reshape(3,21).T)
            optimize(q,stop,True)
        if time.monotonic() < end-.035:
            optimize(state(best_a),end-.018)
    return best
