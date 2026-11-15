import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'seed_fraction': (0.73, 0.4, 0.9),
    'iterations': (160, 50, 350),
    'safety': (2e-14, 5e-15, 1e-10, 'log')
}
I, J = np.triu_indices(21, 1)
K = np.arange(21)
P = np.arange(210)

class Timeout(Exception):
    pass

def valid(a):
    if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
        return False
    d = np.sqrt((a[I,0]-a[J,0])**2 + (a[I,1]-a[J,1])**2)
    if np.any(d < a[I,2]+a[J,2]):
        return False
    x, y, r = [[Fraction(float(t)) for t in a[:,k]] for k in range(3)]
    return (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r)) +
            max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))) <= 2

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + .70*budget_s
    eps = float(params['safety'])
    best = None
    score = -float('inf')
    best_a = None
    original = None
    try:
        if init is not None:
            original = np.asarray(init['circles'], float) if 'circles' in init else np.asarray(init['state'], float).reshape(3,21).T
            if valid(original):
                best, score, best_a = init, float(np.sum(original[:,2])), original.copy()
    except (ValueError, TypeError, KeyError):
        original = None

    def offer(a):
        nonlocal best, score, best_a
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
        s = float(np.sum(a[:,2]))
        if s > score and valid(a):
            score, best_a = s, a.copy()
            best = {'state': a.T.reshape(-1).tolist()}

    if original is not None:
        offer(original)
    offer(np.array([[2*(i%5),2*(i//5),1.] for i in range(21)]))

    wall = np.zeros((84,64))
    wall[K,K] = -1; wall[K,42+K] = -1; wall[K,63] = .5
    wall[21+K,K] = 1; wall[21+K,42+K] = -1; wall[21+K,63] = .5
    wall[42+K,21+K] = -1; wall[42+K,42+K] = -1; wall[42+K,63] = -.5
    wall[63+K,21+K] = 1; wall[63+K,42+K] = -1; wall[63+K,63] = -.5
    offset = np.r_[np.zeros(42),np.ones(42)]
    objective = np.r_[np.zeros(42),-np.ones(21),0.]
    T = np.zeros((64,34))
    ids = np.r_[np.arange(12),np.arange(3,12)]
    T[K,ids] = 1
    T[42+K,21+ids] = 1
    T[21+np.arange(3,12),12+np.arange(9)] = 1
    T[21+np.arange(12,21),12+np.arange(9)] = -1
    T[63,33] = 1

    def state(a):
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        xy = a[:,:2]-(hi+lo)/2
        return np.r_[xy[:,0],xy[:,1],a[:,2],hi[0]-lo[0]]

    def optimize(v, stop, transform=None):
        reduced = transform is not None
        obj = transform.T@objective if reduced else objective
        bw = wall@transform if reduced else wall
        cache = None
        cv = cj = None
        accepted = v.copy()
        count = 0
        initial = transform@v if reduced else v
        weights = 1/np.maximum(2*(initial[42+I]+initial[42+J]),.08)
        def calc(q):
            nonlocal cache, cv, cj
            if time.monotonic() >= stop:
                raise Timeout()
            if cache is not None and np.array_equal(cache,q):
                return cv,cj
            z = transform@q if reduced else q
            dx = z[I]-z[J]; dy = z[21+I]-z[21+J]
            rs = z[42+I]+z[42+J]
            jac = np.zeros((210,64))
            gx = 2*weights*dx; gy = 2*weights*dy; gr = -2*weights*rs
            jac[P,I] = gx; jac[P,J] = -gx
            jac[P,21+I] = gy; jac[P,21+J] = -gy
            jac[P,42+I] = gr; jac[P,42+J] = gr
            if reduced:
                jac = jac@transform
            cv = np.r_[(dx*dx+dy*dy-rs*rs)*weights,bw@q+offset]
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
            if count % 12 == 0:
                z = transform@q if reduced else q
                offer(z[:63].reshape(3,21).T)
        bounds = [(None,None)]*len(v)
        for k in (range(21,33) if reduced else range(42,63)):
            if time.monotonic() >= stop:
                return
            bounds[k] = (1e-7,.65)
        bounds[-1] = (.12,1.88)
        try:
            result = minimize(fun,v,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda q:calc(q)[0], 'jac':lambda q:calc(q)[1]},
                callback=callback,options={'maxiter':int(params['iterations']),'ftol':6e-14,'disp':False})
            accepted = result.x
        except Timeout:
            pass
        z = transform@accepted if reduced else accepted
        offer(z[:63].reshape(3,21).T)

    def row_seed(first, second, spacing=2.):
        # Three axial circles and two upper rows, reflected below the axis.
        sx = np.array([-spacing,0.,spacing])
        ux = np.r_[(np.arange(first)-(first-1)/2)*2,
                   (np.arange(second)-(second-1)/2)*2]
        uy = np.r_[np.full(first,np.sqrt(3)),np.full(second,2*np.sqrt(3))]
        q = np.r_[sx,ux,uy,np.ones(12),0.]
        z = T@q
        a = z[:63].reshape(3,21).T
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        f = 2/np.sum(hi-lo)
        q[:-1] *= f
        q[-1] = (hi[0]-lo[0])*f
        return q

    if score > 2.30:
        optimize(state(best_a), start+.70*budget_s*.78)
    else:
        seed_end = start+.70*budget_s*float(params['seed_fraction'])
        configurations = [(5,4,2.),(4,5,2.),(5,4,2.65)]
        for index, configuration in enumerate(configurations):
            now = time.monotonic()
            if now >= seed_end-.02:
                break
            stop = now+(seed_end-now)/(len(configurations)-index)
            q = row_seed(*configuration)
            offer((T@q)[:63].reshape(3,21).T)
            optimize(q,stop,T)
    if time.monotonic() < end-.04:
        optimize(state(best_a),end-.012)
    return best
