import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'seed_fraction': (0.72, 0.4, 0.85),
    'iterations': (220, 80, 450),
    'safety': (3e-14, 1e-15, 1e-10, 'log')
}
I, J = np.triu_indices(21, 1)
K = np.arange(21)
P = np.arange(210)

class Timeout(Exception):
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
    start = time.monotonic()
    deadline = start + .69*budget_s
    eps = float(params['safety'])
    best = None
    best_a = None
    bestscore = -float('inf')
    original = None
    try:
        if init is not None:
            original = np.asarray(init['circles'],float) if 'circles' in init else np.asarray(init['state'],float).reshape(3,21).T
            if valid(original):
                best = init
                best_a = original.copy()
                bestscore = float(np.sum(original[:,2]))
    except (KeyError, ValueError, TypeError):
        original = None

    def offer(a):
        nonlocal best, best_a, bestscore
        a = np.asarray(a,float).copy()
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
        ratio = min(1.,float(np.min(d/(a[I,2]+a[J,2]))))
        if ratio <= 0:
            return
        a[:,2] *= ratio*(1-eps)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        scale = (2-12*eps)/float(np.sum(hi-lo))
        a[:,:2] = (a[:,:2]-(lo+hi)/2)*scale
        a[:,2] *= scale
        s = float(np.sum(a[:,2]))
        if s > bestscore and valid(a):
            bestscore = s
            best_a = a
            best = {'state': a.T.reshape(-1).tolist()}

    if original is not None:
        offer(original)
    offer(np.array([[2*(i%3),2*(i//3),1.] for i in range(21)],float))

    def state(a):
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        xy = a[:,:2]-(lo+hi)/2
        return np.r_[xy[:,0],xy[:,1],a[:,2],hi[0]-lo[0]]

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
    T[-1,-1] = 1

    def optimize(v, stop, reduced=False):
        transform = T if reduced else None
        obj = T.T@objective if reduced else objective
        bw = wall@T if reduced else wall
        cache_v = None
        cache_c = None
        cache_j = None
        last = v.copy()
        count = 0
        z0 = T@v if reduced else v
        weights = 1/np.maximum(2*(z0[42+I]+z0[42+J]),.08)
        def calc(q):
            nonlocal cache_v, cache_c, cache_j
            if time.monotonic() >= stop:
                raise Timeout()
            if cache_v is not None and np.array_equal(q,cache_v):
                return cache_c,cache_j
            z = T@q if reduced else q
            dx = z[I]-z[J]
            dy = z[21+I]-z[21+J]
            rs = z[42+I]+z[42+J]
            pair = (dx*dx+dy*dy-rs*rs)*weights
            jac = np.zeros((210,64))
            gx = 2*weights*dx; gy = 2*weights*dy; gr = -2*weights*rs
            jac[P,I] = gx; jac[P,J] = -gx
            jac[P,21+I] = gy; jac[P,21+J] = -gy
            jac[P,42+I] = gr; jac[P,42+J] = gr
            if reduced:
                jac = jac@T
            cache_v = q.copy()
            cache_c = np.r_[pair,bw@q+offset]
            cache_j = np.vstack((jac,bw))
            return cache_c,cache_j
        def fun(q):
            if time.monotonic() >= stop:
                raise Timeout()
            return float(obj@q),obj
        def callback(q):
            nonlocal last,count
            if time.monotonic() >= stop:
                raise Timeout()
            last = q.copy()
            count += 1
            if count % 8 == 0:
                z = T@q if reduced else q
                offer(z[:63].reshape(3,21).T)
        bounds = [(None,None)]*len(v)
        for k in (range(21,33) if reduced else range(42,63)):
            if time.monotonic() >= stop:
                return
            bounds[k] = (1e-6,.6)
        bounds[-1] = (.15,1.85)
        try:
            res = minimize(fun,v,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda q:calc(q)[0],'jac':lambda q:calc(q)[1]},
                callback=callback,options={'maxiter':int(params['iterations']),'ftol':3e-13,'disp':False})
            last = res.x
        except Timeout:
            pass
        z = T@last if reduced else last
        offer(z[:63].reshape(3,21).T)

    def lattice(counts):
        singles = []
        positive = []
        negative = []
        for row,n in enumerate(counts):
            if time.monotonic() >= deadline:
                raise Timeout()
            for col in range(n):
                if time.monotonic() >= deadline:
                    raise Timeout()
                p = [row*np.sqrt(3.),2.*col-(n-1),1.]
                if p[1] == 0:
                    singles.append(p)
                elif p[1] > 0:
                    positive.append(p)
                    negative.append([p[0],-p[1],1.])
        a = np.array(singles+positive+negative,float)
        z = state(a)
        scale = 2/(z[-1]+np.ptp(a[:,1])+2)
        z[:63] *= scale
        z[-1] *= scale
        return np.r_[z[:12],z[24:33],z[42:54],z[-1]]

    cold = bestscore < 2.25
    if cold:
        patterns = [(4,3,4,3,4,3),(5,4,3,4,5),(3,4,5,4,5)]
        search_end = start + .69*budget_s*float(params['seed_fraction'])
        try:
            for n,counts in enumerate(patterns):
                now = time.monotonic()
                if now >= search_end-.015:
                    break
                q = lattice(counts)
                z = T@q
                offer(z[:63].reshape(3,21).T)
                stop = now+(search_end-now)/(len(patterns)-n)
                optimize(q,stop,True)
        except Timeout:
            pass
    if time.monotonic() < deadline-.025:
        optimize(state(best_a),deadline-.008,False)
    return best
