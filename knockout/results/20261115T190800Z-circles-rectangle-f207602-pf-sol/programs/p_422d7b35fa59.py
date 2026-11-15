import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {
    'first_share': (0.48, 0.25, 0.7),
    'second_share': (0.28, 0.1, 0.45),
    'iterations': (230, 80, 400),
    'safety': (1e-14, 4e-15, 1e-10, 'log')
}
I,J = np.triu_indices(21,1)
K = np.arange(21)
P = np.arange(210)

class StopSearch(Exception):
    pass

def valid(a):
    if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
        return False
    d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
    if np.any(d < a[I,2]+a[J,2]):
        return False
    x,y,r = [[Fraction(float(v)) for v in a[:,k]] for k in range(3)]
    return (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))+
            max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))) <= 2

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    eps = float(params['safety'])
    best = None
    score = -float('inf')
    best_a = None
    original = None
    try:
        if init is not None:
            original = (np.asarray(init['circles'],float) if 'circles' in init else
                        np.asarray(init['state'],float).reshape(3,21).T)
            if valid(original):
                best, best_a, score = init, original.copy(), float(np.sum(original[:,2]))
    except (ValueError,TypeError,KeyError):
        original = None

    def offer(a):
        nonlocal best, best_a, score
        a = np.asarray(a,float).copy()
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
        shrink = min(1.,float(np.min(d/(a[I,2]+a[J,2]))))
        if shrink <= 0:
            return
        a[:,2] *= shrink*(1-eps)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        span = float(np.sum(hi-lo))
        if span <= 0:
            return
        scale = (2-8*eps)/span
        a[:,:2] = (a[:,:2]-(hi+lo)/2)*scale
        a[:,2] *= scale
        s = float(np.sum(a[:,2]))
        if s > score and valid(a):
            score, best_a = s, a
            best = {'state':a.T.reshape(-1).tolist()}

    if original is not None:
        offer(original)
    offer(np.array([[2*(i%3),2*(i//3),1.] for i in range(21)],float))

    def lattice(counts):
        pts = []
        for row,n in enumerate(counts):
            if time.monotonic() >= deadline:
                raise StopSearch()
            for col in range(n):
                if time.monotonic() >= deadline:
                    raise StopSearch()
                pts.append([row*np.sqrt(3.),2*col-(n-1),1.])
        a = np.asarray(pts,float)
        lo = np.min(a[:,:2]-1,axis=0)
        hi = np.max(a[:,:2]+1,axis=0)
        a[:,:2] -= (hi+lo)/2
        a *= 2/np.sum(hi-lo)
        return a

    def state(a):
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        xy = a[:,:2]-(hi+lo)/2
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
    T[24+np.arange(9),12+np.arange(9)] = 1
    T[33+np.arange(9),12+np.arange(9)] = -1
    T[-1,-1] = 1

    def quotient(a):
        z = state(a)
        a = z[:63].reshape(3,21).T
        singles = np.argsort(np.abs(a[:,1]))[:3]
        rem = np.setdiff1d(K,singles)
        order = rem[np.argsort(a[rem,1])]
        neg,pos = order[:9],order[9:]
        cost = ((a[pos,None,0]-a[None,neg,0])**2+
                (a[pos,None,1]+a[None,neg,1])**2+
                (a[pos,None,2]-a[None,neg,2])**2)
        _,cc = linear_sum_assignment(cost)
        neg = neg[cc]
        return np.r_[a[singles,0],(a[pos,0]+a[neg,0])/2,
                     (a[pos,1]-a[neg,1])/2,a[singles,2],
                     (a[pos,2]+a[neg,2])/2,z[-1]]

    def solve(v, stop, symmetric):
        transform = T if symmetric else np.eye(64)
        obj = transform.T@objective
        bw = wall@transform
        cache = None
        cval = None
        jval = None
        last = v.copy()
        count = 0
        def calc(q):
            nonlocal cache,cval,jval
            if time.monotonic() >= stop:
                raise StopSearch()
            if cache is not None and np.array_equal(q,cache):
                return cval,jval
            z = transform@q
            dx = z[I]-z[J]
            dy = z[21+I]-z[21+J]
            rs = z[42+I]+z[42+J]
            jac = np.zeros((210,64))
            jac[P,I] = 2*dx; jac[P,J] = -2*dx
            jac[P,21+I] = 2*dy; jac[P,21+J] = -2*dy
            jac[P,42+I] = -2*rs; jac[P,42+J] = -2*rs
            cval = np.r_[dx*dx+dy*dy-rs*rs,bw@q+offset]
            jval = np.vstack((jac@transform,bw))
            cache = q.copy()
            return cval,jval
        def fun(q):
            if time.monotonic() >= stop:
                raise StopSearch()
            return float(obj@q),obj
        def callback(q):
            nonlocal last,count
            if time.monotonic() >= stop:
                raise StopSearch()
            last = q.copy()
            count += 1
            if count % 12 == 0:
                z = transform@q
                offer(z[:63].reshape(3,21).T)
        bounds = [(None,None)]*len(v)
        for k in (range(21,33) if symmetric else range(42,63)):
            if time.monotonic() >= stop:
                return
            bounds[k] = (1e-7,.6)
        bounds[-1] = (.15,1.85)
        try:
            res = minimize(fun,v,jac=True,method='SLSQP',bounds=bounds,
                           constraints={'type':'ineq','fun':lambda q:calc(q)[0],
                                        'jac':lambda q:calc(q)[1]},callback=callback,
                           options={'maxiter':int(params['iterations']),'ftol':3e-13})
            last = res.x
        except StopSearch:
            pass
        z = transform@last
        offer(z[:63].reshape(3,21).T)

    try:
        seeds = [lattice([4,3,4,3,4,3]),lattice([5,4,3,4,5])]
        for a in seeds:
            if time.monotonic() >= deadline:
                return best
            offer(a)
        # Every starting geometry competes on exact feasible score first.
        if score > 2.30:
            seeds.insert(0,best_a.copy())
        else:
            seeds.sort(key=lambda a:-float(np.sum(a[:,2])))
        span = deadline-time.monotonic()
        first_stop = time.monotonic()+float(params['first_share'])*span
        solve(quotient(seeds[0]),first_stop,True)
        if time.monotonic() < deadline-.08:
            second_stop = min(deadline-.06, first_stop+float(params['second_share'])*span)
            solve(quotient(seeds[1]),second_stop,True)
        if time.monotonic() < deadline-.04:
            solve(state(best_a),deadline-.012,False)
    except StopSearch:
        pass
    return best
