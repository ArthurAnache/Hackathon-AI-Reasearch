import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {
    'full_fraction': (0.79, 0.45, 0.95),
    'iterations': (320, 80, 600),
    'safety': (1e-14, 4e-15, 1e-10, 'log'),
    'cold_threshold': (1.8, 0.1, 2.25)
}
I, J = np.triu_indices(21, 1)
K = np.arange(21)
P = np.arange(len(I))

class Timeout(Exception):
    pass

def valid(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return False
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return False
    x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
    return max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))+max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)) <= 2

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + .70*budget_s
    eps = float(params['safety'])
    best = None
    bestscore = -float('inf')
    original = None
    try:
        if init is not None:
            original = np.asarray(init['circles'], float) if 'circles' in init else np.asarray(init['state'], float).reshape(3,21).T
            if valid(original):
                best, bestscore = init, float(np.sum(original[:,2]))
    except (ValueError, TypeError, KeyError):
        original = None

    def offer(a):
        nonlocal best, bestscore
        a = np.asarray(a, float).copy()
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return None
        d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
        ratio = min(1., float(np.min(d/(a[I,2]+a[J,2]))))
        if ratio <= 0:
            return None
        a[:,2] *= ratio*(1-eps)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        scale = (2-8*eps)/float(np.sum(hi-lo))
        a[:,:2] = (a[:,:2]-(hi+lo)/2)*scale
        a[:,2] *= scale
        score = float(np.sum(a[:,2]))
        if score > bestscore and valid(a):
            bestscore = score
            best = {'state': a.T.reshape(-1).tolist()}
        return a

    if original is not None:
        offer(original)
    fallback = np.array([[2*(i%3),2*(i//3),1.] for i in range(21)],float)
    offer(fallback)

    def best_array():
        return np.asarray(best['circles'],float).copy() if 'circles' in best else np.asarray(best['state'],float).reshape(3,21).T.copy()

    # Cold starts need a contact-rich geometry rather than tiny random disks.
    # Rotate the staggered lattice so its reflection axis is y=0.
    cold = original is None
    if original is not None:
        cold = original.shape != (21,3) or not np.isfinite(original).all() or np.any(original[:,2] <= 0)
        if not cold:
            cold = float(np.sum(original[:,2])) < float(params['cold_threshold'])
    if cold:
        lattice = []
        for row in range(6):
            if time.monotonic() >= deadline:
                return best
            count = 4 if row % 2 == 0 else 3
            for col in range(count):
                if time.monotonic() >= deadline:
                    return best
                lattice.append([row*np.sqrt(3.0), 2.0*col-(count-1), 1.0])
        offer(np.asarray(lattice))
        original = best_array()

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

    def optimize(v, stop, transform=None):
        reduced = transform is not None
        obj = transform.T@objective if reduced else objective
        bw = wall@transform if reduced else wall
        cache_v = None
        cache_c = None
        cache_j = None
        last_accepted = v.copy()
        count = 0
        initial = transform@v if reduced else v
        weights = 1/np.maximum(2*(initial[42+I]+initial[42+J]),.08)
        def calc(q):
            nonlocal cache_v, cache_c, cache_j
            if time.monotonic() >= stop:
                raise Timeout()
            if cache_v is not None and np.array_equal(q,cache_v):
                return cache_c,cache_j
            z = transform@q if reduced else q
            dx = z[I]-z[J]; dy = z[21+I]-z[21+J]; rs = z[42+I]+z[42+J]
            pair = (dx*dx+dy*dy-rs*rs)*weights
            jac = np.zeros((210,64))
            gx = 2*weights*dx; gy = 2*weights*dy; gr = -2*weights*rs
            jac[P,I] = gx; jac[P,J] = -gx
            jac[P,21+I] = gy; jac[P,21+J] = -gy
            jac[P,42+I] = gr; jac[P,42+J] = gr
            if reduced:
                jac = jac@transform
            cache_v = q.copy()
            cache_c = np.r_[pair,bw@q+offset]
            cache_j = np.vstack((jac,bw))
            return cache_c,cache_j
        def fun(q):
            if time.monotonic() >= stop:
                raise Timeout()
            return float(obj@q),obj
        def callback(q):
            nonlocal count,last_accepted
            if time.monotonic() >= stop:
                raise Timeout()
            last_accepted = q.copy()
            count += 1
            if count % 10 == 0:
                z = transform@q if reduced else q
                offer(z[:63].reshape(3,21).T)
        bounds = [(None,None)]*len(v)
        ridx = range(21,33) if reduced else range(42,63)
        for k in ridx:
            if time.monotonic() >= stop:
                return v
            bounds[k] = (1e-7,.55)
        bounds[-1] = (.15,1.85)
        try:
            res = minimize(fun,v,method='SLSQP',jac=True,bounds=bounds,
                constraints={'type':'ineq','fun':lambda q:calc(q)[0],'jac':lambda q:calc(q)[1]},
                callback=callback,options={'maxiter':int(params['iterations']),'ftol':8e-14,'disp':False})
            last_accepted = res.x
        except Timeout:
            pass
        z = transform@last_accepted if reduced else last_accepted
        offer(z[:63].reshape(3,21).T)
        return last_accepted

    a = original
    if a is None or a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
        a = best_array()
    fullstop = start + .70*budget_s*float(params['full_fraction'])
    optimize(state(a),fullstop)
    if time.monotonic() >= deadline-.035:
        return best

    choices = []
    a = best_array()
    for swap in (False,True):
        if time.monotonic() >= deadline-.03:
            return best
        b = a.copy()
        if swap:
            b[:,:2] = b[:,[1,0]]
        z = state(b)
        b = z[:63].reshape(3,21).T
        refl = b.copy(); refl[:,1] *= -1
        cost = np.sum((b[:,None,:]-refl[None,:,:])**2,axis=2)
        rr,cc = linear_sum_assignment(cost)
        choices.append((float(cost[rr,cc].sum()),b,z[-1]))
    _,a,w = min(choices,key=lambda t:t[0])
    singles = np.argsort(np.abs(a[:,1]))[:3]
    remaining = np.setdiff1d(K,singles)
    ordered = remaining[np.argsort(a[remaining,1])]
    neg,pos = ordered[:9],ordered[9:]
    cost = (a[pos,None,0]-a[None,neg,0])**2+(a[pos,None,1]+a[None,neg,1])**2+(a[pos,None,2]-a[None,neg,2])**2
    _,cc = linear_sum_assignment(cost)
    neg = neg[cc]
    q = np.r_[a[singles,0],.5*(a[pos,0]+a[neg,0]),.5*(a[pos,1]-a[neg,1]),a[singles,2],.5*(a[pos,2]+a[neg,2]),w]
    T = np.zeros((64,34))
    ids = np.r_[np.arange(12),np.arange(3,12)]
    T[K,ids] = 1
    T[42+K,21+ids] = 1
    T[21+np.arange(3,12),12+np.arange(9)] = 1
    T[21+np.arange(12,21),12+np.arange(9)] = -1
    T[-1,-1] = 1
    now = time.monotonic()
    if now < deadline-.025:
        optimize(q,now+.82*(deadline-now),T)
    if time.monotonic() < deadline-.045:
        optimize(state(best_array()),deadline-.008)
    return best
