import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'first_share': (0.31, 0.20, 0.45),
    'second_share': (0.28, 0.18, 0.40),
    'third_share': (0.24, 0.12, 0.35),
    'iterations': (180, 80, 300),
    'safety': (1e-14, 4e-15, 1e-11, 'log')
}
I, J = np.triu_indices(21, 1)

class Timeout(Exception):
    pass

def valid(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return False
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return False
    x = list(map(Fraction, map(float, a[:, 0])))
    y = list(map(Fraction, map(float, a[:, 1])))
    r = list(map(Fraction, map(float, a[:, 2])))
    return (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r)) +
            max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))) <= 2

def repair(a, eps):
    a = np.array(a, dtype=float, copy=True)
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return None
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    ratio = min(1., float(np.min(d/(a[I, 2]+a[J, 2]))))
    if ratio <= 0:
        return None
    a[:, 2] *= ratio*(1-eps)
    lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
    hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
    scale = (2-8*eps)/float(np.sum(hi-lo))
    a[:, :2] = (a[:, :2]-(hi+lo)/2)*scale
    a[:, 2] *= scale
    return a if valid(a) else None

def model(sym, deadline):
    n = 34 if sym else 64
    x = np.zeros((21, n)); y = x.copy(); r = x.copy()
    if sym:
        ids = np.r_[np.arange(12), np.arange(3, 12)]
        x[np.arange(21), ids] = 1
        y[np.arange(3, 12), np.arange(12, 21)] = 1
        y[np.arange(12, 21), np.arange(12, 21)] = -1
        r[np.arange(21), 21+ids] = 1
        pairs = []
        for i in range(12):
            if time.monotonic() >= deadline:
                raise Timeout
            pairs.extend((i,j) for j in range(i+1,12))
        for i in range(3,12):
            if time.monotonic() >= deadline:
                raise Timeout
            pairs.extend((i,j+9) for j in range(i,12))
        pi,pj = np.array(pairs).T
        boundary = np.arange(12)
        ridx = np.arange(21,33)
    else:
        x[np.arange(21),np.arange(21)] = 1
        y[np.arange(21),21+np.arange(21)] = 1
        r[np.arange(21),42+np.arange(21)] = 1
        pi,pj = I,J
        boundary = np.arange(21)
        ridx = np.arange(42,63)
    w = np.zeros(n); w[-1] = 1
    walls = np.concatenate((x[boundary]-r[boundary], w-x[boundary]-r[boundary],
                            -.5*w-y[boundary]-r[boundary], -.5*w+y[boundary]-r[boundary]))
    off = np.r_[np.zeros(2*len(boundary)),np.ones(2*len(boundary))]
    bounds = [(None,None)]*n
    for k in ridx:
        if time.monotonic() >= deadline:
            raise Timeout
        bounds[int(k)] = (1e-7,.5)
    bounds[-1] = (.12,1.88)
    return x,y,r,x[pi]-x[pj],y[pi]-y[pj],r[pi]+r[pj],walls,off,-r.sum(axis=0),bounds

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    span = .70*budget_s
    deadline = start+span
    eps = float(params['safety'])
    best = None
    score_best = -np.inf
    bestarray = None
    try:
        a = np.asarray(init['circles'],float) if 'circles' in init else np.asarray(init['state'],float).reshape(3,21).T
        if valid(a):
            best = init
            score_best = float(np.sum(a[:,2]))
            bestarray = a.copy()
    except (TypeError,KeyError,ValueError):
        pass

    def offer(a):
        nonlocal best,score_best,bestarray
        a = repair(a,eps)
        if a is not None:
            score = float(np.sum(a[:,2]))
            if score > score_best:
                score_best = score
                bestarray = a.copy()
                best = {'state': a.T.reshape(-1).tolist()}
        return a

    offer(np.column_stack((.2*(np.arange(21)%7),.2*(np.arange(21)//7),np.full(21,.09))))

    def encode(a):
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        return np.r_[a[:,0]-lo[0],a[:,1]-(lo[1]+hi[1])/2,a[:,2],hi[0]-lo[0]]

    def unpack(z,m):
        return np.column_stack((m[0]@z,m[1]@z,m[2]@z))

    def lattice(counts):
        singles = []; pairs = []
        for k,c in enumerate(counts):
            if time.monotonic() >= deadline:
                raise Timeout
            xx = np.sqrt(3.)*k
            if c%2:
                singles.append((xx,0.,1.))
            for yy in range(2 if c%2 else 1,c,2):
                if time.monotonic() >= deadline:
                    raise Timeout
                pairs.append((xx,float(yy),1.))
        a = np.array(singles+pairs+[(x,-y,r) for x,y,r in pairs])
        a = offer(a)
        z = encode(a)
        return np.r_[z[:12],z[24:33],z[42:54],z[-1]]

    def optimize(z,m,stop):
        if time.monotonic() >= stop:
            return z
        x,y,r,dxm,dym,rsm,walls,off,obj,bounds = m
        weights = 1/np.maximum(2*(rsm@z),.05)
        cache = [None,None,None]
        last = [z.copy()]
        accepted = [z.copy()]
        count = [0]
        def calc(v):
            if time.monotonic() >= stop:
                raise Timeout
            if cache[0] is not None and np.array_equal(v,cache[0]):
                return cache[1],cache[2]
            dx = dxm@v; dy = dym@v; rs = rsm@v
            c = np.r_[(dx*dx+dy*dy-rs*rs)*weights,walls@v+off]
            jac = np.concatenate((2*weights[:,None]*(dx[:,None]*dxm+dy[:,None]*dym-rs[:,None]*rsm),walls))
            cache[:] = [v.copy(),c,jac]
            last[0] = v.copy()
            return c,jac
        def fun(v):
            if time.monotonic() >= stop:
                raise Timeout
            return float(obj@v),obj
        def callback(v):
            if time.monotonic() >= stop:
                raise Timeout
            accepted[0] = v.copy()
            count[0] += 1
            if count[0]%12 == 0:
                offer(unpack(v,m))
        try:
            result = minimize(fun,z,jac=True,method='SLSQP',bounds=bounds,
                              constraints={'type':'ineq','fun':lambda v:calc(v)[0],
                                           'jac':lambda v:calc(v)[1]},callback=callback,
                              options={'maxiter':int(params['iterations']),'ftol':7e-14,'disp':False})
            z = result.x
        except Timeout:
            z = accepted[0]
        offer(unpack(z,m))
        if time.monotonic() < deadline-.006:
            offer(unpack(last[0],m))
        return z

    try:
        sm = model(True,deadline)
        fm = model(False,deadline)
        if score_best > 2.36:
            optimize(encode(bestarray),fm,deadline-.02)
            return best
        families = [(5,4,3,4,5),(3,4,3,4,3,4),(4,5,4,5,3)]
        shares = [params['first_share'],params['second_share'],params['third_share']]
        for counts,share in zip(families,shares):
            now = time.monotonic()
            if now >= deadline-.08:
                break
            z = lattice(counts)
            optimize(z,sm,min(deadline-.065,now+float(share)*span))
        if bestarray is not None and time.monotonic() < deadline-.035:
            optimize(encode(bestarray),fm,deadline-.018)
    except Timeout:
        pass
    return best
