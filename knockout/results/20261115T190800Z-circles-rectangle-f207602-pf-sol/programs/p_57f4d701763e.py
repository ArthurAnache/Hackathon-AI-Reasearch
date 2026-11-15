import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {
    'full_fraction': (0.76, 0.5, 0.95),
    'iterations': (400, 100, 700),
    'safety': (1.2e-14, 4e-15, 1e-10, 'log')
}
I, J = np.triu_indices(21, 1)

class StopSearch(Exception):
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

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + .70*budget_s
    eps = float(params['safety'])
    best = None
    score = -np.inf
    original = None
    try:
        if init is not None:
            original = np.array(init['circles'], dtype=float) if 'circles' in init else np.array(init['state'], dtype=float).reshape(3,21).T
            if valid(original):
                best = init
                score = float(np.sum(original[:,2]))
    except (ValueError, TypeError, KeyError):
        original = None

    def offer(a):
        nonlocal best, score
        a = np.asarray(a, dtype=float).copy()
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
        f = min(1., float(np.min(d/(a[I,2]+a[J,2]))))
        if f <= 0:
            return
        a[:,2] *= f*(1-eps)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        fac = (2-8*eps)/np.sum(hi-lo)
        a[:,:2] = (a[:,:2]-(lo+hi)/2)*fac
        a[:,2] *= fac
        s = float(np.sum(a[:,2]))
        if s > score and valid(a):
            score = s
            best = {'state': a.T.reshape(-1).tolist()}

    if original is not None:
        offer(original)
    fallback = np.array([[2*(k+(.5 if n==4 else 0)), np.sqrt(3.)*j, 1.] for j,n in enumerate((4,5,4,5,3)) for k in range(n)])
    offer(fallback)
    if best is None:
        return {'circles': [[.2*(i%3),.2*(i//3),.099999999999] for i in range(21)]}

    def best_array():
        return np.array(best['circles'],dtype=float) if 'circles' in best else np.array(best['state'],dtype=float).reshape(3,21).T

    def normalize(a):
        a = a.copy()
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        a[:,0] -= lo[0]
        a[:,1] -= (lo[1]+hi[1])/2
        return a, hi[0]-lo[0]

    def optimize(z, symmetric, stop):
        if symmetric:
            n = 34
            ids = np.r_[np.arange(12),np.arange(3,12)]
            xi = ids
            yi = np.maximum(12+ids-3,0)
            ys = np.r_[np.zeros(3),np.ones(9),-np.ones(9)]
            ri = 21+ids
            pairs = [(i,j) for i in range(12) for j in range(i+1,12)]
            pairs += [(i,12+j-3) for i in range(3,12) for j in range(i,12)]
            pi,pj = np.array(pairs).T
            wallids = np.arange(12)
            ridx = np.arange(21,33)
        else:
            n = 64
            xi = np.arange(21)
            yi = 21+xi
            ys = np.ones(21)
            ri = 42+xi
            pi,pj = I,J
            wallids = np.arange(21)
            ridx = np.arange(42,63)
        m = len(pi)
        rows = np.arange(m)
        objective = np.zeros(n)
        np.add.at(objective,ri,-1.)
        wall = np.zeros((4*len(wallids),n))
        k = len(wallids)
        wr = np.arange(k)
        wall[wr,xi[wallids]] = 1
        wall[wr,ri[wallids]] = -1
        wall[k+wr,xi[wallids]] = -1
        wall[k+wr,ri[wallids]] = -1
        wall[k+wr,-1] = 1
        wall[2*k+wr,yi[wallids]] = -ys[wallids]
        wall[2*k+wr,ri[wallids]] = -1
        wall[2*k+wr,-1] = -.5
        wall[3*k+wr,yi[wallids]] = ys[wallids]
        wall[3*k+wr,ri[wallids]] = -1
        wall[3*k+wr,-1] = -.5
        offset = np.r_[np.zeros(2*k),np.ones(2*k)]
        weights = 1/np.maximum(2*(z[ri[pi]]+z[ri[pj]]),.08)
        cache = [None,None,None]
        last = [z.copy()]
        calls = [0]

        def unpack(v):
            return np.column_stack((v[xi],v[yi]*ys,v[ri]))

        def calc(v):
            if time.monotonic() >= stop:
                raise StopSearch()
            if cache[0] is not None and np.array_equal(cache[0],v):
                return cache[1],cache[2]
            dx = v[xi[pi]]-v[xi[pj]]
            dy = v[yi[pi]]*ys[pi]-v[yi[pj]]*ys[pj]
            rs = v[ri[pi]]+v[ri[pj]]
            values = np.r_[(dx*dx+dy*dy-rs*rs)*weights, wall@v+offset]
            jac = np.zeros((m+4*k,n))
            # Repeated column indices occur only in the reflection quotient.
            if symmetric:
                np.add.at(jac,(rows,xi[pi]),2*weights*dx)
                np.add.at(jac,(rows,xi[pj]),-2*weights*dx)
                np.add.at(jac,(rows,yi[pi]),2*weights*dy*ys[pi])
                np.add.at(jac,(rows,yi[pj]),-2*weights*dy*ys[pj])
                np.add.at(jac,(rows,ri[pi]),-2*weights*rs)
                np.add.at(jac,(rows,ri[pj]),-2*weights*rs)
            else:
                jac[rows,xi[pi]] = 2*weights*dx
                jac[rows,xi[pj]] = -2*weights*dx
                jac[rows,yi[pi]] = 2*weights*dy
                jac[rows,yi[pj]] = -2*weights*dy
                jac[rows,ri[pi]] = -2*weights*rs
                jac[rows,ri[pj]] = -2*weights*rs
            jac[m:] = wall
            cache[:] = [v.copy(),values,jac]
            last[0] = v.copy()
            return values,jac

        def fun(v):
            if time.monotonic() >= stop:
                raise StopSearch()
            return float(objective@v),objective

        def callback(v):
            if time.monotonic() >= stop:
                raise StopSearch()
            calls[0] += 1
            if calls[0] % 8 == 0:
                offer(unpack(v))
        bounds = [(None,None)]*n
        for j in ridx:
            if time.monotonic() >= stop:
                return z
            bounds[int(j)] = (1e-7,.6)
        bounds[-1] = (.1,1.9)
        try:
            res = minimize(fun,z,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda v:calc(v)[0],'jac':lambda v:calc(v)[1]},
                callback=callback,options={'maxiter':int(params['iterations']),'ftol':8e-14,'disp':False})
            z = res.x
        except StopSearch:
            z = last[0]
        offer(unpack(z))
        return z

    a = original
    if a is None or a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
        a = best_array()
    a,w = normalize(a)
    z = np.r_[a[:,0],a[:,1],a[:,2],w]
    optimize(z,False,min(deadline-.025,start+.70*budget_s*float(params['full_fraction'])))
    if time.monotonic() >= deadline-.025:
        return best
    a = best_array()
    choices = []
    for swap in (False,True):
        if time.monotonic() >= deadline-.02:
            return best
        b = a.copy()
        if swap:
            b[:,:2] = b[:,[1,0]]
        b,w = normalize(b)
        reflected = b.copy()
        reflected[:,1] *= -1
        cost = np.sum((b[:,None,:]-reflected[None,:,:])**2,axis=2)
        rows,cols = linear_sum_assignment(cost)
        choices.append((float(cost[rows,cols].sum()),b,w))
    _,a,w = min(choices,key=lambda t:t[0])
    singles = np.argsort(np.abs(a[:,1]))[:3]
    mask = np.ones(21,dtype=bool)
    mask[singles] = False
    remaining = np.flatnonzero(mask)
    order = remaining[np.argsort(a[remaining,1])]
    neg,pos = order[:9],order[9:]
    cost = ((a[pos,None,0]-a[None,neg,0])**2 +
            (a[pos,None,1]+a[None,neg,1])**2 +
            (a[pos,None,2]-a[None,neg,2])**2)
    _,cols = linear_sum_assignment(cost)
    neg = neg[cols]
    z = np.r_[a[singles,0],.5*(a[pos,0]+a[neg,0]),.5*(a[pos,1]-a[neg,1]),
              a[singles,2],.5*(a[pos,2]+a[neg,2]),w]
    if time.monotonic() < deadline-.02:
        optimize(z,True,deadline-.008)
    return best
