import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import lstsq

PARAMS = {
    'symmetric_fraction': (0.53, 0.25, 0.8),
    'iterations': (350, 100, 600),
    'safety': (1.2e-14, 4e-15, 1e-10, 'log')
}
I, J = np.triu_indices(21, 1)

class Halt(Exception):
    pass

def valid(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return False
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return False
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    perimeter = (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r)) +
                 max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)))
    return perimeter <= 2

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + .70*budget_s
    eps = float(params['safety'])
    best = None
    best_a = None
    score = -np.inf
    original = None
    try:
        if init is not None:
            original = (np.asarray(init['circles'], float) if 'circles' in init else
                        np.asarray(init['state'], float).reshape(3,21).T.copy())
            if valid(original):
                best, best_a = init, original.copy()
                score = float(np.sum(original[:,2]))
    except (ValueError, TypeError, KeyError):
        original = None

    def offer(a):
        nonlocal best, best_a, score
        a = np.asarray(a, float).copy()
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
        f = min(1., float(np.min(d/(a[I,2]+a[J,2]))))
        if f <= 0:
            return
        a[:,2] *= f*(1-eps)
        lo = np.min(a[:,:2]-a[:,2,None], axis=0)
        hi = np.max(a[:,:2]+a[:,2,None], axis=0)
        scale = (2-8*eps)/float(np.sum(hi-lo))
        a[:,:2] = (a[:,:2]-(hi+lo)/2)*scale
        a[:,2] *= scale
        value = float(np.sum(a[:,2]))
        if value > score and valid(a):
            best_a, score = a, value
            best = {'state': a.T.reshape(-1).tolist()}

    if original is not None:
        offer(original)
    # Three axial circles, with rows of five and four in each half-plane.
    axis = np.column_stack((np.array([-3.,-1.,1.]), np.zeros(3), np.ones(3)))
    upper = np.array([[x, np.sqrt(3.), 1.] for x in (-4.,-2.,0.,2.,4.)] +
                     [[x, 2*np.sqrt(3.), 1.] for x in (-3.,-1.,1.,3.)])
    lower = upper.copy()
    lower[:,1] *= -1
    symmetric_seed = np.concatenate((axis,upper,lower))
    offer(symmetric_seed)
    if best is None:
        fallback = np.array([[(i%3)*.2,(i//3)*.2,.09] for i in range(21)])
        offer(fallback)
    if time.monotonic() >= end-.04:
        return best

    def centered(a):
        a = a.copy()
        lo = np.min(a[:,:2]-a[:,2,None], axis=0)
        hi = np.max(a[:,:2]+a[:,2,None], axis=0)
        a[:,:2] -= (hi+lo)/2
        return a, float(hi[0]-lo[0])

    def model(sym):
        if sym:
            n = 34
            ids = np.r_[np.arange(12),np.arange(3,12)]
            xi = ids
            yi = np.maximum(12+ids-3,0)
            yi = np.where(ids>=3, yi, 0)
            ys = np.r_[np.zeros(3),np.ones(9),-np.ones(9)]
            ri = 21+ids
            pairs = [(i,j) for i in range(12) for j in range(i+1,12)]
            pairs += [(i,12+j-3) for i in range(3,12) for j in range(i,12)]
            pi,pj = np.asarray(pairs).T
            wall_ids = np.arange(12)
        else:
            n = 64
            xi = np.arange(21)
            yi = 21+xi
            ys = np.ones(21)
            ri = 42+xi
            pi,pj = I,J
            wall_ids = np.arange(21)
        h = len(wall_ids)
        k = np.arange(h)
        b = np.zeros((4*h,n))
        b[k,xi[wall_ids]] = 1
        b[h+k,xi[wall_ids]] = -1
        b[k,ri[wall_ids]] = -1
        b[h+k,ri[wall_ids]] = -1
        b[:2*h,-1] = .5
        b[2*h+k,yi[wall_ids]] = ys[wall_ids]
        b[3*h+k,yi[wall_ids]] = -ys[wall_ids]
        b[2*h+k,ri[wall_ids]] -= 1
        b[3*h+k,ri[wall_ids]] -= 1
        b[2*h:,-1] = -.5
        off = np.r_[np.zeros(2*h),np.ones(2*h)]
        objective = np.zeros(n)
        np.add.at(objective,ri,-1.)
        bounds = [(None,None)]*n
        for q in np.unique(ri):
            if time.monotonic() >= end:
                break
            bounds[int(q)] = (1e-8,.65)
        bounds[-1] = (.12,1.88)
        return n,xi,yi,ys,ri,pi,pj,b,off,objective,bounds

    def pack(a, sym):
        a,w = centered(a)
        if sym:
            return np.r_[a[:12,0],a[3:12,1],a[:12,2],w]
        return np.r_[a.T.reshape(-1),w]

    def solve(z, mod, stop):
        n,xi,yi,ys,ri,pi,pj,b,off,obj,bounds = mod
        rows = np.arange(len(pi))
        weights = 1/np.maximum(2*(z[ri[pi]]+z[ri[pj]]),.07)
        cache = [None,None,None]
        last = [z.copy()]
        feasible = [None,-np.inf]
        count = [0]
        def unpack(v):
            return np.column_stack((v[xi],ys*v[yi],v[ri]))
        def calc(v):
            if time.monotonic() >= stop:
                raise Halt()
            if cache[0] is not None and np.array_equal(cache[0],v):
                return cache[1],cache[2]
            x,y,r = v[xi],ys*v[yi],v[ri]
            dx,dy,rr = x[pi]-x[pj],y[pi]-y[pj],r[pi]+r[pj]
            c = np.r_[(dx*dx+dy*dy-rr*rr)*weights,b@v+off]
            jac = np.zeros((len(rows)+len(b),n))
            tx,ty,tr = 2*weights*dx,2*weights*dy,-2*weights*rr
            jac[rows,xi[pi]] = tx
            np.add.at(jac,(rows,xi[pj]),-tx)
            np.add.at(jac,(rows,yi[pi]),ty*ys[pi])
            np.add.at(jac,(rows,yi[pj]),-ty*ys[pj])
            np.add.at(jac,(rows,ri[pi]),tr)
            np.add.at(jac,(rows,ri[pj]),tr)
            jac[len(rows):] = b
            cache[:] = [v.copy(),c,jac]
            last[0] = v.copy()
            if np.min(c)>-1e-8 and -float(obj@v)>feasible[1]:
                feasible[:] = [v.copy(),-float(obj@v)]
            return c,jac
        def fun(v):
            if time.monotonic() >= stop:
                raise Halt()
            return float(obj@v),obj
        def callback(v):
            if time.monotonic() >= stop:
                raise Halt()
            count[0] += 1
            if count[0]%18 == 0:
                offer(unpack(v))
        try:
            result = minimize(fun,z,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda v:calc(v)[0],
                             'jac':lambda v:calc(v)[1]},callback=callback,
                options={'maxiter':int(params['iterations']),'ftol':5e-14,'disp':False})
            z = result.x
        except Halt:
            z = last[0]
        offer(unpack(z))
        if feasible[0] is not None:
            offer(unpack(feasible[0]))
        for unused in range(3):
            if time.monotonic() >= stop-.008:
                break
            try:
                c,jac = calc(z)
            except Halt:
                break
            active = np.flatnonzero(np.abs(c)<2e-7)
            if len(active)<n:
                break
            delta,_,rank,_ = lstsq(jac[active],-c[active],lapack_driver='gelsy',check_finite=False)
            if rank<n or np.max(np.abs(delta))>3e-4:
                break
            z = z+delta
            offer(unpack(z))
        return z

    full = model(False)
    # Strong incumbents go directly to unrestricted polishing; cold inputs use
    # the much smaller quotient to establish a useful contact graph first.
    if score > 2.30:
        solve(pack(best_a,False),full,end-.018)
        return best
    sym = model(True)
    a = symmetric_seed.copy()
    a[:,:2] -= (np.max(a[:,:2]+a[:,2,None],axis=0)+np.min(a[:,:2]-a[:,2,None],axis=0))/2
    scale = (2-8*eps)/np.sum(np.ptp(a[:,:2],axis=0)+2)
    a *= scale
    symstop = min(end-.035,start+.70*budget_s*float(params['symmetric_fraction']))
    solve(pack(a,True),sym,symstop)
    if time.monotonic() < end-.035:
        solve(pack(best_a,False),full,end-.018)
    return best
