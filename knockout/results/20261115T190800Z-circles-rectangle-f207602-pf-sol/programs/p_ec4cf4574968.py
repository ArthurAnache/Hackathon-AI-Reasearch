import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment
from scipy.linalg import lstsq

PARAMS = {
    'full_fraction': (0.76, 0.45, 0.9),
    'sym_fraction': (0.78, 0.5, 0.95),
    'iterations': (350, 100, 600),
    'safety': (1.5e-14, 3e-15, 1e-10, 'log')
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
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    return (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r)) +
            max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))) <= 2

def repair(a, eps):
    a = np.asarray(a, dtype=float).copy()
    if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
        return None
    d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
    ratio = min(1., float(np.min(d/(a[I,2]+a[J,2]))))
    if ratio <= 0:
        return None
    a[:,2] *= ratio*(1-eps)
    lo = np.min(a[:,:2]-a[:,2,None], axis=0)
    hi = np.max(a[:,:2]+a[:,2,None], axis=0)
    s = (2-8*eps)/float(np.sum(hi-lo))
    a[:,:2] = (a[:,:2]-(lo+hi)/2)*s
    a[:,2] *= s
    return a if valid(a) else None

def model(sym):
    if sym:
        n = 34
        ids = np.r_[np.arange(12),np.arange(3,12)]
        signs = np.r_[np.ones(12),-np.ones(9)]
        xm = np.zeros((21,n)); ym = xm.copy(); rm = xm.copy()
        xm[np.arange(21),ids] = 1
        ix = np.flatnonzero(ids >= 3)
        ym[ix,12+ids[ix]-3] = signs[ix]
        rm[np.arange(21),21+ids] = 1
        pairs = [(i,j) for i in range(12) for j in range(i+1,12)]
        pairs += [(i,12+j-3) for i in range(3,12) for j in range(i,12)]
        pi,pj = np.array(pairs).T
        boundary_ids = np.arange(12)
        ridx = np.arange(21,33)
    else:
        n = 64
        xm = np.zeros((21,n)); ym = xm.copy(); rm = xm.copy()
        xm[np.arange(21),np.arange(21)] = 1
        ym[np.arange(21),21+np.arange(21)] = 1
        rm[np.arange(21),42+np.arange(21)] = 1
        pi,pj = I,J
        boundary_ids = np.arange(21)
        ridx = np.arange(42,63)
    w = np.zeros(n); w[-1] = 1
    b = np.concatenate((xm[boundary_ids]-rm[boundary_ids],
                        w-xm[boundary_ids]-rm[boundary_ids],
                        -.5*w-ym[boundary_ids]-rm[boundary_ids],
                        -.5*w+ym[boundary_ids]-rm[boundary_ids]))
    off = np.r_[np.zeros(2*len(boundary_ids)),np.ones(2*len(boundary_ids))]
    return (xm,ym,rm,pi,pj,xm[pi]-xm[pj],ym[pi]-ym[pj],rm[pi]+rm[pj],b,off,-rm.sum(axis=0),ridx)

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + .70*budget_s
    eps = float(params['safety'])
    best = None
    score = -np.inf
    original = None
    try:
        if init is not None:
            original = np.asarray(init['circles'],float) if 'circles' in init else np.asarray(init['state'],float).reshape(3,21).T
            if valid(original):
                best = init
                score = float(np.sum(original[:,2]))
    except (ValueError,TypeError,KeyError):
        original = None

    def offer(a):
        nonlocal best,score
        b = repair(a,eps)
        if b is not None:
            s = float(np.sum(b[:,2]))
            if s > score:
                score = s
                best = {'state': b.T.reshape(-1).tolist()}
        return b

    if original is not None:
        offer(original)
    lattice = []
    for row,count in enumerate((4,5,4,5,3)):
        if time.monotonic() >= deadline:
            break
        for col in range(count):
            if time.monotonic() >= deadline:
                break
            lattice.append([2*(col+(.5 if count == 4 else 0)),np.sqrt(3.)*row,1.])
    if len(lattice) == 21:
        offer(lattice)
    if best is None:
        return {'circles': [[(i%3)*.2,(i//3)*.2,.099999999999] for i in range(21)]}

    def best_array():
        return np.asarray(best['circles'],float).copy() if 'circles' in best else np.asarray(best['state'],float).reshape(3,21).T.copy()

    def state(a):
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        return np.r_[a[:,0]-lo[0],a[:,1]-(hi[1]+lo[1])/2,a[:,2],hi[0]-lo[0]]

    full = model(False)
    quotient = model(True)

    def unpack(z,m):
        if len(z) == 64:
            return z[:63].reshape(3,21).T.copy()
        return np.column_stack((m[0]@z,m[1]@z,m[2]@z))

    def optimize(z,m,stop):
        xm,ym,rm,pi,pj,dxm,dym,rsm,b,off,obj,ridx = m
        weight = 1/np.maximum(2*(rsm@z),.08)
        last = [z.copy()]
        cache = [None,None,None]
        count = [0]
        n = len(z)
        def calc(v):
            if time.monotonic() >= stop:
                raise Timeout()
            if cache[0] is not None and np.array_equal(v,cache[0]):
                return cache[1],cache[2]
            if n == 64:
                dx = v[pi]-v[pj]
                dy = v[21+pi]-v[21+pj]
                rs = v[42+pi]+v[42+pj]
            else:
                dx = dxm@v; dy = dym@v; rs = rsm@v
            c = np.r_[(dx*dx+dy*dy-rs*rs)*weight,b@v+off]
            jac = np.empty((len(c),n))
            if n == 64:
                jac[:len(pi)] = 0
                rows = np.arange(len(pi))
                jac[rows,pi] = 2*weight*dx
                jac[rows,pj] = -2*weight*dx
                jac[rows,21+pi] = 2*weight*dy
                jac[rows,21+pj] = -2*weight*dy
                jac[rows,42+pi] = -2*weight*rs
                jac[rows,42+pj] = -2*weight*rs
            else:
                jac[:len(pi)] = 2*weight[:,None]*(dx[:,None]*dxm+dy[:,None]*dym-rs[:,None]*rsm)
            jac[len(pi):] = b
            cache[:] = [v.copy(),c,jac]
            return c,jac
        def fun(v):
            if time.monotonic() >= stop:
                raise Timeout()
            return float(obj@v),obj
        def callback(v):
            if time.monotonic() >= stop:
                raise Timeout()
            last[0] = v.copy()
            count[0] += 1
            if count[0] % 10 == 0:
                offer(unpack(v,m))
        bounds = [(None,None)]*n
        for k in ridx:
            if time.monotonic() >= stop:
                return z
            bounds[int(k)] = (1e-7,.5)
        bounds[-1] = (.15,1.85)
        try:
            result = minimize(fun,z,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda v:calc(v)[0],'jac':lambda v:calc(v)[1]},
                callback=callback,options={'maxiter':int(params['iterations']),'ftol':8e-14,'disp':False})
            z = result.x
        except Timeout:
            z = last[0]
        offer(unpack(z,m))
        for _ in range(3):
            if time.monotonic() >= stop-.006:
                break
            try:
                c,jac = calc(z)
            except Timeout:
                break
            active = np.flatnonzero(np.abs(c)<2e-7)
            if len(active)<n:
                break
            delta,_,rank,_ = lstsq(jac[active],-c[active],lapack_driver='gelsy',check_finite=False)
            if rank<n or np.max(np.abs(delta))>2e-4:
                break
            z = z+delta
            offer(unpack(z,m))
        return z

    a = original if original is not None else best_array()
    if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2]<=0):
        a = best_array()
    stop = min(deadline-.025,start+.70*budget_s*float(params['full_fraction']))
    z = optimize(state(a),full,stop)
    if time.monotonic() >= deadline-.035:
        return best
    # Project the recovered geometry, not an unrelated fallback lattice.
    recovered = repair(unpack(z,full),eps)
    a = recovered if recovered is not None and float(np.sum(recovered[:,2])) >= score-.025 else best_array()
    choices = []
    for swap in (False,True):
        if time.monotonic() >= deadline-.025:
            return best
        b = a.copy()
        if swap:
            b[:,:2] = b[:,[1,0]]
        lo = np.min(b[:,:2]-b[:,2,None],axis=0)
        hi = np.max(b[:,:2]+b[:,2,None],axis=0)
        b[:,0] -= lo[0]
        b[:,1] -= (lo[1]+hi[1])/2
        refl = b.copy(); refl[:,1] *= -1
        cost = np.sum((b[:,None,:]-refl[None,:,:])**2,axis=2)
        rows,cols = linear_sum_assignment(cost)
        choices.append((float(cost[rows,cols].sum()),b,hi[0]-lo[0]))
    _,a,width = min(choices,key=lambda item:item[0])
    singles = np.argsort(np.abs(a[:,1]))[:3]
    mask = np.ones(21,dtype=bool); mask[singles] = False
    rem = np.flatnonzero(mask)
    ordered = rem[np.argsort(a[rem,1])]
    neg,pos = ordered[:9],ordered[9:]
    cost = ((a[pos,None,0]-a[None,neg,0])**2+(a[pos,None,1]+a[None,neg,1])**2+(a[pos,None,2]-a[None,neg,2])**2)
    _,cols = linear_sum_assignment(cost)
    neg = neg[cols]
    base = np.r_[a[singles,0],.5*(a[pos,0]+a[neg,0]),.5*(a[pos,1]-a[neg,1]),a[singles,2],.5*(a[pos,2]+a[neg,2]),width]
    now = time.monotonic()
    symstop = min(deadline-.015,now+float(params['sym_fraction'])*(deadline-now))
    base = optimize(base,quotient,symstop)
    if time.monotonic() < deadline-.045:
        a = best_array()
        optimize(state(a),full,deadline-.012)
    return best
