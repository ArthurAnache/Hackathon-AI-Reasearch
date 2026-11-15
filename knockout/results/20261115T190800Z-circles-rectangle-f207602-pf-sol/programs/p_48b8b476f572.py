import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment
from scipy.linalg import lstsq

PARAMS = {
    'full_fraction': (0.94, 0.6, 1.0),
    'maxiter': (350, 100, 600),
    'safety': (1.5e-14, 5e-15, 1e-10, 'log'),
    'restart_scale': (0.004, 0.001, 0.04)
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
    x = [Fraction(float(t)) for t in a[:, 0]]
    y = [Fraction(float(t)) for t in a[:, 1]]
    r = [Fraction(float(t)) for t in a[:, 2]]
    return (max(t+s for t,s in zip(x,r))-min(t-s for t,s in zip(x,r)) +
            max(t+s for t,s in zip(y,r))-min(t-s for t,s in zip(y,r))) <= 2

def repair(a, eps):
    a = np.asarray(a, dtype=float).copy()
    if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
        return None
    d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
    f = min(1., float(np.min(d/(a[I,2]+a[J,2]))))
    if f <= 0:
        return None
    a[:,2] *= f*(1-eps)
    lo = np.min(a[:,:2]-a[:,2,None],axis=0)
    hi = np.max(a[:,:2]+a[:,2,None],axis=0)
    s = (2-8*eps)/float(np.sum(hi-lo))
    a[:,:2] = (a[:,:2]-(lo+hi)/2)*s
    a[:,2] *= s
    return a if valid(a) else None

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + .70*budget_s
    rng = np.random.default_rng(seed)
    eps = float(params['safety'])
    best = None
    score = -np.inf
    best_a = None
    original = None
    try:
        if init is not None:
            original = np.asarray(init['circles'],float) if 'circles' in init else np.asarray(init['state'],float).reshape(3,21).T.copy()
            if valid(original):
                best, best_a = init, original.copy()
                score = float(np.sum(original[:,2]))
    except (ValueError, KeyError, TypeError):
        original = None

    def offer(a):
        nonlocal best, score, best_a
        b = repair(a,eps)
        if b is not None:
            v = float(np.sum(b[:,2]))
            if v > score:
                score = v
                best_a = b
                best = {'state': b.T.reshape(-1).tolist()}
        return b

    if original is not None:
        offer(original)
    lattice = []
    for row, count in enumerate((4,5,4,5,3)):
        if time.monotonic() >= end:
            break
        for col in range(count):
            if time.monotonic() >= end:
                break
            lattice.append([2*(col+(.5 if count == 4 else 0)),np.sqrt(3.)*row,1.])
    if len(lattice)==21:
        offer(lattice)
    if best is None:
        fallback = np.array([[(i%3)*.2,(i//3)*.2,.099] for i in range(21)])
        offer(fallback)
    if best is None:
        return {'circles': [[(i%3)*.2,(i//3)*.2,.09] for i in range(21)]}

    def state(a):
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        return np.r_[a[:,0]-(hi[0]+lo[0])/2,a[:,1]-(hi[1]+lo[1])/2,a[:,2],hi[0]-lo[0]]

    def array(v):
        return v[:63].reshape(3,21).T

    # The model is expressed in centered coordinates. Every Jacobian entry is
    # assembled explicitly rather than by dense coordinate-matrix products.
    wall = np.zeros((84,64))
    k = np.arange(21)
    wall[k,k] = -1
    wall[k+21,k] = 1
    wall[k+42,k+21] = -1
    wall[k+63,k+21] = 1
    wall[k,k+42] = -1
    wall[k+21,k+42] = -1
    wall[k+42,k+42] = -1
    wall[k+63,k+42] = -1
    wall[:42,63] = .5
    wall[42:,63] = -.5
    offset = np.r_[np.zeros(42),np.ones(42)]
    obj = np.r_[np.zeros(42),-np.ones(21),0.]
    rows = np.arange(210)

    def solve(z, stop, transform=None):
        if time.monotonic() >= stop:
            return z
        if transform is None:
            expand = lambda v: v
            wjac = wall
            objective = obj
            bounds = [(None,None)]*42 + [(1e-8,.5)]*21 + [(.15,1.85)]
        else:
            expand = lambda v: transform @ v
            wjac = wall @ transform
            objective = obj @ transform
            bounds = [(None,None)]*21 + [(1e-8,.5)]*12 + [(.15,1.85)]
        zz = expand(z)
        weights = 1/np.maximum(2*(zz[42+I]+zz[42+J]),.06)
        cache = [None,None,None]
        last = [z.copy()]
        feasible = [None, -np.inf]
        iterations = [0]

        def calc(v):
            if time.monotonic() >= stop:
                raise Halt()
            if cache[0] is not None and np.array_equal(cache[0],v):
                return cache[1],cache[2]
            q = expand(v)
            dx = q[I]-q[J]
            dy = q[21+I]-q[21+J]
            rs = q[42+I]+q[42+J]
            values = np.r_[(dx*dx+dy*dy-rs*rs)*weights,wall@q+offset]
            jac = np.zeros((210,64))
            ax = 2*weights*dx
            ay = 2*weights*dy
            ar = -2*weights*rs
            jac[rows,I] = ax
            jac[rows,J] = -ax
            jac[rows,21+I] = ay
            jac[rows,21+J] = -ay
            jac[rows,42+I] = ar
            jac[rows,42+J] = ar
            if transform is not None:
                jac = jac @ transform
            jac = np.concatenate((jac,wjac),axis=0)
            cache[:] = [v.copy(),values,jac]
            last[0] = v.copy()
            if np.min(values) > -2e-8:
                value = -float(objective@v)
                if value > feasible[1]:
                    feasible[:] = [v.copy(),value]
            return values,jac

        def fun(v):
            if time.monotonic() >= stop:
                raise Halt()
            return float(objective@v),objective

        def callback(v):
            if time.monotonic() >= stop:
                raise Halt()
            iterations[0] += 1
            if iterations[0] % 18 == 0:
                offer(array(expand(v)))

        try:
            result = minimize(fun,z,method='SLSQP',jac=True,bounds=bounds,
                constraints={'type':'ineq','fun':lambda v:calc(v)[0], 'jac':lambda v:calc(v)[1]},
                callback=callback, options={'maxiter':int(params['maxiter']),'ftol':8e-14,'disp':False})
            z = result.x
        except Halt:
            z = last[0]
        offer(array(expand(z)))
        if feasible[0] is not None:
            offer(array(expand(feasible[0])))
        # Newton polish only a numerically determined full-rank contact set.
        for _ in range(3):
            if time.monotonic() >= stop-.008:
                break
            try:
                c,jac = calc(z)
            except Halt:
                break
            active = np.flatnonzero(np.abs(c)<2e-7)
            if len(active)<len(z):
                break
            delta,_,rank,_ = lstsq(jac[active],-c[active],lapack_driver='gelsy',check_finite=False)
            if rank<len(z) or np.max(np.abs(delta))>2e-4:
                break
            z = z+delta
            offer(array(expand(z)))
        return z

    a = original
    if a is None or a.shape!=(21,3) or not np.isfinite(a).all() or np.any(a[:,2]<=0):
        a = best_a
    fullstop = min(end-.015,start+.70*budget_s*float(params['full_fraction']))
    z = solve(state(a),fullstop)
    if time.monotonic() >= end-.045:
        return best

    # Infer reflection only after unrestricted topology recovery has converged.
    choices = []
    for swap in (False,True):
        if time.monotonic() >= end-.035:
            return best
        b = best_a.copy()
        if swap:
            b[:,:2] = b[:,[1,0]]
        v = state(b)
        b = array(v).copy()
        reflected = b.copy()
        reflected[:,1] *= -1
        cost = np.sum((b[:,None,:]-reflected[None,:,:])**2,axis=2)
        rr,cc = linear_sum_assignment(cost)
        choices.append((float(cost[rr,cc].sum()),b,v[63]))
    _,a,width = min(choices,key=lambda x:x[0])
    singles = np.argsort(np.abs(a[:,1]))[:3]
    mask = np.ones(21,bool)
    mask[singles] = False
    rem = np.flatnonzero(mask)
    rem = rem[np.argsort(a[rem,1])]
    neg,pos = rem[:9],rem[9:]
    costs = ((a[pos,None,0]-a[None,neg,0])**2 + (a[pos,None,1]+a[None,neg,1])**2 + (a[pos,None,2]-a[None,neg,2])**2)
    _,perm = linear_sum_assignment(costs)
    neg = neg[perm]
    base = np.r_[a[singles,0],.5*(a[pos,0]+a[neg,0]),.5*(a[pos,1]-a[neg,1]),a[singles,2],.5*(a[pos,2]+a[neg,2]),width]
    transform = np.zeros((64,34))
    ids = np.r_[np.arange(12),np.arange(3,12)]
    signs = np.r_[np.ones(12),-np.ones(9)]
    transform[np.arange(21),ids] = 1
    mask = ids>=3
    transform[21+np.arange(21)[mask],12+ids[mask]-3] = signs[mask]
    transform[42+np.arange(21),21+ids] = 1
    transform[63,33] = 1
    base = solve(base,end-.014,transform)
    for attempt in range(4):
        if time.monotonic() >= end-.05:
            break
        trial = base.copy()
        trial[:21] += rng.normal(0,float(params['restart_scale']),21)
        base = solve(trial,end-.014,transform)
    return best
