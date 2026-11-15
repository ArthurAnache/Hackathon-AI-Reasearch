import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {
    'iterations': (350, 100, 600),
    'tolerance': (1e-12, 1e-14, 1e-8, 'log'),
    'safety': (1e-14, 4e-15, 1e-10, 'log'),
    'restart_noise': (0.006, 0.001, 0.04)
}
I, J = np.triu_indices(21, 1)
NPAIR = len(I)

class Deadline(Exception):
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
    scale = (2-12*eps)/float(np.sum(hi-lo))
    a[:,:2] = (a[:,:2]-(hi+lo)/2)*scale
    a[:,2] *= scale
    return a if valid(a) else None

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + .70*budget_s
    rng = np.random.default_rng(seed)
    eps = float(params['safety'])
    best = None
    bestscore = -np.inf
    original = None
    try:
        if init is not None:
            original = np.asarray(init['circles'], dtype=float) if 'circles' in init else np.asarray(init['state'], dtype=float).reshape(3,21).T
            if valid(original):
                best = init
                bestscore = float(np.sum(original[:,2]))
    except (KeyError, ValueError, TypeError):
        original = None

    def offer(a):
        nonlocal best, bestscore
        if time.monotonic() >= end:
            return
        b = repair(a, eps)
        if b is not None:
            score = float(np.sum(b[:,2]))
            if score > bestscore:
                bestscore = score
                best = {'state': b.T.reshape(-1).tolist()}

    if original is not None:
        offer(original)
    fallback = np.array([[.2*(k%3), .2*(k//3), .099999999999] for k in range(21)])
    offer(fallback)
    if best is None:
        best = {'circles': fallback.tolist()}
        bestscore = float(np.sum(fallback[:,2]))
    def current():
        return np.asarray(best['circles'],dtype=float).copy() if 'circles' in best else np.asarray(best['state'],dtype=float).reshape(3,21).T.copy()
    if original is None or original.shape != (21,3) or not np.isfinite(original).all() or np.any(original[:,2] <= 0):
        original = current()

    def state(a):
        lo = np.min(a[:,:2]-a[:,2,None], axis=0)
        hi = np.max(a[:,:2]+a[:,2,None], axis=0)
        xy = a[:,:2]-(hi+lo)/2
        return np.r_[xy[:,0],xy[:,1],a[:,2],hi[0]-lo[0]]

    # Full Jacobian consists of six nonzero entries per pair plus linear walls.
    pairrows = np.arange(NPAIR)
    walls = np.zeros((84,64))
    kk = np.arange(21)
    walls[kk,kk] = 1
    walls[21+kk,kk] = -1
    walls[42+kk,21+kk] = 1
    walls[63+kk,21+kk] = -1
    walls[kk,42+kk] = -1
    walls[21+kk,42+kk] = -1
    walls[42+kk,42+kk] = -1
    walls[63+kk,42+kk] = -1
    walls[:42,63] = .5
    walls[42:,63] = -.5
    offset = np.r_[np.zeros(42),np.ones(42)]
    objective = np.r_[np.zeros(42),-np.ones(21),0.]

    def run(z, transform=None, limit=None):
        if limit is None:
            limit = end-.012
        dim = len(z)
        obj = objective if transform is None else objective@transform
        linear = walls if transform is None else walls@transform
        full0 = z if transform is None else transform@z
        weights = 1/np.maximum(2*(full0[42+I]+full0[42+J]), .08)
        cache = [None,None,None]
        last = [z.copy()]
        calls = [0]
        if transform is None:
            selected = np.arange(NPAIR)
            selected_walls = np.arange(84)
        else:
            # Keep one representative from every reflection orbit.
            pairs = set()
            selected_list = []
            mirror = np.r_[np.arange(3),np.arange(12,21),np.arange(3,12)]
            for k in range(NPAIR):
                if time.monotonic() >= limit:
                    return z
                p = (int(I[k]),int(J[k]))
                q = tuple(sorted((int(mirror[I[k]]),int(mirror[J[k]]))))
                key = min(p,q)
                if key not in pairs:
                    pairs.add(key)
                    selected_list.append(k)
            selected = np.asarray(selected_list,dtype=int)
            selected_walls = np.r_[np.arange(12),21+np.arange(12),42+np.arange(12),63+np.arange(12)]
        pi, pj = I[selected], J[selected]
        rows = np.arange(len(selected))
        w = weights[selected]
        lin = linear[selected_walls]
        off = offset[selected_walls]
        def calc(v):
            if time.monotonic() >= limit:
                raise Deadline()
            if cache[0] is not None and np.array_equal(v,cache[0]):
                return cache[1],cache[2]
            u = v if transform is None else transform@v
            dx = u[pi]-u[pj]
            dy = u[21+pi]-u[21+pj]
            rs = u[42+pi]+u[42+pj]
            vals = np.r_[(dx*dx+dy*dy-rs*rs)*w,lin@v+off]
            jac = np.zeros((len(pi),64))
            jac[rows,pi] = 2*w*dx
            jac[rows,pj] = -2*w*dx
            jac[rows,21+pi] = 2*w*dy
            jac[rows,21+pj] = -2*w*dy
            jac[rows,42+pi] = -2*w*rs
            jac[rows,42+pj] = -2*w*rs
            if transform is not None:
                jac = jac@transform
            jac = np.vstack((jac,lin))
            cache[:] = [v.copy(),vals,jac]
            last[0] = v.copy()
            return vals,jac
        def fun(v):
            if time.monotonic() >= limit:
                raise Deadline()
            return float(obj@v),obj
        def callback(v):
            if time.monotonic() >= limit:
                raise Deadline()
            calls[0] += 1
            if calls[0]%8 == 0:
                u = v if transform is None else transform@v
                offer(u[:63].reshape(3,21).T)
        bounds = [(None,None)]*dim
        ridx = range(42,63) if transform is None else range(21,33)
        for k in ridx:
            if time.monotonic() >= limit:
                return z
            bounds[k] = (1e-7,.5)
        bounds[-1] = (.15,1.85)
        try:
            result = minimize(fun,z,jac=True,method='SLSQP',bounds=bounds,
                              constraints={'type':'ineq','fun':lambda v:calc(v)[0], 'jac':lambda v:calc(v)[1]},
                              callback=callback,options={'ftol':float(params['tolerance']),'maxiter':int(params['iterations']),'disp':False})
            z = result.x
        except Deadline:
            z = last[0]
        u = z if transform is None else transform@z
        offer(u[:63].reshape(3,21).T)
        return z

    # No fixed early cutoff: recover the unconstrained contact topology first.
    z = run(state(original))
    if time.monotonic() >= end-.055:
        return best
    a = current()
    choices = []
    for swap in (False,True):
        if time.monotonic() >= end-.045:
            return best
        b = a.copy()
        if swap:
            b[:,:2] = b[:,[1,0]]
        zz = state(b)
        b = zz[:63].reshape(3,21).T.copy()
        reflection = b.copy()
        reflection[:,1] *= -1
        cost = np.sum((b[:,None,:]-reflection[None,:,:])**2,axis=2)
        ri,ci = linear_sum_assignment(cost)
        choices.append((float(cost[ri,ci].sum()),b,zz[-1]))
    _,a,width = min(choices,key=lambda t:t[0])
    singles = np.argsort(np.abs(a[:,1]))[:3]
    rem = np.setdiff1d(np.arange(21),singles)
    order = rem[np.argsort(a[rem,1])]
    neg,pos = order[:9],order[9:]
    cost = ((a[pos,None,0]-a[None,neg,0])**2 + (a[pos,None,1]+a[None,neg,1])**2 + (a[pos,None,2]-a[None,neg,2])**2)
    _,cols = linear_sum_assignment(cost)
    neg = neg[cols]
    base = np.r_[a[singles,0],.5*(a[pos,0]+a[neg,0]), .5*(a[pos,1]-a[neg,1]),a[singles,2],.5*(a[pos,2]+a[neg,2]),width]
    transform = np.zeros((64,34))
    ids = np.r_[np.arange(12),np.arange(3,12)]
    signs = np.r_[np.ones(12),-np.ones(9)]
    transform[np.arange(21),ids] = 1
    mask = ids>=3
    transform[21+np.arange(21)[mask],12+ids[mask]-3] = signs[mask]
    transform[42+np.arange(21),21+ids] = 1
    transform[63,33] = 1
    base = run(base,transform)
    for attempt in range(4):
        if time.monotonic() >= end-.065:
            break
        trial = base.copy()
        amplitude = float(params['restart_noise'])*(1+attempt)
        trial[:21] += rng.normal(0,amplitude,21)
        trial[12:21] = np.abs(trial[12:21])
        candidate = run(trial,transform)
        if objective@(transform@candidate) < objective@(transform@base):
            base = candidate
    return best
