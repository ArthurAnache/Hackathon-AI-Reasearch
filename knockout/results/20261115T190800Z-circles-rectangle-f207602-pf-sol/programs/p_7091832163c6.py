import time
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linprog, linear_sum_assignment

PARAMS = {
    'linear_fraction': (0.34, 0.1, 0.6),
    'linear_steps': (70, 10, 160),
    'iterations': (220, 80, 400),
    'safety': (1e-14, 4e-15, 1e-10, 'log')
}
I, J = np.triu_indices(21, 1)

class StopSearch(Exception):
    pass

def valid(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return False
    d = np.sqrt((a[I, 0]-a[J, 0])**2+(a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return False
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    return (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))+
            max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))) <= 2

def repair(a, eps):
    a = np.asarray(a, float).copy()
    if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
        return None
    d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
    ratio = min(1., float(np.min(d/(a[I,2]+a[J,2]))))
    if ratio <= 0:
        return None
    a[:,2] *= ratio*(1-eps)
    lo = np.min(a[:,:2]-a[:,2,None], axis=0)
    hi = np.max(a[:,:2]+a[:,2,None], axis=0)
    scale = (2-8*eps)/np.sum(hi-lo)
    a[:,:2] = (a[:,:2]-(hi+lo)/2)*scale
    a[:,2] *= scale
    return a if valid(a) else None

def model(sym):
    if sym:
        n = 34
        ids = np.r_[np.arange(12), np.arange(3,12)]
        signs = np.r_[np.ones(12), -np.ones(9)]
        x = np.zeros((21,n)); y = x.copy(); r = x.copy()
        x[np.arange(21),ids] = 1
        mask = ids >= 3
        y[np.arange(21)[mask],12+ids[mask]-3] = signs[mask]
        r[np.arange(21),21+ids] = 1
        pairs = [(i,j) for i in range(12) for j in range(i+1,12)]
        pairs += [(i,12+j-3) for i in range(3,12) for j in range(i,12)]
        pi,pj = np.array(pairs).T
        boundary = np.arange(12)
        ridx = np.arange(21,33)
    else:
        n = 64
        x = np.zeros((21,n)); y = x.copy(); r = x.copy()
        x[np.arange(21),np.arange(21)] = 1
        y[np.arange(21),21+np.arange(21)] = 1
        r[np.arange(21),42+np.arange(21)] = 1
        pi,pj = I,J
        boundary = np.arange(21)
        ridx = np.arange(42,63)
    w = np.zeros(n); w[-1] = 1
    b = np.concatenate((x[boundary]-r[boundary],w-x[boundary]-r[boundary],
                        -.5*w-y[boundary]-r[boundary],-.5*w+y[boundary]-r[boundary]))
    off = np.r_[np.zeros(2*len(boundary)),np.ones(2*len(boundary))]
    return x,y,r,x[pi]-x[pj],y[pi]-y[pj],r[pi]+r[pj],b,off,-r.sum(axis=0),ridx

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + .70*budget_s
    eps = float(params['safety'])
    best = None
    bestscore = -np.inf
    original = None
    try:
        if init is not None:
            original = (np.asarray(init['circles'],float) if 'circles' in init else
                        np.asarray(init['state'],float).reshape(3,21).T.copy())
            if valid(original):
                best = init
                bestscore = float(np.sum(original[:,2]))
    except (ValueError, TypeError, KeyError):
        original = None

    def offer(a):
        nonlocal best,bestscore
        b = repair(a,eps)
        if b is not None:
            score = float(np.sum(b[:,2]))
            if score > bestscore:
                bestscore = score
                best = {'state':b.T.reshape(-1).tolist()}
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
            lattice.append([2*(col+(.5 if count==4 else 0)),np.sqrt(3.)*row,1.])
    if len(lattice)==21:
        offer(lattice)
    if best is None:
        return {'circles':[[.2*(i%3),.2*(i//3),.09] for i in range(21)]}

    def best_array():
        return (np.asarray(best['circles'],float).copy() if 'circles' in best else
                np.asarray(best['state'],float).reshape(3,21).T.copy())

    def full_state(a):
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        p = a[:,:2]-lo
        p[:,1] -= .5*(hi[1]-lo[1])
        return np.r_[p[:,0],p[:,1],a[:,2],hi[0]-lo[0]]

    full = model(False)
    def unpack(z,m):
        return np.column_stack((m[0]@z,m[1]@z,m[2]@z))

    a = original
    if a is None or a.shape!=(21,3) or not np.isfinite(a).all() or np.any(a[:,2]<=0):
        a = best_array()
    else:
        fixed = repair(a,eps)
        a = fixed if fixed is not None else best_array()
    z = full_state(a)
    x,y,r,dxm,dym,rsm,b,off,obj,ridx = full
    bounds = [(None,None)]*64
    for k in ridx:
        if time.monotonic() >= deadline:
            return best
        bounds[int(k)] = (1e-7,.5)
    bounds[-1] = (.15,1.85)
    lpstop = start + float(params['linear_fraction'])*.70*budget_s
    lastscore = -np.inf
    for it in range(int(params['linear_steps'])):
        if time.monotonic() >= lpstop:
            break
        dx = dxm@z; dy = dym@z
        d = np.maximum(np.hypot(dx,dy),1e-15)
        planes = (dx/d)[:,None]*dxm+(dy/d)[:,None]*dym-rsm
        amat = -np.concatenate((planes,b))
        rhs = np.r_[np.zeros(len(dx)),off]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = linprog(obj,A_ub=amat,b_ub=rhs,bounds=bounds,method='highs-ds',
                             options={'threads':1,'time_limit':max(.001,lpstop-time.monotonic()),
                                      'dual_feasibility_tolerance':1e-8,
                                      'primal_feasibility_tolerance':1e-9})
        if not result.success:
            break
        z = result.x
        score = -float(obj@z)
        if it%5==0 or score-lastscore < 1e-9:
            offer(unpack(z,full))
        if abs(score-lastscore)<2e-10:
            break
        lastscore = score
    offer(unpack(z,full))

    def optimize(z,m,stop):
        x,y,r,dxm,dym,rsm,b,off,obj,ridx = m
        weights = 1/np.maximum(2*(rsm@z),.08)
        cache = [None,None,None]
        accepted = [z.copy()]
        last = [z.copy()]
        count = [0]
        def calc(v):
            if time.monotonic() >= stop:
                raise StopSearch
            if cache[0] is not None and np.array_equal(cache[0],v):
                return cache[1],cache[2]
            dx = dxm@v; dy = dym@v; rs = rsm@v
            c = np.r_[(dx*dx+dy*dy-rs*rs)*weights,b@v+off]
            jac = np.concatenate((2*weights[:,None]*(dx[:,None]*dxm+dy[:,None]*dym-rs[:,None]*rsm),b))
            cache[:] = [v.copy(),c,jac]
            last[0] = v.copy()
            return c,jac
        def fun(v):
            if time.monotonic() >= stop:
                raise StopSearch
            return float(obj@v),obj
        def callback(v):
            if time.monotonic() >= stop:
                raise StopSearch
            accepted[0] = v.copy()
            count[0] += 1
            if count[0]%12==0:
                offer(unpack(v,m))
        bounds = [(None,None)]*len(z)
        for k in ridx:
            if time.monotonic() >= stop:
                return z
            bounds[int(k)] = (1e-7,.5)
        bounds[-1] = (.15,1.85)
        try:
            result = minimize(fun,z,jac=True,method='SLSQP',bounds=bounds,
                              constraints={'type':'ineq','fun':lambda v:calc(v)[0],
                                           'jac':lambda v:calc(v)[1]},callback=callback,
                              options={'maxiter':int(params['iterations']),'ftol':6e-14,'disp':False})
            z = result.x
        except StopSearch:
            z = accepted[0]
        offer(unpack(z,m))
        offer(unpack(last[0],m))
        return z

    # The nonlinear solve is uninterrupted: symmetry is attempted only after convergence.
    if time.monotonic() < deadline-.05:
        z = optimize(z,full,deadline-.09)
    if time.monotonic() >= deadline-.065:
        return best
    a = unpack(z,full)
    if float(np.sum(a[:,2])) < bestscore-1e-5:
        a = best_array()
    choices = []
    for swap in (False,True):
        if time.monotonic() >= deadline-.045:
            return best
        c = a.copy()
        if swap:
            c[:,:2] = c[:,[1,0]]
        lo = np.min(c[:,:2]-c[:,2,None],axis=0)
        hi = np.max(c[:,:2]+c[:,2,None],axis=0)
        c[:,0] -= lo[0]
        c[:,1] -= .5*(lo[1]+hi[1])
        reflected = c.copy(); reflected[:,1] *= -1
        cost = np.sum((c[:,None,:]-reflected[None,:,:])**2,axis=2)
        rr,cc = linear_sum_assignment(cost)
        choices.append((float(cost[rr,cc].sum()),c,hi[0]-lo[0]))
    _,a,width = min(choices,key=lambda v:v[0])
    singles = np.argsort(np.abs(a[:,1]))[:3]
    mask = np.ones(21,dtype=bool); mask[singles] = False
    rem = np.flatnonzero(mask)
    ordered = rem[np.argsort(a[rem,1])]
    neg,pos = ordered[:9],ordered[9:]
    cost = ((a[pos,None,0]-a[None,neg,0])**2+(a[pos,None,1]+a[None,neg,1])**2+
            (a[pos,None,2]-a[None,neg,2])**2)
    _,cols = linear_sum_assignment(cost)
    neg = neg[cols]
    q = np.r_[a[singles,0],.5*(a[pos,0]+a[neg,0]),.5*(a[pos,1]-a[neg,1]),
              a[singles,2],.5*(a[pos,2]+a[neg,2]),width]
    if time.monotonic() < deadline-.035:
        optimize(q,model(True),deadline-.012)
    return best
