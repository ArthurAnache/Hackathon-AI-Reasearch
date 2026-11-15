import time
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linprog, linear_sum_assignment
from scipy.linalg import lstsq

PARAMS = {
    'amplitude': (0.032, 0.003, 0.12),
    'attempts': (8, 2, 20),
    'trial_seconds': (0.19, 0.06, 0.4),
    'safety': (7e-15, 2e-15, 1e-11, 'log')
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
    x, y, r = ([Fraction(float(t)) for t in a[:, k]] for k in range(3))
    return (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r)) +
            max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))) <= 2

def repair(a, eps):
    a = np.array(a, dtype=float, copy=True)
    if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
        return None
    d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
    ratio = min(1., float(np.min(d/(a[I,2]+a[J,2]))))
    if ratio <= 0:
        return None
    a[:,2] *= ratio*(1-eps)
    lo = np.min(a[:,:2]-a[:,2,None], axis=0)
    hi = np.max(a[:,:2]+a[:,2,None], axis=0)
    scale = (2-8*eps)/float(np.sum(hi-lo))
    a[:,:2] = (a[:,:2]-(lo+hi)/2)*scale
    a[:,2] *= scale
    return a if valid(a) else None

def make_model(sym):
    n = 34 if sym else 64
    x = np.zeros((21,n)); y = x.copy(); r = x.copy()
    if sym:
        ids = np.r_[np.arange(12),np.arange(3,12)]
        signs = np.r_[np.ones(12),-np.ones(9)]
        x[np.arange(21),ids] = 1
        mask = ids >= 3
        y[np.arange(21)[mask],9+ids[mask]] = signs[mask]
        r[np.arange(21),21+ids] = 1
        pairs = [(i,j) for i in range(12) for j in range(i+1,12)]
        pairs += [(i,12+j-3) for i in range(3,12) for j in range(i,12)]
        pi,pj = np.array(pairs).T
        wall = np.arange(12)
        ri = np.arange(21,33)
    else:
        x[np.arange(21),np.arange(21)] = 1
        y[np.arange(21),21+np.arange(21)] = 1
        r[np.arange(21),42+np.arange(21)] = 1
        pi,pj = I,J
        wall = np.arange(21)
        ri = np.arange(42,63)
    w = np.zeros(n); w[-1] = 1
    b = np.concatenate((x[wall]-r[wall],w-x[wall]-r[wall],
                        -.5*w-y[wall]-r[wall],-.5*w+y[wall]-r[wall]))
    off = np.r_[np.zeros(2*len(wall)),np.ones(2*len(wall))]
    bounds = [(None,None)]*n
    for k in ri:
        bounds[int(k)] = (1e-6,.5)
    bounds[-1] = (.2,1.8)
    return x,y,r,x[pi]-x[pj],y[pi]-y[pj],r[pi]+r[pj],b,off,-r.sum(axis=0),bounds

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + .70*budget_s
    rng = np.random.default_rng(seed)
    eps = float(params['safety'])
    best = init
    bestscore = -np.inf
    a = None
    if init is not None:
        try:
            a = (np.asarray(init['circles'],float) if 'circles' in init else
                 np.asarray(init['state'],float).reshape(3,21).T.copy())
            if valid(a):
                bestscore = float(np.sum(a[:,2]))
            else:
                a = None
        except (KeyError,ValueError,TypeError):
            a = None
    def offer(c):
        nonlocal best,bestscore
        if time.monotonic() >= deadline:
            return
        fixed = repair(c,eps)
        if fixed is not None:
            score = float(np.sum(fixed[:,2]))
            if score > bestscore:
                bestscore = score
                best = {'state':fixed.T.reshape(-1).tolist()}
    if a is None:
        a = np.array([[.19*(i%3),.19*(i//3),.094] for i in range(21)])
        offer(a)
    full = make_model(False)
    sym = make_model(True)
    def unpack(z,m):
        return np.column_stack((m[0]@z,m[1]@z,m[2]@z))
    def encode(c):
        lo = np.min(c[:,:2]-c[:,2,None],axis=0)
        hi = np.max(c[:,:2]+c[:,2,None],axis=0)
        p = c[:,:2]-lo
        p[:,1] -= .5*(hi[1]-lo[1])
        return np.r_[p[:,0],p[:,1],c[:,2],hi[0]-lo[0]]

    # A conservative tangent LP can remove residual numerical slack cheaply.
    z = encode(a)
    lpstop = min(deadline-.07,started+.13*budget_s)
    for iteration in range(3):
        if time.monotonic() >= lpstop:
            break
        x,y,r,dxm,dym,rsm,b,off,obj,bounds = full
        dx = dxm@z; dy = dym@z
        d = np.maximum(np.hypot(dx,dy),1e-15)
        planes = (dx/d)[:,None]*dxm+(dy/d)[:,None]*dym-rsm
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = linprog(obj,A_ub=-np.concatenate((planes,b)),
                             b_ub=np.r_[np.zeros(len(d)),off],bounds=bounds,
                             method='highs-ds',options={'threads':1,
                             'time_limit':max(.001,lpstop-time.monotonic()),
                             'primal_feasibility_tolerance':1e-9,
                             'dual_feasibility_tolerance':1e-9})
        if not result.success:
            break
        old = z
        z = result.x
        offer(unpack(z,full))
        if np.max(np.abs(z-old)) < 1e-9:
            break

    def optimize(z,m,stop,maxiter=100):
        x,y,r,dxm,dym,rsm,b,off,obj,bounds = m
        weights = 1/np.maximum(2*(rsm@z),.08)
        cache = [None,None,None]
        last = [z.copy()]
        def calc(v):
            if time.monotonic() >= stop:
                raise StopSearch
            if cache[0] is not None and np.array_equal(cache[0],v):
                return cache[1],cache[2]
            dx = dxm@v; dy = dym@v; rs = rsm@v
            c = np.r_[(dx*dx+dy*dy-rs*rs)*weights,b@v+off]
            jac = np.concatenate((2*weights[:,None]*(dx[:,None]*dxm+dy[:,None]*dym-rs[:,None]*rsm),b))
            cache[:] = [v.copy(),c,jac]
            return c,jac
        def fun(v):
            if time.monotonic() >= stop:
                raise StopSearch
            return float(obj@v),obj
        def callback(v):
            if time.monotonic() >= stop:
                raise StopSearch
            last[0] = v.copy()
        try:
            result = minimize(fun,z,jac=True,method='SLSQP',bounds=bounds,
                              constraints={'type':'ineq','fun':lambda v:calc(v)[0],
                                           'jac':lambda v:calc(v)[1]},callback=callback,
                              options={'maxiter':maxiter,'ftol':4e-14,'disp':False})
            z = result.x
        except StopSearch:
            z = last[0]
        offer(unpack(z,m))
        for k in range(3):
            if time.monotonic() >= stop-.004:
                break
            try:
                c,jac = calc(z)
            except StopSearch:
                break
            active = np.flatnonzero(np.abs(c)<2e-8)
            if len(active) < len(z):
                break
            delta,_,rank,_ = lstsq(jac[active],-c[active],lapack_driver='gelsy',check_finite=False)
            if rank < len(z) or np.max(np.abs(delta)) > 2e-5:
                break
            z += delta
            offer(unpack(z,m))
        return z

    # Identify the physical reflection axis before quotienting.
    choices = []
    for swap in (False,True):
        if time.monotonic() >= deadline-.04:
            return best
        c = a.copy()
        if swap:
            c[:,:2] = c[:,[1,0]]
        lo = np.min(c[:,:2]-c[:,2,None],axis=0)
        hi = np.max(c[:,:2]+c[:,2,None],axis=0)
        c[:,0] -= lo[0]
        c[:,1] -= .5*(lo[1]+hi[1])
        ref = c.copy(); ref[:,1] *= -1
        cost = np.sum((c[:,None,:]-ref[None,:,:])**2,axis=2)
        rr,cc = linear_sum_assignment(cost)
        choices.append((float(cost[rr,cc].sum()),c,hi[0]-lo[0]))
    _,a,width = min(choices,key=lambda t:t[0])
    singles = np.argsort(np.abs(a[:,1]))[:3]
    mask = np.ones(21,bool); mask[singles] = False
    rem = np.flatnonzero(mask)
    ordered = rem[np.argsort(a[rem,1])]
    neg,pos = ordered[:9],ordered[9:]
    cost = ((a[pos,None,0]-a[None,neg,0])**2+
            (a[pos,None,1]+a[None,neg,1])**2+
            (a[pos,None,2]-a[None,neg,2])**2)
    _,cols = linear_sum_assignment(cost)
    neg = neg[cols]
    base = np.r_[a[singles,0],.5*(a[pos,0]+a[neg,0]),
                 .5*(a[pos,1]-a[neg,1]),a[singles,2],
                 .5*(a[pos,2]+a[neg,2]),width]
    if time.monotonic() < deadline-.04:
        base = optimize(base,sym,min(deadline-.025,time.monotonic()+.12),60)
    anchor = base.copy()
    for attempt in range(int(params['attempts'])):
        now = time.monotonic()
        if now >= deadline-.045:
            break
        trial = anchor.copy()
        amplitude = float(params['amplitude'])*(0.55,1.,1.8,0.8)[attempt%4]
        # Move a connected triple, including its reflected copies, together.
        centers = np.column_stack((trial[:12],np.r_[np.zeros(3),trial[12:21]]))
        pivot = int(rng.integers(12))
        distances = np.sum((centers-centers[pivot])**2,axis=1)
        near = np.argsort(distances)[:6]
        chosen = np.r_[pivot,rng.choice(near[near!=pivot],2,replace=False)]
        direction = rng.normal(size=2)
        direction /= max(float(np.linalg.norm(direction)),1e-15)
        for k in chosen:
            if time.monotonic() >= deadline-.03:
                return best
            trial[k] += amplitude*(direction[0]+.5*rng.normal())
            if k >= 3:
                trial[9+k] = abs(trial[9+k]+amplitude*(direction[1]+.5*rng.normal()))
            trial[21+k] *= 1+0.15*rng.normal()
        trial[-1] = np.clip(trial[-1]+rng.normal(0,.25*amplitude),.3,1.7)
        candidate = optimize(trial,sym,min(deadline-.015,now+float(params['trial_seconds'])),100)
        if float(sym[8]@candidate) < float(sym[8]@base)-1e-11:
            base = candidate.copy()
        if attempt%2 == 1 and float(sym[8]@base) < float(sym[8]@anchor)-1e-9:
            anchor = base.copy()
    return best
