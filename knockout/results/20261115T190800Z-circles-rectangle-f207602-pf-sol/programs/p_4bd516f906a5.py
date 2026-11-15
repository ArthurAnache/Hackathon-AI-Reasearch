import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment
from scipy.linalg import lstsq, null_space

PARAMS = {
    'release_count': (5, 3, 9),
    'step': (0.055, 0.005, 0.15),
    'restart_seconds': (0.19, 0.08, 0.4),
    'safety': (8e-15, 2e-15, 1e-11, 'log'),
    'iterations': (160, 50, 400)
}
I, J = np.triu_indices(21, 1)

class Stop(Exception):
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
    end = start + .70*budget_s
    rng = np.random.default_rng(seed)
    eps = params['safety']
    best, best_a, best_score = None, None, -np.inf
    original = None
    try:
        if init is not None:
            original = (np.array(init['circles'],float) if 'circles' in init else
                        np.array(init['state'],float).reshape(3,21).T)
            if valid(original):
                best, best_a, best_score = init, original.copy(), float(np.sum(original[:,2]))
    except (KeyError, ValueError, TypeError):
        original = None

    def offer(a):
        nonlocal best,best_a,best_score
        if time.monotonic() >= end:
            return
        a = np.array(a,dtype=float,copy=True)
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
        q = min(1.,float(np.min(d/(a[I,2]+a[J,2]))))
        if q <= 0:
            return
        a[:,2] *= q*(1-eps)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        factor = (2-10*eps)/float(np.sum(hi-lo))
        a[:,:2] = (a[:,:2]-(lo+hi)/2)*factor
        a[:,2] *= factor
        score = float(np.sum(a[:,2]))
        if score > best_score and valid(a):
            best,best_a,best_score = {'state':a.T.ravel().tolist()},a,score

    if original is not None:
        offer(original)
    lattice = []
    for row,count in enumerate((3,4,5,4,5)):
        if time.monotonic() >= end:
            break
        for k in range(count):
            if time.monotonic() >= end:
                break
            lattice.append([row*np.sqrt(3.),2*k-count+1,1.])
    if len(lattice)==21:
        offer(lattice)
    if best_a is None:
        a = np.array([[.2*(i%3),.2*(i//3),.09] for i in range(21)])
        if valid(a):
            best,best_a,best_score = {'circles':a.tolist()},a,float(np.sum(a[:,2]))
    if time.monotonic() >= end-.04:
        return best

    def project(a):
        choices = []
        for swap in (False,True):
            if time.monotonic() >= end-.02:
                break
            b = a.copy()
            if swap:
                b[:,:2] = b[:,[1,0]]
            lo = np.min(b[:,:2]-b[:,2,None],axis=0)
            hi = np.max(b[:,:2]+b[:,2,None],axis=0)
            b[:,0] -= lo[0]
            b[:,1] -= (lo[1]+hi[1])/2
            singles = np.argsort(abs(b[:,1]))[:3]
            rem = np.setdiff1d(np.arange(21),singles)
            order = rem[np.argsort(b[rem,1])]
            neg,pos = order[:9],order[9:]
            cost = ((b[pos,None,0]-b[None,neg,0])**2+
                    (b[pos,None,1]+b[None,neg,1])**2+
                    (b[pos,None,2]-b[None,neg,2])**2)
            rr,cc = linear_sum_assignment(cost)
            neg = neg[cc]
            error = float(cost[rr,cc].sum()+np.sum(b[singles,1]**2))
            z = np.r_[b[singles,0],(b[pos,0]+b[neg,0])/2,
                      (b[pos,1]-b[neg,1])/2,b[singles,2],
                      (b[pos,2]+b[neg,2])/2,hi[0]-lo[0]]
            choices.append((error,z))
        return min(choices,key=lambda q:q[0])[1] if choices else None

    def model(sym):
        if sym:
            T = np.zeros((64,34))
            ids = np.r_[np.arange(12),np.arange(3,12)]
            signs = np.r_[np.zeros(3),np.ones(9),-np.ones(9)]
            T[np.arange(21),ids] = 1
            mask = ids>=3
            T[21+np.arange(21)[mask],12+ids[mask]-3] = signs[mask]
            T[42+np.arange(21),21+ids] = 1
            T[-1,-1] = 1
            pairs = [(i,j) for i in range(12) for j in range(i+1,12)]
            pairs += [(i,12+j-3) for i in range(3,12) for j in range(i,12)]
            pi,pj = np.array(pairs).T
            wallids = np.arange(12)
            bounds = [(None,None)]*34
            bounds[12:21] = [(0,None)]*9
            bounds[21:33] = [(1e-6,.6)]*12
        else:
            T = np.eye(64)
            pi,pj = I,J
            wallids = np.arange(21)
            bounds = [(None,None)]*64
            bounds[42:63] = [(1e-6,.6)]*21
        bounds[-1] = (.2,1.8)
        X,Y,R,W = T[:21],T[21:42],T[42:63],T[-1]
        dx,dy,rs = X[pi]-X[pj],Y[pi]-Y[pj],R[pi]+R[pj]
        xx,yy,rr = X[wallids],Y[wallids],R[wallids]
        walls = np.concatenate((xx-rr,W-xx-rr,-.5*W-yy-rr,-.5*W+yy-rr))
        off = np.r_[np.zeros(2*len(wallids)),np.ones(2*len(wallids))]
        obj = -R.sum(axis=0)
        def calc(v):
            u,t,s = dx@v,dy@v,rs@v
            c = np.r_[u*u+t*t-s*s,walls@v+off]
            jac = np.concatenate((2*(u[:,None]*dx+t[:,None]*dy-s[:,None]*rs),walls))
            return c,jac
        return T,bounds,obj,calc,len(pi)

    def solve(z,mod,stop):
        T,bounds,obj,raw,_ = mod
        last = [z.copy()]
        cache = [None,None,None]
        def calc(v):
            if time.monotonic() >= stop:
                raise Stop
            if cache[0] is None or not np.array_equal(v,cache[0]):
                c,j = raw(v)
                cache[:] = [v.copy(),c,j]
                last[0] = v.copy()
            return cache[1],cache[2]
        def fun(v):
            if time.monotonic() >= stop:
                raise Stop
            return float(obj@v),obj
        try:
            res = minimize(fun,z,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda v:calc(v)[0],
                             'jac':lambda v:calc(v)[1]},
                options={'maxiter':int(params['iterations']),'ftol':4e-14,'disp':False})
            z = res.x
        except Stop:
            z = last[0]
        offer((T@z)[:63].reshape(3,21).T)
        for k in range(3):
            if time.monotonic() >= stop-.004:
                break
            c,j = raw(z)
            active = np.flatnonzero(abs(c)<1e-7)
            if len(active)<len(z)-1:
                break
            delta,_,rank,_ = lstsq(j[active],-c[active],lapack_driver='gelsy',check_finite=False)
            if rank<len(z)-1 or np.max(abs(delta))>1e-4:
                break
            z = z+delta
            offer((T@z)[:63].reshape(3,21).T)
        return z

    sym = model(True)
    z = project(best_a)
    if z is not None and time.monotonic() < end-.12:
        z = solve(z,sym,min(end-.13,time.monotonic()+.24*budget_s))
    anchor = project(best_a)
    attempt = 0
    while anchor is not None and time.monotonic() < end-.22:
        now = time.monotonic()
        trial = anchor.copy()
        c,j = sym[3](anchor)
        active = np.flatnonzero(abs(c)<2e-6)
        contacts = active[active<sym[4]]
        direction = None
        if len(contacts)>=params['release_count']:
            released = rng.choice(contacts,int(params['release_count']),replace=False)
            keep = active[~np.isin(active,released)]
            basis = null_space(j[keep],rcond=1e-8)
            if basis.shape[1]:
                direction = basis@rng.normal(size=basis.shape[1])
        if direction is None:
            direction = rng.normal(size=34)
            direction[21:33] *= .25
            direction[-1] *= .2
        norm = np.linalg.norm(direction[:21])
        if norm>0:
            direction *= params['step']*(1+.55*(attempt%3))/norm
        trial += direction
        trial[12:21] = np.maximum(trial[12:21],1e-5)
        trial[21:33] = np.clip(trial[21:33],1e-5,.59)
        trial[-1] = np.clip(trial[-1],.21,1.79)
        solve(trial,sym,min(end-.16,now+params['restart_seconds']))
        anchor = project(best_a)
        attempt += 1
    if time.monotonic() < end-.035:
        a = best_a
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        zfull = np.r_[a[:,0]-lo[0],a[:,1]-(lo[1]+hi[1])/2,a[:,2],hi[0]-lo[0]]
        solve(zfull,model(False),end-.008)
    return best
