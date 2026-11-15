import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {
    'warm_fraction': (0.43, 0.25, 0.6),
    'symmetric_fraction': (0.36, 0.2, 0.5),
    'iterations': (200, 60, 400),
    'restart_noise': (0.018, 0.0, 0.08),
}
N = 21
II, JJ = np.triu_indices(N, 1)

class Deadline(Exception):
    pass

def valid(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return False
    d = np.sqrt((a[II, 0]-a[JJ, 0])**2+(a[II, 1]-a[JJ, 1])**2)
    if np.any(d < a[II, 2]+a[JJ, 2]):
        return False
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    return (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
            +max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))) <= 2

def model(sym):
    dim = 34 if sym else 64
    X = np.zeros((21, dim)); Y = X.copy(); R = X.copy()
    if sym:
        ids = np.r_[np.arange(12), np.arange(3,12)]
        signs = np.r_[np.ones(12), -np.ones(9)]
        X[np.arange(21), ids] = 1
        mask = ids >= 3
        Y[np.arange(21)[mask], 12+ids[mask]-3] = signs[mask]
        R[np.arange(21), 21+ids] = 1
        pairs = [(i,j) for i in range(12) for j in range(i+1,12)]
        pairs += [(i,12+j-3) for i in range(3,12) for j in range(i,12)]
        pi = np.array([p[0] for p in pairs]); pj = np.array([p[1] for p in pairs])
        boundary = np.arange(12)
    else:
        X[np.arange(21),np.arange(21)] = 1
        Y[np.arange(21),21+np.arange(21)] = 1
        R[np.arange(21),42+np.arange(21)] = 1
        pi,pj = II,JJ
        boundary = np.arange(21)
    W = np.zeros(dim); W[-1] = 1
    xb,yb,rb = X[boundary],Y[boundary],R[boundary]
    B = np.concatenate((xb-rb,W-xb-rb,-.5*W-yb-rb,-.5*W+yb-rb))
    offset = np.r_[np.zeros(2*len(boundary)),np.ones(2*len(boundary))]
    return X,Y,R,X[pi]-X[pj],Y[pi]-Y[pj],R[pi]+R[pj],B,offset,-R.sum(axis=0)

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + .70*budget_s
    rng = np.random.default_rng(seed)
    best = None
    best_score = -np.inf

    def retain(a, original=None):
        nonlocal best,best_score
        if valid(a):
            s = float(np.sum(a[:,2]))
            if s > best_score:
                best_score = s
                best = original if original is not None else {'circles':a.tolist()}

    def offer(a):
        if a.shape != (21,3) or not np.isfinite(a).all():
            return
        a = a.copy()
        a[:,2] = np.maximum(a[:,2],1e-10)
        d = np.sqrt((a[II,0]-a[JJ,0])**2+(a[II,1]-a[JJ,1])**2)
        ratio = min(1.,float(np.min(d/(a[II,2]+a[JJ,2]))))
        if ratio <= 0:
            return
        a[:,2] *= ratio*(1-4e-14)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        scale = (2-8e-14)/float(np.sum(hi-lo))
        a[:,:2] = (a[:,:2]-lo)*scale
        a[:,2] *= scale
        retain(a)

    raw = None
    if init is not None:
        try:
            a = np.asarray(init['circles'],dtype=float)
            if a.shape == (21,3) and np.isfinite(a).all():
                raw = a.copy()
                retain(a,init)
                offer(a)
        except (ValueError,TypeError,KeyError):
            pass
    xx,yy = np.meshgrid(np.arange(5),np.arange(4))
    p = np.column_stack((xx.ravel()+.5*(yy.ravel()%2), yy.ravel()*np.sqrt(3)/2))
    p = np.vstack((p,[2.5,4*np.sqrt(3)/2]))
    offer(np.column_stack((p,np.full(21,.499))))

    def normalize(a):
        a = a.copy()
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        scale = 2/float(np.sum(hi-lo))
        a[:,:2] = (a[:,:2]-lo)*scale
        a[:,2] *= scale
        w = (hi[0]-lo[0])*scale
        a[:,1] -= (2-w)*.5
        return a,w

    def full_vector(a):
        a,w = normalize(a)
        return np.r_[a[:,0],a[:,1],a[:,2],w]

    full = model(False)
    symmetric = model(True)

    def optimize(z,m,deadline):
        X,Y,R,DX,DY,RS,B,offset,obj = m
        last = [z.copy()]
        cached = [None,None,None]
        count = [0]
        def check():
            if time.monotonic() >= deadline:
                raise Deadline()
        def constraints(v):
            check()
            if cached[0] is not None and np.array_equal(v,cached[0]):
                return cached[1],cached[2]
            dx = DX@v; dy = DY@v
            d = np.sqrt(dx*dx+dy*dy+1e-28)
            c = np.r_[d-RS@v,B@v+offset]
            jac = np.concatenate(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RS,B))
            cached[:] = [v.copy(),c,jac]
            return c,jac
        def objective(v):
            check()
            return float(obj@v),obj
        def callback(v):
            last[0] = v.copy()
            check()
            count[0] += 1
            if count[0] % 5 == 0:
                offer(np.column_stack((X@v,Y@v,R@v)))
        bounds = [(None,None)]*len(z)
        first = 21 if len(z)==34 else 42
        for k in range(first,len(z)-1):
            if time.monotonic() >= deadline:
                return z
            bounds[k] = (1e-8,.65)
        bounds[-1] = (.08,1.92)
        try:
            result = minimize(objective,z,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda v:constraints(v)[0],
                             'jac':lambda v:constraints(v)[1]},callback=callback,
                options={'maxiter':int(params['iterations']),'ftol':3e-12})
            last[0] = result.x
        except Deadline:
            pass
        z = last[0]
        offer(np.column_stack((X@z,Y@z,R@z)))
        return z

    total = end-start
    warm_end = start+float(params['warm_fraction'])*total
    if time.monotonic() < warm_end:
        a = raw if raw is not None else np.asarray(best['circles'])
        optimize(full_vector(a),full,warm_end)

    def projection(a):
        chosen = None
        error = np.inf
        for swap in (False,True):
            if time.monotonic() >= end:
                break
            c = a.copy()
            if swap:
                c[:,:2] = c[:,[1,0]]
            c,w = normalize(c)
            order = np.argsort(c[:,1])
            bottom,axis,top = order[:9],order[9:12],order[12:]
            delta = c[top,None,:]-c[None,bottom,:]
            delta[:,:,1] = c[top,None,1]+c[None,bottom,1]
            cost = delta[:,:,0]**2+delta[:,:,1]**2+2*delta[:,:,2]**2
            rr,cc = linear_sum_assignment(cost)
            u,v = top[rr],bottom[cc]
            xs = np.r_[c[axis,0],.5*(c[u,0]+c[v,0])]
            ys = .5*(c[u,1]-c[v,1])
            rs = np.r_[c[axis,2],.5*(c[u,2]+c[v,2])]
            err = float(cost[rr,cc].sum()+4*np.sum(c[axis,1]**2))
            if err < error:
                error = err
                chosen = np.r_[xs,ys,rs,w]
        return chosen

    sym_end = min(end-.025,start+(float(params['warm_fraction'])+float(params['symmetric_fraction']))*total)
    if time.monotonic() < sym_end:
        z = projection(np.asarray(best['circles'],dtype=float))
        if z is not None:
            optimize(z,symmetric,sym_end)

    attempt = 0
    while time.monotonic() < end-.025:
        a = np.asarray(best['circles'],dtype=float)
        z = full_vector(a)
        if attempt:
            z[:42] += rng.normal(0,float(params['restart_noise']),42)
        before = time.monotonic()
        optimize(z,full,end-.012)
        attempt += 1
        if time.monotonic()-before < .003:
            break
    return best
