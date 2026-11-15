import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'time_fraction': (0.69, 0.4, 0.7),
    'symmetric_iterations': (150, 50, 240),
    'full_iterations': (65, 20, 160),
    'jitter': (0.024, 0.002, 0.08)
}

II, JJ = np.triu_indices(21, 1)
IX = 3*np.arange(21)
IR = IX+2

class StopSearch(Exception):
    pass


def exact_score(data):
    try:
        a = np.asarray(data['circles'], dtype=np.float64)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        xs = [Fraction(float(v)) for v in a[:, 0]]
        ys = [Fraction(float(v)) for v in a[:, 1]]
        rs = [Fraction(float(v)) for v in a[:, 2]]
        w = max(x+r for x,r in zip(xs,rs))-min(x-r for x,r in zip(xs,rs))
        h = max(y+r for y,r in zip(ys,rs))-min(y-r for y,r in zip(ys,rs))
        if w+h > 2:
            return -float('inf')
        for p,q in itertools.combinations(a, 2):
            time.monotonic()
            if np.sqrt((p[0]-q[0])**2+(p[1]-q[1])**2) < p[2]+q[2]:
                return -float('inf')
        return float(np.sum(a[:, 2]))
    except Exception:
        return -float('inf')


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + float(params['time_fraction'])*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = exact_score(init)
    grad = np.zeros(64)
    grad[IR] = -1.0
    rr = np.arange(21)
    base = np.zeros((84,64))
    base[rr,IX] = 1
    base[rr,IR] = -1
    base[rr,63] = 0.5
    base[21+rr,IX] = -1
    base[21+rr,IR] = -1
    base[21+rr,63] = 0.5
    base[42+rr,IX+1] = 1
    base[42+rr,IR] = -1
    base[63+rr,IX+1] = -1
    base[63+rr,IR] = -1
    base[63+rr,63] = -1
    offset = np.r_[np.zeros(63), np.full(21,2.0)]
    full_bounds = [(-1,1),(0,2),(1e-7,0.75)]*21+[(0.15,1.85)]

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def normalize(c):
        c = np.asarray(c, dtype=float).reshape(21,3).copy()
        lo = np.min(c[:,:2]-c[:,2,None],axis=0)
        hi = np.max(c[:,:2]+c[:,2,None],axis=0)
        extent = hi-lo
        c[:,0] -= (lo[0]+hi[0])*0.5
        c[:,1] -= lo[1]
        f = 2/float(np.sum(extent))
        c *= f
        return np.r_[c.ravel(),extent[0]*f]

    def consider(z):
        nonlocal best,best_score
        check()
        a = np.asarray(z[:63]).reshape(21,3).copy()
        if not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        dx = a[II,0]-a[JJ,0]
        dy = a[II,1]-a[JJ,1]
        ratio = float(np.min(np.sqrt(dx*dx+dy*dy)/(a[II,2]+a[JJ,2])))
        if ratio <= 0:
            return
        a[:,2] *= min(1.0,ratio)*(1-5e-15)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        a[:,:2] -= lo
        extent = np.max(a[:,:2]+a[:,2,None],axis=0)
        a *= (2-3e-14)/float(np.sum(extent))
        if float(np.sum(a[:,2])) > best_score:
            candidate = {'circles':a.tolist()}
            score = exact_score(candidate)
            if score > best_score:
                best,best_score = candidate,score

    def values(z, pi=II, pj=JJ):
        check()
        c = z[:63].reshape(21,3)
        dx = c[pi,0]-c[pj,0]
        dy = c[pi,1]-c[pj,1]
        return np.r_[base@z+offset,np.sqrt(dx*dx+dy*dy)-c[pi,2]-c[pj,2]]

    def jac(z, pi=II, pj=JJ):
        check()
        c = z[:63].reshape(21,3)
        dx = c[pi,0]-c[pj,0]
        dy = c[pi,1]-c[pj,1]
        d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-25)
        j = np.zeros((84+len(pi),64))
        j[:84] = base
        t = 84+np.arange(len(pi))
        j[t,3*pi] = dx/d
        j[t,3*pj] = -dx/d
        j[t,3*pi+1] = dy/d
        j[t,3*pj+1] = -dy/d
        j[t,3*pi+2] = -1
        j[t,3*pj+2] = -1
        return j

    # Nine reflected pairs and three circles on the central axis.
    transform = np.zeros((64,34))
    for k in range(9):
        check()
        transform[6*k,3*k] = 1
        transform[6*k+3,3*k] = -1
        transform[6*k+1,3*k+1] = 1
        transform[6*k+4,3*k+1] = 1
        transform[6*k+2,3*k+2] = 1
        transform[6*k+5,3*k+2] = 1
    for k in range(3):
        check()
        transform[3*(18+k)+1,27+2*k] = 1
        transform[3*(18+k)+2,28+2*k] = 1
    transform[63,33] = 1
    red_grad = transform.T@grad
    red_bounds = [(1e-6,0.95),(0,2),(1e-7,0.75)]*9 + [(0,2),(1e-7,0.75)]*3 + [(0.15,1.85)]

    def lattice(variant=0):
        q = np.zeros(34)
        k = 0
        for y in (1.,3.,5.,7.):
            check()
            for x in (1.,3.):
                check()
                q[3*k:3*k+3] = [x/8,y/8,1/8]
                k += 1
        q[24:27] = [2/8,(4 if variant == 0 else 2)/8,(np.sqrt(2)-1)/8]
        q[27:33] = [2/8,(np.sqrt(2)-1)/8,4/8,(np.sqrt(2)-1)/8,6/8,(np.sqrt(2)-1)/8]
        q[33] = 1
        return q

    def symmetric_solve(q, iterations):
        check()
        def fun(v):
            check()
            return float(red_grad@v)
        result = minimize(fun,q,jac=lambda v:red_grad,method='SLSQP',bounds=red_bounds,
            constraints={'type':'ineq','fun':lambda v:values(transform@v),
                         'jac':lambda v:jac(transform@v)@transform},
            callback=lambda v:consider(transform@v),
            options={'maxiter':iterations,'ftol':2e-13,'disp':False})
        z = transform@result.x
        consider(z)
        return z

    def full_solve(z, iterations):
        check()
        # Include a generous contact shell, then rebuild it on the next stage.
        selected = values(z)[84:] < 0.14
        pi,pj = II[selected],JJ[selected]
        def fun(v):
            check()
            return float(grad@v)
        result = minimize(fun,z,jac=lambda v:grad,method='SLSQP',bounds=full_bounds,
            constraints={'type':'ineq','fun':lambda v:values(v,pi,pj),
                         'jac':lambda v:jac(v,pi,pj)},
            callback=consider,
            options={'maxiter':iterations,'ftol':1e-13,'disp':False})
        consider(result.x)
        return result.x

    try:
        q0 = lattice()
        consider(transform@q0)
        # Strong supplied packings retain priority; tiny-radius cold starts do not.
        supplied_score = exact_score(init)
        if supplied_score > 2.2:
            z = normalize(init['circles'])
            z = full_solve(z,int(params['full_iterations']))
            if np.min(values(z)) < -1e-7:
                z = full_solve(z,35)
        else:
            z = symmetric_solve(q0,int(params['symmetric_iterations']))
            z = full_solve(z,int(params['full_iterations']))
        attempt = 0
        while time.monotonic() < deadline:
            check()
            if attempt == 0 and supplied_score <= 2.2:
                q = lattice(1)
                z = symmetric_solve(q,int(params['symmetric_iterations']))
            else:
                c = np.asarray(best['circles'],dtype=float).copy()
                strength = float(params['jitter'])*(1+0.35*(attempt//3))
                center = int(rng.integers(21))
                nearby = np.argsort(np.sum((c[:,:2]-c[center,:2])**2,axis=1))[:8]
                c[nearby,:2] += rng.normal(0,strength,(len(nearby),2))
                z = normalize(c)
            z = full_solve(z,int(params['full_iterations']))
            if np.min(values(z)) < -1e-7:
                z = full_solve(z,30)
            attempt += 1
    except (StopSearch,ValueError,FloatingPointError,np.linalg.LinAlgError):
        pass
    return best
