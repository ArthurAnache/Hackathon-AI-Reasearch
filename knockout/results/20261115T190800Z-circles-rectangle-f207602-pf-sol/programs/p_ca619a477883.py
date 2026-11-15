import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {
    'bias': (0.22, 0.03, 0.6),
    'cycles': (4, 1, 10),
    'jitter': (0.006, 0.0, 0.05),
    'safety': (4e-15, 1e-15, 1e-11, 'log')
}
I, J = np.triu_indices(21, 1)

class Timeout(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return -np.inf
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    p = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
    p += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
    return float(np.sum(a[:, 2])) if p <= 2 else -np.inf

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + .70*budget_s
    rng = np.random.default_rng(seed)
    best, best_a, best_s = init, None, -np.inf
    eps = params['safety']

    def check():
        if time.monotonic() >= deadline:
            raise Timeout()

    def offer(a):
        nonlocal best, best_a, best_s
        if time.monotonic() >= deadline:
            return
        a = np.asarray(a, float).copy()
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
        ratio = min(1., float(np.min(d/(a[I,2]+a[J,2]))))
        if ratio <= 0:
            return
        a[:,2] *= ratio*(1-eps)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        scale = (2-8*eps)/np.sum(hi-lo)
        a[:,:2] = (a[:,:2]-(hi+lo)/2)*scale
        a[:,2] *= scale
        if np.sum(a[:,2]) <= best_s:
            return
        s = exact_score(a)
        if s > best_s:
            best_s, best_a = s, a.copy()
            best = {'state':a.T.reshape(-1).tolist()}

    if init is not None:
        try:
            a = np.asarray(init['circles'],float) if 'circles' in init else np.asarray(init['state'],float).reshape(3,21).T
            s = exact_score(a)
            if np.isfinite(s):
                best_s, best_a = s, a.copy()
            offer(a)
        except (ValueError, KeyError, TypeError):
            pass
    base = np.array([[.2*(k%3), .2*(k//3), .099999999999] for k in range(21)])
    offer(base)
    if best_a is None:
        return best

    # Detect the reflection axis and pair the two half-packings.
    candidates = []
    for swap in (False, True):
        if time.monotonic() >= deadline:
            return best
        a = best_a.copy()
        if swap:
            a[:,[0,1]] = a[:,[1,0]]
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        a[:,:2] -= (lo+hi)/2
        singles = np.argsort(np.abs(a[:,1]))[:3]
        rest = np.setdiff1d(np.arange(21),singles)
        order = rest[np.argsort(a[rest,1])]
        lower, upper = order[:9], order[9:]
        u = a[upper].copy()
        l = a[lower].copy()
        l[:,1] *= -1
        cost = np.sum((u[:,None,:]-l[None,:,:])**2,axis=2)
        rows,cols = linear_sum_assignment(cost)
        half = .5*(u[rows]+l[cols])
        ax = a[singles].copy()
        err = float(cost[rows,cols].sum()+np.sum(ax[:,1]**2))
        ax[:,1] = 0
        q = np.vstack((ax,half))
        z = np.r_[q[:,0],q[3:,1],q[:,2],hi[0]-lo[0]]
        candidates.append((err,z))
    z0 = min(candidates,key=lambda t:t[0])[1]

    ids = np.r_[np.arange(12),np.arange(3,12)]
    xi = ids.copy()
    yi = np.maximum(ids+9,0)
    yi[:3] = 0
    ys = np.r_[np.zeros(3),np.ones(9),-np.ones(9)]
    ri = 21+ids
    pairs = [(i,j) for i in range(12) for j in range(i+1,12)]
    for i in range(3,12):
        check()
        pairs.extend((i,12+j-3) for j in range(i,12))
    pi,pj = np.array(pairs).T
    m = len(pi)
    rows = np.arange(m)
    B = np.zeros((48,34))
    for k in range(12):
        check()
        B[k,k] = 1
        B[12+k,k] = -1
        B[k,21+k] = B[12+k,21+k] = -1
        B[k,33] = B[12+k,33] = .5
        B[24+k,yi[k]] += ys[k]
        B[36+k,yi[k]] -= ys[k]
        B[24+k,21+k] = B[36+k,21+k] = -1
        B[24+k,33] = B[36+k,33] = -.5
    offset = np.r_[np.zeros(24),np.ones(24)]
    bounds = [(None,None)]*34
    for k in range(21,33):
        check()
        bounds[k] = (1e-6,.5)
    bounds[-1] = (.2,1.8)
    for k in range(12,21):
        check()
        bounds[k] = (0.,1.)
    multiplicity = np.r_[np.ones(3),np.full(9,2.)]

    def unpack(z):
        return np.column_stack((z[xi],ys*z[yi],z[ri]))

    def solve(z, weights, stop):
        cache = [None,None,None]
        last = [z.copy()]
        objective = np.zeros(34)
        objective[21:33] = -multiplicity*weights
        def calc(v):
            if time.monotonic() >= stop:
                raise Timeout()
            if cache[0] is not None and np.array_equal(v,cache[0]):
                return cache[1],cache[2]
            x,y,r = v[xi],ys*v[yi],v[ri]
            dx,dy,rr = x[pi]-x[pj],y[pi]-y[pj],r[pi]+r[pj]
            c = np.r_[4*(dx*dx+dy*dy-rr*rr),B@v+offset]
            jac = np.zeros((m+48,34))
            np.add.at(jac,(rows,xi[pi]),8*dx)
            np.add.at(jac,(rows,xi[pj]),-8*dx)
            np.add.at(jac,(rows,yi[pi]),8*dy*ys[pi])
            np.add.at(jac,(rows,yi[pj]),-8*dy*ys[pj])
            np.add.at(jac,(rows,ri[pi]),-8*rr)
            np.add.at(jac,(rows,ri[pj]),-8*rr)
            jac[m:] = B
            cache[:] = [v.copy(),c,jac]
            return c,jac
        def fun(v):
            if time.monotonic() >= stop:
                raise Timeout()
            return float(objective@v),objective
        def callback(v):
            if time.monotonic() >= stop:
                raise Timeout()
            last[0] = v.copy()
        try:
            res = minimize(fun,z,method='SLSQP',jac=True,bounds=bounds,
                constraints={'type':'ineq','fun':lambda v:calc(v)[0],'jac':lambda v:calc(v)[1]},
                callback=callback,options={'maxiter':180,'ftol':2e-14,'disp':False})
            out = res.x
        except Timeout:
            out = last[0]
        offer(unpack(out))
        return out

    try:
        remaining = deadline-time.monotonic()
        anchor = solve(z0,np.ones(12),time.monotonic()+.16*remaining)
        cycles = int(params['cycles'])
        for attempt in range(cycles):
            check()
            remaining = deadline-time.monotonic()
            if remaining < .06:
                break
            slot = remaining/(cycles-attempt)
            begin = time.monotonic()
            z = anchor.copy()
            # Bias several independent mirrored radii together, not one contact.
            direction = rng.normal(size=12)
            direction -= np.dot(multiplicity,direction)/21
            direction /= max(np.max(np.abs(direction)),1e-12)
            weights = 1+params['bias']*direction
            if attempt > 0:
                z[:21] += rng.normal(0,params['jitter'],21)
                z[12:21] = np.maximum(z[12:21],1e-4)
            z = solve(z,weights,min(deadline,begin+.36*slot))
            polished = solve(z,np.ones(12),min(deadline-.005,begin+.98*slot))
            if np.sum(unpack(polished)[:,2]) > np.sum(unpack(anchor)[:,2])+1e-9:
                anchor = polished
    except Timeout:
        pass
    return best
