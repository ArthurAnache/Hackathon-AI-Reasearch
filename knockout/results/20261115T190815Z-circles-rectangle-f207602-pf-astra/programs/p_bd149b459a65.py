import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linprog

PARAMS = {
    'mu0': (10.0, 1.0, 100.0, 'log'),
    'growth': (4.0, 1.5, 10.0),
    'stages': (12, 4, 24),
    'maxiter': (100, 20, 300),
    'contraction': (0.55, 0.1, 0.9)
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class Deadline(Exception):
    pass


def perimeter(p, r):
    x = [Fraction(float(v)) for v in p[:, 0]]
    y = [Fraction(float(v)) for v in p[:, 1]]
    q = [Fraction(float(v)) for v in r]
    return (max(a+b for a,b in zip(x,q)) - min(a-b for a,b in zip(x,q))
            + max(a+b for a,b in zip(y,q)) - min(a-b for a,b in zip(y,q)))


def score(p, r):
    if p.shape != (N, 2) or r.shape != (N,):
        return -np.inf
    if not np.isfinite(p).all() or not np.isfinite(r).all() or np.any(r <= 0):
        return -np.inf
    d = np.sqrt((p[I,0]-p[J,0])**2 + (p[I,1]-p[J,1])**2)
    if np.any(d < r[I]+r[J]) or perimeter(p,r) > 2:
        return -np.inf
    return float(np.sum(r))


def constraints(z):
    x, y, r, w = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
    dx, dy = x[I]-x[J], y[I]-y[J]
    d = np.sqrt(dx*dx+dy*dy+1e-30)
    g = np.concatenate((r[I]+r[J]-d, r-x, r+x-w, r-y, r+y+w-2))
    return g, dx, dy, d


def objective(z, mu, multipliers, deadline):
    if time.monotonic() >= deadline:
        raise Deadline()
    g, dx, dy, d = constraints(z)
    force = np.maximum(0.0, multipliers+mu*g)
    f = -z[2*N:3*N].sum() + (np.dot(force,force)-np.dot(multipliers,multipliers))/(2*mu)
    c = force[:M]
    gx = np.bincount(I, weights=-c*dx/d, minlength=N) + np.bincount(J, weights=c*dx/d, minlength=N)
    gy = np.bincount(I, weights=-c*dy/d, minlength=N) + np.bincount(J, weights=c*dy/d, minlength=N)
    gr = -np.ones(N) + np.bincount(I, weights=c, minlength=N) + np.bincount(J, weights=c, minlength=N)
    b = force[M:].reshape(4,N)
    gx += -b[0]+b[1]
    gy += -b[2]+b[3]
    gr += b.sum(axis=0)
    gw = -b[1].sum()+b[3].sum()
    return float(f), np.concatenate((gx,gy,gr,[gw]))


def repair(p, r, deadline):
    p, r = p.copy(), r.copy()
    if not np.isfinite(p).all() or not np.isfinite(r).all() or np.any(r <= 0):
        return None
    d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
    factor = min(1.0, float(np.min(d/(r[I]+r[J]))))
    r *= factor*(1.0-4e-15)
    for _ in range(6):
        if time.monotonic() >= deadline:
            return None
        hp = perimeter(p,r)
        if hp > 2:
            s = (2.0/float(hp))*(1.0-4e-15)
            p *= s
            r *= s
        val = score(p,r)
        if np.isfinite(val):
            return p,r,val
        r *= 1.0-2e-14
    return None


def lp_radii(p, deadline):
    if time.monotonic() >= deadline:
        return None
    a = np.zeros((M+4*N+1,N+4))
    b = np.zeros(M+4*N+1)
    a[np.arange(M),I] = 1
    a[np.arange(M),J] = 1
    b[:M] = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
    for k,(col,sign,coord) in enumerate(((N,1,0),(N+1,-1,0),(N+2,1,1),(N+3,-1,1))):
        if time.monotonic() >= deadline:
            return None
        rows = M+k*N+np.arange(N)
        a[rows,np.arange(N)] = 1
        a[rows,col] = sign
        b[rows] = sign*p[:,coord]
    a[-1,N:] = [-1,1,-1,1]
    b[-1] = 2
    remaining = deadline-time.monotonic()
    if remaining < 0.025:
        return None
    result = linprog(-np.r_[np.ones(N),np.zeros(4)], A_ub=a, b_ub=b,
                     bounds=[(1e-10,None)]*N+[(None,None)]*4,
                     method='highs', options={'time_limit': max(0.001,remaining*0.7)})
    return result.x[:N] if result.success else None


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start+0.70*budget_s
    descent_end = start+0.57*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf
    if init is not None:
        data = np.asarray(init['circles'],dtype=float)
        p, r = data[:,:2].copy(), data[:,2].copy()
        best_score = score(p,r)
    else:
        ids = np.arange(N)
        p = np.column_stack(((ids%5+0.5)/5,(ids//5+0.5)/5))
        r = np.full(N,0.075)
        best = {'circles': np.column_stack((p,r)).tolist()}
        best_score = score(p,r)
        p += rng.normal(0,0.015,p.shape)
    lo = np.min(p-r[:,None],axis=0)
    sides = np.max(p+r[:,None],axis=0)-lo
    scale = 2.0/float(sides.sum())
    p = (p-lo)*scale
    r *= scale
    z = np.r_[p[:,0],p[:,1],r,sides[0]*scale]
    multipliers = np.zeros(M+4*N)
    mu = float(params['mu0'])
    previous_violation = np.inf
    chosen_p = p.copy()
    chosen_rank = -np.inf
    last = z.copy()

    def retain(pp, rr):
        nonlocal best, best_score
        fixed = repair(pp,rr,deadline)
        if fixed is not None:
            fp,fr,val = fixed
            if val > best_score:
                best_score = val
                best = {'circles': np.column_stack((fp,fr)).tolist()}
            return val
        return -np.inf

    def callback(zz):
        nonlocal last
        if time.monotonic() >= descent_end:
            raise Deadline()
        last = zz.copy()

    bounds = [(-0.2,2.2)]*(2*N)+[(1e-9,0.6)]*N+[(0.15,1.85)]
    for stage in range(int(params['stages'])):
        if time.monotonic() >= descent_end:
            break
        last = z.copy()
        try:
            result = minimize(objective,z,args=(mu,multipliers,descent_end),jac=True,
                              method='L-BFGS-B',bounds=bounds,callback=callback,
                              options={'maxiter':int(params['maxiter']), 'ftol':1e-14,
                                       'gtol':2e-9,'maxls':25,'maxcor':15})
            z = result.x
        except Deadline:
            z = last
        pp = np.column_stack((z[:N],z[N:2*N]))
        rank = retain(pp,z[2*N:3*N])
        if rank > chosen_rank:
            chosen_rank = rank
            chosen_p = pp.copy()
        if time.monotonic() >= descent_end:
            break
        g,_,_,_ = constraints(z)
        violation = float(np.max(np.maximum(g,0)))
        # Accumulated contact forces remove the O(1/mu) penalty bias.
        # Increase stiffness only when feasibility progress stalls.
        multipliers = np.maximum(0.0,multipliers+mu*g)
        if violation > float(params['contraction'])*previous_violation:
            mu *= float(params['growth'])
        previous_violation = violation
    # Keep the parent's fixed-centre radius optimization as final recovery.
    if time.monotonic() < deadline-0.04:
        rr = lp_radii(chosen_p,deadline)
        if rr is not None and time.monotonic() < deadline:
            retain(chosen_p,rr)
    return best
