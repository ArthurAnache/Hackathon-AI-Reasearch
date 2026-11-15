import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'bias': (2.8, 1.2, 6.0),
    'cluster': (5, 3, 10),
    'explore_fraction': (0.38, 0.2, 0.6),
    'branch_seconds': (0.42, 0.15, 0.8),
    'attempts': (8, 1, 30),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
K = np.arange(N)

class Deadline(Exception):
    pass

def exact_score(c):
    a = np.asarray(c['circles'], dtype=float)
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    p, r = a[:, :2], a[:, 2]
    d = np.sqrt((p[I, 0]-p[J, 0])**2 + (p[I, 1]-p[J, 1])**2)
    if np.any(d < r[I]+r[J]):
        return -np.inf
    x, y, q = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
    perimeter = max(u+v for u,v in zip(x,q))-min(u-v for u,v in zip(x,q))
    perimeter += max(u+v for u,v in zip(y,q))-min(u-v for u,v in zip(y,q))
    if perimeter > 2:
        return -np.inf
    return float(np.sum(r))

def repaired(z):
    p = np.column_stack((z[:N], z[N:2*N])).copy()
    r = np.maximum(z[2*N:3*N], 1e-12).copy()
    d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
    factor = min(1.0, float(np.min(d/(r[I]+r[J]))))
    r *= factor*(1-8e-15)
    lo = np.min(p-r[:,None], axis=0)
    hi = np.max(p+r[:,None], axis=0)
    scale = min(1.0, 2.0/float(np.sum(hi-lo)))*(1-8e-15)
    p = (p-lo)*scale
    r *= scale
    return {'circles': np.column_stack((p,r)).tolist()}

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a = np.array([[(i%5+0.5)/5, (i//5+0.5)/5, 0.095] for i in range(N)])
        init = {'circles': a.tolist()}
    best = init
    best_score = exact_score(init)
    a = np.asarray(init['circles'], dtype=float)
    p, r = a[:,:2].copy(), a[:,2].copy()
    lo = np.min(p-r[:,None], axis=0)
    sides = np.max(p+r[:,None], axis=0)-lo
    s = 2.0/float(sides.sum())
    p = (p-lo)*s
    r *= s
    origin = np.r_[p[:,0], p[:,1], r, sides[0]*s]
    lower = [(None,None)]*(2*N)+[(1e-9,0.9)]*N+[(0.08,1.92)]
    last = origin.copy()
    phase_end = end

    def guard():
        if time.monotonic() >= phase_end:
            raise Deadline()

    def constraints(z):
        guard()
        x,y,q,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
        dx,dy = x[I]-x[J],y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        return np.r_[d-q[I]-q[J], x-q, w-x-q, y-q, 2-w-y-q]

    def jacobian(z):
        guard()
        x,y = z[:N],z[N:2*N]
        dx,dy = x[I]-x[J],y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        jac = np.zeros((M+4*N,3*N+1))
        row = np.arange(M)
        jac[row,I] = dx/d
        jac[row,J] = -dx/d
        jac[row,N+I] = dy/d
        jac[row,N+J] = -dy/d
        jac[row,2*N+I] = -1
        jac[row,2*N+J] = -1
        for k in range(4):
            rr = M+k*N+K
            jac[rr,2*N+K] = -1
        jac[M+K,K] = 1
        jac[M+N+K,K] = -1
        jac[M+N+K,-1] = 1
        jac[M+2*N+K,N+K] = 1
        jac[M+3*N+K,N+K] = -1
        jac[M+3*N+K,-1] = -1
        return jac

    def solve(z, weights, until, iterations):
        nonlocal phase_end,last
        phase_end = min(until,end)
        last = z.copy()
        gradient = np.r_[np.zeros(2*N),-weights,0.0]
        def objective(v):
            guard()
            return float(gradient@v), gradient
        def callback(v):
            nonlocal last
            guard()
            last = v.copy()
        try:
            result = minimize(objective,z,jac=True,method='SLSQP',bounds=lower,
                constraints={'type':'ineq','fun':constraints,'jac':jacobian},
                callback=callback,options={'maxiter':iterations,'ftol':2e-12,'disp':False})
            return result.x
        except Deadline:
            return last.copy()

    def consider(z):
        nonlocal best,best_score
        c = repaired(z)
        score = exact_score(c)
        if score > best_score:
            best,best_score = c,score

    if time.monotonic() < end:
        polished = solve(origin,np.ones(N),min(end,time.monotonic()+0.15),65)
        consider(polished)
    else:
        return best
    branch_base = origin.copy()
    for attempt in range(int(params['attempts'])):
        now = time.monotonic()
        if now+0.075 >= end:
            break
        duration = min(float(params['branch_seconds']),end-now)
        branch_end = now+duration
        z = branch_base.copy()
        # Correlated weights grow or evacuate a whole spatial neighborhood.
        anchor = int(rng.integers(N))
        pp = np.column_stack((z[:N],z[N:2*N]))
        nearest = np.argsort(np.sum((pp-pp[anchor])**2,axis=1))
        count = int(params['cluster'])
        weights = np.ones(N)
        if attempt % 3 == 0:
            weights[nearest[:count]] = float(params['bias'])
        elif attempt % 3 == 1:
            weights[nearest[:count]] = 1/float(params['bias'])
        else:
            direction = rng.normal(size=2)
            projection = pp@direction
            projection = (projection-projection.mean())/(np.std(projection)+1e-12)
            weights = np.exp(np.clip(0.65*projection,-1.1,1.1))
        weighted = solve(z,weights,now+duration*float(params['explore_fraction']),45)
        if time.monotonic() >= end:
            break
        restored = solve(weighted,np.ones(N),branch_end,100)
        consider(restored)
        # A persistent every-third branch avoids resetting every excursion.
        if attempt % 3 == 1:
            branch_base = restored.copy()
        else:
            branch_base = origin.copy()
    return best
