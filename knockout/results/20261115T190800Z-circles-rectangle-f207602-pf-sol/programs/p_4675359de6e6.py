import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'minimum_delete': (3, 3, 7),
    'maximum_delete': (7, 3, 10),
    'samples': (650, 200, 2000),
    'iterations': (85, 20, 200),
    'time_fraction': (0.68, 0.4, 0.75),
}
N = 21
I, J = np.triu_indices(N, 1)

class StopSearch(Exception):
    pass

def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2] + a[J, 2]):
        return -np.inf
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    perimeter = (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
                 + max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)))
    if perimeter > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))

def repaired(z, deadline):
    a = np.column_stack((z[:N], z[N:2*N], np.maximum(z[2*N:3*N], 1e-12)))
    if not np.isfinite(a).all():
        return None
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    factor = min(1.0, float(np.min(d/(a[I, 2]+a[J, 2]))))
    if factor <= 0:
        return None
    a[:, 2] *= factor*(1-2e-14)
    for unused in range(5):
        if time.monotonic() > deadline:
            return None
        lo = np.min(a[:, :2]-a[:, 2,None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2,None], axis=0)
        s = min(1.0, 2.0/float(np.sum(hi-lo))) * (1-2e-15)
        a[:, :2] = (a[:, :2]-lo)*s
        a[:, 2] *= s
        if exact_score(a) > -np.inf:
            return a
    return None

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + float(params['time_fraction'])*budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a0 = np.array([[(i % 7 + .5)/3.5, (i//7+.5)/1.5, .02] for i in range(N)])
        a0[:, :2] *= .5
        best = {'circles': a0.tolist()}
    else:
        a0 = np.asarray(init['circles'], dtype=float).copy()
        best = init
    best_score = exact_score(a0)
    lo = np.min(a0[:, :2]-a0[:, 2,None], axis=0)
    side = np.max(a0[:, :2]+a0[:, 2,None], axis=0)-lo
    scale = 2.0/side.sum()
    base = a0.copy()
    base[:, :2] = (base[:, :2]-lo)*scale
    base[:, 2] *= scale
    width = float(side[0]*scale)
    height = 2-width
    objective_gradient = np.r_[np.zeros(2*N), -np.ones(N), 0.0]
    last = [None]

    def check_time():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def objective(z):
        check_time()
        return -float(z[2*N:3*N].sum())

    def constraints(z):
        check_time()
        x,y,r,w = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
        dx,dy = x[I]-x[J], y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        return np.r_[d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r]

    def jacobian(z):
        check_time()
        x,y = z[:N], z[N:2*N]
        dx,dy = x[I]-x[J], y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        m = len(I)
        g = np.zeros((m+4*N,3*N+1))
        k = np.arange(m)
        g[k,I],g[k,J] = dx/d,-dx/d
        g[k,N+I],g[k,N+J] = dy/d,-dy/d
        g[k,2*N+I],g[k,2*N+J] = -1,-1
        v = np.arange(N)
        g[m+v,v],g[m+v,2*N+v] = 1,-1
        g[m+N+v,v],g[m+N+v,2*N+v],g[m+N+v,-1] = -1,-1,1
        g[m+2*N+v,N+v],g[m+2*N+v,2*N+v] = 1,-1
        g[m+3*N+v,N+v],g[m+3*N+v,2*N+v],g[m+3*N+v,-1] = -1,-1,-1
        return g

    def callback(z):
        last[0] = z.copy()
        check_time()

    attempt = 0
    while time.monotonic() < deadline - .06:
        count = int(params['minimum_delete']) + attempt % (max(0,int(params['maximum_delete'])-int(params['minimum_delete']))+1)
        anchor = int(rng.integers(N))
        distances = np.linalg.norm(base[:,:2]-base[anchor,:2], axis=1)
        priority = distances + rng.uniform(-.045,.045,N)
        removed = np.argsort(priority)[:count]
        keep = np.ones(N,dtype=bool)
        keep[removed] = False
        a = base.copy()
        live = list(np.flatnonzero(keep))
        try:
            for index in removed:
                check_time()
                points = rng.random((int(params['samples']),2))*[width,height]
                points = np.vstack((points, base[removed,:2]))
                clearance = np.minimum.reduce((points[:,0], width-points[:,0], points[:,1], height-points[:,1]))
                if live:
                    others = a[live]
                    gap = np.linalg.norm(points[:,None,:]-others[None,:,:2],axis=2)-others[None,:,2]
                    clearance = np.minimum(clearance,gap.min(axis=1))
                # Deliberately vary the first inserted disk to explore alternative cavity decompositions.
                top = np.argsort(clearance)[-min(5,len(points)):]
                chosen = int(top[-1] if len(live)>N-count else top[int(rng.integers(len(top)))])
                q = np.r_[points[chosen],max(1e-7,clearance[chosen])]
                others = a[live].copy()
                def disk_constraints(v):
                    check_time()
                    dd = np.linalg.norm(v[:2]-others[:,:2],axis=1)
                    return np.r_[v[0]-v[2],width-v[0]-v[2],v[1]-v[2],height-v[1]-v[2],dd-v[2]-others[:,2]]
                res = minimize(lambda v: -v[2],q,jac=lambda v: np.array([0.,0.,-1.]),
                               constraints={'type':'ineq','fun':disk_constraints},method='SLSQP',
                               bounds=[(0,width),(0,height),(1e-8,min(width,height)/2)],
                               options={'maxiter':15,'ftol':1e-9})
                a[index] = res.x
                live.append(int(index))
            z = np.r_[a[:,0],a[:,1],a[:,2],width]
            last[0] = z.copy()
            result = minimize(objective,z,jac=lambda z: objective_gradient,method='SLSQP',
                              constraints={'type':'ineq','fun':constraints,'jac':jacobian},
                              bounds=[(0,2)]*(2*N)+[(1e-9,1)]*N+[(.15,1.85)],
                              callback=callback,
                              options={'maxiter':int(params['iterations']),'ftol':2e-12})
            last[0] = result.x
        except StopSearch:
            break
        if last[0] is not None and time.monotonic() < deadline:
            candidate = repaired(last[0],deadline)
            if candidate is not None:
                score = exact_score(candidate)
                if score > best_score:
                    best_score = score
                    best = {'circles':candidate.tolist()}
        attempt += 1
    return best
