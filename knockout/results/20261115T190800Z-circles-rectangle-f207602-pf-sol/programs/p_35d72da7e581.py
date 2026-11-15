import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linprog

PARAMS = {
    'aspect': (1.0, 0.5, 2.0, 'log'),
    'rows': (5, 3, 7),
    'jitter': (0.03, 0.0, 0.15),
    'init_jitter': (0.0, 0.0, 0.05),
    'radius_equalization': (1.0, 0.0, 1.0),
    'mu0': (10.0, 1.0, 1000.0, 'log'),
    'mu_growth': (8.0, 2.0, 30.0),
    'stages': (6, 2, 12),
    'maxiter': (150, 30, 1000)
}
N = 21
I, J = np.triu_indices(N, 1)

class StopSearch(Exception):
    pass

def perimeter(p, r):
    x = list(map(lambda v: Fraction(float(v)), p[:, 0]))
    y = list(map(lambda v: Fraction(float(v)), p[:, 1]))
    q = list(map(lambda v: Fraction(float(v)), r))
    return (max(a+b for a,b in zip(x,q))-min(a-b for a,b in zip(x,q))
            +max(a+b for a,b in zip(y,q))-min(a-b for a,b in zip(y,q)))

def valid(p, r):
    if p.shape != (N,2) or r.shape != (N,):
        return False
    if not np.isfinite(p).all() or not np.isfinite(r).all() or not (r > 0).all():
        return False
    d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
    return not np.any(d < r[I]+r[J]) and perimeter(p,r) <= 2

def penalty(z, mu):
    x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
    dx,dy = x[I]-x[J],y[I]-y[J]
    d = np.sqrt(dx*dx+dy*dy+1e-18)
    v = np.maximum(r[I]+r[J]-d,0)
    a = np.maximum(r-x,0)
    b = np.maximum(r+x-w,0)
    c = np.maximum(r-y,0)
    e = np.maximum(r+y-(2-w),0)
    neg = np.maximum(-r,0)
    f = -r.sum()+mu*(v@v+a@a+b@b+c@c+e@e+neg@neg)
    gx,gy,gr = np.zeros(N),np.zeros(N),-np.ones(N)
    t = 2*mu*v
    np.add.at(gr,I,t)
    np.add.at(gr,J,t)
    np.add.at(gx,I,-t*dx/d)
    np.add.at(gx,J,t*dx/d)
    np.add.at(gy,I,-t*dy/d)
    np.add.at(gy,J,t*dy/d)
    gr += 2*mu*(a+b+c+e-neg)
    gx += 2*mu*(-a+b)
    gy += 2*mu*(-c+e)
    return f,np.r_[gx,gy,gr,2*mu*(-b.sum()+e.sum())]

def lp_radii(p, remaining):
    m = len(I)
    a = np.zeros((m+4*N+1,N+4))
    b = np.zeros(m+4*N+1)
    a[np.arange(m),I] = 1
    a[np.arange(m),J] = 1
    b[:m] = np.sqrt(((p[I]-p[J])**2).sum(1))
    ix = np.arange(N)
    a[m+ix,ix] = 1
    a[m+ix,N] = 1
    b[m+ix] = p[:,0]
    a[m+N+ix,ix] = 1
    a[m+N+ix,N+1] = -1
    b[m+N+ix] = -p[:,0]
    a[m+2*N+ix,ix] = 1
    a[m+2*N+ix,N+2] = 1
    b[m+2*N+ix] = p[:,1]
    a[m+3*N+ix,ix] = 1
    a[m+3*N+ix,N+3] = -1
    b[m+3*N+ix] = -p[:,1]
    a[-1,N:] = [-1,1,-1,1]
    b[-1] = 2
    res = linprog(-np.r_[np.ones(N),np.zeros(4)], A_ub=a,b_ub=b,
                  bounds=[(0,None)]*N+[(None,None)]*4,method='highs',
                  options={'time_limit':max(0.001,remaining)})
    return res.x[:N] if res.success else None

def repair(p, r, deadline):
    r = np.maximum(r,1e-12)*(1-1e-12)
    d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
    ratio = min(1.0,float(np.min(d/(r[I]+r[J]))))
    if ratio < 1:
        r *= ratio*(1-1e-12)
    for unused in range(60):
        if time.monotonic() >= deadline:
            break
        excess = perimeter(p,r)-2
        if excess <= 0:
            break
        r *= 1-max(2*float(excess)/float(r.sum()),1e-12)
    return r

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start+0.70*budget_s
    opt_end = start+0.57*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf
    if init is not None:
        data = np.asarray(init['circles'],dtype=float)
        p0,r0 = data[:,:2].copy(),data[:,2].copy()
        if valid(p0,r0):
            best_score = float(np.sum(r0))
        lo = (p0-np.abs(r0)[:,None]).min(0)
        sides = (p0+np.abs(r0)[:,None]).max(0)-lo
        scale = 2/sides.sum()
        p,r,w = (p0-lo)*scale,np.abs(r0)*scale,sides[0]*scale
        p += rng.normal(0,float(params['init_jitter']),p.shape)
        candidates = [p.copy()]
        # The sole search change: reset initial radius heterogeneity.
        mix = float(params['radius_equalization'])
        r = (1-mix)*r+mix*float(r.mean())
    else:
        aspect = float(params['aspect'])
        w = 2*aspect/(1+aspect)
        rows = int(params['rows'])
        cols = int(np.ceil(N/rows))
        k = np.arange(N)
        p = np.column_stack(((k%cols+0.5)/cols+(k//cols%2)*0.25/cols,
                             (k//cols+0.5)/rows))
        p = np.clip(p+rng.normal(0,float(params['jitter']),p.shape),0.02,0.98)
        p *= np.array([w,2-w])
        r = np.full(N,0.05)
        candidates = []
        safe_p = np.column_stack((np.linspace(0.05,1.95,N),np.zeros(N)))
        best = {'circles':np.column_stack((safe_p,np.full(N,0.001))).tolist()}
        best_score = float(np.sum(np.full(N,0.001)))
    z = np.r_[p[:,0],p[:,1],r,w]
    latest = [z.copy()]
    def callback(zz):
        latest[0] = zz.copy()
        if time.monotonic() >= opt_end:
            raise StopSearch()
    mu = float(params['mu0'])
    for stage in range(int(params['stages'])):
        if time.monotonic() >= opt_end:
            break
        latest[0] = z.copy()
        def objective(zz):
            if time.monotonic() >= opt_end:
                raise StopSearch()
            return penalty(zz,mu)
        try:
            z = minimize(objective,z,jac=True,method='L-BFGS-B',callback=callback,
                         options={'maxiter':int(params['maxiter'])}).x
        except StopSearch:
            z = latest[0]
            break
        mu *= float(params['mu_growth'])
    candidates.append(np.column_stack((z[:N],z[N:2*N])))
    for pc in reversed(candidates):
        now = time.monotonic()
        if now >= deadline-0.015:
            break
        rc = lp_radii(pc,deadline-now-0.01)
        if rc is None:
            continue
        rc = repair(pc,rc,deadline)
        score = float(np.sum(rc))
        if score > best_score and valid(pc,rc):
            best_score = score
            best = {'circles':np.column_stack((pc,rc)).tolist()}
    return best
