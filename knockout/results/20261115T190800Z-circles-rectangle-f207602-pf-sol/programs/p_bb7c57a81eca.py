import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linprog

PARAMS = {
    'aspect': (1.0, 0.5, 2.0, 'log'),
    'rows': (5, 3, 7),
    'jitter': (0.03, 0.0, 0.15),
    'init_jitter': (0.045, 0.0, 0.05),
    'mu0': (10.0, 1.0, 1000.0, 'log'),
    'mu_growth': (8.0, 2.0, 30.0),
    'stages': (6, 2, 12),
    'maxiter': (150, 30, 1000),
}
N = 21
I, J = np.triu_indices(N, 1)

class Deadline(Exception):
    pass

def perimeter(p, r):
    x = list(map(lambda v: Fraction(float(v)), p[:, 0]))
    y = list(map(lambda v: Fraction(float(v)), p[:, 1]))
    q = list(map(lambda v: Fraction(float(v)), r))
    return (max(a+b for a,b in zip(x,q)) - min(a-b for a,b in zip(x,q))
            + max(a+b for a,b in zip(y,q)) - min(a-b for a,b in zip(y,q)))

def distances(p):
    return np.sqrt((p[I,0]-p[J,0])**2 + (p[I,1]-p[J,1])**2)

def valid(p, r):
    return (p.shape == (21,2) and r.shape == (21,) and np.isfinite(p).all()
            and np.isfinite(r).all() and (r > 0).all()
            and perimeter(p,r) <= 2 and not (distances(p) < r[I]+r[J]).any())

def repair(p, r, deadline):
    r = np.maximum(r, 1e-12) * (1-1e-12)
    d = distances(p)
    for k in range(5):
        if time.monotonic() >= deadline:
            break
        if not (d < r[I]+r[J]).any():
            break
        for i,j,dd in zip(I,J,d):
            if time.monotonic() >= deadline:
                break
            if r[i]+r[j] > dd*(1-1e-12):
                s = dd*(1-1e-12)/(r[i]+r[j])
                r[i] *= s
                r[j] *= s
    for k in range(60):
        if time.monotonic() >= deadline:
            break
        excess = perimeter(p,r)-2
        if excess <= 0:
            break
        r *= 1-max(2*float(excess)/float(r.sum()),1e-12)
    return r

def lp_radii(p, deadline):
    m = len(I)
    a = np.zeros((m+4*N+1,N+4))
    b = np.zeros(m+4*N+1)
    a[np.arange(m),I] = 1
    a[np.arange(m),J] = 1
    b[:m] = distances(p)
    for k,(col,sign,coord) in enumerate([(N,1,0),(N+1,-1,0),(N+2,1,1),(N+3,-1,1)]):
        if time.monotonic() >= deadline:
            return None
        rows = m+k*N+np.arange(N)
        a[rows,np.arange(N)] = 1
        a[rows,col] = sign
        b[rows] = sign*p[:,coord]
    a[-1,N:] = [-1,1,-1,1]
    b[-1] = 2
    remaining = deadline-time.monotonic()
    if remaining <= 0:
        return None
    res = linprog(-np.r_[np.ones(N),np.zeros(4)], A_ub=a, b_ub=b,
                  bounds=[(0,None)]*N+[(None,None)]*4,
                  method='highs', options={'time_limit': max(0.001,remaining)})
    return res.x[:N] if res.success else None

def penalty(z, mu):
    x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
    dx,dy = x[I]-x[J],y[I]-y[J]
    d = np.sqrt(dx*dx+dy*dy+1e-18)
    v = np.maximum(r[I]+r[J]-d,0)
    b0 = np.maximum(r-x,0)
    b1 = np.maximum(r+x-w,0)
    b2 = np.maximum(r-y,0)
    b3 = np.maximum(r+y-(2-w),0)
    neg = np.maximum(-r,0)
    f = -r.sum()+mu*(v@v+b0@b0+b1@b1+b2@b2+b3@b3+neg@neg)
    gx,gy,gr = np.zeros(N),np.zeros(N),-np.ones(N)
    c = 2*mu*v
    np.add.at(gr,I,c)
    np.add.at(gr,J,c)
    np.add.at(gx,I,-c*dx/d)
    np.add.at(gx,J,c*dx/d)
    np.add.at(gy,I,-c*dy/d)
    np.add.at(gy,J,c*dy/d)
    gr += 2*mu*(b0+b1+b2+b3-neg)
    gx += 2*mu*(-b0+b1)
    gy += 2*mu*(-b2+b3)
    return f,np.r_[gx,gy,gr,2*mu*(-b1.sum()+b3.sum())]

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start+0.70*budget_s
    optimize_end = start+0.56*budget_s
    rng = np.random.default_rng(seed)
    best = None
    best_score = -np.inf
    if init is not None:
        a = np.asarray(init['circles'], dtype=float)
        p0,r0 = a[:,:2].copy(),a[:,2].copy()
        if valid(p0,r0):
            best,best_score = init,float(np.sum(r0))
        lo = (p0-np.abs(r0)[:,None]).min(axis=0)
        side = (p0+np.abs(r0)[:,None]).max(axis=0)-lo
        scale = 2/side.sum()
        p = (p0-lo)*scale
        r = np.abs(r0)*scale
        w = side[0]*scale
        p += rng.normal(0,float(params['init_jitter']),p.shape)
        candidates = [p.copy()]
    else:
        aspect = float(params['aspect'])
        w = 2*aspect/(1+aspect)
        rows = int(params['rows'])
        per_row = int(np.ceil(N/rows))
        points = []
        for k in range(rows):
            if time.monotonic() >= end:
                break
            count = min(per_row,N-len(points))
            if count <= 0:
                break
            off = 0.25/per_row if k%2 else 0.0
            points.extend([((i+0.5)/per_row+off,(k+0.5)/rows) for i in range(count)])
        if len(points) != N:
            p = np.column_stack((np.linspace(0.02,0.98,N),np.full(N,0.5)))
        else:
            p = np.clip(np.array(points)+rng.normal(0,float(params['jitter']),(N,2)),0.02,0.98)
        p *= np.array([w,2-w])
        r = np.full(N,0.05)
        candidates = []
    z = np.r_[p[:,0],p[:,1],r,w]
    mu = float(params['mu0'])
    current = [z.copy()]
    def objective(v):
        if time.monotonic() >= optimize_end:
            raise Deadline()
        return penalty(v,mu)
    def callback(v):
        current[0] = v.copy()
        if time.monotonic() >= optimize_end:
            raise Deadline()
    for stage in range(int(params['stages'])):
        if time.monotonic() >= optimize_end:
            break
        current[0] = z.copy()
        try:
            res = minimize(objective,z,jac=True,method='L-BFGS-B',callback=callback,
                           options={'maxiter':int(params['maxiter'])})
            z = res.x
        except Deadline:
            z = current[0]
            break
        mu *= float(params['mu_growth'])
    candidates.append(np.column_stack((z[:N],z[N:2*N])))
    for pc in reversed(candidates):
        if time.monotonic() >= end:
            break
        rc = lp_radii(pc,end)
        if rc is None:
            continue
        rc = repair(pc,rc,end)
        if valid(pc,rc):
            score = float(np.sum(rc))
            if score > best_score:
                best_score = score
                best = {'circles':np.column_stack((pc,rc)).tolist()}
    if best is None:
        # Explicitly separated fallback, independent of a failed optimization.
        pc = np.column_stack((np.linspace(0.02,0.98,N),np.full(N,0.5)))
        best = {'circles':np.column_stack((pc,np.full(N,0.001))).tolist()}
    return best
