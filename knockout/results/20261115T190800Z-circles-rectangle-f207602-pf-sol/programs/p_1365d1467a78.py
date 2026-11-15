import time
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize

PARAMS = {
    'aspect': (1.0, 0.5, 2.0, 'log'),
    'rows': (5, 3, 7),
    'jitter': (0.03, 0.0, 0.15),
    'init_jitter': (0.008, 0.0, 0.05),
    'mu0': (10.0, 1.0, 1000.0, 'log'),
    'mu_growth': (8.0, 2.0, 30.0),
    'stages': (6, 2, 12),
    'maxiter': (150, 30, 1000),
}
N = 21
I, J = np.triu_indices(N, 1)

class StopSearch(Exception):
    pass


def exact_score(data):
    try:
        a = np.asarray(data['circles'], dtype=np.float64)
        if a.shape != (N, 3) or not np.isfinite(a).all():
            return -np.inf
        p, r = a[:, :2], a[:, 2]
        if np.any(r <= 0):
            return -np.inf
        d = np.sqrt((p[I, 0]-p[J, 0])**2 + (p[I, 1]-p[J, 1])**2)
        if np.any(d < r[I]+r[J]):
            return -np.inf
        if half_perimeter(p, r) > 2:
            return -np.inf
        return float(np.sum(r))
    except (ValueError, TypeError, KeyError, OverflowError):
        return -np.inf


def half_perimeter(p, r):
    x = list(map(lambda v: Fraction(float(v)), p[:, 0]))
    y = list(map(lambda v: Fraction(float(v)), p[:, 1]))
    q = list(map(lambda v: Fraction(float(v)), r))
    return (max(a+b for a,b in zip(x,q))-min(a-b for a,b in zip(x,q))
            + max(a+b for a,b in zip(y,q))-min(a-b for a,b in zip(y,q)))


def layout(rows, jitter, w, h, rng, deadline):
    per = int(np.ceil(N/rows))
    pts = []
    for k in range(rows):
        if time.monotonic() >= deadline:
            break
        count = min(per, N-len(pts))
        if count <= 0:
            break
        off = 0.25/per if k % 2 else 0.0
        pts.extend([((i+0.5)/per+off, (k+0.5)/rows) for i in range(count)])
    if len(pts) < N:
        pts.extend(rng.uniform(0.1, 0.9, (N-len(pts), 2)).tolist())
    return np.clip(np.asarray(pts[:N])+rng.normal(0, jitter, (N,2)), 0.02, 0.98)*[w,h]


def penalty(z, mu):
    x, y, r, w = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
    dx, dy = x[I]-x[J], y[I]-y[J]
    d = np.sqrt(dx*dx+dy*dy+1e-18)
    v = np.maximum(r[I]+r[J]-d, 0)
    b = np.maximum(np.array([r-x, r+x-w, r-y, r+y-(2-w)]), 0)
    neg = np.maximum(-r, 0)
    f = -r.sum()+mu*(np.sum(v*v)+np.sum(b*b)+np.sum(neg*neg))
    gx, gy, gr = np.zeros(N), np.zeros(N), -np.ones(N)
    c = 2*mu*v
    np.add.at(gr, I, c)
    np.add.at(gr, J, c)
    np.add.at(gx, I, -c*dx/d)
    np.add.at(gx, J, c*dx/d)
    np.add.at(gy, I, -c*dy/d)
    np.add.at(gy, J, c*dy/d)
    gr += 2*mu*b.sum(axis=0)-2*mu*neg
    gx += 2*mu*(-b[0]+b[1])
    gy += 2*mu*(-b[2]+b[3])
    gw = 2*mu*(-b[1].sum()+b[3].sum())
    return f, np.r_[gx, gy, gr, gw]


def lp_radii(p, deadline):
    m = len(I)
    a = np.zeros((m+4*N+1, N+4))
    b = np.zeros(m+4*N+1)
    a[np.arange(m), I] = 1
    a[np.arange(m), J] = 1
    b[:m] = np.sqrt(np.sum((p[I]-p[J])**2, axis=1))
    for k, (col, sign, coord) in enumerate([(N,1,0),(N+1,-1,0),(N+2,1,1),(N+3,-1,1)]):
        if time.monotonic() >= deadline:
            return None
        rows = m+k*N+np.arange(N)
        a[rows, np.arange(N)] = 1
        a[rows, col] = sign
        b[rows] = sign*p[:,coord]
    a[-1,N:] = [-1,1,-1,1]
    b[-1] = 2
    remaining = deadline-time.monotonic()
    if remaining <= 0:
        return None
    res = linprog(-np.r_[np.ones(N),np.zeros(4)], A_ub=a, b_ub=b,
                  bounds=[(0,None)]*N+[(None,None)]*4, method='highs',
                  options={'time_limit': max(0.001, remaining)})
    return res.x[:N] if res.success else None


def repair(p, radii, deadline):
    r = np.maximum(np.asarray(radii, dtype=float), 1e-12)*(1-1e-12)
    d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
    for _ in range(5):
        if time.monotonic() >= deadline:
            break
        bad = np.flatnonzero(d < r[I]+r[J])
        if not len(bad):
            break
        for k in bad:
            if time.monotonic() >= deadline:
                break
            i,j = I[k],J[k]
            if r[i]+r[j] > d[k]*(1-1e-12):
                s = d[k]*(1-1e-12)/(r[i]+r[j])
                r[i] *= s
                r[j] *= s
    for _ in range(60):
        if time.monotonic() >= deadline:
            break
        excess = half_perimeter(p,r)-2
        if excess <= 0:
            break
        r *= 1-max(2*float(excess)/float(r.sum()),1e-12)
    return {'circles': np.column_stack((p,r)).tolist()}


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start+0.70*budget_s
    search_end = start+0.54*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = exact_score(init) if init is not None else -np.inf
    if init is not None:
        a = np.asarray(init['circles'], dtype=float)
        p0,r0 = a[:,:2],a[:,2]
        lo = (p0-np.abs(r0)[:,None]).min(axis=0)
        side = (p0+np.abs(r0)[:,None]).max(axis=0)-lo
        scale = 2/side.sum()
        p = (p0-lo)*scale
        r = np.abs(r0)*scale
        w = side[0]*scale
        candidates = [p.copy()]
        p += rng.normal(0,float(params['init_jitter']),p.shape)
    else:
        aspect = float(params['aspect'])
        w = 2*aspect/(1+aspect)
        p = layout(int(params['rows']),float(params['jitter']),w,2-w,rng,deadline)
        r = np.full(N,0.05)
        candidates = []
    z = np.r_[p[:,0],p[:,1],r,w]
    latest = [z.copy()]
    def objective(v, mu):
        if time.monotonic() >= search_end:
            raise StopSearch()
        return penalty(v,mu)
    def callback(v):
        latest[0] = v.copy()
        if time.monotonic() >= search_end:
            raise StopSearch()
    mu = float(params['mu0'])
    for _ in range(int(params['stages'])):
        if time.monotonic() >= search_end:
            break
        latest[0] = z.copy()
        try:
            z = minimize(objective,z,args=(mu,),jac=True,method='L-BFGS-B',
                         callback=callback,options={'maxiter':int(params['maxiter'])}).x
        except StopSearch:
            z = latest[0]
            break
        mu *= float(params['mu_growth'])
    candidates.append(np.column_stack((z[:N],z[N:2*N])))
    for pc in candidates:
        if time.monotonic() >= deadline:
            break
        rc = lp_radii(pc,deadline)
        if rc is None:
            continue
        candidate = repair(pc,rc,deadline)
        score = exact_score(candidate)
        if score > best_score:
            best,best_score = candidate,score
    if best is None:
        # Deterministic valid fallback for a cold call.
        best = {'circles': [[float(i)*0.08,0.0,0.001] for i in range(N)]}
    return best
