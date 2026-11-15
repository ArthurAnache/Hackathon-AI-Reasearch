import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linprog

PARAMS = {
    "aspect": (1.0, 0.5, 2.0, "log"),
    "rows": (5, 3, 7),
    "jitter": (0.03, 0.0, 0.15),
    "init_jitter": (0.0, 0.0, 0.05),
    "mu0": (0.5, 0.01, 1000.0, "log"),
    "mu_growth": (8.0, 2.0, 30.0),
    "stages": (8, 2, 12),
    "maxiter": (150, 30, 1000)
}
N = 21
I, J = np.triu_indices(N, 1)

class Deadline(Exception):
    pass

def layout(rows, jitter, w, h, rng, deadline):
    per = int(np.ceil(N / rows))
    pts = []
    for k in range(rows):
        if time.monotonic() >= deadline:
            raise Deadline
        count = min(per, N - len(pts))
        if count <= 0:
            break
        off = 0.25 / per if k % 2 else 0.0
        pts.extend([((i + 0.5) / per + off, (k + 0.5) / rows) for i in range(count)])
    p = np.clip(np.asarray(pts[:N]) + rng.normal(0, jitter, (N, 2)), 0.02, 0.98)
    return p * [w, h]

def penalty(z, mu):
    x, y, r, w = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
    dx, dy = x[I] - x[J], y[I] - y[J]
    d = np.sqrt(dx*dx + dy*dy + 1e-18)
    v = np.maximum(r[I] + r[J] - d, 0)
    b0 = np.maximum(r-x, 0)
    b1 = np.maximum(r+x-w, 0)
    b2 = np.maximum(r-y, 0)
    b3 = np.maximum(r+y-(2-w), 0)
    neg = np.maximum(-r, 0)
    f = -r.sum() + mu * (v@v + b0@b0 + b1@b1 + b2@b2 + b3@b3 + neg@neg)
    c = 2*mu*v
    gx, gy, gr = np.zeros(N), np.zeros(N), -np.ones(N)
    np.add.at(gr, I, c)
    np.add.at(gr, J, c)
    np.add.at(gx, I, -c*dx/d)
    np.add.at(gx, J, c*dx/d)
    np.add.at(gy, I, -c*dy/d)
    np.add.at(gy, J, c*dy/d)
    gr += 2*mu*(b0+b1+b2+b3-neg)
    gx += 2*mu*(-b0+b1)
    gy += 2*mu*(-b2+b3)
    gw = 2*mu*(-b1.sum()+b3.sum())
    return f, np.r_[gx, gy, gr, gw]

def half_perimeter(p, r):
    xs = list(map(lambda v: Fraction(float(v)), p[:, 0]))
    ys = list(map(lambda v: Fraction(float(v)), p[:, 1]))
    rs = list(map(lambda v: Fraction(float(v)), r))
    return (max(x+q for x,q in zip(xs,rs)) - min(x-q for x,q in zip(xs,rs))
            + max(y+q for y,q in zip(ys,rs)) - min(y-q for y,q in zip(ys,rs)))

def distances(p):
    return np.sqrt((p[I,0]-p[J,0])**2 + (p[I,1]-p[J,1])**2)

def score(p, r):
    if p.shape != (N,2) or r.shape != (N,):
        return -np.inf
    if not np.isfinite(p).all() or not np.isfinite(r).all() or not (r > 0).all():
        return -np.inf
    if (distances(p) < r[I]+r[J]).any() or half_perimeter(p,r) > 2:
        return -np.inf
    return float(np.sum(r))

def lp_radii(p, deadline):
    if time.monotonic() >= deadline:
        raise Deadline
    m = len(I)
    a = np.zeros((m+4*N+1,N+4))
    b = np.zeros(m+4*N+1)
    a[np.arange(m),I] = 1
    a[np.arange(m),J] = 1
    b[:m] = distances(p)
    for k,(col,sign,coord) in enumerate([(N,1,0),(N+1,-1,0),(N+2,1,1),(N+3,-1,1)]):
        if time.monotonic() >= deadline:
            raise Deadline
        rr = m+k*N+np.arange(N)
        a[rr,np.arange(N)] = 1
        a[rr,col] = sign
        b[rr] = sign*p[:,coord]
    a[-1,N:] = [-1,1,-1,1]
    b[-1] = 2
    remaining = deadline-time.monotonic()
    if remaining <= 0:
        raise Deadline
    res = linprog(-np.r_[np.ones(N),np.zeros(4)], A_ub=a, b_ub=b,
                  bounds=[(0,None)]*N+[(None,None)]*4, method='highs',
                  options={'time_limit': max(0.001, remaining)})
    return res.x[:N] if res.success else None

def repair(p, r, deadline):
    r = np.maximum(np.asarray(r),1e-12)*(1-1e-12)
    d = distances(p)
    for _ in range(5):
        if time.monotonic() >= deadline:
            raise Deadline
        if not (d < r[I]+r[J]).any():
            break
        for i,j,dd in zip(I,J,d):
            if time.monotonic() >= deadline:
                raise Deadline
            if r[i]+r[j] > dd*(1-1e-12):
                s = dd*(1-1e-12)/(r[i]+r[j])
                r[i] *= s
                r[j] *= s
    for _ in range(60):
        if time.monotonic() >= deadline:
            raise Deadline
        excess = half_perimeter(p,r)-2
        if excess <= 0:
            break
        r *= 1-max(2*float(excess)/float(r.sum()),1e-12)
    return r

def output(p,r):
    return {'circles': np.column_stack((p,r)).tolist()}

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    optimization_end = start + 0.57*budget_s
    rng = np.random.default_rng(seed)
    best, best_score = None, -np.inf
    if init is not None:
        ar = np.asarray(init['circles'],dtype=float)
        p0, r0 = ar[:,:2].copy(), ar[:,2].copy()
        best_score = score(p0,r0)
        if np.isfinite(best_score):
            best = init
        lo = (p0-np.abs(r0)[:,None]).min(0)
        side = (p0+np.abs(r0)[:,None]).max(0)-lo
        scale = 2/side.sum()
        p = (p0-lo)*scale
        r = np.abs(r0)*scale
        w = side[0]*scale
        p += rng.normal(0,float(params['init_jitter']),p.shape)
        candidates = [p.copy()]
    else:
        w = 2*float(params['aspect'])/(1+float(params['aspect']))
        try:
            p = layout(int(params['rows']),float(params['jitter']),w,2-w,rng,deadline)
        except Deadline:
            p = np.column_stack((np.arange(N)*0.08,np.zeros(N)))
        r = np.full(N,0.05)
        candidates = []
    z = np.r_[p[:,0],p[:,1],r,w]
    last = z.copy()
    def fun(v,mu):
        nonlocal last
        if time.monotonic() >= optimization_end:
            raise Deadline
        last = v.copy()
        return penalty(v,mu)
    mu = float(params['mu0'])
    for _ in range(int(params['stages'])):
        if time.monotonic() >= optimization_end:
            break
        try:
            result = minimize(fun,z,args=(mu,),jac=True,method='L-BFGS-B',
                              options={'maxiter':int(params['maxiter'])})
            z = result.x
        except Deadline:
            z = last
            break
        mu *= float(params['mu_growth'])
    candidates.append(np.column_stack((z[:N],z[N:2*N])))
    # The converged/rearranged candidate gets the first share of the LP reserve.
    for pc in reversed(candidates):
        if time.monotonic() >= deadline:
            break
        try:
            rc = lp_radii(pc,deadline)
            if rc is None:
                continue
            rc = repair(pc,rc,deadline)
            val = score(pc,rc)
            if val > best_score:
                best_score = val
                best = output(pc,rc)
        except Deadline:
            break
    if best is None:
        # Universal valid cold-start fallback, independent of optimization progress.
        pc = np.column_stack((0.04+0.08*np.arange(N),np.full(N,0.04)))
        best = output(pc,np.full(N,0.039))
    return best
