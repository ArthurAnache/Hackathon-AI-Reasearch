import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linprog

PARAMS = {
    'aspect': (1.0, 0.5, 2.0, 'log'),
    'rows': (5, 3, 7),
    'jitter': (0.03, 0.0, 0.15),
    'init_jitter': (0.065, 0.0, 0.15),
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
    x = list(map(Fraction, map(float, p[:, 0])))
    y = list(map(Fraction, map(float, p[:, 1])))
    q = list(map(Fraction, map(float, r)))
    return (max(a+b for a,b in zip(x,q)) - min(a-b for a,b in zip(x,q))
            + max(a+b for a,b in zip(y,q)) - min(a-b for a,b in zip(y,q)))

def distances(p):
    return np.sqrt((p[I,0]-p[J,0])**2 + (p[I,1]-p[J,1])**2)

def score(candidate):
    try:
        a = np.asarray(candidate['circles'], dtype=np.float64)
        if a.shape != (N,3) or not np.isfinite(a).all():
            return -np.inf
        p, r = a[:,:2], a[:,2]
        if not (r > 0).all() or perimeter(p,r) > 2:
            return -np.inf
        if (distances(p) < r[I]+r[J]).any():
            return -np.inf
        return float(np.sum(r))
    except (ValueError, TypeError, KeyError, OverflowError):
        return -np.inf

def output(p,r):
    return {'circles': np.column_stack((p,r)).tolist()}

def layout(rows,jitter,w,h,rng,end):
    per_row = int(np.ceil(N/rows))
    pts = []
    for k in range(rows):
        if time.monotonic() >= end:
            raise Deadline
        count = min(per_row,N-len(pts))
        if count <= 0:
            break
        off = 0.25/per_row if k % 2 else 0.0
        pts.extend(((i+0.5)/per_row+off,(k+0.5)/rows) for i in range(count))
    p = np.clip(np.asarray(pts[:N])+rng.normal(0,jitter,(N,2)),0.02,0.98)
    return p*np.array([w,h])

def penalty(z,mu):
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
    gw = 2*mu*(-b1.sum()+b3.sum())
    return f,np.concatenate((gx,gy,gr,[gw]))

def lp_radii(p,end):
    if time.monotonic() >= end-0.02:
        return None
    m = len(I)
    a = np.zeros((m+4*N+1,N+4))
    b = np.zeros(m+4*N+1)
    a[np.arange(m),I] = 1
    a[np.arange(m),J] = 1
    b[:m] = distances(p)
    for k,(col,sign,coord) in enumerate(((N,1,0),(N+1,-1,0),(N+2,1,1),(N+3,-1,1))):
        if time.monotonic() >= end-0.02:
            return None
        rows = m+k*N+np.arange(N)
        a[rows,np.arange(N)] = 1
        a[rows,col] = sign
        b[rows] = sign*p[:,coord]
    a[-1,N:] = [-1,1,-1,1]
    b[-1] = 2
    remaining = end-time.monotonic()-0.02
    if remaining <= 0:
        return None
    res = linprog(-np.r_[np.ones(N),np.zeros(4)],A_ub=a,b_ub=b,
                  bounds=[(0,None)]*N+[(None,None)]*4,method='highs',
                  options={'time_limit': remaining})
    return res.x[:N] if res.success else None

def repair(p,r,end):
    r = np.maximum(np.asarray(r,dtype=float),1e-12)*(1-1e-12)
    d = distances(p)
    for unused in range(5):
        if time.monotonic() >= end:
            return None
        if not (d < r[I]+r[J]).any():
            break
        for i,j,dd in zip(I,J,d):
            if time.monotonic() >= end:
                return None
            if r[i]+r[j] > dd*(1-1e-12):
                s = dd*(1-1e-12)/(r[i]+r[j])
                r[i] *= s
                r[j] *= s
    for unused in range(60):
        if time.monotonic() >= end:
            return None
        excess = perimeter(p,r)-2
        if excess <= 0:
            return r
        r *= 1-max(2*float(excess)/float(r.sum()),1e-12)
    return r

def construct(params,seed,budget_s,init=None):
    start = time.monotonic()
    end = start+0.70*budget_s
    opt_end = start+0.53*budget_s
    rng = np.random.default_rng(seed)
    # A cold-start fallback is independently feasible.
    ids = np.arange(N)
    fallback = output(np.column_stack((0.1+0.15*(ids%5),0.1+0.15*(ids//5))),np.full(N,0.01))
    best = init if init is not None else fallback
    best_score = score(best)
    if best_score == -np.inf:
        best,best_score = fallback,score(fallback)
    if time.monotonic() >= opt_end:
        return best
    if init is not None:
        a = np.asarray(init['circles'],dtype=float)
        p0,r0 = a[:,:2],np.abs(a[:,2])
        lo = (p0-r0[:,None]).min(axis=0)
        side = (p0+r0[:,None]).max(axis=0)-lo
        scale = 2/side.sum()
        p,r,w = (p0-lo)*scale,r0*scale,side[0]*scale
        p = p+rng.normal(0,float(params['init_jitter']),p.shape)
        original_centres = p.copy()
    else:
        aspect = float(params['aspect'])
        w = 2*aspect/(1+aspect)
        try:
            p = layout(int(params['rows']),float(params['jitter']),w,2-w,rng,opt_end)
        except Deadline:
            return best
        r = np.full(N,0.05)
        original_centres = None
    z = np.concatenate((p[:,0],p[:,1],r,[w]))
    mu = float(params['mu0'])
    last = [z.copy()]
    def objective(v):
        if time.monotonic() >= opt_end:
            raise Deadline
        last[0] = v.copy()
        return penalty(v,mu)
    for stage in range(int(params['stages'])):
        if time.monotonic() >= opt_end:
            break
        try:
            result = minimize(objective,z,jac=True,method='L-BFGS-B',
                              options={'maxiter':int(params['maxiter'])})
            z = result.x
        except Deadline:
            z = last[0]
            break
        mu *= float(params['mu_growth'])
    candidates = [np.column_stack((z[:N],z[N:2*N]))]
    if original_centres is not None:
        candidates.append(original_centres)
    for pc in candidates:
        if time.monotonic() >= end-0.03:
            break
        rc = lp_radii(pc,end)
        if rc is None:
            continue
        rc = repair(pc,rc,end)
        if rc is None or time.monotonic() >= end:
            continue
        candidate = output(pc,rc)
        value = score(candidate)
        if value > best_score:
            best,best_score = candidate,value
    return best
