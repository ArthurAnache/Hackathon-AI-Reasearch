import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linprog

PARAMS = {
    'aspect': (1.0, 0.5, 2.0, 'log'),
    'rows': (5, 3, 7),
    'jitter': (0.03, 0.0, 0.15),
    'init_jitter': (0.0, 0.0, 0.05),
    'mu0': (10.0, 1.0, 1000.0, 'log'),
    'mu_growth': (8.0, 2.0, 30.0),
    'stages': (6, 2, 12),
    'maxiter': (150, 30, 1000),
    'contact_threshold': (2e-7, 1e-10, 1e-5, 'log'),
    'polish_steps': (8, 1, 15)
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class Stop(Exception):
    pass

def perimeter(p, r):
    xs = [Fraction(float(x)) for x in p[:, 0]]
    ys = [Fraction(float(y)) for y in p[:, 1]]
    rs = [Fraction(float(q)) for q in r]
    return (max(x+q for x,q in zip(xs,rs)) - min(x-q for x,q in zip(xs,rs))
            + max(y+q for y,q in zip(ys,rs)) - min(y-q for y,q in zip(ys,rs)))

def score(a):
    if a.shape != (21,3) or not np.isfinite(a).all() or not (a[:,2] > 0).all():
        return -np.inf
    p, r = a[:,:2], a[:,2]
    d = np.sqrt((p[I,0]-p[J,0])**2 + (p[I,1]-p[J,1])**2)
    if (d < r[I]+r[J]).any() or perimeter(p,r) > 2:
        return -np.inf
    return float(np.sum(r))

def repair(p, r, end):
    r = np.maximum(np.asarray(r).copy(), 1e-12) * (1-1e-12)
    d = np.sqrt((p[I,0]-p[J,0])**2 + (p[I,1]-p[J,1])**2)
    for _ in range(5):
        if time.monotonic() >= end:
            break
        if not (d < r[I]+r[J]).any():
            break
        for i,j,dd in zip(I,J,d):
            if time.monotonic() >= end:
                break
            if r[i]+r[j] > dd*(1-1e-12):
                s = dd*(1-1e-12)/(r[i]+r[j])
                r[i] *= s
                r[j] *= s
    for _ in range(60):
        if time.monotonic() >= end:
            break
        excess = perimeter(p,r)-2
        if excess <= 0:
            break
        r *= 1-max(2*float(excess)/float(r.sum()),1e-12)
    return np.column_stack((p,r))

def penalty(z, mu, end):
    if time.monotonic() >= end:
        raise Stop()
    x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
    dx,dy = x[I]-x[J],y[I]-y[J]
    d = np.sqrt(dx*dx+dy*dy+1e-18)
    v = np.maximum(r[I]+r[J]-d,0)
    b = [np.maximum(r-x,0),np.maximum(r+x-w,0),np.maximum(r-y,0),np.maximum(r+y-(2-w),0)]
    neg = np.maximum(-r,0)
    f = -r.sum()+mu*((v*v).sum()+sum((bb*bb).sum() for bb in b)+(neg*neg).sum())
    gx,gy,gr = np.zeros(N),np.zeros(N),-np.ones(N)
    c = 2*mu*v
    np.add.at(gr,I,c)
    np.add.at(gr,J,c)
    np.add.at(gx,I,-c*dx/d)
    np.add.at(gx,J,c*dx/d)
    np.add.at(gy,I,-c*dy/d)
    np.add.at(gy,J,c*dy/d)
    gr += 2*mu*(b[0]+b[1]+b[2]+b[3])-2*mu*neg
    gx += 2*mu*(-b[0]+b[1])
    gy += 2*mu*(-b[2]+b[3])
    return f,np.r_[gx,gy,gr,2*mu*(-b[1].sum()+b[3].sum())]

def lp_radii(p, end):
    if time.monotonic() >= end-0.08:
        return None
    a = np.zeros((M+4*N+1,N+4))
    b = np.zeros(M+4*N+1)
    a[np.arange(M),I] = 1
    a[np.arange(M),J] = 1
    b[:M] = np.sqrt(((p[I]-p[J])**2).sum(1))
    for k,(col,sign,coord) in enumerate([(N,1,0),(N+1,-1,0),(N+2,1,1),(N+3,-1,1)]):
        if time.monotonic() >= end:
            return None
        rows = M+k*N+np.arange(N)
        a[rows,np.arange(N)] = 1
        a[rows,col] = sign
        b[rows] = sign*p[:,coord]
    a[-1,N:] = [-1,1,-1,1]
    b[-1] = 2
    res = linprog(-np.r_[np.ones(N),np.zeros(4)],A_ub=a,b_ub=b,
                  bounds=[(0,None)]*N+[(None,None)]*4,method='highs',
                  options={'time_limit': max(0.001,min(0.12,end-time.monotonic()-0.03))})
    return res.x[:N] if res.success else None

def constraints(z):
    x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
    dx,dy = x[I]-x[J],y[I]-y[J]
    d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-30)
    g = np.r_[d-r[I]-r[J],x-r,w-x-r,y-r,2-w-y-r]
    a = np.zeros((M+4*N,3*N+1))
    k = np.arange(M)
    a[k,I],a[k,J] = dx/d,-dx/d
    a[k,N+I],a[k,N+J] = dy/d,-dy/d
    a[k,2*N+I],a[k,2*N+J] = -1,-1
    k = np.arange(N)
    a[M+k,k] = 1
    a[M+N+k,k] = -1
    a[M+N+k,-1] = 1
    a[M+2*N+k,N+k] = 1
    a[M+3*N+k,N+k] = -1
    a[M+3*N+k,-1] = -1
    a[M+k,2*N+k] = -1
    a[M+N+k,2*N+k] = -1
    a[M+2*N+k,2*N+k] = -1
    a[M+3*N+k,2*N+k] = -1
    return g,a

def polish(z, params, end):
    z = z.copy()
    g,_ = constraints(z)
    active = np.flatnonzero(np.abs(g) < params['contact_threshold'])
    if len(active) < 60:
        return z
    for _ in range(int(params['polish_steps'])):
        if time.monotonic() >= end:
            break
        g,a = constraints(z)
        step = np.linalg.lstsq(a[active],-g[active],rcond=1e-12)[0]
        if not np.isfinite(step).all() or np.linalg.norm(step) > 0.02:
            break
        z += step
        if np.max(np.abs(step)) < 2e-15:
            break
    return z

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start+0.70*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf
    if init is not None:
        a = np.asarray(init['circles'],dtype=float)
        best_score = score(a)
        p,r = a[:,:2].copy(),a[:,2].copy()
        lo = (p-np.abs(r)[:,None]).min(0)
        side = (p+np.abs(r)[:,None]).max(0)-lo
        s = 2/side.sum()
        p,r,w = (p-lo)*s,np.abs(r)*s,side[0]*s
    else:
        aspect = params['aspect']
        w = 2*aspect/(1+aspect)
        rows = int(params['rows'])
        cols = int(np.ceil(N/rows))
        k = np.arange(N)
        p = np.column_stack(((k%cols+0.5)/cols,(k//cols+0.5)/rows))
        p += rng.normal(0,params['jitter'],p.shape)
        p = np.clip(p,0.02,0.98)*[w,2-w]
        r = np.full(N,0.025)
        fallback = np.column_stack((np.linspace(0.02,1.98,N),np.zeros(N),np.full(N,0.001)))
        best = {'circles':fallback.tolist()}
        best_score = score(fallback)
    def consider(pc,rc):
        nonlocal best,best_score
        if time.monotonic() >= end:
            return
        a = repair(pc,rc,end)
        value = score(a)
        if value > best_score:
            best,best_score = {'circles':a.tolist()},value
    z = np.r_[p[:,0],p[:,1],r,w]
    # Numerical addition: solve the nearly active contact equations before
    # the existing penalty continuation can perturb the warm-start graph.
    if time.monotonic() < end:
        zp = polish(z,params,min(end,start+0.20*budget_s))
        consider(np.column_stack((zp[:N],zp[N:2*N])),zp[2*N:3*N])
    if time.monotonic() < end:
        rr = lp_radii(p,end)
        if rr is not None:
            consider(p,rr)
    z[:2*N] += rng.normal(0,params['init_jitter'],2*N)
    mu = params['mu0']
    opt_end = min(end-0.18,start+0.55*budget_s)
    last = z.copy()
    def callback(x):
        nonlocal last
        last = x.copy()
        if time.monotonic() >= opt_end:
            raise Stop()
    for _ in range(int(params['stages'])):
        if time.monotonic() >= opt_end:
            break
        try:
            z = minimize(penalty,z,args=(mu,opt_end),jac=True,method='L-BFGS-B',
                         callback=callback,options={'maxiter':int(params['maxiter'])}).x
        except Stop:
            z = last
            break
        mu *= params['mu_growth']
    if time.monotonic() < end-0.08:
        pc = np.column_stack((z[:N],z[N:2*N]))
        rc = lp_radii(pc,end)
        if rc is not None:
            consider(pc,rc)
    return best
