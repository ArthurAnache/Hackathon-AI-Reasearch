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
    'polish_iterations': (35, 5, 100)
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
K = np.arange(N)

class Stop(Exception):
    pass


def perimeter(p, r):
    x = list(map(lambda v: Fraction(float(v)), p[:, 0]))
    y = list(map(lambda v: Fraction(float(v)), p[:, 1]))
    q = list(map(lambda v: Fraction(float(v)), r))
    return (max(a+b for a,b in zip(x,q)) - min(a-b for a,b in zip(x,q))
            + max(a+b for a,b in zip(y,q)) - min(a-b for a,b in zip(y,q)))


def score(p, r):
    if p.shape != (N, 2) or r.shape != (N,):
        return -np.inf
    if not np.isfinite(p).all() or not np.isfinite(r).all() or not (r > 0).all():
        return -np.inf
    if perimeter(p, r) > 2:
        return -np.inf
    d = np.sqrt((p[I,0]-p[J,0])**2 + (p[I,1]-p[J,1])**2)
    if np.any(d < r[I]+r[J]):
        return -np.inf
    return float(np.sum(r))


def repair(p, r, end):
    r = np.maximum(np.asarray(r).copy(), 1e-12) * (1-1e-12)
    d = np.sqrt((p[I,0]-p[J,0])**2 + (p[I,1]-p[J,1])**2)
    for unused in range(5):
        if time.monotonic() >= end:
            break
        bad = np.flatnonzero(d < r[I]+r[J])
        if not len(bad):
            break
        for k in bad:
            if time.monotonic() >= end:
                break
            i, j = I[k], J[k]
            s = min(1.0, d[k]*(1-1e-12)/(r[i]+r[j]))
            r[i] *= s
            r[j] *= s
    for unused in range(60):
        if time.monotonic() >= end:
            break
        excess = perimeter(p,r)-2
        if excess <= 0:
            break
        r *= 1-max(2*float(excess)/float(r.sum()), 1e-12)
    return r


def penalty(z, mu):
    x, y, r, w = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
    dx, dy = x[I]-x[J], y[I]-y[J]
    d = np.sqrt(dx*dx+dy*dy+1e-18)
    v = np.maximum(r[I]+r[J]-d, 0)
    b0, b1 = np.maximum(r-x,0), np.maximum(r+x-w,0)
    b2, b3 = np.maximum(r-y,0), np.maximum(r+y-(2-w),0)
    neg = np.maximum(-r,0)
    f = -r.sum()+mu*(v@v+b0@b0+b1@b1+b2@b2+b3@b3+neg@neg)
    gx, gy, gr = np.zeros(N), np.zeros(N), -np.ones(N)
    c = 2*mu*v
    np.add.at(gr,I,c)
    np.add.at(gr,J,c)
    np.add.at(gx,I,-c*dx/d)
    np.add.at(gx,J,c*dx/d)
    np.add.at(gy,I,-c*dy/d)
    np.add.at(gy,J,c*dy/d)
    gx += 2*mu*(-b0+b1)
    gy += 2*mu*(-b2+b3)
    gr += 2*mu*(b0+b1+b2+b3-neg)
    return f, np.r_[gx,gy,gr,2*mu*(-b1.sum()+b3.sum())]


def constraints(z):
    x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
    d = np.sqrt((x[I]-x[J])**2+(y[I]-y[J])**2)
    return np.r_[d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r]


def constraint_jac(z):
    x,y = z[:N],z[N:2*N]
    dx,dy = x[I]-x[J],y[I]-y[J]
    d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-30)
    a = np.zeros((M+4*N,3*N+1))
    t = np.arange(M)
    a[t,I], a[t,J] = dx/d,-dx/d
    a[t,N+I], a[t,N+J] = dy/d,-dy/d
    a[t,2*N+I], a[t,2*N+J] = -1,-1
    a[M+K,K] = 1
    a[M+N+K,K] = -1
    a[M+N+K,-1] = 1
    a[M+2*N+K,N+K] = 1
    a[M+3*N+K,N+K] = -1
    a[M+3*N+K,-1] = -1
    a[M+K,2*N+K] = -1
    a[M+N+K,2*N+K] = -1
    a[M+2*N+K,2*N+K] = -1
    a[M+3*N+K,2*N+K] = -1
    return a


def lp_radii(p):
    a = np.zeros((M+4*N+1,N+4))
    b = np.zeros(M+4*N+1)
    a[np.arange(M),I] = 1
    a[np.arange(M),J] = 1
    b[:M] = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
    a[M+K,K],a[M+K,N] = 1,1
    b[M+K] = p[:,0]
    a[M+N+K,K],a[M+N+K,N+1] = 1,-1
    b[M+N+K] = -p[:,0]
    a[M+2*N+K,K],a[M+2*N+K,N+2] = 1,1
    b[M+2*N+K] = p[:,1]
    a[M+3*N+K,K],a[M+3*N+K,N+3] = 1,-1
    b[M+3*N+K] = -p[:,1]
    a[-1,N:] = [-1,1,-1,1]
    b[-1] = 2
    res = linprog(-np.r_[np.ones(N),np.zeros(4)], A_ub=a,b_ub=b,
                  bounds=[(0,None)]*N+[(None,None)]*4,method='highs')
    return res.x[:N] if res.success else None


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start+0.70*budget_s
    rng = np.random.default_rng(seed)
    # A valid cold fallback is available before optimization starts.
    fallback = np.array([[0.04+0.08*i,0.04,0.03] for i in range(N)])
    best = {'circles': fallback.tolist()}
    best_score = score(fallback[:,:2],fallback[:,2])
    if init is not None:
        raw = np.asarray(init['circles'],dtype=float)
        s0 = score(raw[:,:2],raw[:,2])
        if s0 >= best_score:
            best,best_score = init,s0
        p,r = raw[:,:2].copy(),np.abs(raw[:,2]).copy()
        lo = (p-r[:,None]).min(axis=0)
        side = (p+r[:,None]).max(axis=0)-lo
        scale = 2/side.sum()
        p,r,w = (p-lo)*scale,r*scale,side[0]*scale
        p += rng.normal(0,float(params['init_jitter']),p.shape)
    else:
        w = 2*params['aspect']/(1+params['aspect'])
        rows = int(params['rows'])
        cols = int(np.ceil(N/rows))
        k = np.arange(N)
        p = np.column_stack(((k%cols+0.5)/cols,(k//cols+0.5)/rows))
        p = np.clip(p+rng.normal(0,params['jitter'],p.shape),0.02,0.98)
        p *= [w,2-w]
        r = np.full(N,0.05)
    z0 = np.r_[p[:,0],p[:,1],r,w]

    def consider(z):
        nonlocal best,best_score
        if time.monotonic() >= end:
            return
        pp = np.column_stack((z[:N],z[N:2*N]))
        rr = repair(pp,z[2*N:3*N],end)
        ss = score(pp,rr)
        if ss > best_score:
            best_score = ss
            best = {'circles': np.column_stack((pp,rr)).tolist()}

    # The single method change: a joint, hard-constraint numerical polish.
    objective_jac = np.r_[np.zeros(2*N),-np.ones(N),0.0]
    def polish(z, deadline):
        last = z.copy()
        def check():
            if time.monotonic() >= deadline:
                raise Stop()
        def fun(q):
            check()
            return -float(q[2*N:3*N].sum())
        def con(q):
            check()
            return constraints(q)
        def jac(q):
            check()
            return constraint_jac(q)
        def callback(q):
            nonlocal last
            last = q.copy()
            check()
        try:
            result = minimize(fun,z,jac=lambda q: objective_jac,
                constraints={'type':'ineq','fun':con,'jac':jac},
                bounds=[(None,None)]*(2*N)+[(1e-9,None)]*N+[(0.01,1.99)],
                method='SLSQP',callback=callback,
                options={'ftol':1e-13,'maxiter':int(params['polish_iterations'])})
            last = result.x
        except Stop:
            pass
        return last

    zp = polish(z0,min(end,start+0.23*budget_s))
    consider(zp)
    z = z0.copy()
    penalty_end = start+0.43*budget_s
    mu = float(params['mu0'])
    last = z.copy()
    def cb(q):
        nonlocal last
        last = q.copy()
        if time.monotonic() >= penalty_end:
            raise Stop()
    def pf(q, m):
        if time.monotonic() >= penalty_end:
            raise Stop()
        return penalty(q,m)
    for unused in range(int(params['stages'])):
        if time.monotonic() >= penalty_end:
            break
        try:
            z = minimize(pf,z,args=(mu,),jac=True,method='L-BFGS-B',callback=cb,
                         options={'maxiter':int(params['maxiter'])}).x
            last = z.copy()
        except Stop:
            z = last
            break
        mu *= params['mu_growth']
    if time.monotonic() < start+0.55*budget_s:
        z = polish(z,start+0.55*budget_s)
    consider(z)
    for q in (zp,z,z0):
        if time.monotonic() >= end-0.12*budget_s:
            break
        pp = np.column_stack((q[:N],q[N:2*N]))
        rr = lp_radii(pp)
        if rr is not None:
            qq = q.copy()
            qq[2*N:3*N] = rr
            consider(qq)
    return best
