import time
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linprog

PARAMS = {
    'mu0': (10.0, 1.0, 1000.0, 'log'),
    'mu_growth': (8.0, 2.0, 30.0),
    'stages': (3, 1, 6),
    'maxiter': (150, 30, 400),
    'polish_iterations': (180, 30, 400)
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
K = np.arange(N)

class Deadline(Exception):
    pass

def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2] + a[J, 2]):
        return -np.inf
    x = list(map(Fraction, map(float, a[:, 0])))
    y = list(map(Fraction, map(float, a[:, 1])))
    r = list(map(Fraction, map(float, a[:, 2])))
    perimeter = (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
                 +max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)))
    if perimeter > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))

def repair(p, r):
    p = np.asarray(p).copy()
    r = np.maximum(np.asarray(r).copy(), 1e-12)
    if not np.isfinite(p).all() or not np.isfinite(r).all():
        return None
    d = np.sqrt((p[I, 0]-p[J, 0])**2 + (p[I, 1]-p[J, 1])**2)
    if np.min(d) <= 0:
        return None
    ratio = min(1.0, float(np.min(d / (r[I]+r[J]))))
    r *= ratio * (1.0-2e-13)
    lo = np.min(p-r[:, None], axis=0)
    hi = np.max(p+r[:, None], axis=0)
    scale = (2.0-2e-13)/float(np.sum(hi-lo))
    p = (p-lo)*scale
    r *= scale
    return np.column_stack((p, r))

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70*budget_s
    search_end = start + 0.65*budget_s
    phase_end = min(search_end, start + 0.30*budget_s)
    best = None
    best_score = -np.inf
    if init is not None:
        a = np.asarray(init['circles'], dtype=float)
        score = exact_score(a)
        if score > best_score:
            best, best_score = init, score
    else:
        a = np.array([[(k%5+0.5)/5, (k//5+0.5)/5, 0.0999999999999] for k in range(N)])

    def consider(p, r):
        nonlocal best, best_score
        if time.monotonic() >= end:
            return
        b = repair(p, r)
        if b is not None and float(np.sum(b[:, 2])) > best_score:
            score = exact_score(b)
            if score > best_score:
                best_score = score
                best = {'circles': b.tolist()}

    # Always have a valid fallback, including for invalid supplied starts.
    fallback = np.array([[(k%5+0.5)/5, (k//5+0.5)/5, 0.0999999999999] for k in range(N)])
    consider(fallback[:, :2], fallback[:, 2])
    if a.shape != (N, 3) or not np.isfinite(a).all():
        a = fallback
    p, r = a[:, :2].copy(), np.maximum(np.abs(a[:, 2]), 1e-8)
    lo = np.min(p-r[:, None], axis=0)
    side = np.max(p+r[:, None], axis=0)-lo
    scale = 2.0/float(side.sum())
    p, r = (p-lo)*scale, r*scale
    w = side[0]*scale
    z = np.r_[p[:, 0], p[:, 1], r, w]
    consider(p, r)
    latest = z.copy()

    def penalty(z, mu):
        if time.monotonic() >= phase_end:
            raise Deadline
        x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
        dx,dy = x[I]-x[J],y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-24)
        v = np.maximum(r[I]+r[J]-d,0)
        b = np.maximum(np.array([r-x,r+x-w,r-y,r+y-2+w]),0)
        neg = np.maximum(-r,0)
        f = -r.sum()+mu*(v@v+np.sum(b*b)+neg@neg)
        c = 2*mu*v
        gx = np.bincount(I,weights=-c*dx/d,minlength=N)+np.bincount(J,weights=c*dx/d,minlength=N)
        gy = np.bincount(I,weights=-c*dy/d,minlength=N)+np.bincount(J,weights=c*dy/d,minlength=N)
        gr = -np.ones(N)+np.bincount(I,weights=c,minlength=N)+np.bincount(J,weights=c,minlength=N)
        gx += 2*mu*(-b[0]+b[1])
        gy += 2*mu*(-b[2]+b[3])
        gr += 2*mu*(b.sum(axis=0)-neg)
        gw = 2*mu*(-b[1].sum()+b[3].sum())
        return f,np.r_[gx,gy,gr,gw]

    def penalty_callback(v):
        nonlocal latest
        latest = v.copy()
        if time.monotonic() >= phase_end:
            raise Deadline

    mu = float(params['mu0'])
    for stage in range(int(params['stages'])):
        if time.monotonic() >= phase_end:
            break
        try:
            z = minimize(penalty,z,args=(mu,),jac=True,method='L-BFGS-B',callback=penalty_callback,
                         options={'maxiter':int(params['maxiter']), 'ftol':1e-12}).x
            latest = z.copy()
        except Deadline:
            z = latest.copy()
            break
        consider(np.column_stack((z[:N],z[N:2*N])), z[2*N:3*N])
        mu *= float(params['mu_growth'])

    # Replace the high-mu continuation tail with direct constrained polishing.
    wall_jac = np.zeros((4*N,3*N+1))
    wall_jac[K,K]=1
    wall_jac[K,2*N+K]=-1
    wall_jac[N+K,K]=-1
    wall_jac[N+K,2*N+K]=-1
    wall_jac[N+K,-1]=1
    wall_jac[2*N+K,N+K]=1
    wall_jac[2*N+K,2*N+K]=-1
    wall_jac[3*N+K,N+K]=-1
    wall_jac[3*N+K,2*N+K]=-1
    wall_jac[3*N+K,-1]=-1
    rows = np.arange(M)
    objective_jac = np.r_[np.zeros(2*N),-np.ones(N),0.0]

    def objective(v):
        if time.monotonic() >= search_end:
            raise Deadline
        return -float(v[2*N:3*N].sum()),objective_jac

    def constraints(v):
        if time.monotonic() >= search_end:
            raise Deadline
        x,y,r,w = v[:N],v[N:2*N],v[2*N:3*N],v[-1]
        d = np.sqrt((x[I]-x[J])**2+(y[I]-y[J])**2+1e-30)
        return np.r_[d-r[I]-r[J],x-r,w-x-r,y-r,2-w-y-r]

    def jacobian(v):
        if time.monotonic() >= search_end:
            raise Deadline
        dx,dy = v[I]-v[J],v[N+I]-v[N+J]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        q = np.zeros((M+4*N,3*N+1))
        q[rows,I]=dx/d
        q[rows,J]=-dx/d
        q[rows,N+I]=dy/d
        q[rows,N+J]=-dy/d
        q[rows,2*N+I]=-1
        q[rows,2*N+J]=-1
        q[M:]=wall_jac
        return q

    def callback(v):
        nonlocal latest
        if time.monotonic() >= search_end:
            raise Deadline
        latest = v.copy()
        consider(np.column_stack((v[:N],v[N:2*N])),v[2*N:3*N])

    if time.monotonic() < search_end:
        try:
            result = minimize(objective,z,jac=True,method='SLSQP',
                bounds=[(None,None)]*(2*N)+[(1e-8,1.0)]*N+[(0.05,1.95)],
                constraints={'type':'ineq','fun':constraints,'jac':jacobian},callback=callback,
                options={'maxiter':int(params['polish_iterations']),'ftol':1e-12})
            latest = result.x
        except Deadline:
            pass
    p = np.column_stack((latest[:N],latest[N:2*N]))
    consider(p, latest[2*N:3*N])

    # The original fixed-centre LP remains a final optional radius repair.
    if end-time.monotonic() > 0.08:
        mat = np.zeros((M+4*N+1,N+4))
        rhs = np.zeros(M+4*N+1)
        mat[rows,I]=1
        mat[rows,J]=1
        rhs[:M]=np.sqrt(np.sum((p[I]-p[J])**2,axis=1))
        for k,(col,sign,coord) in enumerate([(N,1,0),(N+1,-1,0),(N+2,1,1),(N+3,-1,1)]):
            if time.monotonic() >= end:
                return best
            rr=M+k*N+K
            mat[rr,K]=1
            mat[rr,col]=sign
            rhs[rr]=sign*p[:,coord]
        mat[-1,N:]=[-1,1,-1,1]
        rhs[-1]=2
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result=linprog(-np.r_[np.ones(N),np.zeros(4)],A_ub=mat,b_ub=rhs,
                bounds=[(1e-10,None)]*N+[(None,None)]*4,method='highs',
                options={'threads':1,'time_limit':max(0.001,end-time.monotonic()-0.015)})
        if result.success:
            consider(p,result.x[:N])
    return best
