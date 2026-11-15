import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'penalty_iterations': (55, 20, 150),
    'sqp_iterations': (180, 40, 400),
    'restart_noise': (0.025, 0.0, 0.1),
    'search_fraction': (0.70, 0.4, 0.8),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
PAIR_ROWS = np.arange(M)

class Deadline(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + float(params['search_fraction']) * budget_s
    rng = np.random.default_rng(seed)
    best = None
    best_score = -np.inf

    def check():
        if time.monotonic() >= end:
            raise Deadline()

    def valid(a):
        if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return False
        p, r = a[:, :2], a[:, 2]
        d = np.sqrt((p[I, 0]-p[J, 0])**2 + (p[I, 1]-p[J, 1])**2)
        if np.any(d < r[I]+r[J]):
            return False
        x = [Fraction(float(v)) for v in p[:, 0]]
        y = [Fraction(float(v)) for v in p[:, 1]]
        q = [Fraction(float(v)) for v in r]
        perimeter = (max(a+b for a,b in zip(x,q)) - min(a-b for a,b in zip(x,q))
                     + max(a+b for a,b in zip(y,q)) - min(a-b for a,b in zip(y,q)))
        return perimeter <= 2

    def retain(a, original=None):
        nonlocal best, best_score
        if valid(a):
            score = float(np.sum(a[:, 2]))
            if score > best_score:
                best_score = score
                best = original if original is not None else {'circles': a.tolist()}

    def repair(z):
        p = np.column_stack((z[:N], z[N:2*N])).copy()
        r = np.maximum(z[2*N:3*N], 1e-10).copy()
        if not np.isfinite(p).all() or not np.isfinite(r).all():
            return
        d = np.sqrt((p[I, 0]-p[J, 0])**2 + (p[I, 1]-p[J, 1])**2)
        factor = min(1.0, float(np.min(d/(r[I]+r[J]))))
        if factor <= 0:
            return
        r *= factor * (1.0-2e-12)
        lo = np.min(p-r[:, None], axis=0)
        hi = np.max(p+r[:, None], axis=0)
        scale = (2.0-2e-12)/float(np.sum(hi-lo))
        p = (p-lo)*scale
        r *= scale*(1.0-2e-12)
        retain(np.column_stack((p,r)))

    if init is not None:
        try:
            a0 = np.asarray(init['circles'], dtype=float)
            retain(a0, init)
        except (ValueError, KeyError, TypeError):
            a0 = None
    else:
        a0 = None

    # A guaranteed feasible fallback, independent of the quality of the input.
    xx, yy = np.meshgrid(np.arange(3), np.arange(7))
    pgrid = np.column_stack((xx.ravel()+0.5*(yy.ravel()%2), yy.ravel()*np.sqrt(3)/2))
    zg = np.r_[pgrid[:,0], pgrid[:,1], np.full(N,0.49), 1.0]
    repair(zg)

    # All box inequalities are linear in (x,y,r,width), with height=2-width.
    A = np.zeros((4*N, 3*N+1))
    b = np.zeros(4*N)
    k = np.arange(N)
    A[k,k] = 1; A[k,2*N+k] = -1
    A[N+k,k] = -1; A[N+k,2*N+k] = -1; A[N+k,-1] = 1
    A[2*N+k,N+k] = 1; A[2*N+k,2*N+k] = -1
    A[3*N+k,N+k] = -1; A[3*N+k,2*N+k] = -1; A[3*N+k,-1] = -1
    b[3*N:] = 2
    objective_gradient = np.r_[np.zeros(2*N), -np.ones(N), 0.0]

    def objective(z):
        check()
        return -float(np.sum(z[2*N:3*N]))

    def contacts(z):
        check()
        dx = z[I]-z[J]
        dy = z[N+I]-z[N+J]
        rs = z[2*N+I]+z[2*N+J]
        return dx*dx+dy*dy-rs*rs

    def contacts_jac(z):
        check()
        dx = 2*(z[I]-z[J]); dy = 2*(z[N+I]-z[N+J])
        rs = -2*(z[2*N+I]+z[2*N+J])
        jac = np.zeros((M, 3*N+1))
        jac[PAIR_ROWS,I] = dx; jac[PAIR_ROWS,J] = -dx
        jac[PAIR_ROWS,N+I] = dy; jac[PAIR_ROWS,N+J] = -dy
        jac[PAIR_ROWS,2*N+I] = rs; jac[PAIR_ROWS,2*N+J] = rs
        return jac

    def penalty(z, mu):
        check()
        dx = z[I]-z[J]; dy = z[N+I]-z[N+J]
        dist = np.sqrt(dx*dx+dy*dy+1e-24)
        v = np.maximum(z[2*N+I]+z[2*N+J]-dist,0)
        wall = np.minimum(A@z+b,0)
        c = 2*mu*v
        gx = np.bincount(I, -c*dx/dist, minlength=N)+np.bincount(J,c*dx/dist,minlength=N)
        gy = np.bincount(I, -c*dy/dist, minlength=N)+np.bincount(J,c*dy/dist,minlength=N)
        gr = -np.ones(N)+np.bincount(I,c,minlength=N)+np.bincount(J,c,minlength=N)
        grad = np.r_[gx,gy,gr,0.0]+2*mu*(A.T@wall)
        return -z[2*N:3*N].sum()+mu*(v@v+wall@wall), grad

    bounds = [(None,None)]*(2*N)+[(1e-8,0.5)]*N+[(0.05,1.95)]
    attempt = 0
    try:
        while time.monotonic() < end:
            check()
            if attempt == 0 and a0 is not None and a0.shape == (N,3) and np.isfinite(a0).all():
                p = a0[:,:2].copy()
                lo = np.min(p,axis=0); span = np.ptp(p,axis=0)
                scale = 1.65/max(float(span.sum()),1e-8)
                p = (p-lo)*scale+0.08
                w = float(np.ptp(p[:,0])+0.18)
                z = np.r_[p[:,0],p[:,1],np.full(N,0.06),np.clip(w,0.2,1.8)]
            else:
                a = np.asarray(best['circles'])
                p = a[:,:2]+rng.normal(0,float(params['restart_noise']),(N,2))
                z = np.r_[p[:,0],p[:,1],a[:,2],float(np.ptp(a[:,0])+2*np.max(a[:,2]))]
            for mu in (20.0, 200.0, 2000.0):
                check()
                result = minimize(penalty,z,args=(mu,),jac=True,method='L-BFGS-B',bounds=bounds,
                                  options={'maxiter':int(params['penalty_iterations']),'ftol':1e-10,'maxls':20})
                z = result.x
                repair(z)
            last = [z.copy()]
            count = [0]
            def callback(q):
                check()
                last[0] = q.copy()
                count[0] += 1
                if count[0] % 4 == 0:
                    repair(q)
            try:
                result = minimize(objective,z,jac=lambda q: objective_gradient,method='SLSQP',bounds=bounds,
                    constraints=[{'type':'ineq','fun':lambda q:A@q+b,'jac':lambda q:A},
                                 {'type':'ineq','fun':contacts,'jac':contacts_jac}],
                    callback=callback,options={'maxiter':int(params['sqp_iterations']),'ftol':2e-11})
                repair(result.x)
            except Deadline:
                repair(last[0])
                break
            attempt += 1
    except Deadline:
        pass
    return best
