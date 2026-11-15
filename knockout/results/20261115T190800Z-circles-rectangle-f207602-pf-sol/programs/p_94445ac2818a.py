import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'penalty_iterations': (55, 20, 150),
    'sqp_iterations': (180, 40, 400),
    'restart_noise': (0.025, 0.0, 0.1),
    'search_fraction': (0.70, 0.4, 0.75),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
ROWS = np.arange(M)

class Deadline(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    end = time.monotonic() + float(params['search_fraction']) * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf

    def check():
        if time.monotonic() >= end:
            raise Deadline()

    def valid(a):
        if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return False
        d = np.sqrt((a[I,0]-a[J,0])**2 + (a[I,1]-a[J,1])**2)
        if np.any(d < a[I,2]+a[J,2]):
            return False
        xs = [Fraction(float(v)) for v in a[:,0]]
        ys = [Fraction(float(v)) for v in a[:,1]]
        rs = [Fraction(float(v)) for v in a[:,2]]
        width = max(x+r for x,r in zip(xs,rs))-min(x-r for x,r in zip(xs,rs))
        height = max(y+r for y,r in zip(ys,rs))-min(y-r for y,r in zip(ys,rs))
        return width+height <= 2

    def retain(a, original=None):
        nonlocal best, best_score
        if valid(a):
            score = float(np.sum(a[:,2]))
            if score > best_score:
                best_score = score
                best = original if original is not None else {'circles': a.tolist()}

    def repair(z):
        p = np.column_stack((z[:N], z[N:2*N])).copy()
        r = np.maximum(z[2*N:3*N], 1e-10).copy()
        if not np.isfinite(p).all() or not np.isfinite(r).all():
            return
        d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
        factor = min(1.0, float(np.min(d/(r[I]+r[J]))))
        if factor <= 0:
            return
        r *= factor*(1-2e-14)
        lo = np.min(p-r[:,None], axis=0)
        hi = np.max(p+r[:,None], axis=0)
        scale = (2-2e-14)/float(np.sum(hi-lo))
        p = (p-lo)*scale
        r *= scale*(1-2e-14)
        retain(np.column_stack((p,r)))

    a0 = None
    if init is not None:
        try:
            a = np.asarray(init['circles'], dtype=float)
            if valid(a):
                a0 = a.copy()
                retain(a, init)
        except (ValueError, KeyError, TypeError):
            pass

    xx, yy = np.meshgrid(np.arange(3), np.arange(7))
    pg = np.column_stack((xx.ravel()+0.5*(yy.ravel()%2), yy.ravel()*np.sqrt(3)/2))
    repair(np.r_[pg[:,0],pg[:,1],np.full(N,0.49),1.0])

    A = np.zeros((4*N,3*N+1))
    b = np.zeros(4*N)
    k = np.arange(N)
    A[k,k] = 1; A[k,2*N+k] = -1
    A[N+k,k] = -1; A[N+k,2*N+k] = -1; A[N+k,-1] = 1
    A[2*N+k,N+k] = 1; A[2*N+k,2*N+k] = -1
    A[3*N+k,N+k] = -1; A[3*N+k,2*N+k] = -1; A[3*N+k,-1] = -1
    b[3*N:] = 2
    og = np.r_[np.zeros(2*N),-np.ones(N),0.0]

    def objective(z):
        check()
        return -float(np.sum(z[2*N:3*N]))

    def contacts(z):
        check()
        dx = z[I]-z[J]; dy = z[N+I]-z[N+J]
        rs = z[2*N+I]+z[2*N+J]
        return dx*dx+dy*dy-rs*rs

    def contacts_jac(z):
        check()
        dx = 2*(z[I]-z[J]); dy = 2*(z[N+I]-z[N+J])
        rs = -2*(z[2*N+I]+z[2*N+J])
        out = np.zeros((M,3*N+1))
        out[ROWS,I] = dx; out[ROWS,J] = -dx
        out[ROWS,N+I] = dy; out[ROWS,N+J] = -dy
        out[ROWS,2*N+I] = rs; out[ROWS,2*N+J] = rs
        return out

    def penalty(z, mu):
        check()
        dx = z[I]-z[J]; dy = z[N+I]-z[N+J]
        dist = np.sqrt(dx*dx+dy*dy+1e-24)
        v = np.maximum(z[2*N+I]+z[2*N+J]-dist,0)
        wall = np.minimum(A@z+b,0)
        c = 2*mu*v
        gx = np.bincount(I,-c*dx/dist,minlength=N)+np.bincount(J,c*dx/dist,minlength=N)
        gy = np.bincount(I,-c*dy/dist,minlength=N)+np.bincount(J,c*dy/dist,minlength=N)
        gr = -np.ones(N)+np.bincount(I,c,minlength=N)+np.bincount(J,c,minlength=N)
        grad = np.r_[gx,gy,gr,0.0]+2*mu*(A.T@wall)
        return -z[2*N:3*N].sum()+mu*(v@v+wall@wall), grad

    def normalized(a):
        p = a[:,:2].copy()
        r = a[:,2].copy()
        lo = np.min(p-r[:,None],axis=0)
        hi = np.max(p+r[:,None],axis=0)
        scale = 2/float(np.sum(hi-lo))
        p = (p-lo)*scale
        r *= scale
        return np.r_[p[:,0],p[:,1],r,(hi[0]-lo[0])*scale]

    bounds = [(None,None)]*(2*N)+[(1e-8,0.5)]*N+[(0.05,1.95)]
    attempt = 0
    try:
        while time.monotonic() < end:
            check()
            direct = attempt == 0 and a0 is not None
            if direct:
                z = normalized(a0)
            else:
                a = np.asarray(best['circles'],dtype=float)
                z = normalized(a)
                z[:2*N] += rng.normal(0,float(params['restart_noise']),2*N)
                for mu in (20.0,200.0,2000.0):
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
                result = minimize(objective,z,jac=lambda q:og,method='SLSQP',bounds=bounds,
                    constraints=[{'type':'ineq','fun':lambda q:A@q+b,'jac':lambda q:A},
                                 {'type':'ineq','fun':contacts,'jac':contacts_jac}],
                    callback=callback,
                    options={'maxiter':int(params['sqp_iterations']),'ftol':2e-13 if direct else 2e-11})
                repair(result.x)
            except Deadline:
                repair(last[0])
                break
            attempt += 1
    except Deadline:
        pass
    return best
