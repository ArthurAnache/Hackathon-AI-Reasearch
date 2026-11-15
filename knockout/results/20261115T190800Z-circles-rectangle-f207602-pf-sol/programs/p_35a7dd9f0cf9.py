import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'search_fraction': (0.70, 0.4, 0.75),
    'iterations': (75, 25, 160),
    'restart_noise': (0.027, 0.002, 0.09),
    'slice_seconds': (0.27, 0.08, 0.6),
}
N = 21
I, J = np.triu_indices(N, 1)
ROWS = np.arange(len(I))

class Deadline(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + float(params['search_fraction']) * budget_s
    rng = np.random.default_rng(seed)
    best = None
    best_score = -np.inf
    local_end = end

    def check():
        if time.monotonic() >= min(end, local_end):
            raise Deadline()

    def valid(a):
        if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return False
        dx = a[I, 0] - a[J, 0]
        dy = a[I, 1] - a[J, 1]
        if np.any(np.sqrt(dx*dx + dy*dy) < a[I, 2] + a[J, 2]):
            return False
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        width = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        height = max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        return width + height <= 2

    def retain(a, original=None):
        nonlocal best, best_score
        score = float(np.sum(a[:, 2])) if a.shape == (N,3) else -np.inf
        if score > best_score and valid(a):
            best_score = score
            best = original if original is not None else {'circles': a.tolist()}

    def repair(z):
        p = np.column_stack((z[:N], z[N:2*N])).copy()
        r = np.maximum(z[2*N:3*N], 1e-10).copy()
        if not np.isfinite(p).all() or not np.isfinite(r).all():
            return
        dx = p[I,0]-p[J,0]
        dy = p[I,1]-p[J,1]
        factor = min(1.0, float(np.min(np.sqrt(dx*dx+dy*dy)/(r[I]+r[J]))))
        if factor <= 0:
            return
        r *= factor * (1-3e-14)
        lo = np.min(p-r[:,None], axis=0)
        hi = np.max(p+r[:,None], axis=0)
        scale = (2-8e-14)/float(np.sum(hi-lo))
        p = (p-lo)*scale
        r *= scale*(1-3e-14)
        retain(np.column_stack((p,r)))

    a0 = None
    if init is not None:
        try:
            a0 = np.asarray(init['circles'], dtype=float)
            retain(a0, init)
            if a0.shape != (N,3) or not np.isfinite(a0).all():
                a0 = None
        except (KeyError, TypeError, ValueError):
            a0 = None
    xx, yy = np.meshgrid(np.arange(3), np.arange(7))
    p = np.column_stack((xx.ravel()+0.5*(yy.ravel()%2), yy.ravel()*np.sqrt(3)/2))
    repair(np.r_[p[:,0],p[:,1],np.full(N,0.49),1.0])
    if a0 is not None:
        repair(np.r_[a0[:,0],a0[:,1],a0[:,2],1.0])

    A = np.zeros((4*N,3*N+1))
    b = np.zeros(4*N)
    k = np.arange(N)
    A[k,k] = 1; A[k,2*N+k] = -1
    A[N+k,k] = -1; A[N+k,2*N+k] = -1; A[N+k,-1] = 1
    A[2*N+k,N+k] = 1; A[2*N+k,2*N+k] = -1
    A[3*N+k,N+k] = -1; A[3*N+k,2*N+k] = -1; A[3*N+k,-1] = -1
    b[3*N:] = 2
    gradient = np.r_[np.zeros(2*N),-np.ones(N),0.0]

    def objective(z):
        check()
        return -float(np.sum(z[2*N:3*N]))

    def contacts(z):
        check()
        dx = z[I]-z[J]
        dy = z[N+I]-z[N+J]
        return np.sqrt(dx*dx+dy*dy+1e-30)-z[2*N+I]-z[2*N+J]

    def jacobian(z):
        check()
        dx = z[I]-z[J]
        dy = z[N+I]-z[N+J]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        jac = np.zeros((len(I),3*N+1))
        jac[ROWS,I] = dx/d; jac[ROWS,J] = -dx/d
        jac[ROWS,N+I] = dy/d; jac[ROWS,N+J] = -dy/d
        jac[ROWS,2*N+I] = -1; jac[ROWS,2*N+J] = -1
        return jac

    bounds = [(None,None)]*(2*N)+[(1e-9,0.7)]*N+[(0.1,1.9)]
    constraints = [
        {'type':'ineq','fun':lambda z:A@z+b,'jac':lambda z:A},
        {'type':'ineq','fun':contacts,'jac':jacobian}
    ]
    attempt = 0
    while time.monotonic() < end - 0.025:
        a = np.asarray(best['circles'], dtype=float)
        p = a[:,:2].copy()
        r = a[:,2].copy()
        lo = np.min(p-r[:,None],axis=0)
        p -= lo
        width = float(np.max(p[:,0]+r))
        if attempt:
            factors = (1.0, 1.8, 0.45, 2.7, 1.35)
            amplitude = float(params['restart_noise'])*factors[(attempt-1)%len(factors)]
            displacement = rng.normal(size=(N,2))*amplitude
            if attempt % 3 == 0:
                selected = rng.choice(N, size=5, replace=False)
                displacement[:] = 0
                displacement[selected] = rng.normal(size=(5,2))*amplitude*2
            p += displacement
            r *= np.exp(rng.normal(0,amplitude*0.8,N))
        z = np.r_[p[:,0],p[:,1],r,width]
        last = [z.copy()]
        count = [0]
        local_end = min(end, time.monotonic()+float(params['slice_seconds']))

        def callback(q):
            last[0] = q.copy()
            count[0] += 1
            if count[0] % 5 == 0:
                repair(q)
            check()

        try:
            result = minimize(objective,z,jac=lambda z:gradient,method='SLSQP',
                              bounds=bounds,constraints=constraints,callback=callback,
                              options={'maxiter':int(params['iterations']),'ftol':2e-12,'disp':False})
            repair(result.x)
        except Deadline:
            repair(last[0])
        attempt += 1
    return best
