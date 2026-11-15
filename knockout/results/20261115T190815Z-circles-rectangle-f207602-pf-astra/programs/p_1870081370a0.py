import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import lstsq

PARAMS = {
    'cluster_size': (5, 3, 9),
    'rotation': (0.26, 0.05, 0.7),
    'mu': (24.0, 3.0, 200.0, 'log'),
    'growth': (7.0, 2.0, 20.0),
    'maxiter': (120, 30, 250),
    'ftol': (2e-13, 1e-14, 1e-8, 'log'),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
K = np.arange(N)
T = np.arange(M)
G = np.r_[np.zeros(42), -np.ones(21), 0.0]

class Stop(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    r = a[:, 2]
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < r[I]+r[J]):
        return -np.inf
    x = list(map(lambda v: Fraction(float(v)), a[:, 0]))
    y = list(map(lambda v: Fraction(float(v)), a[:, 1]))
    q = list(map(lambda v: Fraction(float(v)), r))
    w = max(map(lambda u: u[0]+u[1], zip(x,q))) - min(map(lambda u: u[0]-u[1], zip(x,q)))
    h = max(map(lambda u: u[0]+u[1], zip(y,q))) - min(map(lambda u: u[0]-u[1], zip(y,q)))
    if w+h > 2:
        return -np.inf
    return float(np.sum(r))

def unpack(z):
    return np.column_stack((z[:21], z[21:42], z[42:63]))

def pack(a):
    lo = np.min(a[:, :2]-a[:, 2,None], axis=0)
    hi = np.max(a[:, :2]+a[:, 2,None], axis=0)
    p = a[:, :2]-lo
    return np.r_[p[:,0], p[:,1], a[:,2], hi[0]-lo[0]]

def repair(a):
    a = np.array(a, dtype=float, copy=True)
    if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
        return None
    a[:, :2] -= np.mean(a[:, :2], axis=0)
    d = np.sqrt((a[I,0]-a[J,0])**2 + (a[I,1]-a[J,1])**2)
    if np.min(d) <= 0:
        return None
    ratio = min(1.0, float(np.min(d/(a[I,2]+a[J,2]))))
    a[:,2] *= ratio*(1-8e-15)
    lo = np.min(a[:,:2]-a[:,2,None], axis=0)
    hi = np.max(a[:,:2]+a[:,2,None], axis=0)
    perimeter = float(np.sum(hi-lo))
    if perimeter <= 0:
        return None
    a *= (2/perimeter)*(1-8e-15)
    return a

def cons(z):
    x,y,r,w = z[:21],z[21:42],z[42:63],z[-1]
    dx,dy = x[I]-x[J],y[I]-y[J]
    d = np.sqrt(dx*dx+dy*dy)
    return np.r_[d-r[I]-r[J],x-r,w-x-r,y-r,2-w-y-r]

def jac(z):
    dx,dy = z[I]-z[J],z[21+I]-z[21+J]
    d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-15)
    a = np.zeros((M+84,64))
    a[T,I],a[T,J] = dx/d,-dx/d
    a[T,21+I],a[T,21+J] = dy/d,-dy/d
    a[T,42+I],a[T,42+J] = -1,-1
    a[M+K,K],a[M+K,42+K] = 1,-1
    a[M+21+K,K],a[M+21+K,42+K],a[M+21+K,-1] = -1,-1,1
    a[M+42+K,21+K],a[M+42+K,42+K] = 1,-1
    a[M+63+K,21+K],a[M+63+K,42+K],a[M+63+K,-1] = -1,-1,-1
    return a

def penalty(z, mu):
    x,y,r,w = z[:21],z[21:42],z[42:63],z[-1]
    dx,dy = x[I]-x[J],y[I]-y[J]
    d = np.sqrt(dx*dx+dy*dy+1e-30)
    v = np.maximum(r[I]+r[J]-d,0)
    b0 = np.maximum(r-x,0)
    b1 = np.maximum(r+x-w,0)
    b2 = np.maximum(r-y,0)
    b3 = np.maximum(r+y-(2-w),0)
    c = 2*mu*v
    gx = np.bincount(I,weights=-c*dx/d,minlength=N)+np.bincount(J,weights=c*dx/d,minlength=N)
    gy = np.bincount(I,weights=-c*dy/d,minlength=N)+np.bincount(J,weights=c*dy/d,minlength=N)
    gr = -np.ones(N)+np.bincount(I,weights=c,minlength=N)+np.bincount(J,weights=c,minlength=N)
    gx += 2*mu*(-b0+b1)
    gy += 2*mu*(-b2+b3)
    gr += 2*mu*(b0+b1+b2+b3)
    f = -r.sum()+mu*(v@v+b0@b0+b1@b1+b2@b2+b3@b3)
    return f,np.r_[gx,gy,gr,2*mu*(-b1.sum()+b3.sum())]

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start+0.70*budget_s
    rng = np.random.default_rng(seed)
    fallback = np.column_stack(((np.arange(21)%7+0.5)/7,(np.arange(21)//7+0.5)/3,np.full(21,0.06)))
    best = {'circles':fallback.tolist()}
    best_a = fallback.copy()
    best_score = exact_score(fallback)

    def offer(a, original=None):
        nonlocal best,best_a,best_score
        if a is None:
            return
        s = exact_score(a)
        if s > best_score:
            best_score,best_a = s,a.copy()
            best = original if original is not None else {'circles':a.tolist()}

    if init is not None:
        try:
            offer(np.asarray(init['circles'],dtype=float),init)
        except (ValueError,TypeError,KeyError,IndexError):
            pass
    if time.monotonic() >= end:
        return best
    offer(repair(best_a))
    z = pack(best_a)
    bounds = [(None,None)]*42+[(1e-8,0.5)]*21+[(0.05,1.95)]

    def refine(v, local_end):
        v = v.copy()
        active = np.flatnonzero(np.abs(cons(v)) < 2e-7)
        if len(active) < 60:
            return v
        for _ in range(5):
            if time.monotonic() >= local_end:
                break
            c = cons(v)
            delta = lstsq(jac(v)[active],-c[active],cond=1e-11,lapack_driver='gelsy')[0]
            if np.max(np.abs(delta)) > 2e-4:
                break
            v += delta
            offer(repair(unpack(v)))
            if np.max(np.abs(delta)) < 2e-15:
                break
        return v

    def polish(v, local_end):
        latest = [v.copy()]
        iterations = [0]
        def check():
            if time.monotonic() >= local_end:
                raise Stop()
        def objective(q):
            check()
            return -float(np.sum(q[42:63]))
        def cf(q):
            check()
            return cons(q)
        def cj(q):
            check()
            return jac(q)
        def callback(q):
            check()
            latest[0] = q.copy()
            iterations[0] += 1
            if iterations[0] % 4 == 0:
                offer(repair(unpack(q)))
        try:
            result = minimize(objective,v,jac=lambda q:G,method='SLSQP',bounds=bounds,
                              constraints={'type':'ineq','fun':cf,'jac':cj},callback=callback,
                              options={'maxiter':int(params['maxiter']),'ftol':float(params['ftol'])})
            latest[0] = result.x
        except Stop:
            pass
        if time.monotonic() < end:
            offer(repair(unpack(latest[0])))
        return latest[0]

    z = refine(z,min(end,start+0.08*budget_s))
    z = polish(z,min(end,start+0.19*budget_s))
    if time.monotonic() < start+0.23*budget_s:
        refine(z,start+0.23*budget_s)
    if time.monotonic() >= end-0.04:
        return best

    # Move a compact cluster together: this alters several contacts without
    # imposing a global reflection family or independently scattering centres.
    a = best_a.copy()
    anchor = int(rng.integers(N))
    dist = np.sum((a[:,:2]-a[anchor,:2])**2,axis=1)
    group = np.argsort(dist)[:int(params['cluster_size'])]
    centre = np.mean(a[group,:2],axis=0)
    angle = float(params['rotation'])*(1 if rng.integers(2) else -1)
    co,si = np.cos(angle),np.sin(angle)
    rotation = np.array([[co,-si],[si,co]])
    a[group,:2] = centre+(a[group,:2]-centre)@rotation.T
    a[group,2] *= 0.96
    z = pack(a)
    smooth_end = min(end,start+0.40*budget_s)
    mu = float(params['mu'])
    for stage in range(2):
        if time.monotonic() >= smooth_end:
            break
        stage_end = min(smooth_end,time.monotonic()+(smooth_end-time.monotonic())/(2-stage))
        latest = [z.copy()]
        def fun(q):
            if time.monotonic() >= stage_end:
                raise Stop()
            return penalty(q,mu)
        def pcb(q):
            latest[0] = q.copy()
            if time.monotonic() >= stage_end:
                raise Stop()
        try:
            result = minimize(fun,z,jac=True,method='L-BFGS-B',bounds=bounds,callback=pcb,
                              options={'maxiter':90,'ftol':1e-10,'maxls':15})
            z = result.x
        except Stop:
            z = latest[0]
        if time.monotonic() < end:
            offer(repair(unpack(z)))
        mu *= float(params['growth'])
    if time.monotonic() < end-0.025:
        z = polish(z,end-0.015)
    if time.monotonic() < end:
        refine(z,end)
    return best
