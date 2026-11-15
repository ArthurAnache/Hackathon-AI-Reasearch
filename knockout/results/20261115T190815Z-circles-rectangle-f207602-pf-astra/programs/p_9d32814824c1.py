import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'penalty_fraction': (0.23, 0.08, 0.35),
    'mu0': (12.0, 1.0, 100.0, 'log'),
    'growth': (7.0, 2.0, 20.0),
    'penalty_iterations': (100, 30, 200),
    'polish_iterations': (140, 40, 300),
    'jitter': (0.009, 0.0, 0.04),
    'aspect': (1.1, 0.8, 1.4)
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
K = np.arange(N)
ROWS = np.arange(M)
REP = np.r_[np.arange(12), np.arange(3,12)]

class StopSearch(Exception):
    pass


def exact_score(a):
    if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
        return -np.inf
    r = a[:,2]
    d = np.sqrt((a[I,0]-a[J,0])**2 + (a[I,1]-a[J,1])**2)
    if np.any(d < r[I]+r[J]):
        return -np.inf
    x = [Fraction(float(v)) for v in a[:,0]]
    y = [Fraction(float(v)) for v in a[:,1]]
    q = [Fraction(float(v)) for v in r]
    w = max(u+v for u,v in zip(x,q))-min(u-v for u,v in zip(x,q))
    h = max(u+v for u,v in zip(y,q))-min(u-v for u,v in zip(y,q))
    return float(np.sum(r)) if w+h <= 2 else -np.inf


def repair(a):
    a = np.asarray(a, dtype=float).copy()
    if a.shape != (21,3) or not np.isfinite(a).all():
        return None
    a[:,:2] -= a[:,:2].mean(axis=0)
    r = np.maximum(a[:,2], 1e-12)
    d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
    ratio = min(1.0, float(np.min(d/(r[I]+r[J]))))
    if ratio <= 0:
        return None
    a[:,2] = r*ratio*(1-4e-13)
    lo = np.min(a[:,:2]-a[:,2,None],axis=0)
    hi = np.max(a[:,:2]+a[:,2,None],axis=0)
    span = float(np.sum(hi-lo))
    if not span > 0:
        return None
    a *= 2*(1-4e-13)/span
    return a


def pack(a):
    lo = np.min(a[:,:2]-a[:,2,None],axis=0)
    hi = np.max(a[:,:2]+a[:,2,None],axis=0)
    p = a[:,:2]-lo
    return np.r_[p[:,0],p[:,1],a[:,2],hi[0]-lo[0]]


def unpack(z):
    return np.column_stack((z[:N],z[N:2*N],z[2*N:3*N]))


def expand(q):
    mid = 1-q[-1]/2
    return np.r_[q[:12][REP],np.r_[np.full(3,mid),mid+q[12:21],mid-q[12:21]],q[21:33][REP],q[-1]]


def reduced_gradient(g):
    return np.r_[np.bincount(REP,weights=g[:N],minlength=12),
                 g[N+3:N+12]-g[N+12:2*N],
                 np.bincount(REP,weights=g[2*N:3*N],minlength=12),
                 g[-1]-.5*np.sum(g[N:2*N])]


def penalty(z,mu):
    x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
    dx,dy = x[I]-x[J],y[I]-y[J]
    d = np.sqrt(dx*dx+dy*dy+1e-24)
    v = np.maximum(r[I]+r[J]-d,0)
    b0,b1 = np.maximum(r-x,0),np.maximum(r+x-w,0)
    b2,b3 = np.maximum(r-y,0),np.maximum(r+y-2+w,0)
    neg = np.maximum(-r,0)
    f = -r.sum()+mu*(v@v+b0@b0+b1@b1+b2@b2+b3@b3+neg@neg)
    c = 2*mu*v
    gx = np.bincount(I,weights=-c*dx/d,minlength=N)+np.bincount(J,weights=c*dx/d,minlength=N)+2*mu*(-b0+b1)
    gy = np.bincount(I,weights=-c*dy/d,minlength=N)+np.bincount(J,weights=c*dy/d,minlength=N)+2*mu*(-b2+b3)
    gr = -np.ones(N)+np.bincount(I,weights=c,minlength=N)+np.bincount(J,weights=c,minlength=N)+2*mu*(b0+b1+b2+b3-neg)
    return f,np.r_[gx,gy,gr,2*mu*(-b1.sum()+b3.sum())]


def construct(params,seed,budget_s,init=None):
    start = time.monotonic()
    deadline = start+.70*budget_s
    rng = np.random.default_rng(seed)
    fallback = np.column_stack(((K%7)*.2,(K//7)*.2,np.full(N,.1-1e-12)))
    best = {'circles':fallback.tolist()}
    best_array = fallback.copy()
    best_score = exact_score(fallback)
    original = None

    def offer(a, original_object=None):
        nonlocal best,best_array,best_score
        if a is None:
            return
        s = exact_score(a)
        if s > best_score:
            best_score,best_array = s,a.copy()
            best = original_object if original_object is not None else {'circles':a.tolist()}

    if init is not None:
        try:
            original = np.asarray(init['circles'],dtype=float)
            offer(original,init)
            offer(repair(original))
        except (ValueError,TypeError,KeyError,IndexError):
            original = None

    strong = best_score > 2.2
    z = pack(best_array)
    if not strong and time.monotonic() < deadline:
        w = 2*float(params['aspect'])/(1+float(params['aspect']))
        h = 2-w
        x = np.r_[(np.arange(3)+.5)/3,np.linspace(.1,.9,4),np.linspace(.1,.9,5)]*w
        x += rng.normal(0,float(params['jitter']),12)
        dy = np.r_[np.full(4,.205*h),np.full(5,.405*h)]
        dy += rng.normal(0,float(params['jitter']),9)
        q = np.r_[x,dy,np.full(12,.085),w]
        mu = float(params['mu0'])
        penalty_end = min(deadline,start+float(params['penalty_fraction'])*budget_s)
        for stage in range(4):
            if time.monotonic() >= penalty_end:
                break
            latest = [q.copy()]
            def fun(v):
                if time.monotonic() >= penalty_end:
                    raise StopSearch()
                f,g = penalty(expand(v),mu)
                return f,reduced_gradient(g)
            def cb(v):
                latest[0] = v.copy()
                if time.monotonic() >= penalty_end:
                    raise StopSearch()
            try:
                res = minimize(fun,q,jac=True,method='L-BFGS-B',callback=cb,
                               options={'maxiter':int(params['penalty_iterations']),'ftol':1e-11,'gtol':1e-7,'maxls':20})
                q = res.x
            except StopSearch:
                q = latest[0]
            z = expand(q)
            offer(repair(unpack(z)))
            mu *= float(params['growth'])

    linear = np.zeros((4*N,3*N+1))
    linear[K,K] = 1
    linear[K,2*N+K] = -1
    linear[N+K,K] = -1
    linear[N+K,2*N+K] = -1
    linear[N+K,3*N] = 1
    linear[2*N+K,N+K] = 1
    linear[2*N+K,2*N+K] = -1
    linear[3*N+K,N+K] = -1
    linear[3*N+K,2*N+K] = -1
    linear[3*N+K,3*N] = -1
    objgrad = np.r_[np.zeros(2*N),-np.ones(N),0.0]
    bounds = [(None,None)]*(2*N)+[(1e-8,.5)]*N+[(.05,1.95)]

    for attempt in range(3):
        now = time.monotonic()
        if now >= deadline-.025:
            break
        if attempt:
            a = best_array.copy()
            a[:,:2] += rng.normal(0,float(params['jitter']),size=(N,2))
            z = pack(a)
        local_end = deadline if not strong or attempt else min(deadline,now+.44*budget_s)
        latest = [z.copy()]
        count = [0]
        def check():
            if time.monotonic() >= local_end:
                raise StopSearch()
        def objective(v):
            check()
            return -float(np.sum(v[2*N:3*N]))
        def constraints(v):
            check()
            x,y,r,w = v[:N],v[N:2*N],v[2*N:3*N],v[-1]
            d = np.sqrt((x[I]-x[J])**2+(y[I]-y[J])**2+1e-30)
            return np.r_[d-r[I]-r[J],x-r,w-x-r,y-r,2-w-y-r]
        def jacobian(v):
            check()
            dx,dy = v[I]-v[J],v[N+I]-v[N+J]
            d = np.sqrt(dx*dx+dy*dy+1e-30)
            a = np.zeros((M+4*N,3*N+1))
            a[ROWS,I],a[ROWS,J] = dx/d,-dx/d
            a[ROWS,N+I],a[ROWS,N+J] = dy/d,-dy/d
            a[ROWS,2*N+I],a[ROWS,2*N+J] = -1,-1
            a[M:] = linear
            return a
        def callback(v):
            latest[0] = v.copy()
            check()
            count[0] += 1
            if count[0]%4 == 0:
                offer(repair(unpack(v)))
        try:
            res = minimize(objective,z,jac=lambda v:objgrad,method='SLSQP',bounds=bounds,
                           constraints={'type':'ineq','fun':constraints,'jac':jacobian},callback=callback,
                           options={'maxiter':int(params['polish_iterations']),'ftol':2e-12,'disp':False})
            latest[0] = res.x
        except StopSearch:
            pass
        if time.monotonic() < start+.76*budget_s:
            offer(repair(unpack(latest[0])))
    return best
