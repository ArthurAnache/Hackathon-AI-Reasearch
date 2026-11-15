import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'symmetric_fraction': (0.47, 0.2, 0.6),
    'finish_fraction': (0.70, 0.6, 0.75),
    'iterations': (110, 30, 250),
    'jitter': (0.009, 0.0, 0.04),
    'restart_noise': (0.012, 0.001, 0.05)
}
N = 21
I, J = np.triu_indices(N, 1)

class Stop(Exception):
    pass

def exact_score(data):
    try:
        c = np.asarray(data['circles'], dtype=np.float64)
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return -np.inf
        d = c[I, :2] - c[J, :2]
        if np.any(np.sqrt(d[:, 0]**2 + d[:, 1]**2) < c[I, 2] + c[J, 2]):
            return -np.inf
        x, y, r = [[Fraction(float(v)) for v in c[:, k]] for k in range(3)]
        perimeter = max(a+b for a,b in zip(x,r))-min(a-b for a,b in zip(x,r))
        perimeter += max(a+b for a,b in zip(y,r))-min(a-b for a,b in zip(y,r))
        if perimeter > 2:
            return -np.inf
        return float(np.sum(c[:, 2]))
    except (TypeError, KeyError, ValueError, OverflowError):
        return -np.inf

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    end = started + float(params['finish_fraction'])*budget_s
    symend = started + float(params['symmetric_fraction'])*budget_s
    rng = np.random.default_rng(seed)
    best = init
    bestscore = exact_score(init)
    bestz = None

    def retain(z):
        nonlocal best, bestscore, bestz
        if time.monotonic() >= end:
            return
        c = np.asarray(z[:63]).reshape(3,21).T.copy()
        if not np.isfinite(c).all():
            return
        p = c[:, :2]
        r = np.maximum(c[:, 2], 1e-10)
        delta = p[I]-p[J]
        dist = np.sqrt(np.sum(delta*delta,axis=1))
        shrink = min(1.0, float(np.min(dist/(r[I]+r[J]))))
        if shrink <= 0:
            return
        r = r * shrink * (1-2e-13)
        lo = np.min(p-r[:,None],axis=0)
        hi = np.max(p+r[:,None],axis=0)
        scale = 2/np.sum(hi-lo)*(1-2e-13)
        p = (p-lo)*scale
        r = r*scale
        candidate = {'circles': np.column_stack((p,r)).tolist()}
        score = exact_score(candidate)
        if score > bestscore:
            bestscore = score
            best = candidate
            bestz = np.r_[p[:,0],p[:,1],r,(hi[0]-lo[0])*scale]

    if bestscore > -np.inf:
        c = np.asarray(init['circles'],dtype=float)
        lo = np.min(c[:,:2]-c[:,2,None],axis=0)
        hi = np.max(c[:,:2]+c[:,2,None],axis=0)
        scale = 2/np.sum(hi-lo)*(1-1e-13)
        p = (c[:,:2]-lo)*scale
        z = np.r_[p[:,0],p[:,1],c[:,2]*scale,(hi[0]-lo[0])*scale]
        retain(z)
        if bestz is None:
            bestz = z.copy()

    def optimize(z0, T, deadline, maxiter):
        # z = T @ v uses circle-major blocks x, y, r, then rectangle width.
        if time.monotonic() >= deadline:
            return
        if T is None:
            T = np.eye(64)
            v0 = z0.copy()
        else:
            v0 = z0.copy()
        X, Y, R, W = T[:21], T[21:42], T[42:63], T[63]
        dxmap = X[I]-X[J]
        dymap = Y[I]-Y[J]
        rrmap = R[I]+R[J]
        walljac = np.vstack((X-R, W-X-R, Y-R, -W-Y-R))
        gradient = -R.sum(axis=0)
        latest = v0.copy()
        cached_v = None
        cached_c = None
        cached_j = None
        calls = 0

        def check():
            if time.monotonic() >= deadline:
                raise Stop

        def compute(v):
            nonlocal cached_v, cached_c, cached_j
            check()
            if cached_v is not None and np.array_equal(v,cached_v):
                return
            z = T@v
            x,y,r,w = z[:21],z[21:42],z[42:63],z[63]
            dx,dy = x[I]-x[J],y[I]-y[J]
            d = np.sqrt(dx*dx+dy*dy+1e-30)
            cached_c = np.r_[d-r[I]-r[J],x-r,w-x-r,y-r,2-w-y-r,r-1e-9,w-.05,1.95-w]
            pairjac = (dx/d)[:,None]*dxmap + (dy/d)[:,None]*dymap-rrmap
            cached_j = np.vstack((pairjac,walljac,R,W,-W))
            cached_v = v.copy()

        def obj(v):
            check()
            return float(gradient@v),gradient

        def fun(v):
            compute(v)
            return cached_c

        def jac(v):
            compute(v)
            return cached_j

        def callback(v):
            nonlocal latest,calls
            latest = v.copy()
            calls += 1
            check()
            if calls % 5 == 0:
                retain(T@v)

        try:
            result = minimize(obj,v0,jac=True,method='SLSQP',
                              constraints={'type':'ineq','fun':fun,'jac':jac},
                              callback=callback,
                              options={'maxiter':maxiter,'ftol':2e-12,'disp':False})
            latest = result.x
        except Stop:
            pass
        retain(T@latest)

    def lattice(rows, trial):
        # Each pair has offset from the vertical symmetry axis, height and radius.
        nvar = 1 + sum(3*(k//2)+2*(k%2) for k in rows)
        T = np.zeros((64,nvar))
        T[63,-1] = 1
        T[:21,-1] = .5
        v = np.zeros(nvar)
        w = 1.0
        v[-1] = w
        rad = min(w/(2*max(rows)), 1.0/(2+(len(rows)-1)*np.sqrt(3)))
        idx = 0
        q = 0
        for row,k in enumerate(rows):
            if time.monotonic() >= end:
                raise Stop
            yy = .11 + row*.78/(len(rows)-1)
            if trial:
                yy += rng.normal(0,float(params['jitter']))
            for j in range(k//2):
                if time.monotonic() >= end:
                    raise Stop
                u = (j+.5 if k%2==0 else j+1)*2*rad
                u = min(u,.5-rad)
                T[idx,q],T[idx+1,q] = -1,1
                T[21+idx,q+1] = T[21+idx+1,q+1] = 1
                T[42+idx,q+2] = T[42+idx+1,q+2] = 1
                v[q:q+3] = [u,yy,rad*.92]
                idx += 2
                q += 3
            if k%2:
                T[21+idx,q] = 1
                T[42+idx,q+1] = 1
                v[q:q+2] = [yy,rad*.92]
                idx += 1
                q += 2
        return v,T

    # An already substantial incumbent merits an unrestricted recovery pass.
    if bestscore > 2.0 and bestz is not None:
        optimize(bestz,None,min(symend,started+.22*budget_s),int(params['iterations']))

    layouts = [(4,5,3,5,4),(3,5,5,5,3),(4,4,5,4,4),(5,4,3,4,5),(4,5,4,5,3),(3,4,7,4,3)]
    for trial,rows in enumerate(layouts):
        now = time.monotonic()
        if now >= symend:
            break
        try:
            v,T = lattice(rows,trial)
        except Stop:
            break
        retain(T@v)
        # Keep enough time to examine competing symmetric contact graphs.
        deadline = min(symend,now+max(.075*budget_s,(symend-now)/max(1,4-trial)))
        optimize(v,T,deadline,int(params['iterations']))

    if bestz is not None and time.monotonic() < end:
        optimize(bestz.copy(),None,min(end,time.monotonic()+.15*budget_s),int(params['iterations']))

    attempt = 0
    while time.monotonic() < end-.035*budget_s and bestz is not None:
        z = bestz.copy()
        amplitude = float(params['restart_noise'])*(1+attempt%3)
        z[:42] += rng.normal(0,amplitude,42)
        z[42:63] *= rng.uniform(.96,1.02,21)
        optimize(z,None,min(end,time.monotonic()+.095*budget_s),65)
        attempt += 1
    return best
