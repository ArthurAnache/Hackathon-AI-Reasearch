import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'penalty_fraction': (0.24, 0.10, 0.40),
    'contact_margin': (0.075, 0.025, 0.20),
    'stage_iterations': (24, 10, 60),
    'restart_noise': (0.012, 0.002, 0.06),
}
N = 21
I, J = np.triu_indices(N, 1)

class Done(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70 * budget_s
    penalty_end = start + float(params['penalty_fraction']) * budget_s
    rng = np.random.default_rng(seed)
    best = None
    best_score = -np.inf
    best_z = None
    last = None

    def check():
        if time.monotonic() >= end:
            raise Done()

    def valid(a):
        if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return False
        d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
        if np.any(d < a[I, 2]+a[J, 2]):
            return False
        x = [Fraction(float(t)) for t in a[:, 0]]
        y = [Fraction(float(t)) for t in a[:, 1]]
        r = [Fraction(float(t)) for t in a[:, 2]]
        return (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r)) +
                max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))) <= 2

    def remember(z):
        nonlocal best, best_score, best_z
        if time.monotonic() >= end:
            return
        p = np.column_stack((z[:N], z[N:2*N])).copy()
        r = np.maximum(z[2*N:3*N], 1e-12).copy()
        if not np.isfinite(p).all() or not np.isfinite(r).all():
            return
        d = np.sqrt((p[I,0]-p[J,0])**2 + (p[I,1]-p[J,1])**2)
        shrink = min(1.0, float(np.min(d/(r[I]+r[J]))))
        if shrink <= 0:
            return
        r *= shrink * (1-3e-13)
        p -= (p-r[:,None]).min(axis=0)
        sides = (p+r[:,None]).max(axis=0)-(p-r[:,None]).min(axis=0)
        scale = 2.0 / sides.sum() * (1-3e-13)
        p *= scale
        r *= scale
        score = float(np.sum(r))
        if score <= best_score:
            return
        a = np.column_stack((p,r))
        if valid(a):
            best = {'circles': a.tolist()}
            best_score = score
            best_z = np.r_[p[:,0],p[:,1],r,sides[0]*scale]

    # A valid, deterministic safety construction also supplies a fallback start.
    points = []
    for row, count in enumerate((4,5,4,5,3)):
        check()
        for col in range(count):
            check()
            points.append((2*col + (1 if count < 5 else 0), np.sqrt(3)*row))
    p0 = np.asarray(points, dtype=float)
    p0 -= (p0-1).min(axis=0)
    sides = np.ptp(p0,axis=0)+2
    scale = 2/sides.sum()
    fallback = np.r_[p0[:,0]*scale,p0[:,1]*scale,np.full(N,scale),sides[0]*scale]
    remember(fallback)
    z = fallback.copy()
    if init is not None:
        try:
            a = np.asarray(init['circles'],dtype=float)
            if a.shape == (N,3) and np.isfinite(a).all():
                if valid(a):
                    sc = float(np.sum(a[:,2]))
                    if sc > best_score:
                        best, best_score = init, sc
                p = a[:,:2].copy()
                r = np.maximum(np.abs(a[:,2]),1e-9)
                lo = (p-r[:,None]).min(axis=0)
                sides = (p+r[:,None]).max(axis=0)-lo
                s = 2/max(float(sides.sum()),1e-12)
                p = (p-lo)*s
                z = np.r_[p[:,0],p[:,1],r*s,sides[0]*s]
                remember(z)
        except (KeyError, ValueError, TypeError):
            pass

    def penalty(v,mu):
        nonlocal last
        check()
        if time.monotonic() >= penalty_end:
            raise Done()
        last = v.copy()
        x,y,r,w = v[:N],v[N:2*N],v[2*N:3*N],v[-1]
        dx,dy = x[I]-x[J],y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-26)
        o = np.maximum(r[I]+r[J]-d,0)
        b = np.maximum(np.array([r-x,r+x-w,r-y,r+y-2+w]),0)
        c = 2*mu*o
        gx = np.bincount(I,-c*dx/d,minlength=N)+np.bincount(J,c*dx/d,minlength=N)
        gy = np.bincount(I,-c*dy/d,minlength=N)+np.bincount(J,c*dy/d,minlength=N)
        gr = -np.ones(N)+np.bincount(I,c,minlength=N)+np.bincount(J,c,minlength=N)
        gx += 2*mu*(-b[0]+b[1])
        gy += 2*mu*(-b[2]+b[3])
        gr += 2*mu*b.sum(axis=0)
        return -r.sum()+mu*(o@o+np.sum(b*b)), np.r_[gx,gy,gr,2*mu*(-b[1].sum()+b[3].sum())]

    bounds = [(None,None)]*(2*N)+[(1e-9,0.6)]*N+[(0.05,1.95)]
    for mu in (10.,100.,1000.):
        if time.monotonic() >= penalty_end:
            break
        last = z.copy()
        try:
            res = minimize(penalty,z,args=(mu,),jac=True,method='L-BFGS-B',bounds=bounds,
                           options={'maxiter':75,'ftol':1e-11,'maxls':16})
            z = res.x
        except Done:
            z = last.copy()
            break
        remember(z)
    remember(z)

    grad = np.r_[np.zeros(2*N),-np.ones(N),0.]
    eye = np.eye(N)
    rounds = 0
    stagnant = 0
    previous = -np.inf
    while time.monotonic() < end-0.035:
        check()
        dx,dy = z[I]-z[J],z[N+I]-z[N+J]
        d = np.sqrt(dx*dx+dy*dy)
        gap = d-z[2*N+I]-z[2*N+J]
        active = gap < float(params['contact_margin'])
        # Include each circle's nearest competitors even in dilute initial states.
        for k in range(N):
            check()
            ids = np.flatnonzero((I==k)|(J==k))
            active[ids[np.argsort(gap[ids])[:5]]] = True
        ai,aj = I[active],J[active]
        m = len(ai)
        rows = np.arange(m)
        base = np.zeros((m+4*N,3*N+1))
        base[rows,2*N+ai] = -1
        base[rows,2*N+aj] = -1
        for k in range(4):
            check()
            sl = slice(m+k*N,m+(k+1)*N)
            base[sl,2*N:3*N] = -eye
            offset = 0 if k<2 else N
            base[sl,offset:offset+N] = eye if k%2==0 else -eye
        base[m+N:m+2*N,-1] = 1
        base[m+3*N:m+4*N,-1] = -1

        def objective(v):
            check()
            return -float(v[2*N:3*N].sum())
        def con(v):
            check()
            x,y,r,w = v[:N],v[N:2*N],v[2*N:3*N],v[-1]
            dist = np.sqrt((x[ai]-x[aj])**2+(y[ai]-y[aj])**2+1e-30)
            return np.r_[dist-r[ai]-r[aj],x-r,w-x-r,y-r,2-w-y-r]
        def jac(v):
            check()
            dx,dy = v[ai]-v[aj],v[N+ai]-v[N+aj]
            dist = np.sqrt(dx*dx+dy*dy+1e-30)
            a = base.copy()
            a[rows,ai],a[rows,aj] = dx/dist,-dx/dist
            a[rows,N+ai],a[rows,N+aj] = dy/dist,-dy/dist
            return a
        count = [0]
        def callback(v):
            nonlocal last
            check()
            last = v.copy()
            count[0] += 1
            if count[0]%3 == 0:
                remember(v)
        last = z.copy()
        try:
            result = minimize(objective,z,jac=lambda v:grad,method='SLSQP',bounds=bounds,
                              constraints={'type':'ineq','fun':con,'jac':jac},callback=callback,
                              options={'maxiter':int(params['stage_iterations']),'ftol':2e-12})
            z = result.x
        except Done:
            break
        remember(z)
        rounds += 1
        if best_score <= previous+1e-8:
            stagnant += 1
        else:
            stagnant = 0
        previous = best_score
        if stagnant >= 2 and best_z is not None:
            z = best_z.copy()
            z[:2*N] += rng.normal(0,float(params['restart_noise']),2*N)
            z[2*N:3*N] *= 0.97
            stagnant = 0
    return best
