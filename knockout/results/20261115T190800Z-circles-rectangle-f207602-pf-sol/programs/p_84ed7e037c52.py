import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'exchange_count': (4, 3, 7),
    'tether': (160.0, 10.0, 1000.0, 'log'),
    'locality': (0.45, 0.1, 1.0),
}
N = 21
I, J = np.triu_indices(N, 1)

class StopSearch(Exception):
    pass

def exact_score(c):
    a = np.asarray(c['circles'], dtype=float)
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    r = a[:, 2]
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < r[I]+r[J]):
        return -np.inf
    x, y, q = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
    perimeter = (max(u+v for u,v in zip(x,q))-min(u-v for u,v in zip(x,q))
                 +max(u+v for u,v in zip(y,q))-min(u-v for u,v in zip(y,q)))
    if perimeter > 2:
        return -np.inf
    return float(np.sum(r))

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        pts = np.array([[(i%5+.5)*.2, (i//5+.5)*.2, .085] for i in range(N)])
        init = {'circles': pts.tolist()}
    best = init
    best_score = exact_score(init)
    a = np.asarray(init['circles'], float).copy()
    lo = np.min(a[:, :2]-a[:, 2,None], axis=0)
    sides = np.max(a[:, :2]+a[:, 2,None], axis=0)-lo
    scale = 2/sides.sum()
    a[:, :2] = (a[:, :2]-lo)*scale
    a[:, 2] *= scale
    base = np.r_[a[:,0], a[:,1], a[:,2], sides[0]*scale]
    last = [base.copy()]

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def constraints(z, jac=False):
        check()
        x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
        dx,dy = x[I]-x[J],y[I]-y[J]
        d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-25)
        v = np.r_[d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r]
        if not jac:
            return v
        g = np.zeros((len(v), 64))
        k = np.arange(len(I))
        g[k,I] = dx/d; g[k,J] = -dx/d
        g[k,N+I] = dy/d; g[k,N+J] = -dy/d
        g[k,2*N+I] = -1; g[k,2*N+J] = -1
        off = len(I)
        for group in range(4):
            check()
            rows = off + group*N + np.arange(N)
            g[rows,2*N+np.arange(N)] = -1
            if group == 0:
                g[rows,np.arange(N)] = 1
            elif group == 1:
                g[rows,np.arange(N)] = -1; g[rows,-1] = 1
            elif group == 2:
                g[rows,N+np.arange(N)] = 1
            else:
                g[rows,N+np.arange(N)] = -1; g[rows,-1] = -1
        return v,g

    def consider(z):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        p = np.column_stack((z[:N],z[N:2*N]))
        r = np.maximum(z[2*N:3*N].copy(), 1e-12)
        d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
        ratio = np.min(d/(r[I]+r[J]))
        r *= min(1.0, ratio)*(1-3e-14)
        lo = np.min(p-r[:,None],axis=0)
        ext = np.max(p+r[:,None],axis=0)-lo
        factor = min(1.0, 2/ext.sum())*(1-3e-14)
        p = (p-lo)*factor
        r *= factor
        c = {'circles': np.column_stack((p,r)).tolist()}
        score = exact_score(c)
        if score > best_score:
            best,best_score = c,score

    bounds = [(None,None)]*(2*N)+[(1e-8,.5)]*N+[(.15,1.85)]
    objective_gradient = np.r_[np.zeros(2*N),-np.ones(N),0.0]
    attempts = 0
    try:
        while time.monotonic() < deadline-0.12:
            check()
            z = base.copy()
            radii = z[2*N:3*N]
            p = np.column_stack((z[:N],z[N:2*N]))
            anchor = int(np.argsort(radii)[-1-(attempts%5)])
            distances = np.linalg.norm(p-p[anchor],axis=1)
            desirability = (radii[anchor]-radii)/(distances+float(params['locality']))
            desirability[anchor] = -np.inf
            k = int(params['exchange_count'])
            pool = np.argsort(desirability)[-min(10,N-1):]
            chosen = np.r_[anchor, rng.choice(pool,k-1,replace=False)]
            target = radii.copy()
            target[chosen] = np.roll(target[chosen],1 if attempts%2 == 0 else -1)
            z[2*N:3*N] = target
            last[0] = z.copy()
            local_end = min(deadline-0.12,time.monotonic()+min(.42,.28*budget_s))
            for mu,tether in [(70.,float(params['tether'])),(350.,15.),(2500.,0.)]:
                check()
                if time.monotonic() >= local_end:
                    break
                def penalty(v):
                    check()
                    c,g = constraints(v,True)
                    neg = np.minimum(c,0)
                    dr = v[2*N:3*N]-target
                    f = -v[2*N:3*N].sum()+mu*np.dot(neg,neg)+tether*np.dot(dr,dr)
                    grad = objective_gradient+2*mu*(neg@g)
                    grad[2*N:3*N] += 2*tether*dr
                    last[0] = v.copy()
                    return f,grad
                z = minimize(penalty,z,jac=True,method='L-BFGS-B',bounds=bounds,
                             options={'maxiter':65,'ftol':1e-11,'maxls':15}).x
            check()
            def obj(v):
                check()
                last[0] = v.copy()
                return -float(v[2*N:3*N].sum())
            result = minimize(obj,z,jac=lambda v:objective_gradient,method='SLSQP',bounds=bounds,
                              constraints={'type':'ineq','fun':constraints,'jac':lambda v:constraints(v,True)[1]},
                              callback=lambda v:check(),
                              options={'maxiter':85,'ftol':2e-12,'disp':False})
            consider(result.x)
            attempts += 1
    except StopSearch:
        pass
    if time.monotonic() < start+0.76*budget_s:
        deadline = start+0.77*budget_s
        consider(last[0])
    return best
