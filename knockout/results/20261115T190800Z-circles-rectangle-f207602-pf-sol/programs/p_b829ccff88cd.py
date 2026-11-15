import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

PARAMS = {
    'target': (2.39, 2.367, 2.48),
    'temperature': (0.009, 0.001, 0.04, 'log'),
    'elastic_iterations': (95, 20, 250),
    'release_iterations': (115, 20, 250),
    'kick': (0.007, 0.0, 0.06),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class Stop(Exception):
    pass


def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0] - a[J, 0])**2 + (a[I, 1] - a[J, 1])**2)
    if np.any(d < a[I, 2] + a[J, 2]):
        return -np.inf
    x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
    perimeter = (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
                 + max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)))
    return float(np.sum(a[:, 2])) if perimeter <= 2 else -np.inf


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf
    if init is not None:
        a0 = np.asarray(init['circles'], dtype=float)
        best_score = exact_score(a0)
    else:
        a0 = np.array([[(k % 7)*0.18+0.09, (k//7)*0.18+0.09, 0.085] for k in range(N)])
        best = {'circles': a0.tolist()}
        best_score = exact_score(a0)

    def check():
        if time.monotonic() >= deadline:
            raise Stop()

    def consider(z):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        a = np.column_stack((z[:N], z[N:2*N], z[2*N:3*N])).copy()
        if not np.isfinite(a).all():
            return
        a[:, 2] = np.maximum(a[:, 2], 1e-10)
        dist = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
        factor = min(1.0, float(np.min(dist/(a[I,2]+a[J,2]))))
        a[:,2] *= factor*(1-3e-14)
        if np.any(a[:,2] <= 0):
            return
        lo = np.min(a[:,:2]-a[:,2,None], axis=0)
        hi = np.max(a[:,:2]+a[:,2,None], axis=0)
        a[:,:2] -= lo
        scale = min(1.0, 2.0/float(np.sum(hi-lo)))*(1-3e-14)
        a *= scale
        s = exact_score(a)
        if s > best_score:
            best_score = s
            best = {'circles': a.tolist()}

    p = a0[:,:2].copy()
    r = a0[:,2].copy()
    lo = np.min(p-r[:,None], axis=0)
    side = np.max(p+r[:,None], axis=0)-lo
    scale = 2/side.sum()
    p = (p-lo)*scale
    r *= scale
    w = side[0]*scale
    base = np.r_[p[:,0], p[:,1], r, w]

    def constraints(z, jacobian=False):
        x, y, rr, ww = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
        dx, dy = x[I]-x[J], y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-28)
        g = np.r_[d-rr[I]-rr[J], x-rr, ww-x-rr, y-rr, 2-ww-y-rr]
        if not jacobian:
            return g
        j = np.zeros((M+4*N, 3*N+1))
        k = np.arange(M)
        j[k,I] = dx/d
        j[k,J] = -dx/d
        j[k,N+I] = dy/d
        j[k,N+J] = -dy/d
        j[k,2*N+I] = -1
        j[k,2*N+J] = -1
        q = np.arange(N)
        for b in range(4):
            check()
            rows = M+b*N+q
            j[rows,2*N+q] = -1
            j[rows,(0 if b < 2 else N)+q] = 1 if b%2 == 0 else -1
        j[M+N:M+2*N,-1] = 1
        j[M+3*N:M+4*N,-1] = -1
        return g, j

    objective_grad = np.r_[np.zeros(2*N), -np.ones(N), 0.0]
    bounds = [(-0.2,2.2)]*(2*N)+[(1e-8,0.6)]*N+[(0.25,1.75)]

    def release(z, iterations):
        check()
        def fun(t):
            check()
            return -float(np.sum(t[2*N:3*N])), objective_grad
        def cf(t):
            check()
            return constraints(t)
        def cj(t):
            check()
            return constraints(t, True)[1]
        res = minimize(fun, z, jac=True, method='SLSQP', bounds=bounds,
                       constraints={'type':'ineq','fun':cf,'jac':cj},
                       callback=lambda t: check(),
                       options={'maxiter':iterations,'ftol':2e-12,'disp':False})
        consider(res.x)
        return res.x

    try:
        target = float(params['target'])
        logits = np.log(np.maximum(r,1e-9))
        logits -= logits.mean()
        u = np.r_[p[:,0], p[:,1], logits, w]
        u[:2*N] += rng.normal(0, float(params['kick']), 2*N)
        ubounds = [(-0.2,2.2)]*(2*N)+[(-9,5)]*N+[(0.25,1.75)]
        for stage in range(3):
            check()
            temperature = float(params['temperature'])*(0.25**stage)
            def elastic(t):
                check()
                ll = t[2*N:3*N]
                fractions = np.exp(ll-logsumexp(ll))
                radii = target*fractions
                z = np.r_[t[:2*N], radii, t[-1]]
                g, j = constraints(z, True)
                v = -g/temperature
                norm = logsumexp(v)
                weights = np.exp(v-norm)
                grad = -weights @ j
                gr = grad[2*N:3*N].copy()
                grad[2*N:3*N] = radii*(gr-np.dot(fractions,gr))
                return float(temperature*norm), grad
            result = minimize(elastic, u, jac=True, method='L-BFGS-B', bounds=ubounds,
                              callback=lambda t: check(),
                              options={'maxiter':int(params['elastic_iterations']),
                                       'ftol':1e-13,'gtol':2e-7,'maxls':15,'maxcor':12})
            u = result.x
        check()
        rr = target*np.exp(u[2*N:3*N]-logsumexp(u[2*N:3*N]))
        z = np.r_[u[:2*N],rr,u[-1]]
        consider(z)
        z = release(z, int(params['release_iterations']))
        # A second, cooler stress excursion starts from the new graph if time permits.
        if time.monotonic() < start + 0.48*budget_s:
            check()
            p2 = np.column_stack((z[:N],z[N:2*N]))
            pick = rng.choice(N,3,replace=False)
            p2[pick] += rng.normal(0,0.025,(3,2))
            z[:N], z[N:2*N] = p2[:,0], p2[:,1]
            release(z, int(params['release_iterations']))
    except Stop:
        pass
    return best
