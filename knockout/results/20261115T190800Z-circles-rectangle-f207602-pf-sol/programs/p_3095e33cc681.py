import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    "strain": (0.16, 0.03, 0.4),
    "explore_mu": (240.0, 20.0, 2000.0, "log"),
    "explore_iterations": (100, 20, 250),
    "relax_iterations": (90, 20, 200),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
K = np.arange(N)

class StopSearch(Exception):
    pass

def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0] - a[J, 0])**2 + (a[I, 1] - a[J, 1])**2)
    if np.any(d < a[I, 2] + a[J, 2]):
        return -np.inf
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    q = (max(u + v for u, v in zip(x, r)) - min(u - v for u, v in zip(x, r))
         + max(u + v for u, v in zip(y, r)) - min(u - v for u, v in zip(y, r)))
    return float(np.sum(a[:, 2])) if q <= 2 else -np.inf

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf
    if init is not None:
        a = np.asarray(init['circles'], dtype=float).copy()
        best_score = exact_score(a)
    else:
        xx, yy = np.meshgrid(np.linspace(.1, .9, 5), np.linspace(.1, .9, 5))
        a = np.column_stack((xx.ravel()[:N], yy.ravel()[:N], np.full(N, .095)))
        best = {'circles': a.tolist()}
        best_score = exact_score(a)
    lo = (a[:, :2] - a[:, 2, None]).min(axis=0)
    side = (a[:, :2] + a[:, 2, None]).max(axis=0) - lo
    scale = 2.0 / side.sum()
    base = np.r_[(a[:, 0] - lo[0])*scale, (a[:, 1] - lo[1])*scale,
                 a[:, 2]*scale, side[0]*scale]
    bounds = [(0.0, 2.0)]*(2*N) + [(1e-7, 0.8)]*N + [(0.15, 1.85)]
    objective_gradient = np.r_[np.zeros(2*N), -np.ones(N), 0.0]

    def check():
        if time.monotonic() >= end:
            raise StopSearch()

    def consider(z):
        nonlocal best, best_score
        check()
        p = np.column_stack((z[:N], z[N:2*N]))
        r = z[2*N:3*N].copy()
        if not np.isfinite(z).all() or np.any(r <= 0):
            return
        distances = np.sqrt((p[I, 0]-p[J, 0])**2 + (p[I, 1]-p[J, 1])**2)
        factor = min(1.0, float(np.min(distances/(r[I]+r[J]))))
        r *= max(0.0, factor)*(1.0-3e-15)
        if np.any(r <= 0):
            return
        lower = (p-r[:, None]).min(axis=0)
        upper = (p+r[:, None]).max(axis=0)
        factor = min(1.0, 2.0/float(np.sum(upper-lower)))*(1.0-3e-15)
        p = (p-lower)*factor
        r *= factor
        candidate = np.column_stack((p, r))
        for _ in range(4):
            check()
            s = exact_score(candidate)
            if s > best_score:
                best_score = s
                best = {'circles': candidate.tolist()}
            if np.isfinite(s):
                return
            candidate[:, 2] *= 1.0-2e-14

    def penalty(z, mu):
        check()
        x, y, r, w = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
        dx, dy = x[I]-x[J], y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-26)
        v = np.maximum(r[I]+r[J]-d, 0.0)
        b0 = np.maximum(r-x, 0.0)
        b1 = np.maximum(r+x-w, 0.0)
        b2 = np.maximum(r-y, 0.0)
        b3 = np.maximum(r+y-2.0+w, 0.0)
        c = 2.0*mu*v
        gx = np.bincount(I, -c*dx/d, N)+np.bincount(J, c*dx/d, N)
        gy = np.bincount(I, -c*dy/d, N)+np.bincount(J, c*dy/d, N)
        gr = -np.ones(N)+np.bincount(I, c, N)+np.bincount(J, c, N)
        gx += 2*mu*(-b0+b1)
        gy += 2*mu*(-b2+b3)
        gr += 2*mu*(b0+b1+b2+b3)
        f = -r.sum()+mu*(v@v+b0@b0+b1@b1+b2@b2+b3@b3)
        return f, np.r_[gx, gy, gr, 2*mu*(-b1.sum()+b3.sum())]

    boundary_jac = np.zeros((4*N, 3*N+1))
    boundary_jac[K, K] = 1
    boundary_jac[K, 2*N+K] = -1
    boundary_jac[N+K, K] = -1
    boundary_jac[N+K, 2*N+K] = -1
    boundary_jac[N+K, -1] = 1
    boundary_jac[2*N+K, N+K] = 1
    boundary_jac[2*N+K, 2*N+K] = -1
    boundary_jac[3*N+K, N+K] = -1
    boundary_jac[3*N+K, 2*N+K] = -1
    boundary_jac[3*N+K, -1] = -1
    cache_z = None
    cache_c = None
    cache_j = None

    def constraints(z):
        nonlocal cache_z, cache_c, cache_j
        check()
        if cache_z is not None and np.array_equal(z, cache_z):
            return cache_c
        x, y, r, w = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
        dx, dy = x[I]-x[J], y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        c = np.r_[d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r]
        jac = np.zeros((M+4*N, 3*N+1))
        rows = np.arange(M)
        jac[rows, I] = dx/d
        jac[rows, J] = -dx/d
        jac[rows, N+I] = dy/d
        jac[rows, N+J] = -dy/d
        jac[rows, 2*N+I] = -1
        jac[rows, 2*N+J] = -1
        jac[M:] = boundary_jac
        cache_z, cache_c, cache_j = z.copy(), c, jac
        return c

    def jacobian(z):
        constraints(z)
        return cache_j

    def objective(z):
        check()
        return -float(z[2*N:3*N].sum())

    trial = 0
    try:
        while time.monotonic() < end:
            check()
            z = base.copy()
            amplitude = float(params['strain'])*(1.0 + .45*(trial//2))
            direction = -1 if (trial + seed) % 2 else 1
            w0 = base[-1]
            w1 = float(np.clip(w0+direction*amplitude, .3, 1.7))
            z[:N] *= w1/w0
            z[N:2*N] *= (2-w1)/(2-w0)
            z[2*N:3*N] *= min(w1/w0, (2-w1)/(2-w0))
            z[-1] = w1
            fixed = bounds[:-1]+[(w1, w1)]
            result = minimize(penalty, z, args=(float(params['explore_mu']),),
                              jac=True, method='L-BFGS-B', bounds=fixed,
                              options={'maxiter': int(params['explore_iterations']), 'ftol': 1e-11, 'maxls': 15})
            z = result.x
            check()
            # Release the box only after the compressed contact network equilibrates.
            result = minimize(penalty, z, args=(float(params['explore_mu'])*8,),
                              jac=True, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 55, 'ftol': 1e-11, 'maxls': 15})
            z = result.x
            consider(z)
            result = minimize(objective, z, jac=lambda z: objective_gradient,
                              method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                              callback=lambda z: consider(z),
                              options={'maxiter': int(params['relax_iterations']), 'ftol': 2e-13, 'disp': False})
            consider(result.x)
            trial += 1
    except StopSearch:
        pass
    return best
