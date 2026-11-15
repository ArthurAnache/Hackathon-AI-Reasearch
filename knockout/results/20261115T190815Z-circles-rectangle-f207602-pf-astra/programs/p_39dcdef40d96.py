import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'repulsion': (0.30, 0.05, 0.8),
    'escape_scale': (0.065, 0.02, 0.15),
    'jitter': (0.012, 0.002, 0.05),
    'escape_iterations': (90, 30, 180),
    'relax_iterations': (110, 30, 220)
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class Deadline(Exception):
    pass

def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or not (a[:, 2] > 0).all():
        return -float('inf')
    r = a[:, 2]
    d = np.sqrt((a[I, 0] - a[J, 0])**2 + (a[I, 1] - a[J, 1])**2)
    if np.any(d < r[I] + r[J]):
        return -float('inf')
    x = list(map(lambda v: Fraction(float(v)), a[:, 0]))
    y = list(map(lambda v: Fraction(float(v)), a[:, 1]))
    q = list(map(lambda v: Fraction(float(v)), r))
    perimeter = (max(u + v for u, v in zip(x, q)) - min(u - v for u, v in zip(x, q))
                 + max(u + v for u, v in zip(y, q)) - min(u - v for u, v in zip(y, q)))
    return float(np.sum(r)) if perimeter <= 2 else -float('inf')

def repaired(z):
    p = np.column_stack((z[:N], z[N:2*N])).copy()
    r = np.maximum(z[2*N:3*N], 1e-10).copy()
    d = np.sqrt((p[I, 0] - p[J, 0])**2 + (p[I, 1] - p[J, 1])**2)
    ratio = min(1.0, float(np.min(d / (r[I] + r[J]))))
    r *= ratio * (1.0 - 3e-14)
    lo = np.min(p - r[:, None], axis=0)
    hi = np.max(p + r[:, None], axis=0)
    scale = (2.0 - 2e-13) / float(np.sum(hi - lo))
    return np.column_stack(((p - lo) * scale, r * scale))

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.69 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -float('inf')
    if init is not None:
        a = np.asarray(init['circles'], dtype=float)
        best_score = exact_score(a)
    else:
        k = np.arange(N)
        a = np.column_stack(((k % 5 + 0.5) / 5, (k // 5 + 0.5) / 5, np.full(N, 0.075)))
        best = {'circles': a.tolist()}
        best_score = exact_score(a)
    lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
    hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
    scale = 2.0 / float(np.sum(hi - lo))
    p = (a[:, :2] - lo) * scale
    r = a[:, 2] * scale
    base = np.r_[p[:, 0], p[:, 1], r, (hi[0] - lo[0]) * scale]
    objective_grad = np.r_[np.zeros(2*N), -np.ones(N), 0.0]
    rows = np.arange(M)
    ix = np.arange(N)
    bounds = [(0.0, 2.0)] * (2*N) + [(1e-7, 0.7)] * N + [(0.2, 1.8)]

    def check():
        if time.monotonic() >= deadline:
            raise Deadline()

    def consider(z):
        nonlocal best, best_score
        candidate = repaired(z)
        score = exact_score(candidate)
        if score > best_score:
            best_score = score
            best = {'circles': candidate.tolist()}

    def penalty(z, mu, amplitude, reference, sigma):
        check()
        x, y, r, w = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
        dx, dy = x[I] - x[J], y[I] - y[J]
        d = np.sqrt(dx*dx + dy*dy + 1e-24)
        v = np.maximum(r[I] + r[J] - d, 0.0)
        b = np.maximum(np.vstack((r-x, r+x-w, r-y, r+y-(2-w))), 0.0)
        f = -float(r.sum()) + mu * (float(v @ v) + float(np.sum(b*b)))
        c = 2*mu*v
        gx = np.bincount(I, weights=-c*dx/d, minlength=N) + np.bincount(J, weights=c*dx/d, minlength=N)
        gy = np.bincount(I, weights=-c*dy/d, minlength=N) + np.bincount(J, weights=c*dy/d, minlength=N)
        gr = -np.ones(N) + np.bincount(I, weights=c, minlength=N) + np.bincount(J, weights=c, minlength=N)
        gx += 2*mu*(-b[0]+b[1])
        gy += 2*mu*(-b[2]+b[3])
        gr += 2*mu*np.sum(b, axis=0)
        gw = 2*mu*(-np.sum(b[1])+np.sum(b[3]))
        g = np.r_[gx, gy, gr, gw]
        delta = z[:2*N] - reference[:2*N]
        denom = N*sigma*sigma
        repel = amplitude*np.exp(-0.5*float(delta @ delta)/denom)
        f += repel
        g[:2*N] -= repel*delta/denom
        return f, g

    def constraints(z):
        check()
        x, y, r, w = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
        dx, dy = x[I]-x[J], y[I]-y[J]
        return np.r_[dx*dx+dy*dy-(r[I]+r[J])**2, x-r, w-x-r, y-r, 2-w-y-r]

    def jacobian(z):
        check()
        x, y, r = z[:N], z[N:2*N], z[2*N:3*N]
        dx, dy, rr = x[I]-x[J], y[I]-y[J], r[I]+r[J]
        j = np.zeros((M+4*N, 3*N+1))
        j[rows, I] = 2*dx
        j[rows, J] = -2*dx
        j[rows, N+I] = 2*dy
        j[rows, N+J] = -2*dy
        j[rows, 2*N+I] = -2*rr
        j[rows, 2*N+J] = -2*rr
        j[M+ix, ix] = 1
        j[M+N+ix, ix] = -1
        j[M+2*N+ix, N+ix] = 1
        j[M+3*N+ix, N+ix] = -1
        j[M+ix, 2*N+ix] = -1
        j[M+N+ix, 2*N+ix] = -1
        j[M+2*N+ix, 2*N+ix] = -1
        j[M+3*N+ix, 2*N+ix] = -1
        j[M+N+ix, -1] = 1
        j[M+3*N+ix, -1] = -1
        return j

    attempt = 0
    while time.monotonic() < deadline - 0.035:
        z = base.copy()
        z[:2*N] += rng.normal(0, params['jitter'], 2*N)
        last = [z.copy()]
        def callback(v):
            last[0] = v.copy()
            check()
        try:
            for mu, strength in ((180.0, 1.0), (1800.0, 0.35)):
                check()
                result = minimize(penalty, z, args=(mu, params['repulsion']*strength, base, params['escape_scale']),
                                  jac=True, method='L-BFGS-B', bounds=bounds, callback=callback,
                                  options={'maxiter': int(params['escape_iterations']), 'ftol': 1e-12, 'maxls': 20})
                z = result.x
            check()
            result = minimize(lambda v: (-float(np.sum(v[2*N:3*N])), objective_grad), z,
                              jac=True, method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                              callback=callback,
                              options={'maxiter': int(params['relax_iterations']), 'ftol': 2e-12})
            z = result.x
            consider(z)
        except Deadline:
            consider(last[0])
            break
        attempt += 1
    return best
