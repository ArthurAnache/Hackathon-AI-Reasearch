import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'equalization': (35.0, 2.0, 100.0, 'log'),
    'soft_penalty': (180.0, 20.0, 2000.0, 'log'),
    'homotopy_iterations': (100, 20, 250),
    'sqp_iterations': (140, 30, 300),
    'jitter': (0.001, 0.0, 0.02)
}
N = 21
I, J = np.triu_indices(N, 1)
K = np.arange(N)
M = len(I)

class Deadline(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0] - a[J, 0])**2 + (a[I, 1] - a[J, 1])**2)
    if np.any(d < a[I, 2] + a[J, 2]):
        return -np.inf
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    perimeter = (max(u + v for u, v in zip(x, r)) - min(u - v for u, v in zip(x, r))
                 + max(u + v for u, v in zip(y, r)) - min(u - v for u, v in zip(y, r)))
    if perimeter > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.69 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf
    if init is not None:
        a0 = np.array(init['circles'], dtype=float)
        best_score = exact_score(a0)
    else:
        a0 = np.array([[(i % 7 + 0.5) / 7, (i // 7 + 0.5) / 3, 0.035] for i in range(N)])

    def check():
        if time.monotonic() >= deadline:
            raise Deadline

    def pack(a):
        lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
        hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
        sides = hi - lo
        scale = 2.0 / np.sum(sides)
        p = (a[:, :2] - lo) * scale
        return np.r_[p[:, 0], p[:, 1], a[:, 2] * scale, sides[0] * scale]

    def consider(z):
        nonlocal best, best_score
        check()
        p = np.column_stack((z[:N], z[N:2*N])).copy()
        r = np.maximum(z[2*N:3*N], 1e-10).copy()
        if not np.isfinite(p).all() or not np.isfinite(r).all():
            return
        dist = np.sqrt(np.sum((p[I] - p[J])**2, axis=1))
        factor = min(1.0, float(np.min(dist / (r[I] + r[J]))))
        r *= max(factor, 0.0) * (1.0 - 2e-14)
        if np.any(r <= 0):
            return
        lo = np.min(p - r[:, None], axis=0)
        hi = np.max(p + r[:, None], axis=0)
        scale = min(1.0, 2.0 / np.sum(hi - lo)) * (1.0 - 2e-14)
        a = np.column_stack(((p - lo) * scale, r * scale))
        score = exact_score(a)
        if score > best_score:
            best_score = score
            best = {'circles': a.tolist()}

    def penalty(z, mu, eq):
        check()
        x, y, r, w = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
        dx, dy = x[I] - x[J], y[I] - y[J]
        d = np.sqrt(dx*dx + dy*dy + 1e-24)
        v = np.maximum(r[I] + r[J] - d, 0.0)
        b0 = np.maximum(r - x, 0.0)
        b1 = np.maximum(r + x - w, 0.0)
        b2 = np.maximum(r - y, 0.0)
        b3 = np.maximum(r + y + w - 2.0, 0.0)
        centered = r - r.mean()
        f = -r.sum() + mu * (v @ v + b0 @ b0 + b1 @ b1 + b2 @ b2 + b3 @ b3) + eq * (centered @ centered)
        gx, gy = np.zeros(N), np.zeros(N)
        gr = -np.ones(N) + 2.0 * eq * centered
        c = 2.0 * mu * v
        np.add.at(gr, I, c)
        np.add.at(gr, J, c)
        np.add.at(gx, I, -c * dx / d)
        np.add.at(gx, J, c * dx / d)
        np.add.at(gy, I, -c * dy / d)
        np.add.at(gy, J, c * dy / d)
        gx += 2*mu*(-b0+b1)
        gy += 2*mu*(-b2+b3)
        gr += 2*mu*(b0+b1+b2+b3)
        gw = 2*mu*(-b1.sum()+b3.sum())
        return f, np.r_[gx, gy, gr, gw]

    objective_gradient = np.r_[np.zeros(2*N), -np.ones(N), 0.0]
    def objective(z):
        check()
        return -float(np.sum(z[2*N:3*N]))

    def constraints(z):
        check()
        x, y, r, w = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
        d = np.sqrt((x[I]-x[J])**2 + (y[I]-y[J])**2 + 1e-30)
        return np.r_[d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r]

    def jacobian(z):
        check()
        x, y = z[:N], z[N:2*N]
        dx, dy = x[I]-x[J], y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        a = np.zeros((M+4*N, 3*N+1))
        rows = np.arange(M)
        a[rows,I] = dx/d
        a[rows,J] = -dx/d
        a[rows,N+I] = dy/d
        a[rows,N+J] = -dy/d
        a[rows,2*N+I] = -1
        a[rows,2*N+J] = -1
        for block, col, sign in ((0,0,1),(1,0,-1),(2,N,1),(3,N,-1)):
            check()
            rr = M+block*N+K
            a[rr,col+K] = sign
            a[rr,2*N+K] = -1
        a[M+N:M+2*N,-1] = 1
        a[M+3*N:M+4*N,-1] = -1
        return a

    bounds = [(None,None)]*(2*N) + [(1e-7,0.6)]*N + [(0.12,1.88)]
    last = pack(a0)
    if init is None:
        consider(last)
    base = last.copy()
    try:
        # Anneal the radius hierarchy rather than merely shaking the centers.
        for attempt in range(3):
            check()
            z = base.copy()
            if attempt:
                z[:2*N] += rng.normal(0, float(params['jitter'])*(attempt+1), 2*N)
            eq = float(params['equalization']) * (1.7 ** attempt)
            mu = float(params['soft_penalty'])
            for multiplier, stiffness in ((1.0,1.0),(0.18,2.5),(0.0,10.0)):
                check()
                res = minimize(penalty, z, args=(mu*stiffness,eq*multiplier), jac=True,
                               method='L-BFGS-B', bounds=bounds,
                               options={'maxiter':int(params['homotopy_iterations']), 'ftol':2e-11, 'gtol':2e-7, 'maxls':15})
                z = res.x
                last = z.copy()
                consider(z)
            check()
            def callback(q):
                nonlocal last
                check()
                last = q.copy()
            res = minimize(objective, z, jac=lambda q: objective_gradient,
                           constraints={'type':'ineq','fun':constraints,'jac':jacobian},
                           bounds=bounds, method='SLSQP', callback=callback,
                           options={'maxiter':int(params['sqp_iterations']), 'ftol':3e-13, 'disp':False})
            last = res.x
            consider(last)
    except Deadline:
        pass
    # Reserve a small, bounded finalization allowance after the search deadline.
    if best is None:
        a = np.array([[(i%7)*0.1,(i//7)*0.1,0.001] for i in range(N)])
        best = {'circles':a.tolist()}
    return best
