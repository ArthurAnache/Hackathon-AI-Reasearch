import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linprog

PARAMS = {
    'aspect': (1.0, 0.5, 2.0, 'log'),
    'rows': (5, 3, 7),
    'jitter': (0.03, 0.0, 0.15),
    'mu0': (10.0, 1.0, 1000.0, 'log'),
    'mu_growth': (8.0, 2.0, 30.0),
    'stages': (4, 2, 8),
    'maxiter': (150, 30, 500),
    'polish_iter': (120, 20, 300)
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
K = np.arange(N)

class Timeout(Exception):
    pass

def exact_score(data):
    try:
        a = np.asarray(data['circles'], dtype=np.float64)
        if a.shape != (N, 3) or not np.isfinite(a).all() or not (a[:, 2] > 0).all():
            return -np.inf
        r = a[:, 2]
        dx = a[I, 0] - a[J, 0]
        dy = a[I, 1] - a[J, 1]
        if np.any(np.sqrt(dx * dx + dy * dy) < r[I] + r[J]):
            return -np.inf
        x, y, q = ([Fraction(float(v)) for v in a[:, k]] for k in range(3))
        hp = (max(u + v for u, v in zip(x, q)) - min(u - v for u, v in zip(x, q))
              + max(u + v for u, v in zip(y, q)) - min(u - v for u, v in zip(y, q)))
        if hp > 2:
            return -np.inf
        return float(np.sum(r))
    except (ValueError, TypeError, KeyError, OverflowError):
        return -np.inf

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    penalty_end = start + 0.36 * budget_s
    polish_end = start + 0.68 * budget_s
    final_end = start + 0.78 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = exact_score(init)

    def consider(p, r):
        nonlocal best, best_score
        if time.monotonic() >= final_end:
            return
        p = np.asarray(p, dtype=float).copy()
        r = np.maximum(np.asarray(r, dtype=float), 1e-10).copy()
        if not np.isfinite(p).all() or not np.isfinite(r).all():
            return
        d = np.sqrt(np.sum((p[I] - p[J]) ** 2, axis=1))
        factor = min(1.0, float(np.min(d / (r[I] + r[J]))))
        if factor <= 0:
            return
        r *= factor * (1 - 3e-13)
        lo = np.min(p - r[:, None], axis=0)
        hi = np.max(p + r[:, None], axis=0)
        scale = 2.0 / float(np.sum(hi - lo)) * (1 - 3e-13)
        p = (p - lo) * scale
        r *= scale
        out = {'circles': np.column_stack((p, r)).tolist()}
        score = exact_score(out)
        if score > best_score:
            best, best_score = out, score

    # A deterministic valid fallback also supplies a sensible grid when needed.
    rows = int(params['rows'])
    cols = int(np.ceil(N / rows))
    points = []
    for k in range(rows):
        if time.monotonic() >= final_end:
            break
        for j in range(cols):
            if time.monotonic() >= final_end or len(points) == N:
                break
            points.append(((j + 0.5 + 0.25 * (k % 2)) / cols, (k + 0.5) / rows))
    p = np.asarray(points)
    if len(p) != N:
        return init
    a = float(params['aspect'])
    w = 2 * a / (1 + a)
    p = np.clip(p + rng.normal(0, float(params['jitter']), p.shape), 0.02, 0.98)
    p *= [w, 2 - w]
    r = np.full(N, 0.05)
    consider(p, r)
    if init is not None:
        try:
            c = np.asarray(init['circles'], dtype=float)
            if c.shape == (N, 3) and np.isfinite(c).all():
                pp, rr = c[:, :2].copy(), np.maximum(np.abs(c[:, 2]), 1e-8)
                lo = np.min(pp - rr[:, None], axis=0)
                side = np.max(pp + rr[:, None], axis=0) - lo
                s = 2 / side.sum()
                p, r, w = (pp - lo) * s, rr * s, side[0] * s
                consider(p, r)
        except (TypeError, ValueError, KeyError):
            pass
    z = np.r_[p[:, 0], p[:, 1], r, w]
    current = z.copy()

    def penalty(z, mu):
        if time.monotonic() >= penalty_end:
            raise Timeout
        x, y, r, w = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
        dx, dy = x[I] - x[J], y[I] - y[J]
        d = np.sqrt(dx*dx + dy*dy + 1e-24)
        v = np.maximum(r[I] + r[J] - d, 0)
        b0, b1 = np.maximum(r-x, 0), np.maximum(r+x-w, 0)
        b2, b3 = np.maximum(r-y, 0), np.maximum(r+y-(2-w), 0)
        neg = np.maximum(-r, 0)
        f = -r.sum() + mu * (v@v + b0@b0 + b1@b1 + b2@b2 + b3@b3 + neg@neg)
        c = 2*mu*v
        gx = np.bincount(I, -c*dx/d, minlength=N) + np.bincount(J, c*dx/d, minlength=N)
        gy = np.bincount(I, -c*dy/d, minlength=N) + np.bincount(J, c*dy/d, minlength=N)
        gr = -np.ones(N) + np.bincount(I, c, minlength=N) + np.bincount(J, c, minlength=N)
        gx += 2*mu*(-b0+b1)
        gy += 2*mu*(-b2+b3)
        gr += 2*mu*(b0+b1+b2+b3-neg)
        gw = 2*mu*(-b1.sum()+b3.sum())
        return f, np.r_[gx, gy, gr, gw]

    def penalty_callback(v):
        nonlocal current
        current = v.copy()
        if time.monotonic() >= penalty_end:
            raise Timeout

    mu = float(params['mu0'])
    for stage in range(int(params['stages'])):
        if time.monotonic() >= penalty_end:
            break
        try:
            res = minimize(penalty, z, args=(mu,), jac=True, method='L-BFGS-B',
                           callback=penalty_callback,
                           options={'maxiter': int(params['maxiter']), 'ftol': 1e-11})
            z = res.x
            current = z.copy()
        except Timeout:
            z = current.copy()
        consider(np.column_stack((z[:N], z[N:2*N])), z[2*N:3*N])
        mu *= float(params['mu_growth'])

    def check():
        if time.monotonic() >= polish_end:
            raise Timeout

    def objective(v):
        check()
        g = np.zeros(3*N+1)
        g[2*N:3*N] = -1
        return -v[2*N:3*N].sum(), g

    def constraints(v):
        check()
        x, y, r, w = v[:N], v[N:2*N], v[2*N:3*N], v[-1]
        d = np.sqrt((x[I]-x[J])**2 + (y[I]-y[J])**2 + 1e-30)
        return np.r_[d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r]

    def jacobian(v):
        check()
        x, y = v[:N], v[N:2*N]
        dx, dy = x[I]-x[J], y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        a = np.zeros((M+4*N, 3*N+1))
        t = np.arange(M)
        a[t,I], a[t,J] = dx/d, -dx/d
        a[t,N+I], a[t,N+J] = dy/d, -dy/d
        a[t,2*N+I], a[t,2*N+J] = -1, -1
        a[M+K,K], a[M+K,2*N+K] = 1, -1
        a[M+N+K,K], a[M+N+K,2*N+K], a[M+N+K,-1] = -1, -1, 1
        a[M+2*N+K,N+K], a[M+2*N+K,2*N+K] = 1, -1
        a[M+3*N+K,N+K], a[M+3*N+K,2*N+K], a[M+3*N+K,-1] = -1, -1, -1
        return a

    def polish_callback(v):
        nonlocal current
        current = v.copy()
        check()
        consider(np.column_stack((v[:N], v[N:2*N])), v[2*N:3*N])

    if time.monotonic() < polish_end:
        try:
            res = minimize(objective, z, jac=True, method='SLSQP',
                           bounds=[(None,None)]*(2*N)+[(1e-9,None)]*N+[(0.05,1.95)],
                           constraints={'type':'ineq','fun':constraints,'jac':jacobian},
                           callback=polish_callback,
                           options={'maxiter':int(params['polish_iter']), 'ftol':1e-11})
            current = res.x
        except Timeout:
            pass
    consider(np.column_stack((current[:N], current[N:2*N])), current[2*N:3*N])

    # Retain the fixed-centre LP finishing step when enough reserve remains.
    if time.monotonic() < start + 0.60*budget_s:
        p = np.column_stack((current[:N], current[N:2*N]))
        a = np.zeros((M+4*N+1,N+4))
        b = np.zeros(M+4*N+1)
        a[np.arange(M),I] = 1
        a[np.arange(M),J] = 1
        b[:M] = np.sqrt(np.sum((p[I]-p[J])**2,axis=1))
        for k, (col, sign, coord) in enumerate(((N,1,0),(N+1,-1,0),(N+2,1,1),(N+3,-1,1))):
            if time.monotonic() >= final_end:
                return best
            t = M+k*N+K
            a[t,K] = 1
            a[t,col] = sign
            b[t] = sign*p[:,coord]
        a[-1,N:] = [-1,1,-1,1]
        b[-1] = 2
        res = linprog(-np.r_[np.ones(N),np.zeros(4)], A_ub=a, b_ub=b,
                      bounds=[(1e-10,None)]*N+[(None,None)]*4, method='highs',
                      options={'time_limit':max(0.01,final_end-time.monotonic())})
        if res.success:
            consider(p,res.x[:N])
    return best
