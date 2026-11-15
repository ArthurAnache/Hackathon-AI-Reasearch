import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linprog

PARAMS = {
    'aspect': (1.0, 0.5, 2.0, 'log'),
    'rows': (5, 3, 7),
    'jitter': (0.03, 0.0, 0.15),
    'init_jitter': (0.065, 0.0, 0.15),
    'mu0': (10.0, 1.0, 1000.0, 'log'),
    'mu_growth': (8.0, 2.0, 30.0),
    'stages': (6, 2, 12),
    'maxiter': (150, 30, 1000),
}
N = 21
I, J = np.triu_indices(N, 1)

class StopSearch(Exception):
    pass

def perimeter(p, r):
    xs = list(map(lambda v: Fraction(float(v)), p[:, 0]))
    ys = list(map(lambda v: Fraction(float(v)), p[:, 1]))
    rs = list(map(lambda v: Fraction(float(v)), r))
    return (max(x + q for x, q in zip(xs, rs)) - min(x - q for x, q in zip(xs, rs))
            + max(y + q for y, q in zip(ys, rs)) - min(y - q for y, q in zip(ys, rs)))

def score(p, r):
    if p.shape != (N, 2) or r.shape != (N,):
        return -np.inf
    if not np.isfinite(p).all() or not np.isfinite(r).all() or not (r > 0).all():
        return -np.inf
    d = np.sqrt((p[I, 0] - p[J, 0])**2 + (p[I, 1] - p[J, 1])**2)
    if np.any(d < r[I] + r[J]) or perimeter(p, r) > 2:
        return -np.inf
    return float(np.sum(r))

def output(p, r):
    return {'circles': np.column_stack((p, r)).tolist()}

def penalty(z, mu):
    x, y, r, w = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
    dx, dy = x[I] - x[J], y[I] - y[J]
    d = np.sqrt(dx*dx + dy*dy + 1e-18)
    v = np.maximum(r[I] + r[J] - d, 0)
    b0 = np.maximum(r - x, 0)
    b1 = np.maximum(r + x - w, 0)
    b2 = np.maximum(r - y, 0)
    b3 = np.maximum(r + y - (2-w), 0)
    neg = np.maximum(-r, 0)
    f = -r.sum() + mu*(v@v + b0@b0 + b1@b1 + b2@b2 + b3@b3 + neg@neg)
    c = 2*mu*v
    gx, gy, gr = np.zeros(N), np.zeros(N), -np.ones(N)
    np.add.at(gr, I, c)
    np.add.at(gr, J, c)
    np.add.at(gx, I, -c*dx/d)
    np.add.at(gx, J, c*dx/d)
    np.add.at(gy, I, -c*dy/d)
    np.add.at(gy, J, c*dy/d)
    gx += 2*mu*(-b0+b1)
    gy += 2*mu*(-b2+b3)
    gr += 2*mu*(b0+b1+b2+b3-neg)
    return f, np.r_[gx, gy, gr, 2*mu*(-b1.sum()+b3.sum())]

def lp_radii(p, deadline):
    remaining = deadline-time.monotonic()
    if remaining <= 0.015:
        return None
    m = len(I)
    a = np.zeros((m+4*N+1, N+4))
    b = np.zeros(m+4*N+1)
    a[np.arange(m), I] = 1
    a[np.arange(m), J] = 1
    b[:m] = np.sqrt((p[I, 0]-p[J, 0])**2 + (p[I, 1]-p[J, 1])**2)
    for k, (col, sign, coord) in enumerate(((N, 1, 0), (N+1, -1, 0), (N+2, 1, 1), (N+3, -1, 1))):
        if time.monotonic() >= deadline:
            return None
        rows = m+k*N+np.arange(N)
        a[rows, np.arange(N)] = 1
        a[rows, col] = sign
        b[rows] = sign*p[:, coord]
    a[-1, N:] = [-1, 1, -1, 1]
    b[-1] = 2
    result = linprog(-np.r_[np.ones(N), np.zeros(4)], A_ub=a, b_ub=b,
                     bounds=[(0, None)]*N+[(None, None)]*4, method='highs',
                     options={'time_limit': max(0.001, deadline-time.monotonic()-0.01)})
    return result.x[:N] if result.success else None

def repair(p, r, deadline):
    r = np.maximum(np.asarray(r), 1e-12).copy()*(1-1e-12)
    d = np.sqrt((p[I, 0]-p[J, 0])**2 + (p[I, 1]-p[J, 1])**2)
    for _ in range(5):
        if time.monotonic() >= deadline:
            return None
        if not np.any(d < r[I]+r[J]):
            break
        for i, j, dd in zip(I, J, d):
            if time.monotonic() >= deadline:
                return None
            if r[i]+r[j] > dd*(1-1e-12):
                s = dd*(1-1e-12)/(r[i]+r[j])
                r[i] *= s
                r[j] *= s
    for _ in range(60):
        if time.monotonic() >= deadline:
            return None
        excess = perimeter(p, r)-2
        if excess <= 0:
            return r
        r *= 1-max(2*float(excess)/float(r.sum()), 1e-12)
    return r

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start+0.70*budget_s
    descent_end = start+0.53*budget_s
    rng = np.random.default_rng(seed)
    fallback_p = np.column_stack(((np.arange(N)%7+0.5)/7, (np.arange(N)//7+0.5)/3))
    fallback_r = np.full(N, 0.04)
    best = output(fallback_p, fallback_r)
    best_score = score(fallback_p, fallback_r)
    original_p = None
    if init is not None:
        arr = np.asarray(init['circles'], dtype=float)
        p0, r0 = arr[:, :2].copy(), arr[:, 2].copy()
        initial_score = score(p0, r0)
        if initial_score >= best_score:
            best, best_score = init, initial_score
        lo = (p0-np.abs(r0)[:, None]).min(axis=0)
        side = (p0+np.abs(r0)[:, None]).max(axis=0)-lo
        scale = 2/side.sum()
        p = (p0-lo)*scale
        original_p = p.copy()
        r = np.abs(r0)*scale
        w = side[0]*scale
        p += rng.normal(0, float(params['init_jitter']), p.shape)
    else:
        w = 2*params['aspect']/(1+params['aspect'])
        rows = int(params['rows'])
        per_row = int(np.ceil(N/rows))
        idx = np.arange(N)
        p = np.column_stack(((idx%per_row+0.5)/per_row + ((idx//per_row)%2)*0.25/per_row,
                             (idx//per_row+0.5)/rows))
        p = np.clip(p+rng.normal(0, params['jitter'], p.shape), 0.02, 0.98)*[w, 2-w]
        r = np.full(N, 0.05)
    z = np.r_[p[:, 0], p[:, 1], r, w]
    latest = z.copy()
    mu = float(params['mu0'])
    def objective(v):
        if time.monotonic() >= descent_end:
            raise StopSearch()
        return penalty(v, mu)
    def callback(v):
        nonlocal latest
        latest = v.copy()
        if time.monotonic() >= descent_end:
            raise StopSearch()
    for _ in range(int(params['stages'])):
        if time.monotonic() >= descent_end:
            break
        try:
            result = minimize(objective, z, jac=True, method='L-BFGS-B', callback=callback,
                              options={'maxiter': int(params['maxiter'])})
            z = result.x
            latest = z.copy()
        except StopSearch:
            z = latest
            break
        mu *= float(params['mu_growth'])
    candidates = [np.column_stack((z[:N], z[N:2*N]))]
    if original_p is not None:
        candidates.append(original_p)
    for pc in candidates:
        if time.monotonic() >= deadline-0.02:
            break
        rc = lp_radii(pc, deadline-0.008)
        if rc is None:
            continue
        rc = repair(pc, rc, deadline)
        if rc is None:
            continue
        value = score(pc, rc)
        if value > best_score:
            best, best_score = output(pc, rc), value
    return best
