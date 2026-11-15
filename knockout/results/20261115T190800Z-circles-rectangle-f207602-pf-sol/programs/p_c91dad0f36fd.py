import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'samples': (2200, 100, 6000),
    'angles': (24, 8, 48),
    'restarts': (5, 1, 12),
    'iterations': (110, 20, 300),
}
N = 21
I, J = np.triu_indices(N, 1)

class StopSearch(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    dx = a[I, 0] - a[J, 0]
    dy = a[I, 1] - a[J, 1]
    if np.any(np.sqrt(dx * dx + dy * dy) < a[I, 2] + a[J, 2]):
        return -np.inf
    x = [Fraction(float(t)) for t in a[:, 0]]
    y = [Fraction(float(t)) for t in a[:, 1]]
    r = [Fraction(float(t)) for t in a[:, 2]]
    perimeter = (max(u + v for u, v in zip(x, r)) - min(u - v for u, v in zip(x, r))
                 + max(u + v for u, v in zip(y, r)) - min(u - v for u, v in zip(y, r)))
    return float(np.sum(a[:, 2])) if perimeter <= 2 else -np.inf

def normalize(a):
    a = a.copy()
    lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
    hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
    scale = 2.0 / np.sum(hi - lo)
    a[:, :2] = (a[:, :2] - lo) * scale
    a[:, 2] *= scale
    return a

def repair(a):
    if not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
        return None
    a = a.copy()
    d = np.sqrt((a[I, 0] - a[J, 0]) ** 2 + (a[I, 1] - a[J, 1]) ** 2)
    ratio = np.min(d / (a[I, 2] + a[J, 2]))
    a[:, 2] *= min(1.0, float(ratio)) * (1 - 3e-14)
    a = normalize(a)
    a[:, :2] *= 1 - 3e-14
    a[:, 2] *= 1 - 6e-14
    return a

def pack(a):
    a = normalize(a)
    w = np.max(a[:, 0] + a[:, 2])
    return np.r_[a[:, 0], a[:, 1], a[:, 2], w]

def unpack(z):
    return np.column_stack((z[:N], z[N:2*N], z[2*N:3*N]))

def constraints(z):
    x, y, r, w = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
    d = np.sqrt((x[I] - x[J]) ** 2 + (y[I] - y[J]) ** 2 + 1e-30)
    return np.r_[d - r[I] - r[J], x-r, w-x-r, y-r, 2-w-y-r]

def jacobian(z):
    x, y = z[:N], z[N:2*N]
    dx, dy = x[I] - x[J], y[I] - y[J]
    d = np.sqrt(dx*dx + dy*dy + 1e-30)
    m = len(I)
    A = np.zeros((m+4*N, 3*N+1))
    k = np.arange(m)
    A[k, I], A[k, J] = dx/d, -dx/d
    A[k, N+I], A[k, N+J] = dy/d, -dy/d
    A[k, 2*N+I] = -1
    A[k, 2*N+J] = -1
    q = np.arange(N)
    A[m+q, q] = 1
    A[m+N+q, q] = -1
    A[m+N+q, -1] = 1
    A[m+2*N+q, N+q] = 1
    A[m+3*N+q, N+q] = -1
    A[m+3*N+q, -1] = -1
    for b in range(4):
        A[m+b*N+q, 2*N+q] = -1
    return A

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    if init is not None:
        original = np.asarray(init['circles'], dtype=float)
        best, best_score = init, exact_score(original)
    else:
        original = np.array([[(k % 5 + .5)/5, (k // 5 + .5)/5, .07] for k in range(N)])
        original = normalize(original)
        best = {'circles': original.tolist()}
        best_score = exact_score(original)
    a = normalize(original)
    center = (np.max(a[:, :2]+a[:, 2,None], axis=0) + np.min(a[:, :2]-a[:, 2,None], axis=0))/2
    a[:, :2] -= center
    p, r = a[:, :2], a[:, 2]
    lo, hi = np.min(p-r[:,None], axis=0), np.max(p+r[:,None], axis=0)
    count = int(params['samples'])
    poles = rng.uniform(lo, hi, (count, 2))
    # Add poles near contact interstices without assuming a particular symmetry.
    near = np.argsort(np.linalg.norm(p[I]-p[J], axis=1)-r[I]-r[J])[:65]
    mids = (p[I[near]]*r[J[near],None] + p[J[near]]*r[I[near],None])/(r[I[near]]+r[J[near]])[:,None]
    extra = mids[:,None,:] + rng.normal(0, .035, (len(mids), 5, 2))
    poles = np.concatenate((poles, extra.reshape(-1,2)))
    angles = np.arange(int(params['angles'])) * (np.pi/2/int(params['angles']))
    co, si = np.cos(angles), np.sin(angles)
    pool = []
    for offset in range(0, len(poles), 128):
        if time.monotonic() > start + .20*budget_s:
            break
        pp = poles[offset:offset+128]
        v = p[None,:,:] - pp[:,None,:]
        den = np.sum(v*v, axis=2) - r[None,:]**2
        valid = np.all(den > 1e-8, axis=1)
        if not np.any(valid):
            continue
        pp, v, den = pp[valid], v[valid], den[valid]
        c, rr = v/den[:,:,None], r[None,:]/den
        xx = c[:,None,:,0]*co[None,:,None] - c[:,None,:,1]*si[None,:,None]
        yy = c[:,None,:,0]*si[None,:,None] + c[:,None,:,1]*co[None,:,None]
        widths = np.max(xx+rr[:,None,:],axis=2)-np.min(xx-rr[:,None,:],axis=2)
        heights = np.max(yy+rr[:,None,:],axis=2)-np.min(yy-rr[:,None,:],axis=2)
        sums = widths+heights
        ai = np.argmin(sums,axis=1)
        scores = 2*np.sum(rr,axis=1)/sums[np.arange(len(rr)),ai]
        for k in np.argsort(scores)[-10:]:
            if time.monotonic() > deadline:
                break
            q = int(ai[k])
            candidate = np.column_stack((xx[k,q], yy[k,q], rr[k]))
            pool.append((float(scores[k]), pp[k].copy(), normalize(candidate)))
    pool.sort(key=lambda t: -t[0])
    selected = []
    for item in pool:
        if time.monotonic() > deadline:
            break
        if all(np.linalg.norm(item[1]-prev[1]) > .045 for prev in selected):
            selected.append(item)
        if len(selected) >= int(params['restarts']):
            break
    gradient = np.r_[np.zeros(2*N), -np.ones(N), 0.0]
    def objective(z):
        if time.monotonic() > deadline:
            raise StopSearch()
        return -float(np.sum(z[2*N:3*N]))
    def consider(candidate):
        nonlocal best, best_score
        b = repair(candidate)
        if b is None:
            return
        value = exact_score(b)
        if value > best_score:
            best_score = value
            best = {'circles': b.tolist()}
    for _, _, candidate in selected:
        if time.monotonic() > deadline - .03:
            break
        consider(candidate)
        last = [pack(candidate)]
        def callback(z):
            last[0] = z.copy()
            if time.monotonic() > deadline:
                raise StopSearch()
        try:
            result = minimize(objective, last[0], jac=lambda z: gradient, method='SLSQP',
                              bounds=[(None,None)]*(2*N)+[(1e-8,None)]*N+[(.03,1.97)],
                              constraints={'type':'ineq','fun':constraints,'jac':jacobian},
                              callback=callback,
                              options={'maxiter':int(params['iterations']), 'ftol':2e-11, 'disp':False})
            last[0] = result.x
        except StopSearch:
            pass
        consider(unpack(last[0]))
    return best
