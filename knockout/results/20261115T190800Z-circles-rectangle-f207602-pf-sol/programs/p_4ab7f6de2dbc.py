import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigh

PARAMS = {
    'amplitude': (0.065, 0.005, 0.18),
    'prestress': (0.035, 0.001, 0.2),
    'modes': (10, 3, 24),
    'iterations': (65, 15, 150),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class StopSearch(Exception):
    pass

def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    x, y, r = ([Fraction(float(v)) for v in a[:, k]] for k in range(3))
    perimeter = (max(u + v for u, v in zip(x, r)) - min(u - v for u, v in zip(x, r))
                 + max(u + v for u, v in zip(y, r)) - min(u - v for u, v in zip(y, r)))
    if perimeter > 2:
        return -np.inf
    d = np.sqrt((a[I, 0] - a[J, 0]) ** 2 + (a[I, 1] - a[J, 1]) ** 2)
    if np.any(d < a[I, 2] + a[J, 2]):
        return -np.inf
    return float(np.sum(a[:, 2]))

def unpack(z):
    return z[:N], z[N:2*N], z[2*N:3*N], z[-1]

def constraints(z):
    x, y, r, w = unpack(z)
    dx, dy = x[I] - x[J], y[I] - y[J]
    d = np.sqrt(dx*dx + dy*dy + 1e-30)
    return np.concatenate((d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r))

def jacobian(z):
    x, y, r, w = unpack(z)
    dx, dy = x[I]-x[J], y[I]-y[J]
    d = np.sqrt(dx*dx+dy*dy+1e-30)
    a = np.zeros((M+4*N, 3*N+1))
    rows = np.arange(M)
    a[rows, I], a[rows, J] = dx/d, -dx/d
    a[rows, N+I], a[rows, N+J] = dy/d, -dy/d
    a[rows, 2*N+I], a[rows, 2*N+J] = -1, -1
    k = np.arange(N)
    a[M+k, k] = 1
    a[M+N+k, k] = -1
    a[M+N+k, -1] = 1
    a[M+2*N+k, N+k] = 1
    a[M+3*N+k, N+k] = -1
    a[M+3*N+k, -1] = -1
    for t in range(4):
        a[M+t*N+k, 2*N+k] = -1
    return a

def repaired(z):
    x, y, r, w = unpack(z)
    p = np.column_stack((x, y)).copy()
    r = r.copy()
    if not np.isfinite(z).all() or np.any(r <= 0):
        return None
    d = np.sqrt((x[I]-x[J])**2+(y[I]-y[J])**2)
    factor = min(1.0, float(np.min(d/(r[I]+r[J]))))
    r *= factor*(1-2e-14)
    lo = np.min(p-r[:, None], axis=0)
    hi = np.max(p+r[:, None], axis=0)
    s = (2-8e-14)/float(np.sum(hi-lo))
    p = (p-lo)*s
    r *= s
    return np.column_stack((p, r))

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a = np.array([[(i%7+0.5)*0.14, (i//7+0.5)*0.14, 0.069] for i in range(N)])
        best = {'circles': a.tolist()}
    else:
        best = init
        a = np.asarray(init['circles'], dtype=float).copy()
    best_score = exact_score(a)
    lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
    hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
    scale = 2/np.sum(hi-lo)
    p = (a[:, :2]-lo)*scale
    z0 = np.r_[p[:, 0], p[:, 1], a[:, 2]*scale, (hi[0]-lo[0])*scale]
    grad = np.r_[np.zeros(2*N), -np.ones(N), 0.0]

    def consider(z):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        b = repaired(z)
        if b is not None and float(np.sum(b[:, 2])) > best_score:
            score = exact_score(b)
            if score > best_score:
                best_score = score
                best = {'circles': b.tolist()}

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def fun(z):
        check()
        return -float(np.sum(z[2*N:3*N]))

    def con(z):
        check()
        return constraints(z)

    active = np.flatnonzero(constraints(z0) < 2e-6)
    A = jacobian(z0)[active]
    # Approximate nonnegative equilibrium contact forces.
    forces = np.maximum(np.linalg.lstsq(A.T, grad, rcond=1e-9)[0], 0.003)
    metric = np.r_[np.ones(2*N), np.full(N, 0.55), 1.0]
    stiffness = A.T @ (forces[:, None]*A)
    H = np.zeros_like(stiffness)
    x, y, r, w = unpack(z0)
    for row, force in zip(active, forces):
        if time.monotonic() >= deadline:
            return best
        if row >= M:
            continue
        i, j = int(I[row]), int(J[row])
        v = np.array([x[i]-x[j], y[i]-y[j]])
        d = np.linalg.norm(v)
        if d < 1e-12:
            continue
        block = -(force/d)*(np.eye(2)-np.outer(v, v)/(d*d))
        ids_i, ids_j = [i, N+i], [j, N+j]
        H[np.ix_(ids_i, ids_i)] += block
        H[np.ix_(ids_j, ids_j)] += block
        H[np.ix_(ids_i, ids_j)] -= block
        H[np.ix_(ids_j, ids_i)] -= block
    K = metric[:, None]*(stiffness+float(params['prestress'])*H)*metric[None, :]
    if time.monotonic() >= deadline:
        return best
    _, vectors = eigh(K, check_finite=False, subset_by_index=(0, int(params['modes'])-1))
    modes = metric[:, None]*vectors
    bounds = [(None, None)]*(2*N)+[(1e-5, 0.8)]*N+[(0.2, 1.8)]
    trial = 0
    while time.monotonic() < deadline:
        if trial == 0:
            direction = modes[:, 0].copy()
        else:
            c = rng.normal(size=modes.shape[1])
            c *= np.exp(-np.arange(len(c))/4)
            direction = modes@c
        direction /= max(np.sqrt(np.mean(direction[:2*N]**2)), 1e-10)
        sign = 1 if trial % 2 == 0 else -1
        amplitude = float(params['amplitude'])*(0.65+0.35*(trial % 3))
        z = z0 + sign*amplitude*direction
        z[2*N:3*N] = np.maximum(z[2*N:3*N], 0.005)
        z[-1] = np.clip(z[-1], 0.3, 1.7)
        last = [z]
        def callback(v):
            check()
            last[0] = v.copy()
        try:
            result = minimize(fun, z, jac=lambda v: grad, method='SLSQP',
                              bounds=bounds,
                              constraints={'type': 'ineq', 'fun': con, 'jac': jacobian},
                              callback=callback,
                              options={'maxiter': int(params['iterations']), 'ftol': 2e-12, 'disp': False})
            consider(result.x)
        except StopSearch:
            break
        trial += 1
    return best
