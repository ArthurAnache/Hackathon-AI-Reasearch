import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'opening': (0.045, 0.01, 0.12),
    'groups': (3, 2, 3),
    'slice': (0.26, 0.12, 0.45),
    'continuation_steps': (4, 2, 8),
    'active_tol': (2e-7, 1e-9, 1e-5, 'log')
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
K = M + 4*N
IDS = np.arange(N)
OBJ = np.r_[np.zeros(42), -np.ones(21), 0.0]

class Deadline(Exception):
    pass


def score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return -np.inf
    x = list(map(lambda v: Fraction(float(v)), a[:, 0]))
    y = list(map(lambda v: Fraction(float(v)), a[:, 1]))
    r = list(map(lambda v: Fraction(float(v)), a[:, 2]))
    w = max(u+v for u, v in zip(x, r))-min(u-v for u, v in zip(x, r))
    h = max(u+v for u, v in zip(y, r))-min(u-v for u, v in zip(y, r))
    return float(np.sum(a[:, 2])) if w+h <= 2 else -np.inf


def geometry(z, jac=False):
    x, y, r, w = z[:21], z[21:42], z[42:63], z[63]
    dx, dy = x[I]-x[J], y[I]-y[J]
    d = np.sqrt(dx*dx+dy*dy)
    c = np.r_[d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r]
    if not jac:
        return c
    a = np.zeros((K, 64))
    rows = np.arange(M)
    d = np.maximum(d, 1e-100)
    a[rows, I], a[rows, J] = dx/d, -dx/d
    a[rows, 21+I], a[rows, 21+J] = dy/d, -dy/d
    a[rows, 42+I] = -1
    a[rows, 42+J] = -1
    a[M+IDS, IDS] = 1
    a[M+N+IDS, IDS] = -1
    a[M+2*N+IDS, N+IDS] = 1
    a[M+3*N+IDS, N+IDS] = -1
    a[M+IDS, 42+IDS] = -1
    a[M+N+IDS, 42+IDS] = -1
    a[M+2*N+IDS, 42+IDS] = -1
    a[M+3*N+IDS, 42+IDS] = -1
    a[M+N+IDS, 63] = 1
    a[M+3*N+IDS, 63] = -1
    return c, a


def pack(z):
    p = np.column_stack((z[:21], z[21:42]))
    r = z[42:63].copy()
    if not np.isfinite(z).all() or np.any(r <= 0):
        return None
    d = np.sqrt((p[I, 0]-p[J, 0])**2+(p[I, 1]-p[J, 1])**2)
    factor = min(1.0, float(np.min(d/(r[I]+r[J]))))
    if factor <= 0:
        return None
    r *= factor*(1-3e-14)
    lo = np.min(p-r[:, None], axis=0)
    hi = np.max(p+r[:, None], axis=0)
    scale = (2-8e-14)/float(np.sum(hi-lo))
    return np.column_stack(((p-lo)*scale, r*scale))


def encode(a):
    p, r = a[:, :2].copy(), a[:, 2].copy()
    lo = np.min(p-r[:, None], axis=0)
    sides = np.max(p+r[:, None], axis=0)-lo
    fac = 2/float(sides.sum())
    p = (p-lo)*fac
    return np.r_[p[:, 0], p[:, 1], r*fac, sides[0]*fac]


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start+0.70*budget_s
    rng = np.random.default_rng(seed)
    fallback = np.column_stack(((IDS % 7)*0.2, (IDS // 7)*0.2, np.full(N, 0.1-1e-13)))
    best = init if init is not None else {'circles': fallback.tolist()}
    best_score = score(np.asarray(best['circles'], dtype=float))
    base = encode(np.asarray(best['circles'], dtype=float))

    def consider(z):
        nonlocal best, best_score
        if time.monotonic() >= end:
            return
        a = pack(z)
        if a is None:
            return
        s = score(a)
        if s > best_score:
            best_score = s
            best = {'circles': a.tolist()}

    bounds = [(None, None)]*42+[(1e-5, 0.45)]*21+[(0.35, 1.65)]

    def polish(z, seconds):
        stop = min(end-0.012, time.monotonic()+seconds)
        last = [z.copy()]
        def check():
            if time.monotonic() >= stop:
                raise Deadline()
        def objective(v):
            check()
            return -float(np.sum(v[42:63]))
        def grad(v):
            check()
            return OBJ
        def fun(v):
            check()
            return geometry(v)
        def jac(v):
            check()
            return geometry(v, True)[1]
        def callback(v):
            check()
            last[0] = v.copy()
            if -OBJ@v > best_score-1e-8:
                consider(v)
        try:
            res = minimize(objective, z, jac=grad, method='SLSQP', bounds=bounds,
                           constraints={'type': 'ineq', 'fun': fun, 'jac': jac},
                           callback=callback,
                           options={'maxiter': 90, 'ftol': 2e-13, 'disp': False})
            last[0] = res.x
        except Deadline:
            pass
        consider(last[0])
        return last[0]

    polish(base, min(0.16, 0.12*budget_s))
    if time.monotonic() >= end-0.04:
        return best
    base = encode(np.asarray(best['circles'], dtype=float))
    c0, a0 = geometry(base, True)
    active = np.flatnonzero(c0 < float(params['active_tol']))
    if len(active) < 10:
        polish(base, end-time.monotonic()-0.02)
        return best

    # Determine which rectangle midline is the incumbent's reflection axis.
    p = np.column_stack((base[:21], base[21:42]))
    reflected_y = p.copy()
    reflected_y[:, 1] = 2-base[63]-reflected_y[:, 1]
    reflected_x = p.copy()
    reflected_x[:, 0] = base[63]-reflected_x[:, 0]
    dy = np.sum((reflected_y[:, None, :]-p[None, :, :])**2, axis=2)
    dx = np.sum((reflected_x[:, None, :]-p[None, :, :])**2, axis=2)
    horizontal = np.min(dy, axis=1).sum() <= np.min(dx, axis=1).sum()
    mirror = np.argmin(dy if horizontal else dx, axis=1)
    lookup = np.full((N, N), -1, dtype=int)
    lookup[I, J] = np.arange(M)
    lookup[J, I] = np.arange(M)
    contact_map = lookup[mirror[I], mirror[J]]
    walls = [0, 1, 3, 2] if horizontal else [1, 0, 2, 3]
    reflected_rows = np.r_[contact_map,
                           M+walls[0]*N+mirror, M+walls[1]*N+mirror,
                           M+walls[2]*N+mirror, M+walls[3]*N+mirror]
    groups = []
    used = set()
    aset = set(active.tolist())
    for row in active:
        if time.monotonic() >= end-0.03:
            return best
        row = int(row)
        if row in used:
            continue
        mate = int(reflected_rows[row])
        group = [row]
        if mate in aset and mate != row:
            group.append(mate)
        groups.append(group)
        used.update(group)

    # Low-cost contact groups are preferred, without excluding stiff releases.
    multipliers = np.linalg.lstsq(a0[active].T, OBJ, rcond=1e-9)[0]
    costs = np.zeros(K)
    costs[active] = np.abs(multipliers)
    weights = np.array([1/(0.15+float(np.sum(costs[g]))) for g in groups])
    weights /= weights.sum()
    attempt = 0
    while time.monotonic() < end-0.065:
        count = min(int(params['groups']), len(groups))
        selected = rng.choice(len(groups), size=count, replace=False, p=weights)
        release = np.concatenate([np.asarray(groups[k], dtype=int) for k in selected])
        target = np.zeros(K)
        opening = float(params['opening'])*(0.75+0.5*(attempt % 3))
        target[release] = opening
        z = base.copy()
        steps = int(params['continuation_steps'])
        for step in range(steps):
            if time.monotonic() >= end-0.045:
                break
            c, a = geometry(z, True)
            desired = target[active]*(step+1)/steps
            delta = np.linalg.lstsq(a[active], desired-c[active], rcond=1e-8)[0]
            length = float(np.max(np.abs(delta)))
            if length > 0.075:
                delta *= 0.075/length
            z += delta
            z[42:63] = np.maximum(z[42:63], 0.015)
            z[63] = np.clip(z[63], 0.4, 1.6)
        if time.monotonic() >= end-0.035:
            break
        consider(z)
        polish(z, float(params['slice']))
        attempt += 1
    return best
