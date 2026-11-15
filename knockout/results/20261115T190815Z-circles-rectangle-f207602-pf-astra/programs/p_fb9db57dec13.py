import time
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog

PARAMS = {
    'contact_tol': (2e-7, 1e-9, 1e-4, 'log'),
    'perturbation': (0.045, 0.002, 0.2, 'log'),
    'steps': (14, 4, 40),
    'orbits': (3, 2, 7),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)


def perimeter(a):
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    return (max(u+v for u, v in zip(x, r)) - min(u-v for u, v in zip(x, r))
            + max(u+v for u, v in zip(y, r)) - min(u-v for u, v in zip(y, r)))


def score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]) or perimeter(a) > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))


def unpack(z):
    return np.column_stack((z[:N], z[N:2*N], z[2*N:3*N]))


def canonical(a):
    lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
    hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
    scale = 2.0 / np.sum(hi-lo)
    p = (a[:, :2]-lo)*scale
    return np.r_[p[:, 0], p[:, 1], a[:, 2]*scale, (hi[0]-lo[0])*scale]


def repair(z, deadline):
    a = unpack(z).copy()
    if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return None
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    fac = min(1.0, float(np.min(d/(a[I, 2]+a[J, 2]))))
    a[:, 2] *= fac*(1.0-3e-15)
    for _ in range(3):
        if time.monotonic() >= deadline:
            return None
        per = perimeter(a)
        if per > 2:
            a *= (2.0/float(per))*(1.0-3e-15)
        if score(a) > -np.inf:
            return a
        a[:, 2] *= 1.0-2e-14
    return None


def geometry(z):
    x, y, r, w = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
    dx, dy = x[I]-x[J], y[I]-y[J]
    d = np.sqrt(dx*dx+dy*dy)
    d = np.maximum(d, 1e-30)
    g = np.r_[d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r]
    a = np.zeros((M+4*N, 64))
    q = np.arange(M)
    a[q, I] = dx/d
    a[q, J] = -dx/d
    a[q, N+I] = dy/d
    a[q, N+J] = -dy/d
    a[q, 2*N+I] = -1
    a[q, 2*N+J] = -1
    k = np.arange(N)
    a[M+k, k] = 1
    a[M+N+k, k] = -1
    a[M+N+k, -1] = 1
    a[M+2*N+k, N+k] = 1
    a[M+3*N+k, N+k] = -1
    a[M+3*N+k, -1] = -1
    a[M+k, 2*N+k] = -1
    a[M+N+k, 2*N+k] = -1
    a[M+2*N+k, 2*N+k] = -1
    a[M+3*N+k, 2*N+k] = -1
    return g, a


def polish(z, tol, deadline):
    z = z.copy()
    g, a = geometry(z)
    active = np.flatnonzero(g < tol)
    if len(active) < 64:
        return z
    for _ in range(7):
        if time.monotonic() >= deadline:
            break
        g, a = geometry(z)
        err = g[active]
        if np.max(np.abs(err)) < 2e-16:
            break
        step = np.linalg.lstsq(a[active], -err, rcond=1e-11)[0]
        if np.max(np.abs(step)) > 0.02:
            break
        z += step
    return z


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a = np.array([[(i % 7+0.5)*0.2, (i//7+0.5)*0.2, 0.1] for i in range(N)])
        a *= 1.0-1e-13
        best = {'circles': a.tolist()}
    else:
        best = init
        a = np.asarray(init['circles'], dtype=float)
    best_score = score(a)
    base = canonical(a)

    def consider(z):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        candidate = repair(z, deadline)
        if candidate is not None:
            value = score(candidate)
            if value > best_score:
                best_score = value
                best = {'circles': candidate.tolist()}

    tol = float(params['contact_tol'])
    polished = polish(base, tol, deadline)
    consider(polished)
    if np.min(geometry(polished)[0]) > -1e-8:
        base = polished

    objective = np.r_[np.zeros(2*N), -np.ones(N), 0.0]
    bounds = [(0.0, 2.0)]*(2*N) + [(1e-8, 0.5)]*N + [(0.05, 1.95)]
    # Identify approximate reflection orbits in the incumbent.
    pts = unpack(base)[:, :2]
    h = 2-base[-1]
    reflected = pts.copy()
    reflected[:, 1] = h-reflected[:, 1]
    mirror = np.argmin(np.sum((reflected[:, None, :]-pts[None, :, :])**2, axis=2), axis=1)
    used = set()
    orbits = []
    for i in range(N):
        if time.monotonic() >= deadline:
            return best
        if i not in used:
            j = int(mirror[i])
            orbit = [i] if j == i or j in used else [i, j]
            used.update(orbit)
            orbits.append(orbit)

    restart = 0
    while time.monotonic() < deadline-0.025:
        z = base.copy()
        if restart:
            selected = rng.choice(len(orbits), min(int(params['orbits']), len(orbits)), replace=False)
            amp = float(params['perturbation'])*(0.5, 1.0, 2.0, 3.0)[(restart-1) % 4]
            for index in selected:
                if time.monotonic() >= deadline:
                    return best
                orbit = orbits[int(index)]
                dx, dy = rng.normal(0.0, amp, 2)
                z[orbit[0]] += dx
                if len(orbit) == 2:
                    z[N+orbit[0]] += dy
                    z[orbit[1]] += dx
                    z[N+orbit[1]] -= dy
        previous = -np.inf
        for iteration in range(int(params['steps'])):
            remaining = deadline-time.monotonic()
            if remaining < 0.025:
                break
            g, jac = geometry(z)
            # g + jac @ (new-z) >= 0 is sufficient for true nonoverlap.
            rhs = g-jac@z
            result = linprog(objective, A_ub=-jac, b_ub=rhs, bounds=bounds,
                             method='highs-ds', options={
                                 'primal_feasibility_tolerance': 1e-9,
                                 'dual_feasibility_tolerance': 1e-9,
                                 'time_limit': max(0.001, min(0.08, remaining-0.015)),
                             })
            if not result.success or time.monotonic() >= deadline:
                break
            z = result.x
            value = float(np.sum(z[2*N:3*N]))
            if value > best_score-2e-7:
                consider(z)
                if time.monotonic() < deadline-0.015:
                    consider(polish(z, tol, deadline))
            if abs(value-previous) < 2e-12:
                break
            previous = value
        restart += 1
    return best
