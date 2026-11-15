import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'power': (2.15, 1.2, 3.5),
    'excursion_fraction': (0.22, 0.08, 0.35),
    'radius_floor': (0.012, 0.001, 0.04),
    'symmetry_break': (0.012, 0.0, 0.1),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
IDS = np.arange(N)

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.69 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf

    def exact(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -np.inf
        d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
        if np.any(d < a[I, 2]+a[J, 2]):
            return -np.inf
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        wh = (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
              +max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)))
        if wh > 2:
            return -np.inf
        return float(np.sum(a[:, 2]))

    if init is not None:
        a = np.array(init['circles'], dtype=float)
        best_score = exact(a)
    else:
        # A deterministic rectangular cold start, used only when no warm start exists.
        a = np.array([[(k % 7 + 0.5)/7, (k // 7 + 0.5)/3, 0.06] for k in range(N)])
        best = {'circles': a.tolist()}
        best_score = exact(a)

    low = np.min(a[:, :2]-a[:, 2, None], axis=0)
    sides = np.max(a[:, :2]+a[:, 2, None], axis=0)-low
    scale = 2.0/np.sum(sides)
    p = (a[:, :2]-low)*scale
    r = a[:, 2]*scale
    z = np.r_[p[:, 0], p[:, 1], r, sides[0]*scale]

    def unpack(v):
        return np.column_stack((v[:N], v[N:2*N], v[2*N:3*N]))

    def retain(v):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        b = unpack(v).copy()
        if not np.isfinite(b).all() or np.min(b[:, 2]) <= 0:
            return
        dx = b[I, 0]-b[J, 0]
        dy = b[I, 1]-b[J, 1]
        d = np.sqrt(dx*dx+dy*dy)
        fac = min(1.0, float(np.min(d/(b[I, 2]+b[J, 2]))))
        b[:, 2] *= fac*(1.0-3e-15)
        lo = np.min(b[:, :2]-b[:, 2, None], axis=0)
        hi = np.max(b[:, :2]+b[:, 2, None], axis=0)
        s = 2.0/float(np.sum(hi-lo))*(1.0-3e-15)
        b[:, :2] = (b[:, :2]-lo)*s
        b[:, 2] *= s
        if float(np.sum(b[:, 2])) <= best_score:
            return
        score = exact(b)
        if score > best_score:
            best_score = score
            best = {'circles': b.tolist()}

    # Squared distance inequalities have inexpensive dense analytic derivatives.
    wall_jac = np.zeros((4*N, 3*N+1))
    wall_jac[IDS, IDS] = 1
    wall_jac[IDS, 2*N+IDS] = -1
    wall_jac[N+IDS, IDS] = -1
    wall_jac[N+IDS, 2*N+IDS] = -1
    wall_jac[N+IDS, -1] = 1
    wall_jac[2*N+IDS, N+IDS] = 1
    wall_jac[2*N+IDS, 2*N+IDS] = -1
    wall_jac[3*N+IDS, N+IDS] = -1
    wall_jac[3*N+IDS, 2*N+IDS] = -1
    wall_jac[3*N+IDS, -1] = -1
    row = np.arange(M)
    cache_x = None
    cache_c = None
    cache_j = None
    phase_end = deadline

    def constraints(v):
        nonlocal cache_x, cache_c, cache_j
        if time.monotonic() >= phase_end:
            raise StopSearch()
        if cache_x is not None and np.array_equal(v, cache_x):
            return cache_c
        x, y, q, w = v[:N], v[N:2*N], v[2*N:3*N], v[-1]
        dx, dy = x[I]-x[J], y[I]-y[J]
        rr = q[I]+q[J]
        # Scaling keeps contact and wall residuals comparable near tangency.
        c = np.r_[4.0*(dx*dx+dy*dy-rr*rr), x-q, w-x-q, y-q, 2-w-y-q]
        jac = np.zeros((M+4*N, 3*N+1))
        jac[row, I] = 8*dx
        jac[row, J] = -8*dx
        jac[row, N+I] = 8*dy
        jac[row, N+J] = -8*dy
        jac[row, 2*N+I] = -8*rr
        jac[row, 2*N+J] = -8*rr
        jac[M:] = wall_jac
        cache_x, cache_c, cache_j = v.copy(), c, jac
        return c

    def jacobian(v):
        constraints(v)
        return cache_j

    latest = z.copy()
    ticks = 0
    def callback(v):
        nonlocal latest, ticks
        latest = v.copy()
        ticks += 1
        if time.monotonic() >= phase_end:
            raise StopSearch()
        if ticks % 5 == 0:
            retain(v)

    weights = 1.0 + float(params['symmetry_break'])*rng.normal(size=N)
    bounds = [(0.0, 2.0)]*(2*N) + [(float(params['radius_floor']), 0.5)]*N + [(0.3, 1.7)]
    # The first two phases follow a radius-dispersion branch; the last releases it.
    phases = [(float(params['power']), start+float(params['excursion_fraction'])*budget_s),
              (1.35, start+(float(params['excursion_fraction'])+0.10)*budget_s),
              (1.0, deadline)]
    for power, end in phases:
        if time.monotonic() >= deadline:
            break
        phase_end = min(end, deadline)
        if time.monotonic() >= phase_end:
            continue
        active_weights = weights if power > 1.01 else np.ones(N)
        def objective(v):
            if time.monotonic() >= phase_end:
                raise StopSearch()
            rr = np.maximum(v[2*N:3*N], 1e-12)
            t = rr/0.11
            value = -0.11*np.sum(active_weights*t**power)/power
            grad = np.zeros(3*N+1)
            grad[2*N:3*N] = -active_weights*t**(power-1)
            return value, grad
        latest = z.copy()
        try:
            result = minimize(objective, z, jac=True, method='SLSQP', bounds=bounds,
                              constraints={'type':'ineq', 'fun':constraints, 'jac':jacobian},
                              callback=callback,
                              options={'maxiter':120, 'ftol':2e-12, 'disp':False})
            z = result.x
        except StopSearch:
            z = latest
        retain(z)
    return best
