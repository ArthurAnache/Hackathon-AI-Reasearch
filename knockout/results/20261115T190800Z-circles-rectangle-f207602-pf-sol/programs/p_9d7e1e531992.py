import time
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog

PARAMS = {
    'trust': (0.065, 0.01, 0.2),
    'weight_spread': (0.85, 0.1, 2.0),
    'force': (0.16, 0.01, 0.6),
    'tilt_steps': (4, 1, 10),
    'restore_steps': (13, 3, 30),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)


def score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    p, r = a[:, :2], a[:, 2]
    d = np.sqrt((p[I, 0]-p[J, 0])**2 + (p[I, 1]-p[J, 1])**2)
    if np.any(d < r[I]+r[J]):
        return -np.inf
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    q = [Fraction(float(v)) for v in r]
    perimeter = (max(v+t for v,t in zip(x,q))-min(v-t for v,t in zip(x,q))
                 +max(v+t for v,t in zip(y,q))-min(v-t for v,t in zip(y,q)))
    if perimeter > 2:
        return -np.inf
    return float(np.sum(r))


def repaired(z, deadline):
    a = np.column_stack((z[:N], z[N:2*N], z[2*N:3*N])).copy()
    if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return None
    p, r = a[:, :2], a[:, 2]
    d = np.sqrt((p[I, 0]-p[J, 0])**2+(p[I, 1]-p[J, 1])**2)
    factor = min(1.0, float(np.min(d/(r[I]+r[J]))))
    r *= factor*(1-3e-15)
    lo = np.min(p-r[:, None], axis=0)
    hi = np.max(p+r[:, None], axis=0)
    extent = float(np.sum(hi-lo))
    if extent > 2-2e-14:
        a *= (2-2e-14)/extent
    for _ in range(4):
        if time.monotonic() >= deadline:
            return None
        if np.isfinite(score(a)):
            return a
        a[:, 2] *= 1-2e-13
    return None


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a = np.array([[(i % 7 + .5)*.19, (i//7+.5)*.19, .09] for i in range(N)])
        best = {'circles': a.tolist()}
        best_value = score(a)
    else:
        a = np.asarray(init['circles'], dtype=float).copy()
        best = init
        best_value = score(a)
    low = np.min(a[:, :2]-a[:, 2, None], axis=0)
    high = np.max(a[:, :2]+a[:, 2, None], axis=0)
    extent = high-low
    scale = 2/float(np.sum(extent))
    p = (a[:, :2]-low)*scale
    r = a[:, 2]*scale
    base = np.r_[p[:, 0], p[:, 1], r, extent[0]*scale]
    # Variables are x, y, radius, and the rectangle width; its height is 2-width.
    mat = np.zeros((M+4*N, 3*N+1))
    rhs = np.zeros(M+4*N)
    row = np.arange(M)
    mat[row, 2*N+I] = 1
    mat[row, 2*N+J] = 1
    for k in range(N):
        j = M+k
        mat[j, k] = -1
        mat[j, 2*N+k] = 1
        j = M+N+k
        mat[j, k] = 1
        mat[j, 2*N+k] = 1
        mat[j, -1] = -1
        j = M+2*N+k
        mat[j, N+k] = -1
        mat[j, 2*N+k] = 1
        j = M+3*N+k
        mat[j, N+k] = 1
        mat[j, 2*N+k] = 1
        mat[j, -1] = 1
        rhs[j] = 2
    true_objective = np.r_[np.zeros(2*N), -np.ones(N), 0.0]

    def step(z, objective, trust):
        if time.monotonic() >= deadline:
            return None
        dx, dy = z[I]-z[J], z[N+I]-z[N+J]
        dd = np.maximum(np.hypot(dx, dy), 1e-14)
        ux, uy = dx/dd, dy/dd
        mat[row, I] = -ux
        mat[row, J] = ux
        mat[row, N+I] = -uy
        mat[row, N+J] = uy
        lower = z-trust
        upper = z+trust
        lower[:2*N] = np.maximum(lower[:2*N], 0)
        upper[:2*N] = np.minimum(upper[:2*N], 2)
        lower[2*N:3*N] = np.maximum(lower[2*N:3*N], 1e-6)
        lower[-1] = max(.08, lower[-1])
        upper[-1] = min(1.92, upper[-1])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = linprog(objective, A_ub=mat, b_ub=rhs,
                          bounds=np.column_stack((lower, upper)), method='highs',
                          options={'threads': 1, 'maxiter': 350,
                                   'primal_feasibility_tolerance': 1e-9,
                                   'dual_feasibility_tolerance': 1e-9,
                                   'time_limit': max(.001, deadline-time.monotonic())})
        if not res.success:
            return None
        return res.x

    def consider(z):
        nonlocal best, best_value
        if time.monotonic() >= deadline:
            return
        aa = repaired(z, deadline)
        if aa is not None:
            value = score(aa)
            if value > best_value:
                best_value = value
                best = {'circles': aa.tolist()}

    # First obtain the tangent-LP fixed point as a reference for coordinated excursions.
    z = base.copy()
    for _ in range(8):
        if time.monotonic() >= deadline:
            break
        zz = step(z, true_objective, float(params['trust']))
        if zz is None:
            break
        delta = float(np.max(np.abs(zz-z)))
        z = zz
        consider(z)
        if delta < 2e-8:
            break
    reference = z.copy()
    attempt = 0
    while time.monotonic() < deadline:
        z = reference.copy()
        trust = float(params['trust'])*[.6, 1.0, 1.7, 2.3][attempt % 4]
        weights = rng.lognormal(-.5*float(params['weight_spread'])**2,
                                float(params['weight_spread']), N)
        forces = rng.normal(0, float(params['force']), (2, N))
        forces -= forces.mean(axis=1, keepdims=True)
        # Correlated forces on close circles encourage whole contact clusters to pivot.
        if attempt % 2:
            pp = np.column_stack((z[:N], z[N:2*N]))
            pivot = pp[int(rng.integers(N))]
            v = rng.normal(size=2)
            envelope = np.exp(-np.sum((pp-pivot)**2, axis=1)/.07)
            forces += float(params['force'])*2*v[:, None]*envelope[None, :]
        objective = np.r_[forces.ravel(), -weights, 0.0]
        for k in range(int(params['tilt_steps'])):
            if time.monotonic() >= deadline:
                break
            zz = step(z, objective, trust)
            if zz is None:
                break
            z = zz
        for k in range(int(params['restore_steps'])):
            if time.monotonic() >= deadline:
                break
            zz = step(z, true_objective, trust)
            if zz is None:
                break
            delta = float(np.max(np.abs(zz-z)))
            z = zz
            consider(z)
            if delta < 3e-8:
                break
        if float(np.sum(z[2*N:3*N])) > float(np.sum(reference[2*N:3*N]))+1e-9:
            reference = z.copy()
        attempt += 1
    return best
