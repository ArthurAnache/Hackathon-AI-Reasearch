import time
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog

PARAMS = {
    'trust': (0.19, 0.03, 0.5),
    'steps': (12, 4, 30),
    'hole_samples': (320, 80, 1000),
    'double_probability': (0.4, 0.0, 1.0),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)


def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2] + a[J, 2]):
        return -np.inf
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    perimeter = (max(u+v for u, v in zip(x, r))-min(u-v for u, v in zip(x, r))
                 + max(u+v for u, v in zip(y, r))-min(u-v for u, v in zip(y, r)))
    return float(np.sum(a[:, 2])) if perimeter <= 2 else -np.inf


def repaired(z):
    a = np.column_stack((z[:N], z[N:2*N], z[2*N:3*N])).copy()
    if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return None
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    factor = min(1.0, float(np.min(d / (a[I, 2]+a[J, 2]))))
    a[:, 2] *= factor * (1.0-8e-15)
    lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
    hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
    a[:, :2] -= lo
    a *= min(1.0, 2.0/float(np.sum(hi-lo))) * (1.0-8e-15)
    return a


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf
    if init is not None:
        a = np.asarray(init['circles'], dtype=float).copy()
        best_score = exact_score(a)
    else:
        a = np.array([[(i % 7 + .5)*.2, (i//7+.5)*.2, .1*(1-1e-13)] for i in range(N)])
        best = {'circles': a.tolist()}
        best_score = exact_score(a)
    lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
    hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
    scale = 2.0/np.sum(hi-lo)
    a[:, :2] = (a[:, :2]-lo)*scale
    a[:, 2] *= scale
    w = float((hi[0]-lo[0])*scale)
    base = np.r_[a[:, 0], a[:, 1], a[:, 2], w]

    A = np.zeros((M+4*N, 3*N+1))
    b = np.zeros(M+4*N)
    ii = np.arange(N)
    rr = np.arange(M)
    A[rr, 2*N+I] = 1
    A[rr, 2*N+J] = 1
    A[M+ii, ii] = -1
    A[M+ii, 2*N+ii] = 1
    A[M+N+ii, ii] = 1
    A[M+N+ii, 2*N+ii] = 1
    A[M+N+ii, -1] = -1
    A[M+2*N+ii, N+ii] = -1
    A[M+2*N+ii, 2*N+ii] = 1
    A[M+3*N+ii, N+ii] = 1
    A[M+3*N+ii, 2*N+ii] = 1
    A[M+3*N+ii, -1] = 1
    b[M+3*N:] = 2
    objective = np.r_[np.zeros(2*N), -np.ones(N), 0.0]
    trust = float(params['trust'])
    pool = []
    attempt = 0

    while time.monotonic() < deadline-0.025:
        if attempt == 0:
            z = base.copy()
            steps = 3
        else:
            if pool and attempt % 4 == 0:
                z = pool[int(rng.integers(len(pool)))][1].copy()
            else:
                z = base.copy()
            w = z[-1]
            h = 2-w
            count = 2 if rng.random() < float(params['double_probability']) else 1
            small = np.argsort(z[2*N:3*N])[:12]
            chosen = rng.choice(small, size=count, replace=False)
            for k in chosen:
                if time.monotonic() >= deadline-0.025:
                    break
                pts = rng.random((int(params['hole_samples']), 2))*[w, h]
                # Include near-wall holes as well as interior holes.
                pts[:40, 0] = rng.choice([0.025, max(.026, w-.025)], 40)
                pts[40:80, 1] = rng.choice([0.025, max(.026, h-.025)], 40)
                centers = np.column_stack((z[:N], z[N:2*N]))
                radii = z[2*N:3*N]
                distances = np.sqrt(np.sum((pts[:, None, :]-centers[None, :, :])**2, axis=2))-radii
                distances[:, k] = np.inf
                clearance = np.minimum(np.min(distances, axis=1), np.min(np.column_stack((pts, np.array([w, h])-pts)), axis=1))
                moved = np.linalg.norm(pts-centers[k], axis=1) > max(.14, 1.6*radii[k])
                clearance[~moved] = -10
                order = np.argsort(clearance)[-6:]
                pick = order[int(rng.integers(len(order)))]
                z[k], z[N+k] = pts[pick]
                z[2*N+k] = max(1e-5, clearance[pick])
            steps = int(params['steps'])
        previous = -np.inf
        for iteration in range(steps):
            remaining = deadline-time.monotonic()
            if remaining < .025:
                break
            dx = z[I]-z[J]
            dy = z[N+I]-z[N+J]
            norm = np.maximum(np.hypot(dx, dy), 1e-14)
            ux, uy = dx/norm, dy/norm
            A[rr, I] = -ux
            A[rr, J] = ux
            A[rr, N+I] = -uy
            A[rr, N+J] = uy
            bounds = np.empty((3*N+1, 2))
            bounds[:2*N, 0] = np.maximum(0, z[:2*N]-trust)
            bounds[:2*N, 1] = np.minimum(2, z[:2*N]+trust)
            bounds[2*N:3*N] = [1e-7, .65]
            bounds[-1] = [max(.12, z[-1]-trust), min(1.88, z[-1]+trust)]
            result = linprog(objective, A_ub=A, b_ub=b, bounds=bounds,
                             method='highs', options={'time_limit': min(.08, remaining*.8),
                             'dual_feasibility_tolerance': 1e-9,
                             'primal_feasibility_tolerance': 1e-9})
            if not result.success:
                break
            z = result.x
            value = float(np.sum(z[2*N:3*N]))
            if value > best_score-2e-8:
                candidate = repaired(z)
                if candidate is not None:
                    score = exact_score(candidate)
                    if score > best_score:
                        best_score = score
                        best = {'circles': candidate.tolist()}
            if iteration >= 4 and value < 2.345:
                break
            if abs(value-previous) < 2e-12:
                break
            previous = value
        if attempt > 0 and time.monotonic() < deadline:
            value = float(np.sum(z[2*N:3*N]))
            if value > 2.35 and np.linalg.norm(z-base) > .04:
                pool.append((value, z.copy()))
                pool.sort(key=lambda item: item[0], reverse=True)
                pool = pool[:4]
        attempt += 1
    return best
