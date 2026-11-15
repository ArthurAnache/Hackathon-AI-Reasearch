import time
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog

PARAMS = {
    'trust': (0.065, 0.01, 0.2),
    'tilt': (1.3, 0.2, 3.0),
    'recovery_steps': (18, 5, 40),
    'relocate_count': (2, 1, 4),
}
N = 21
I, J = np.triu_indices(N, 1)


def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return -np.inf
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    s = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
    s += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
    if s > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))


def repaired(z):
    a = np.column_stack((z[:N], z[N:2*N], z[2*N:3*N])).copy()
    if not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
        return None
    d = np.sqrt((a[I, 0]-a[J, 0])**2+(a[I, 1]-a[J, 1])**2)
    factor = min(1.0, float(np.min(d/(a[I, 2]+a[J, 2]))))
    a[:, 2] *= factor*(1.0-8e-15)
    lo = np.min(a[:, :2]-a[:, 2,None], axis=0)
    hi = np.max(a[:, :2]+a[:, 2,None], axis=0)
    a[:, :2] -= (lo+hi)/2
    scale = min(1.0, 2.0/float(np.sum(hi-lo)))*(1.0-8e-15)
    a *= scale
    return a


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.69*budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a = np.array([[(i%7+0.5)/7, (i//7+0.5)/3, 0.04] for i in range(N)], float)
        best = {'circles': a.tolist()}
        best_score = exact_score(a)
    else:
        a = np.asarray(init['circles'], dtype=float).copy()
        best = init
        best_score = exact_score(a)
    lo = np.min(a[:, :2]-a[:, 2,None], axis=0)
    hi = np.max(a[:, :2]+a[:, 2,None], axis=0)
    sides = hi-lo
    a[:, :2] -= lo
    scale = 2.0/float(sides.sum())
    a *= scale
    zbase = np.r_[a[:, 0], a[:, 1], a[:, 2], sides[0]*scale]
    best_z = zbase.copy()
    m = len(I)
    rows = np.arange(m)
    A = np.zeros((m+4*N, 3*N+1))
    b = np.zeros(m+4*N)
    A[rows, 2*N+I] = 1
    A[rows, 2*N+J] = 1
    k = np.arange(N)
    A[m+k, k] = -1
    A[m+k, 2*N+k] = 1
    A[m+N+k, k] = 1
    A[m+N+k, 2*N+k] = 1
    A[m+N+k, -1] = -1
    A[m+2*N+k, N+k] = -1
    A[m+2*N+k, 2*N+k] = 1
    A[m+3*N+k, N+k] = 1
    A[m+3*N+k, 2*N+k] = 1
    A[m+3*N+k, -1] = 1
    b[m+3*N:] = 2
    trust = float(params['trust'])
    trial = 0
    bank = [zbase.copy()]
    while time.monotonic() < deadline:
        if trial == 0:
            z = best_z.copy()
            tilt_steps = 0
        else:
            z = (best_z if trial % 4 else bank[int(rng.integers(len(bank)))]).copy()
            tilt_steps = 2 + trial % 2
            if trial % 3 == 0:
                order = np.argsort(z[2*N:3*N])
                count = int(params['relocate_count'])
                chosen = order[:count]
                z[2*N+chosen] = 1e-7
                for idx in chosen:
                    if time.monotonic() >= deadline:
                        break
                    pts = rng.random((160, 2))*np.array([z[-1], 2-z[-1]])
                    others = np.arange(N) != idx
                    centers = np.column_stack((z[:N], z[N:2*N]))[others]
                    radii = z[2*N:3*N][others]
                    clearance = np.min(np.sqrt(np.sum((pts[:,None,:]-centers[None,:,:])**2, axis=2))-radii, axis=1)
                    clearance = np.minimum(clearance, np.min(np.column_stack((pts, np.array([z[-1],2-z[-1]])-pts)), axis=1))
                    j = int(np.argmax(clearance))
                    z[idx], z[N+idx] = pts[j]
                    z[2*N+idx] = max(1e-7, clearance[j]*0.98)
                tilt_steps = 0
        weights = np.exp(np.clip(rng.normal(0, float(params['tilt']), N), -2, 2))
        previous = -np.inf
        for step in range(tilt_steps+int(params['recovery_steps'])):
            if time.monotonic() >= deadline:
                break
            dx = z[I]-z[J]
            dy = z[N+I]-z[N+J]
            dist = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-15)
            ux, uy = dx/dist, dy/dist
            A[rows, I] = -ux
            A[rows, J] = ux
            A[rows, N+I] = -uy
            A[rows, N+J] = uy
            c = np.zeros(3*N+1)
            c[2*N:3*N] = -(weights if step < tilt_steps else np.ones(N))
            delta = trust if step < tilt_steps else trust*1.8
            lower = z-delta
            upper = z+delta
            lower[2*N:3*N] = 1e-8
            upper[2*N:3*N] = 0.7
            lower[-1] = max(0.12, z[-1]-delta)
            upper[-1] = min(1.88, z[-1]+delta)
            bounds = np.column_stack((lower, upper))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs-ds', options={'primal_feasibility_tolerance':1e-10, 'dual_feasibility_tolerance':1e-10, 'threads':1, 'time_limit':max(0.001, deadline-time.monotonic())})
            if not result.success:
                break
            newz = result.x
            raw_score = float(np.sum(newz[2*N:3*N]))
            if raw_score > best_score:
                candidate = repaired(newz)
                if candidate is not None:
                    score = exact_score(candidate)
                    if score > best_score:
                        best_score = score
                        best = {'circles': candidate.tolist()}
                        best_z = newz.copy()
            movement = float(np.max(np.abs(newz-z)))
            z = newz
            if step >= tilt_steps and movement < 2e-8 and abs(raw_score-previous) < 1e-11:
                break
            previous = raw_score
        if time.monotonic() < deadline and float(np.sum(z[2*N:3*N])) > best_score-0.025:
            bank.append(z.copy())
            if len(bank) > 8:
                bank.pop(1)
        trial += 1
    return best
