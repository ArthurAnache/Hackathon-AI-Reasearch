import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'time_fraction': (0.69, 0.4, 0.7),
    'first_slice': (0.40, 0.15, 0.7),
    'restart_slice': (0.29, 0.1, 0.6),
    'jitter': (0.018, 0.001, 0.09),
    'iterations': (100, 30, 250)
}


def exact_score(data):
    try:
        a = np.asarray(data['circles'], dtype=np.float64)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
        width = max(u+v for u, v in zip(x, r))-min(u-v for u, v in zip(x, r))
        height = max(u+v for u, v in zip(y, r))-min(u-v for u, v in zip(y, r))
        if width+height > 2:
            return -float('inf')
        for i in range(21):
            for j in range(i):
                if np.sqrt((a[i, 0]-a[j, 0])**2+(a[i, 1]-a[j, 1])**2) < a[i, 2]+a[j, 2]:
                    return -float('inf')
        return float(np.sum(a[:, 2]))
    except Exception:
        return -float('inf')


class StopAttempt(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    begun = time.monotonic()
    deadline = begun + params['time_fraction']*budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    rows = np.arange(21)
    best = init
    best_score = exact_score(init)
    best_array = None
    if best_score > -float('inf'):
        best_array = np.array(init['circles'], dtype=float)

    def repair(a):
        a = np.asarray(a, dtype=float).reshape(21, 3).copy()
        if not np.isfinite(a).all():
            return None
        a[:, 2] = np.maximum(a[:, 2], 1e-9)
        delta = a[ii, :2]-a[jj, :2]
        distances = np.sqrt(np.sum(delta*delta, axis=1))
        factor = min(1.0, float(np.min(distances/(a[ii, 2]+a[jj, 2]))))
        if factor <= 0:
            return None
        a[:, 2] *= factor*(1-3e-13)
        a[:, :2] -= np.min(a[:, :2]-a[:, 2, None], axis=0)
        extent = np.max(a[:, :2]+a[:, 2, None], axis=0)
        a *= (2-2e-12)/float(np.sum(extent))
        return a

    def consider(a):
        nonlocal best, best_score, best_array
        a = repair(a)
        if a is not None and float(np.sum(a[:, 2])) > best_score:
            candidate = {'circles': a.tolist()}
            s = exact_score(candidate)
            if s > best_score:
                best, best_score, best_array = candidate, s, a.copy()
        return a

    if best_array is None:
        try:
            a = np.asarray(init['circles'], dtype=float)
            if a.shape != (21, 3):
                raise ValueError()
            consider(a)
        except Exception:
            pass
    if best_array is None:
        a = np.array([[(i % 7)*0.1+0.05, (i//7)*0.1+0.05, 0.049] for i in range(21)])
        consider(a)

    gradient = np.zeros(64)
    gradient[2:63:3] = -1
    boundary = np.zeros((84, 64))
    boundary[rows, 3*rows] = 1
    boundary[rows, 3*rows+2] = -1
    boundary[21+rows, 3*rows] = -1
    boundary[21+rows, 3*rows+2] = -1
    boundary[21+rows, 63] = 1
    boundary[42+rows, 3*rows+1] = 1
    boundary[42+rows, 3*rows+2] = -1
    boundary[63+rows, 3*rows+1] = -1
    boundary[63+rows, 3*rows+2] = -1
    boundary[63+rows, 63] = -1
    original = best_array.copy()
    attempt = 0
    while time.monotonic() < deadline:
        local_deadline = min(deadline, time.monotonic() + (params['first_slice'] if attempt == 0 else params['restart_slice'])*budget_s)
        base = best_array.copy() if attempt % 4 != 3 else original.copy()
        base[:, :2] -= np.min(base[:, :2]-base[:, 2, None], axis=0)
        if attempt:
            amplitude = params['jitter'] * (1.0, 0.5, 1.7, 2.5)[(attempt-1) % 4]
            noise = rng.normal(0, amplitude, (21, 2))
            if attempt % 2 == 0:
                chosen = rng.choice(21, size=5, replace=False)
                noise[:] = 0
                noise[chosen] = rng.normal(0, amplitude*2, (5, 2))
            base[:, :2] += noise
            base[:, 2] *= rng.uniform(0.97, 1.015, 21)
        base = repair(base)
        if base is None:
            attempt += 1
            continue
        z = np.r_[base.ravel(), np.max(base[:, 0]+base[:, 2])]
        # Distant pairs are inactive throughout a short local solve. Every
        # saved iterate is nevertheless repaired and checked against all pairs.
        delta = base[ii, :2]-base[jj, :2]
        gap = np.sqrt(np.sum(delta*delta, axis=1))-base[ii, 2]-base[jj, 2]
        mask = gap < 0.20
        pi, pj = ii[mask], jj[mask]
        k = np.arange(len(pi))+84

        def check():
            if time.monotonic() >= local_deadline:
                raise StopAttempt()

        def constraints(q):
            check()
            a = q[:63].reshape(21, 3)
            x, y, r = a.T
            dx, dy = x[pi]-x[pj], y[pi]-y[pj]
            d = np.sqrt(dx*dx+dy*dy+1e-30)
            return np.concatenate((x-r, q[63]-x-r, y-r, 2-q[63]-y-r, d-r[pi]-r[pj]))

        def jacobian(q):
            check()
            a = q[:63].reshape(21, 3)
            dx = a[pi, 0]-a[pj, 0]
            dy = a[pi, 1]-a[pj, 1]
            d = np.sqrt(dx*dx+dy*dy+1e-30)
            out = np.zeros((84+len(pi), 64))
            out[:84] = boundary
            out[k, 3*pi] = dx/d
            out[k, 3*pj] = -dx/d
            out[k, 3*pi+1] = dy/d
            out[k, 3*pj+1] = -dy/d
            out[k, 3*pi+2] = -1
            out[k, 3*pj+2] = -1
            return out

        def callback(q):
            check()
            consider(q[:63])

        try:
            result = minimize(lambda q: -np.sum(q[2:63:3]), z, jac=lambda q: gradient,
                              method='SLSQP', bounds=[(0, 2), (0, 2), (1e-8, 0.6)]*21+[(0.15, 1.85)],
                              constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                              callback=callback,
                              options={'maxiter': int(params['iterations']), 'ftol': 2e-12, 'disp': False})
            if time.monotonic() < deadline:
                consider(result.x[:63])
        except (StopAttempt, ValueError, FloatingPointError):
            pass
        attempt += 1
    return best
