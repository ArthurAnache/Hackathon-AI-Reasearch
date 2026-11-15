import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'time_fraction': (0.69, 0.4, 0.7),
    'iterations': (75, 20, 180),
    'jitter': (0.023, 0.001, 0.10)
}


def exact_score(data):
    try:
        a = np.asarray(data['circles'], dtype=np.float64)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
        w = max(u+v for u, v in zip(x, r))-min(u-v for u, v in zip(x, r))
        h = max(u+v for u, v in zip(y, r))-min(u-v for u, v in zip(y, r))
        if w+h > 2:
            return -float('inf')
        for i in range(21):
            for j in range(i):
                if np.sqrt((a[i, 0]-a[j, 0])**2+(a[i, 1]-a[j, 1])**2) < a[i, 2]+a[j, 2]:
                    return -float('inf')
        return float(np.sum(a[:, 2]))
    except Exception:
        return -float('inf')


class Timeout(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    deadline = time.monotonic()+float(params['time_fraction'])*budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    rows = np.arange(21)
    k = np.arange(210)+84
    best = init
    best_score = exact_score(init)
    best_array = None
    if best_score > -float('inf'):
        best_array = np.asarray(init['circles'], dtype=float).copy()

    def check():
        if time.monotonic() >= deadline:
            raise Timeout()

    def repair(a):
        a = np.asarray(a, dtype=float).reshape(21, 3).copy()
        if not np.isfinite(a).all():
            return None
        a[:, 2] = np.maximum(a[:, 2], 1e-10)
        dx = a[ii, 0]-a[jj, 0]
        dy = a[ii, 1]-a[jj, 1]
        f = min(1.0, float(np.min(np.sqrt(dx*dx+dy*dy)/(a[ii, 2]+a[jj, 2]))))
        if f <= 0:
            return None
        a[:, 2] *= f*(1-3e-13)
        a[:, :2] -= np.min(a[:, :2]-a[:, 2, None], axis=0)
        extent = np.max(a[:, :2]+a[:, 2, None], axis=0)
        a *= (2-2e-12)/float(np.sum(extent))
        return a

    def consider(z):
        nonlocal best, best_score, best_array
        a = repair(z[:63])
        if a is not None and float(np.sum(a[:, 2])) > best_score:
            candidate = {'circles': a.tolist()}
            score = exact_score(candidate)
            if score > best_score:
                best, best_score, best_array = candidate, score, a

    base = np.zeros((294, 64))
    base[rows, 3*rows] = 1
    base[rows, 3*rows+2] = -1
    base[21+rows, 3*rows] = -1
    base[21+rows, 3*rows+2] = -1
    base[21+rows, 63] = 1
    base[42+rows, 3*rows+1] = 1
    base[42+rows, 3*rows+2] = -1
    base[63+rows, 3*rows+1] = -1
    base[63+rows, 3*rows+2] = -1
    base[63+rows, 63] = -1
    base[k, 3*ii+2] = -1
    base[k, 3*jj+2] = -1

    def constraints(z):
        check()
        x, y, r = z[:63].reshape(21, 3).T
        dx, dy = x[ii]-x[jj], y[ii]-y[jj]
        return np.concatenate((x-r, z[63]-x-r, y-r, 2-z[63]-y-r,
                               np.sqrt(dx*dx+dy*dy+1e-30)-r[ii]-r[jj]))

    def jacobian(z):
        check()
        a = z[:63].reshape(21, 3)
        dx, dy = a[ii, 0]-a[jj, 0], a[ii, 1]-a[jj, 1]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        out = base.copy()
        out[k, 3*ii] = dx/d
        out[k, 3*jj] = -dx/d
        out[k, 3*ii+1] = dy/d
        out[k, 3*jj+1] = -dy/d
        return out

    gradient = np.zeros(64)
    gradient[2:63:3] = -1

    def objective(z):
        check()
        return -float(np.sum(z[2:63:3]))

    def callback(z):
        check()
        consider(z)

    def normalize(a):
        a = a.copy()
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        a[:, :2] -= lo
        extent = np.max(a[:, :2]+a[:, 2, None], axis=0)
        a *= 2/float(np.sum(extent))
        return a, float(extent[0]*2/np.sum(extent))

    if best_array is None:
        r = 0.099999999999
        fallback = np.array([[(i % 7)*2*r+r, (i//7)*2*r+r, r] for i in range(21)])
        consider(fallback.ravel())
    if init is not None:
        try:
            raw = np.asarray(init['circles'], dtype=float)
            if raw.shape == (21, 3) and np.isfinite(raw).all():
                consider(raw.ravel())
        except Exception:
            pass

    anchor = best_array.copy()
    attempt = 0
    while time.monotonic() < deadline:
        a = best_array.copy()
        if attempt == 0:
            a, width = normalize(a)
        else:
            if attempt % 5 == 4:
                a = anchor.copy()
            a, width = normalize(a)
            amplitude = float(params['jitter'])*[0.55, 1.0, 1.8, 2.8][(attempt-1) % 4]
            if attempt % 3 == 1:
                indices = rng.choice(21, size=7, replace=False)
                a[indices, :2] += rng.normal(0, amplitude, (7, 2))
                a[indices, 2] *= rng.uniform(0.88, 1.12, 7)
            else:
                a[:, :2] += rng.normal(0, amplitude, (21, 2))
                a[:, 2] *= rng.uniform(0.94, 1.06, 21)
            a[:, :2] = np.clip(a[:, :2], 0.001, 1.999)
        z = np.r_[a.ravel(), width]
        try:
            result = minimize(objective, z, jac=lambda z: gradient,
                              method='SLSQP',
                              bounds=[(0, 2), (0, 2), (1e-8, 0.7)]*21+[(0.15, 1.85)],
                              constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                              callback=callback,
                              options={'maxiter': int(params['iterations']), 'ftol': 3e-12, 'disp': False})
            consider(result.x)
        except Timeout:
            break
        except (ValueError, FloatingPointError):
            pass
        attempt += 1
    return best
