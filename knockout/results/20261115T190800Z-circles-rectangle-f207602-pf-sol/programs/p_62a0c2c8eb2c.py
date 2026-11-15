import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"iterations": (160, 40, 300), "time_fraction": (0.69, 0.4, 0.7), "jitter": (0.025, 0.0, 0.15)}


def exact_score(data):
    try:
        a = np.asarray(data['circles'], dtype=np.float64)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        xs = [Fraction(float(v)) for v in a[:, 0]]
        ys = [Fraction(float(v)) for v in a[:, 1]]
        rs = [Fraction(float(v)) for v in a[:, 2]]
        w = max(x+r for x, r in zip(xs, rs))-min(x-r for x, r in zip(xs, rs))
        h = max(y+r for y, r in zip(ys, rs))-min(y-r for y, r in zip(ys, rs))
        if w+h > 2:
            return -float('inf')
        for i in range(21):
            for j in range(i):
                d = np.sqrt((a[i, 0]-a[j, 0])**2+(a[i, 1]-a[j, 1])**2)
                if d < a[i, 2]+a[j, 2]:
                    return -float('inf')
        return float(np.sum(a[:, 2]))
    except Exception:
        return -float('inf')


class StopSearch(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + float(params['time_fraction'])*budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    rows = np.arange(21)
    best = init
    best_score = exact_score(init)
    pending = None
    pending_score = best_score

    def repair(z):
        a = np.asarray(z[:63]).reshape(21, 3).copy()
        if not np.isfinite(a).all():
            return None
        a[:, 2] = np.maximum(a[:, 2], 1e-10)
        dx = a[ii, 0]-a[jj, 0]
        dy = a[ii, 1]-a[jj, 1]
        distances = np.sqrt(dx*dx+dy*dy)
        factor = min(1.0, float(np.min(distances/(a[ii, 2]+a[jj, 2]))))
        if factor <= 0:
            return None
        a[:, 2] *= factor*(1-2e-12)
        lower = np.min(a[:, :2]-a[:, 2, None], axis=0)
        a[:, :2] -= lower
        extent = np.max(a[:, :2]+a[:, 2, None], axis=0)
        scale = (2-8e-12)/float(np.sum(extent))
        a *= scale
        return a

    def consider(z):
        nonlocal pending, pending_score
        a = repair(z)
        if a is not None:
            score = float(np.sum(a[:, 2]))
            if score > pending_score:
                pending_score = score
                pending = a

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def constraints(z):
        check()
        a = z[:63].reshape(21, 3)
        x, y, r = a.T
        dx = x[ii]-x[jj]
        dy = y[ii]-y[jj]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        return np.concatenate((x-r, z[63]-x-r, y-r, 2-z[63]-y-r, d-r[ii]-r[jj]))

    def jacobian(z):
        check()
        a = z[:63].reshape(21, 3)
        dx = a[ii, 0]-a[jj, 0]
        dy = a[ii, 1]-a[jj, 1]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        out = np.zeros((294, 64))
        out[rows, 3*rows] = 1
        out[rows, 3*rows+2] = -1
        out[21+rows, 3*rows] = -1
        out[21+rows, 3*rows+2] = -1
        out[21+rows, 63] = 1
        out[42+rows, 3*rows+1] = 1
        out[42+rows, 3*rows+2] = -1
        out[63+rows, 3*rows+1] = -1
        out[63+rows, 3*rows+2] = -1
        out[63+rows, 63] = -1
        k = np.arange(210)+84
        out[k, 3*ii] = dx/d
        out[k, 3*jj] = -dx/d
        out[k, 3*ii+1] = dy/d
        out[k, 3*jj+1] = -dy/d
        out[k, 3*ii+2] = -1
        out[k, 3*jj+2] = -1
        return out

    gradient = np.zeros(64)
    gradient[2:63:3] = -1

    def callback(z):
        check()
        consider(z)

    patterns = [[5, 4, 5, 4, 3], [4, 4, 5, 4, 4], [4, 5, 4, 5, 3], [6, 5, 5, 5]]
    attempt = 0
    while time.monotonic() < deadline:
        counts = patterns[attempt % len(patterns)]
        coords = []
        for row, count in enumerate(counts):
            if time.monotonic() >= deadline:
                break
            if len(counts) == 5:
                lattice = 1.0 if row % 2 == 0 else 2.0
                if count == 3:
                    lattice += 2.0
                if count == 4 and row % 2 == 0:
                    lattice += 0.0 if row == 0 else 2.0
            else:
                lattice = 1.0 if row % 2 == 0 else 2.0
            for k in range(count):
                coords.append([lattice+2*k, 1+np.sqrt(3)*row, 1.0])
        if len(coords) != 21:
            break
        a = np.asarray(coords, dtype=float)
        if attempt:
            a[:, :2] += rng.normal(0, float(params['jitter']), (21, 2))
        z = np.r_[a.ravel(), 1.0]
        a = repair(z)
        if a is None:
            break
        z[:63] = a.ravel()
        z[63] = np.max(a[:, 0]+a[:, 2])
        consider(z)
        try:
            result = minimize(lambda q: -np.sum(q[2:63:3]), z,
                              jac=lambda q: gradient, method='SLSQP',
                              bounds=[(0, 2), (0, 2), (1e-9, 0.65)]*21+[(0.15, 1.85)],
                              constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                              callback=callback,
                              options={'maxiter': int(params['iterations']), 'ftol': 2e-11, 'disp': False})
            consider(result.x)
        except StopSearch:
            break
        except (ValueError, FloatingPointError):
            pass
        attempt += 1
    if pending is not None:
        candidate = {'circles': pending.tolist()}
        score = exact_score(candidate)
        if score > best_score:
            best, best_score = candidate, score
    if best is None or best_score == -float('inf'):
        r = 1.0/23.0
        fallback = {'circles': [[(i % 7)*2*r+r, (i//7)*2*r+r, r*(1-1e-10)] for i in range(21)]}
        if exact_score(fallback) > best_score:
            best = fallback
    return best
