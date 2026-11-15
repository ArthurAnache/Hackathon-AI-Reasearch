import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'time_fraction': (0.69, 0.4, 0.7),
    'iterations': (100, 30, 220),
    'contact_margin': (0.14, 0.04, 0.35),
    'restart_noise': (0.018, 0.001, 0.12)
}


def exact_score(data):
    try:
        a = np.asarray(data['circles'], dtype=np.float64)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
        extent = max(u+t for u, t in zip(x, r))-min(u-t for u, t in zip(x, r))
        extent += max(u+t for u, t in zip(y, r))-min(u-t for u, t in zip(y, r))
        if extent > 2:
            return -float('inf')
        for i in range(21):
            for j in range(i):
                d = np.sqrt((a[i, 0]-a[j, 0])**2+(a[i, 1]-a[j, 1])**2)
                if d < a[i, 2]+a[j, 2]:
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
    best = init
    best_score = exact_score(init)
    pending = None
    pending_score = best_score

    def check():
        if time.monotonic() >= deadline:
            raise Timeout()

    def repair(a):
        a = np.array(a, dtype=float).reshape(21, 3).copy()
        if not np.isfinite(a).all():
            return None
        a[:, 2] = np.maximum(a[:, 2], 1e-9)
        delta = a[ii, :2]-a[jj, :2]
        dist = np.sqrt(np.sum(delta*delta, axis=1))
        f = min(1.0, float(np.min(dist/(a[ii, 2]+a[jj, 2]))))
        if f <= 0:
            return None
        a[:, 2] *= f*(1-3e-13)
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        a[:, :2] -= lo
        wh = np.max(a[:, :2]+a[:, 2, None], axis=0)
        a *= (2-2e-12)/float(np.sum(wh))
        return a

    def consider(a):
        nonlocal pending, pending_score
        b = repair(a)
        if b is not None:
            s = float(np.sum(b[:, 2]))
            if s > pending_score:
                pending = b
                pending_score = s
        return b

    fallback_r = 0.099999999999
    fallback = np.array([[(i % 7)*0.2+0.1, (i//7)*0.2+0.1, fallback_r] for i in range(21)])
    consider(fallback)
    warm = None
    try:
        warm = repair(np.asarray(init['circles'], dtype=float))
        if warm is not None:
            consider(warm)
    except Exception:
        pass

    grad = np.zeros(64)
    grad[2:63:3] = -1
    bounds = [(0, 2), (0, 2), (1e-8, 0.6)]*21+[(0.1, 1.9)]

    def optimize(a, maxiter):
        check()
        z = np.r_[a.ravel(), np.max(a[:, 0]+a[:, 2])]
        delta = a[ii, :2]-a[jj, :2]
        gaps = np.sqrt(np.sum(delta*delta, axis=1))-a[ii, 2]-a[jj, 2]
        active = gaps < float(params['contact_margin'])
        for phase in range(3):
            check()
            pi, pj = ii[active], jj[active]
            k = np.arange(len(pi))+84
            base = np.zeros((84+len(pi), 64))
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
            base[k, 3*pi+2] = -1
            base[k, 3*pj+2] = -1
            cached_z = None
            cached_c = None
            cached_j = None

            def compute(q):
                nonlocal cached_z, cached_c, cached_j
                check()
                if cached_z is not None and np.array_equal(q, cached_z):
                    return
                b = q[:63].reshape(21, 3)
                x, y, r = b.T
                dx = x[pi]-x[pj]
                dy = y[pi]-y[pj]
                d = np.sqrt(dx*dx+dy*dy+1e-30)
                cached_c = np.concatenate((x-r, q[63]-x-r, y-r, 2-q[63]-y-r, d-r[pi]-r[pj]))
                cached_j = base.copy()
                cached_j[k, 3*pi] = dx/d
                cached_j[k, 3*pj] = -dx/d
                cached_j[k, 3*pi+1] = dy/d
                cached_j[k, 3*pj+1] = -dy/d
                cached_z = q.copy()

            def fun(q):
                compute(q)
                return cached_c

            def jac(q):
                compute(q)
                return cached_j

            def callback(q):
                check()
                consider(q[:63])

            result = minimize(lambda q: -np.sum(q[2:63:3]), z,
                              jac=lambda q: grad, method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': fun, 'jac': jac},
                              callback=callback,
                              options={'maxiter': maxiter, 'ftol': 8e-12, 'disp': False})
            z = result.x
            consider(z[:63])
            b = z[:63].reshape(21, 3)
            d = b[ii, :2]-b[jj, :2]
            gaps = np.sqrt(np.sum(d*d, axis=1))-b[ii, 2]-b[jj, 2]
            missing = (~active) & (gaps < 0.025)
            if not np.any(missing):
                break
            active |= missing

    def lattice():
        coords = []
        for row, count in enumerate([5, 4, 5, 4, 3]):
            check()
            offset = 1 if row % 2 == 0 else 2
            if count == 3:
                offset += 2
            for col in range(count):
                check()
                coords.append([offset+2*col, 1+np.sqrt(3)*row, 1])
        return repair(coords)

    attempt = 0
    while time.monotonic() < deadline:
        try:
            if attempt == 0 and warm is not None:
                a = warm.copy()
            elif attempt == 1:
                a = lattice()
            else:
                a = pending.copy() if pending is not None else lattice()
                amplitude = float(params['restart_noise'])*(1+(attempt % 4))
                ids = rng.choice(21, size=min(21, 5+attempt), replace=False)
                a[ids, :2] += rng.normal(0, amplitude, (len(ids), 2))
                a[ids, 2] *= rng.uniform(0.90, 1.08, len(ids))
                a = repair(a)
            if a is not None:
                optimize(a, int(params['iterations']))
        except Timeout:
            break
        except (ValueError, FloatingPointError):
            pass
        attempt += 1

    if pending is not None:
        candidate = {'circles': pending.tolist()}
        score = exact_score(candidate)
        if score > best_score:
            best, best_score = candidate, score
    if best_score == -float('inf'):
        a = repair(fallback)
        candidate = {'circles': a.tolist()}
        if exact_score(candidate) > best_score:
            best = candidate
    return best
