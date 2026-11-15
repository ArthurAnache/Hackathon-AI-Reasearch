import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import qr

PARAMS = {
    'rotation': (0.67, 0.2, 1.4),
    'cluster_size': (4, 3, 5),
    'stop_fraction': (0.70, 0.50, 0.75),
    'iterations': (65, 20, 120),
}


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + float(params['stop_fraction']) * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    ids = np.arange(n)
    best = init
    best_score = -float('inf')

    class Finished(Exception):
        pass

    def check():
        if time.monotonic() >= deadline:
            raise Finished()

    def exact_score(data):
        check()
        a = np.asarray(data['circles'], dtype=np.float64)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        xs = list(map(lambda v: Fraction(float(v)), a[:, 0]))
        ys = list(map(lambda v: Fraction(float(v)), a[:, 1]))
        rs = list(map(lambda v: Fraction(float(v)), a[:, 2]))
        width = max(x+r for x, r in zip(xs, rs)) - min(x-r for x, r in zip(xs, rs))
        height = max(y+r for y, r in zip(ys, rs)) - min(y-r for y, r in zip(ys, rs))
        if width + height > 2:
            return -float('inf')
        for u, v in itertools.combinations(a, 2):
            check()
            d = np.sqrt((u[0]-v[0])**2 + (u[1]-v[1])**2)
            if d < u[2]+v[2]:
                return -float('inf')
        return float(np.sum(a[:, 2]))

    def accept(z):
        nonlocal best, best_score
        check()
        a = np.column_stack((z[:n], z[n:2*n], z[2*n:3*n])).copy()
        if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return
        d = np.sqrt(np.sum((a[ii, :2]-a[jj, :2])**2, axis=1))
        shrink = max(0.0, float(np.max(a[ii, 2]+a[jj, 2]-d)) / 2.0)
        a[:, 2] -= shrink + 2e-15
        if np.any(a[:, 2] <= 0):
            return
        a[:, :2] -= np.min(a[:, :2]-a[:, 2, None], axis=0)
        xs = [Fraction(float(x)) for x in a[:, 0]]
        ys = [Fraction(float(y)) for y in a[:, 1]]
        rs = [Fraction(float(r)) for r in a[:, 2]]
        perimeter = (max(x+r for x, r in zip(xs, rs))-min(x-r for x, r in zip(xs, rs))
                     + max(y+r for y, r in zip(ys, rs))-min(y-r for y, r in zip(ys, rs)))
        if perimeter > 2:
            a *= float(Fraction(2) / perimeter) * (1.0-4e-15)
        if float(np.sum(a[:, 2])) <= best_score:
            return
        candidate = {'circles': a.tolist()}
        score = exact_score(candidate)
        if score > best_score:
            best, best_score = candidate, score

    def encode(a):
        a = np.asarray(a, dtype=float).copy()
        lower = np.min(a[:, :2]-a[:, 2, None], axis=0)
        upper = np.max(a[:, :2]+a[:, 2, None], axis=0)
        extent = upper-lower
        scale = 2.0 / float(np.sum(extent))
        a[:, :2] = (a[:, :2]-lower) * scale
        a[:, 2] *= scale
        return np.r_[a[:, 0], a[:, 1], a[:, 2], extent[0]*scale]

    def constraints(z):
        check()
        x, y, r, w = z[:n], z[n:2*n], z[2*n:3*n], z[-1]
        dx, dy = x[ii]-x[jj], y[ii]-y[jj]
        d = np.sqrt(dx*dx+dy*dy)
        return np.r_[d-r[ii]-r[jj], x-r, w-x-r, y-r, 2-w-y-r]

    def jacobian(z):
        check()
        x, y = z[:n], z[n:2*n]
        dx, dy = x[ii]-x[jj], y[ii]-y[jj]
        d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-30)
        j = np.zeros((m+4*n, 3*n+1))
        rows = np.arange(m)
        j[rows, ii], j[rows, jj] = dx/d, -dx/d
        j[rows, n+ii], j[rows, n+jj] = dy/d, -dy/d
        j[rows, 2*n+ii] = -1
        j[rows, 2*n+jj] = -1
        for k in range(4):
            check()
            row = m+k*n+ids
            j[row, 2*n+ids] = -1
            j[row, (0 if k < 2 else n)+ids] = 1 if k % 2 == 0 else -1
        j[m+n:m+2*n, -1] = 1
        j[m+3*n:m+4*n, -1] = -1
        return j

    def contact_polish(z):
        check()
        values = constraints(z)
        active = np.flatnonzero(values < 2e-7)
        if len(active) < len(z):
            return z
        mat = jacobian(z)[active]
        _, rr, piv = qr(mat.T, mode='economic', pivoting=True, check_finite=False)
        if min(abs(np.diag(rr)[:len(z)])) < 1e-9:
            return z
        active = active[piv[:len(z)]]
        q = z.copy()
        for unused in range(5):
            check()
            residual = constraints(q)[active]
            if np.max(abs(residual)) < 3e-16:
                break
            try:
                step = np.linalg.solve(jacobian(q)[active], -residual)
            except np.linalg.LinAlgError:
                break
            if np.max(abs(step)) > 0.01:
                break
            q += step
            accept(q)
        return q

    gradient = np.r_[np.zeros(2*n), -np.ones(n), 0.0]

    def objective(z):
        check()
        return -float(np.sum(z[2*n:3*n]))

    bounds = [(None, None)]*(2*n) + [(1e-7, 0.5)]*n + [(0.3, 1.7)]
    callback_count = 0

    def callback(z):
        nonlocal callback_count
        check()
        callback_count += 1
        if callback_count % 3 == 0:
            accept(z)

    try:
        if init is not None:
            best_score = exact_score(init)
            base = encode(init['circles'])
        else:
            a = []
            for row in range(5):
                check()
                count = 5 if row % 2 == 0 else 4
                for col in range(count):
                    check()
                    a.append([col + 0.5*(row % 2), row*np.sqrt(3)/2, 0.49])
            a = np.asarray(a[:21])
            base = encode(a)
            accept(base)
        base = contact_polish(base)
        accept(base)
        result = minimize(objective, base, jac=lambda z: gradient, method='SLSQP',
                          bounds=bounds, constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                          callback=callback, options={'maxiter': 18, 'ftol': 3e-14, 'disp': False})
        accept(result.x)
        contact_polish(result.x)
        attempt = 0
        while time.monotonic() < deadline:
            check()
            base = encode(best['circles']) if best is not None else result.x.copy()
            xy = np.column_stack((base[:n], base[n:2*n]))
            anchor = int(rng.integers(n))
            k = int(params['cluster_size']) + (attempt % 2)
            subset = np.argsort(np.sum((xy-xy[anchor])**2, axis=1))[:min(k, 5)]
            centre = np.mean(xy[subset], axis=0)
            angle = float(params['rotation']) * (-1 if attempt % 2 else 1)
            angle *= float(rng.uniform(0.8, 1.25))
            c, s = np.cos(angle), np.sin(angle)
            rot = np.array([[c, -s], [s, c]])
            xy[subset] = centre + (xy[subset]-centre) @ rot.T
            base[:n], base[n:2*n] = xy[:, 0], xy[:, 1]
            base[2*n+subset] *= 0.9
            result = minimize(objective, base, jac=lambda z: gradient, method='SLSQP',
                              bounds=bounds, constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                              callback=callback, options={'maxiter': int(params['iterations']), 'ftol': 2e-12, 'disp': False})
            accept(result.x)
            contact_polish(result.x)
            attempt += 1
    except Finished:
        pass
    return best
