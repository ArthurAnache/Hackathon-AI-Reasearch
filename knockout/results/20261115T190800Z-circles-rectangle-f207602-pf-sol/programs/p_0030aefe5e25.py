import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {
    'search_fraction': (0.68, 0.3, 0.7),
    'perturbation': (0.027, 0.001, 0.15),
    'iterations': (65, 15, 150),
}


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + float(params['search_fraction']) * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)

    def exact(data):
        a = np.asarray(data['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -np.inf
        xs = [Fraction(float(v)) for v in a[:, 0]]
        ys = [Fraction(float(v)) for v in a[:, 1]]
        rs = [Fraction(float(v)) for v in a[:, 2]]
        w = max(x+r for x, r in zip(xs, rs)) - min(x-r for x, r in zip(xs, rs))
        h = max(y+r for y, r in zip(ys, rs)) - min(y-r for y, r in zip(ys, rs))
        if w+h > 2:
            return -np.inf
        d = np.sqrt((a[ii, 0]-a[jj, 0])**2 + (a[ii, 1]-a[jj, 1])**2)
        if np.any(d < a[ii, 2]+a[jj, 2]):
            return -np.inf
        return float(np.sum(a[:, 2]))

    if init is None:
        a = np.array([[0.2*(j+0.5*(i % 2)), 0.17320508075688773*i, 0.099]
                      for i in range(5) for j in range(5)][:21])
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        a[:, :2] -= (lo+hi)/2
        a *= 1.999999999 / np.sum(hi-lo)
        init = {'circles': a.tolist()}
    best = init
    best_score = exact(init)
    a = np.array(init['circles'], dtype=float)
    lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
    hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
    a[:, :2] -= (lo+hi)/2
    width = float(hi[0]-lo[0])

    def consider(c):
        nonlocal best, best_score
        if time.monotonic() >= deadline or not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return
        c = c.copy()
        d = np.sqrt((c[ii, 0]-c[jj, 0])**2+(c[ii, 1]-c[jj, 1])**2)
        ratio = float(np.min(d/(c[ii, 2]+c[jj, 2])))
        c[:, 2] *= min(1.0, ratio)*(1.0-8e-16)
        xs = [Fraction(float(v)) for v in c[:, 0]]
        ys = [Fraction(float(v)) for v in c[:, 1]]
        rs = [Fraction(float(v)) for v in c[:, 2]]
        p = (max(x+r for x, r in zip(xs, rs))-min(x-r for x, r in zip(xs, rs))
             +max(y+r for y, r in zip(ys, rs))-min(y-r for y, r in zip(ys, rs)))
        c *= float(Fraction(2)/p)*(1.0-1.5e-15)
        for _ in range(3):
            if time.monotonic() >= deadline:
                return
            data = {'circles': c.tolist()}
            score = exact(data)
            if score > best_score:
                best, best_score = data, score
                return
            if np.isfinite(score):
                return
            c[:, 2] *= 1.0-2e-15

    # Find the reflection closest to the supplied packing.
    options = []
    for axis in (0, 1):
        if time.monotonic() >= deadline:
            return best
        reflected = a.copy()
        reflected[:, axis] *= -1
        cost = np.sum((a[:, None, :]-reflected[None, :, :])**2, axis=2)
        _, perm = linear_sum_assignment(cost)
        options.append((float(cost[np.arange(n), perm].sum()), axis, perm))
    error, axis, perm = min(options, key=lambda v: v[0])
    if axis == 0:
        a[:, [0, 1]] = a[:, [1, 0]]
        width = float(hi[1]-lo[1])

    # Reduced linear coordinates: pairs share x and radius, and have opposite y.
    columns = []
    seen = set()
    valid_symmetry = error < 1e-5 and all(int(perm[int(perm[k])]) == k for k in range(n))
    if valid_symmetry:
        for k in range(n):
            if time.monotonic() >= deadline:
                return best
            if k in seen:
                continue
            l = int(perm[k])
            seen.update((k, l))
            for component in (0, 1, 2):
                if k == l and component == 1:
                    continue
                v = np.zeros(64)
                v[3*k+component] = 1
                if l != k:
                    v[3*l+component] = -1 if component == 1 else 1
                columns.append(v)
    else:
        columns = [np.eye(64)[:, k] for k in range(63)]
    v = np.zeros(64)
    v[63] = 1
    columns.append(v)
    U = np.array(columns).T
    zfull = np.r_[a.ravel(), width]
    z = (U.T @ zfull) / np.sum(U*U, axis=0)
    m = len(z)
    X, Y, R = U[0:63:3], U[1:63:3], U[2:63:3]
    DX, DY, DR = X[ii]-X[jj], Y[ii]-Y[jj], R[ii]+R[jj]
    # Remove reflection-equivalent pair constraints.
    keep = []
    keys = set()
    for k in range(len(ii)):
        if time.monotonic() >= deadline:
            return best
        key = (tuple(np.outer(DX[k], DX[k]).ravel()), tuple(np.outer(DY[k], DY[k]).ravel()), tuple(DR[k]))
        if key not in keys:
            keys.add(key)
            keep.append(k)
    DX, DY, DR = DX[keep], DY[keep], DR[keep]
    W = U[63]
    B = np.concatenate((0.5*W+X-R, 0.5*W-X-R, -0.5*W+Y-R, -0.5*W-Y-R, R), axis=0)
    bc = np.r_[np.zeros(42), np.ones(42), np.full(21, -1e-6)]
    rows = np.c_[B, bc]
    _, unique = np.unique(rows, axis=0, return_index=True)
    B, bc = B[unique], bc[unique]
    objective = -np.sum(R, axis=0)

    class Expired(Exception):
        pass

    def check():
        if time.monotonic() >= deadline:
            raise Expired()

    def con(t):
        check()
        x, y = DX@t, DY@t
        return np.r_[np.sqrt(x*x+y*y)-DR@t, B@t+bc]

    def jac(t):
        check()
        x, y = DX@t, DY@t
        d = np.maximum(np.sqrt(x*x+y*y), 1e-30)
        return np.vstack((DX*(x/d)[:, None]+DY*(y/d)[:, None]-DR, B))

    def submit(t):
        c = (U@t)[:63].reshape(21, 3).copy()
        if axis == 0:
            c[:, [0, 1]] = c[:, [1, 0]]
        consider(c)

    def polish_contacts(t):
        t = t.copy()
        active = np.flatnonzero(con(t) < 2e-7)
        for _ in range(7):
            check()
            values = con(t)[active]
            if np.max(np.abs(values), initial=0) < 3e-16:
                break
            J = jac(t)[active]
            step = np.linalg.lstsq(J, -values, rcond=1e-12)[0]
            if np.linalg.norm(step) > 0.02:
                break
            t += step
        submit(t)
        return t

    try:
        z = polish_contacts(z)
        result = minimize(lambda t: float(objective@t), z, jac=lambda t: objective,
                          constraints={'type': 'ineq', 'fun': con, 'jac': jac},
                          method='SLSQP', options={'maxiter': int(params['iterations']), 'ftol': 2e-14})
        z = result.x
        submit(z)
        z = polish_contacts(z)
        anchor = z.copy()
        trial = 0
        while time.monotonic() < deadline:
            trial += 1
            t = anchor.copy()
            amplitude = float(params['perturbation'])*(1.0, 1.8, 3.0, 0.5)[trial % 4]
            chosen = rng.choice(m-1, size=min(m-1, 7+trial % 8), replace=False)
            t[chosen] += rng.normal(0, amplitude, len(chosen))
            t[-1] += rng.normal(0, amplitude*0.5)
            result = minimize(lambda q: float(objective@q), t, jac=lambda q: objective,
                              constraints={'type': 'ineq', 'fun': con, 'jac': jac},
                              method='SLSQP', options={'maxiter': int(params['iterations']), 'ftol': 3e-12})
            submit(result.x)
            if np.min(con(result.x)) > -1e-7:
                candidate = polish_contacts(result.x)
                if objective@candidate < objective@anchor-1e-11:
                    anchor = candidate
    except Expired:
        pass
    return best
