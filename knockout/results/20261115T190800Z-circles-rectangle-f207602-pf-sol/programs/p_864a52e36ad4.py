import time
import numpy as np
from scipy.optimize import minimize
from fractions import Fraction

PARAMS = {
    'iterations': (160, 60, 240),
    'restart_noise': (0.026, 0.001, 0.1),
    'time_fraction': (0.70, 0.5, 0.75)
}

class Deadline(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + float(params['time_fraction']) * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    best = init
    best_score = -float('inf')

    def exact_score(c):
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return -float('inf')
        d = c[ii, :2] - c[jj, :2]
        distances = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
        if np.any(distances < c[ii, 2] + c[jj, 2]):
            return -float('inf')
        xs = [Fraction(float(x)) for x in c[:, 0]]
        ys = [Fraction(float(y)) for y in c[:, 1]]
        rs = [Fraction(float(r)) for r in c[:, 2]]
        width = max(x+r for x, r in zip(xs, rs)) - min(x-r for x, r in zip(xs, rs))
        height = max(y+r for y, r in zip(ys, rs)) - min(y-r for y, r in zip(ys, rs))
        if width + height > 2:
            return -float('inf')
        return float(np.sum(c[:, 2]))

    if init is not None:
        try:
            if 'circles' in init:
                old = np.asarray(init['circles'], dtype=float)
            else:
                old = np.asarray(init['state'], dtype=float).reshape(3, 21).T
            best_score = exact_score(old)
        except Exception:
            pass

    def offer(c):
        nonlocal best, best_score
        c = np.asarray(c, dtype=float).copy()
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return
        delta = c[ii, :2] - c[jj, :2]
        dist = np.sqrt(np.sum(delta * delta, axis=1))
        ratio = float(np.min(dist / (c[ii, 2] + c[jj, 2])))
        c[:, 2] *= min(1.0, ratio) * (1.0 - 3e-15)
        if np.min(c[:, 2]) <= 0:
            return
        lo = np.min(c[:, :2] - c[:, 2, None], axis=0)
        hi = np.max(c[:, :2] + c[:, 2, None], axis=0)
        extent = float(np.sum(hi-lo))
        if extent <= 0:
            return
        c[:, :2] -= (lo+hi)*0.5
        c *= (2.0 / extent) * (1.0 - 5e-15)
        if float(np.sum(c[:, 2])) <= best_score:
            return
        score = exact_score(c)
        if score > best_score:
            best_score = score
            best = {'circles': c.tolist()}

    # A deterministic feasible fallback, independent of the supplied start.
    fallback = np.array([[(k % 7 - 3)*0.2, (k // 7 - 1)*0.2, 0.1] for k in range(21)])
    offer(fallback)

    def model(rows):
        groups = []
        entries = []
        reflect = []
        radius = 1.0 / (10.0 + 5.0*np.sqrt(3.0))
        for row, count in enumerate(rows):
            if time.monotonic() >= deadline:
                raise Deadline
            y = (row - 2.5)*np.sqrt(3.0)*radius
            if count == 3:
                g = len(groups)
                groups.append((0.0, y, True))
                k = len(entries)
                entries.append((g, 0.0))
                reflect.append(k)
                xs = [2.0*radius]
            else:
                xs = [radius, 3.0*radius]
            for x in xs:
                if time.monotonic() >= deadline:
                    raise Deadline
                g = len(groups)
                groups.append((x, y, False))
                k = len(entries)
                entries.extend([(g, 1.0), (g, -1.0)])
                reflect.extend([k+1, k])
        ng = len(groups)
        pairs = [g for g in range(ng) if not groups[g][2]]
        nx = len(pairs)
        dim = nx + 2*ng + 1
        xid = {g: k for k, g in enumerate(pairs)}
        X = np.zeros((21, dim))
        Y = np.zeros((21, dim))
        R = np.zeros((21, dim))
        v = np.zeros(dim)
        for k, (g, sign) in enumerate(entries):
            if time.monotonic() >= deadline:
                raise Deadline
            if sign:
                X[k, xid[g]] = sign
            Y[k, nx+g] = 1.0
            R[k, nx+ng+g] = 1.0
        for g, (x, y, axis) in enumerate(groups):
            if time.monotonic() >= deadline:
                raise Deadline
            if not axis:
                v[xid[g]] = x
            v[nx+g] = y
            v[nx+ng+g] = radius
        v[-1] = 8.0*radius
        pi, pj = [], []
        for a, b in zip(ii, jj):
            if time.monotonic() >= deadline:
                raise Deadline
            reflected = tuple(sorted((reflect[a], reflect[b])))
            if (int(a), int(b)) <= reflected:
                pi.append(a)
                pj.append(b)
        pi = np.asarray(pi)
        pj = np.asarray(pj)
        DX = X[pi]-X[pj]
        DY = Y[pi]-Y[pj]
        RR = R[pi]+R[pj]
        walls = []
        constants = []
        for g, (x, y, axis) in enumerate(groups):
            if time.monotonic() >= deadline:
                raise Deadline
            wall = np.zeros(dim)
            wall[-1] = 0.5
            wall[nx+ng+g] = -1.0
            if not axis:
                wall[xid[g]] = -1.0
            walls.append(wall)
            constants.append(0.0)
            for sign in (-1.0, 1.0):
                wall = np.zeros(dim)
                wall[-1] = -0.5
                wall[nx+ng+g] = -1.0
                wall[nx+g] = sign
                walls.append(wall)
                constants.append(1.0)
        W = np.asarray(walls)
        C = np.asarray(constants)
        objective = -np.sum(R, axis=0)
        bounds = [(1e-6, 1.0)]*nx + [(-1.0, 1.0)]*ng + [(1e-6, 0.5)]*ng + [(0.1, 1.9)]
        return v, X, Y, R, DX, DY, RR, W, C, objective, bounds

    def run(m, v, stop, iterations):
        _, X, Y, R, DX, DY, RR, W, C, objective, bounds = m
        local_best = [v.copy(), -float('inf')]

        def check():
            if time.monotonic() >= stop:
                raise Deadline

        def fun(z):
            check()
            return float(objective @ z)

        def cons(z):
            check()
            dx, dy, rr = DX@z, DY@z, RR@z
            return np.concatenate((dx*dx + dy*dy - rr*rr, W@z+C))

        def jac(z):
            check()
            dx, dy, rr = DX@z, DY@z, RR@z
            return np.vstack((2.0*(dx[:, None]*DX + dy[:, None]*DY - rr[:, None]*RR), W))

        def callback(z):
            check()
            c = np.column_stack((X@z, Y@z, R@z))
            old_score = best_score
            offer(c)
            if best_score > old_score:
                local_best[0] = z.copy()
                local_best[1] = best_score

        offer(np.column_stack((X@v, Y@v, R@v)))
        try:
            result = minimize(fun, v, jac=lambda z: objective, bounds=bounds,
                              constraints={'type': 'ineq', 'fun': cons, 'jac': jac},
                              callback=callback, method='SLSQP',
                              options={'maxiter': iterations, 'ftol': 2e-12, 'disp': False})
            if time.monotonic() < deadline:
                callback(result.x)
            return result.x
        except Deadline:
            return local_best[0]

    try:
        m = model([3, 4, 3, 4, 3, 4])
        remaining = deadline-time.monotonic()
        v = run(m, m[0], time.monotonic()+remaining*0.58, int(params['iterations']))
        trials = 0
        while time.monotonic() < deadline-0.035:
            trials += 1
            if trials == 1:
                alt = model([3, 4, 4, 3, 4, 3])
                remaining = deadline-time.monotonic()
                run(alt, alt[0], time.monotonic()+remaining*0.60, int(params['iterations']))
            else:
                trial = v.copy()
                trial[:-1] += rng.normal(0.0, float(params['restart_noise']), len(trial)-1)
                trial[21:33] = np.maximum(trial[21:33], 0.025)
                remaining = deadline-time.monotonic()
                v = run(m, trial, min(deadline, time.monotonic()+max(0.07, remaining*0.8)), 85)
    except Deadline:
        pass
    return best
