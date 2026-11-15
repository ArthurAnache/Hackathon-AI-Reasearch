import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"iterations": (135, 40, 250), "reserve": (0.70, 0.55, 0.75)}

class StopSearch(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + float(params['reserve']) * budget_s
    n = 21
    ii, jj = np.triu_indices(n, 1)
    best = None
    best_score = -float('inf')

    def exact(c):
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return -float('inf')
        dx = c[ii, 0] - c[jj, 0]
        dy = c[ii, 1] - c[jj, 1]
        if np.any(np.sqrt(dx*dx + dy*dy) < c[ii, 2] + c[jj, 2]):
            return -float('inf')
        xs = list(map(Fraction, map(float, c[:, 0])))
        ys = list(map(Fraction, map(float, c[:, 1])))
        rs = list(map(Fraction, map(float, c[:, 2])))
        width = max(x+r for x, r in zip(xs, rs)) - min(x-r for x, r in zip(xs, rs))
        height = max(y+r for y, r in zip(ys, rs)) - min(y-r for y, r in zip(ys, rs))
        if width + height > 2:
            return -float('inf')
        return float(np.sum(c[:, 2]))

    def keep(c, repair=True):
        nonlocal best, best_score
        c = np.array(c, dtype=float, copy=True)
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return
        if repair:
            c[:, :2] -= np.mean(c[:, :2], axis=0)
            dist = np.sqrt(np.sum((c[ii, :2]-c[jj, :2])**2, axis=1))
            ratio = min(1.0, float(np.min(dist / (c[ii, 2]+c[jj, 2]))))
            if ratio <= 0:
                return
            c[:, 2] *= ratio * (1.0-3e-14)
            extent = np.max(c[:, :2]+c[:, 2, None], axis=0)-np.min(c[:, :2]-c[:, 2, None], axis=0)
            c *= (2.0/float(np.sum(extent))) * (1.0-4e-15)
        score = exact(c)
        if score > best_score:
            best_score = score
            best = {'circles': c.tolist()}

    if init is not None:
        try:
            if 'circles' in init:
                original = np.asarray(init['circles'], dtype=float)
            else:
                original = np.asarray(init['state'], dtype=float).reshape(3, 21).T
            score = exact(original)
            if score > best_score:
                best_score, best = score, init
            keep(original)
        except (ValueError, KeyError, TypeError):
            pass

    def lattice(rows):
        groups = []
        circles = []
        for k, count in enumerate(rows):
            if time.monotonic() >= deadline:
                raise StopSearch
            yy = (k-(len(rows)-1)/2)*np.sqrt(3.0)
            if count % 2:
                groups.append((0.0, yy, False))
                circles.append((len(groups)-1, 0))
            for j in range(count//2):
                xx = 2*j + (1 if count % 2 == 0 else 2)
                groups.append((float(xx), yy, True))
                circles.extend([(len(groups)-1, 1), (len(groups)-1, -1)])
        ng = len(groups)
        npair = sum(g[2] for g in groups)
        dim = npair+2*ng+1
        T = np.zeros((63, dim))
        z = np.zeros(dim)
        xp = {}
        t = 0
        for g, (x, y, paired) in enumerate(groups):
            if time.monotonic() >= deadline:
                raise StopSearch
            if paired:
                xp[g] = t
                z[t] = x
                t += 1
            z[npair+g] = y
            z[npair+ng+g] = 1.0
        for k, (g, sign) in enumerate(circles):
            if time.monotonic() >= deadline:
                raise StopSearch
            if sign:
                T[k, xp[g]] = sign
            T[21+k, npair+g] = 1
            T[42+k, npair+ng+g] = 1
        c = (T@z).reshape(3, 21).T
        extent = np.max(c[:, :2]+c[:, 2, None], axis=0)-np.min(c[:, :2]-c[:, 2, None], axis=0)
        scale = 2/float(np.sum(extent))
        z *= scale
        z[-1] = extent[0]*scale
        bounds = [(0.0, 1.0)]*npair + [(-1.0, 1.0)]*ng + [(1e-7, 0.6)]*ng + [(0.05, 1.95)]
        return T, z, bounds

    def solve(T, z0, bounds, until, iterations):
        dim = len(z0)
        ax = T[ii]-T[jj]
        ay = T[21+ii]-T[21+jj]
        ar = T[42+ii]+T[42+jj]
        # Squared distances are invariant under independent sign changes.
        keys = np.concatenate((np.abs(ax), np.abs(ay), ar), axis=1)
        _, select = np.unique(keys, axis=0, return_index=True)
        ax, ay, ar = ax[select], ay[select], ar[select]
        walls = []
        constants = []
        for k in range(21):
            if time.monotonic() >= until:
                return z0
            for axis in range(2):
                for sign in (-1, 1):
                    a = sign*T[21*axis+k]-T[42+k]
                    a = a.copy()
                    a[-1] += 0.5 if axis == 0 else -0.5
                    walls.append(a)
                    constants.append(0.0 if axis == 0 else 1.0)
        walls = np.asarray(walls)
        constants = np.asarray(constants)
        _, unique = np.unique(np.column_stack((walls, constants)), axis=0, return_index=True)
        walls, constants = walls[unique], constants[unique]
        objective = -np.sum(T[42:63], axis=0)
        last = z0.copy()
        local_quality = -float('inf')
        local_best = z0.copy()
        calls = 0

        def constraints(z):
            if time.monotonic() >= until:
                raise StopSearch
            dx, dy, rr = ax@z, ay@z, ar@z
            return np.concatenate((dx*dx+dy*dy-rr*rr, walls@z+constants))

        def jacobian(z):
            dx, dy, rr = ax@z, ay@z, ar@z
            return np.vstack((2*(dx[:, None]*ax+dy[:, None]*ay-rr[:, None]*ar), walls))

        def callback(z):
            nonlocal last, local_quality, local_best, calls
            if time.monotonic() >= until:
                raise StopSearch
            last = z.copy()
            calls += 1
            c = (T@z).reshape(3, 21).T
            if np.min(c[:, 2]) <= 0:
                return
            d = np.sqrt(np.sum((c[ii, :2]-c[jj, :2])**2, axis=1))
            f = min(1.0, float(np.min(d/(c[ii, 2]+c[jj, 2]))))
            rr = c[:, 2]*f
            ext = np.max(c[:, :2]+rr[:, None], axis=0)-np.min(c[:, :2]-rr[:, None], axis=0)
            quality = float(2*np.sum(rr)/np.sum(ext))
            if quality > local_quality:
                local_quality, local_best = quality, z.copy()

        try:
            res = minimize(lambda z: float(objective@z), z0, jac=lambda z: objective,
                           bounds=bounds, constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                           method='SLSQP', callback=callback,
                           options={'maxiter': int(iterations), 'ftol': 2e-12, 'disp': False})
            last = res.x
            callback(last)
        except StopSearch:
            pass
        keep((T@local_best).reshape(3, 21).T)
        if time.monotonic() < deadline-0.005:
            keep((T@last).reshape(3, 21).T)
        return local_best

    families = [(3, 4, 3, 4, 3, 4), (5, 4, 5, 4, 3), (5, 4, 3, 4, 5)]
    records = []
    search_end = started + 0.55*budget_s
    for index, rows in enumerate(families):
        now = time.monotonic()
        if now >= search_end:
            break
        try:
            T, z, bounds = lattice(rows)
            keep((T@z).reshape(3, 21).T)
            until = min(search_end, now + (search_end-now)/(len(families)-index))
            z = solve(T, z, bounds, until, params['iterations'])
            records.append((float(-np.sum(T[42:63], axis=0)@z), T, z, bounds))
        except StopSearch:
            break
    if records and time.monotonic() < deadline-0.025:
        records.sort(key=lambda r: r[0])
        _, T, z, bounds = records[0]
        solve(T, z, bounds, deadline-0.008, 180)
    if best is None:
        # Guaranteed feasible fallback, independent of the supplied input.
        r = (1.0-1e-12)/26
        c = np.array([[2*r*(k%7-3), 2*r*(k//7-1), r] for k in range(21)])
        keep(c)
    return best
