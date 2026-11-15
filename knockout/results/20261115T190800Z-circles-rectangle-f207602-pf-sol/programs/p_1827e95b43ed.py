import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"search_fraction": (0.69, 0.4, 0.72), "ftol": (2e-13, 1e-14, 1e-8, "log")}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + budget_s * params['search_fraction']
    rng = np.random.default_rng(seed)
    best = init
    best_score = -float('inf')
    ai, aj = np.triu_indices(21, 1)

    def exact(c):
        a = np.asarray(c, dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return -float('inf')
        d = np.sqrt(np.sum((a[ai, :2] - a[aj, :2])**2, axis=1))
        if np.any(d < a[ai, 2] + a[aj, 2]):
            return -float('inf')
        f = [[Fraction(float(v)) for v in row] for row in a]
        w = max(q[0]+q[2] for q in f)-min(q[0]-q[2] for q in f)
        h = max(q[1]+q[2] for q in f)-min(q[1]-q[2] for q in f)
        if w+h > 2:
            return -float('inf')
        return float(np.sum(a[:, 2]))

    if init is not None:
        try:
            best_score = exact(init['circles'])
        except (ValueError, KeyError, TypeError):
            pass

    def repair(c):
        c = np.array(c, dtype=float, copy=True)
        if not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return None
        d = np.sqrt(np.sum((c[ai, :2]-c[aj, :2])**2, axis=1))
        ratio = min(1.0, float(np.min(d/(c[ai, 2]+c[aj, 2]))))
        if ratio <= 0:
            return None
        c[:, 2] *= ratio * (1-1.2e-14)
        c[:, :2] -= np.mean(c[:, :2], axis=0)
        perimeter = np.sum(np.max(c[:, :2]+c[:, 2, None], axis=0)-np.min(c[:, :2]-c[:, 2, None], axis=0))
        c *= (2.0/perimeter)*(1-1.2e-14)
        return c

    def consider(c):
        nonlocal best, best_score
        c = repair(c)
        if c is None or float(np.sum(c[:, 2])) <= best_score:
            return
        s = exact(c)
        if s > best_score:
            best_score = s
            best = {'circles': c.tolist()}

    # A guaranteed valid fallback, independent of the supplied start.
    fallback = np.array([[(i % 5)*0.19, (i // 5)*0.19, 0.094] for i in range(21)])
    consider(fallback)

    def model(rows):
        center = rows[2]
        count = center + rows[3] + rows[4]
        nv = 2*center + 3*(count-center) + 1
        X = np.zeros((21, nv)); Y = X.copy(); R = X.copy()
        z = np.zeros(nv)
        mirror = np.arange(21)
        representatives = []
        index = 0; col = 0
        # Axis circles, then two rows above the axis and their mirrors.
        for k in range(center):
            X[index, col] = 1; R[index, col+1] = 1
            z[col] = 2*k-(center-1); z[col+1] = 0.94
            representatives.append(index)
            index += 1; col += 2
        for level, number in enumerate(rows[3:], 1):
            if time.monotonic() >= deadline:
                raise StopSearch
            for k in range(number):
                p, q = index, index+1
                X[p, col] = X[q, col] = 1
                Y[p, col+1] = 1; Y[q, col+1] = -1
                R[p, col+2] = R[q, col+2] = 1
                z[col] = 2*k-(number-1)
                z[col+1] = level*np.sqrt(3.0)
                z[col+2] = 0.94
                mirror[p] = q; mirror[q] = p
                representatives.append(p)
                index += 2; col += 3
        c = np.column_stack((X@z, Y@z, R@z))
        w = np.ptp(c[:, 0])+2
        h = np.ptp(c[:, 1])+2
        scale = 2/(w+h)
        z[:-1] *= scale
        z[-1] = w*scale
        pairs = []
        seen = set()
        for i, j in zip(ai, aj):
            if time.monotonic() >= deadline:
                raise StopSearch
            a = (int(i), int(j))
            b = tuple(sorted((int(mirror[i]), int(mirror[j]))))
            key = min(a, b)
            if key not in seen:
                seen.add(key); pairs.append(a)
        pi = np.array([p[0] for p in pairs]); pj = np.array([p[1] for p in pairs])
        DX = X[pi]-X[pj]; DY = Y[pi]-Y[pj]; RS = R[pi]+R[pj]
        walls = []; offsets = []
        for i in representatives:
            if time.monotonic() >= deadline:
                raise StopSearch
            a = X[i]-R[i]; a = a.copy(); a[-1] += .5
            walls.append(a); offsets.append(0.)
            a = -X[i]-R[i]; a = a.copy(); a[-1] += .5
            walls.append(a); offsets.append(0.)
            a = -Y[i]-R[i]; a = a.copy(); a[-1] -= .5
            walls.append(a); offsets.append(1.)
        W = np.asarray(walls); offsets = np.asarray(offsets)
        objective = -np.sum(R, axis=0)
        bounds = [(None, None)]*nv
        for k in range(nv-1):
            if time.monotonic() >= deadline:
                raise StopSearch
            if np.any(R[:, k]):
                bounds[k] = (1e-7, 1.)
            elif np.any(Y[:, k]):
                bounds[k] = (0., 1.)
        bounds[-1] = (.15, 1.85)
        def circles(v):
            return np.column_stack((X@v, Y@v, R@v))
        def fun(v):
            dx = DX@v; dy = DY@v
            return np.concatenate((np.hypot(dx, dy)-RS@v, W@v+offsets))
        def jac(v):
            dx = DX@v; dy = DY@v
            d = np.maximum(np.hypot(dx, dy), 1e-16)
            return np.vstack((DX*(dx/d)[:, None]+DY*(dy/d)[:, None]-RS, W))
        return z, circles, fun, jac, objective, bounds

    records = []
    families = [(4,5,3,5,4), (5,4,3,4,5), (4,4,5,4,4)]
    total = max(0., deadline-time.monotonic())
    def run(record, seconds):
        z, circles, fun, jac, objective, bounds = record['model']
        local_end = min(deadline, time.monotonic()+seconds)
        last = record['z'].copy()
        iteration = 0
        def callback(v):
            nonlocal last, iteration
            last = v.copy(); iteration += 1
            if time.monotonic() >= local_end:
                raise StopSearch
            if iteration % 6 == 0:
                consider(circles(v))
        try:
            result = minimize(lambda v: float(objective@v), last,
                              jac=lambda v: objective, method='SLSQP', bounds=bounds,
                              constraints={'type':'ineq', 'fun':fun, 'jac':jac},
                              callback=callback,
                              options={'maxiter':160, 'ftol':params['ftol'], 'disp':False})
            last = result.x
        except StopSearch:
            pass
        record['z'] = last
        repaired = repair(circles(last))
        record['score'] = -1. if repaired is None else float(np.sum(repaired[:,2]))
        consider(circles(last))

    for family in families:
        if time.monotonic() >= deadline:
            break
        try:
            m = model(family)
        except StopSearch:
            break
        rec = {'model':m, 'z':m[0], 'score':0.}
        records.append(rec)
        run(rec, total*.235)
    if records and time.monotonic() < deadline:
        rec = max(records, key=lambda q:q['score'])
        run(rec, max(0., deadline-time.monotonic()-.015))
    return best
