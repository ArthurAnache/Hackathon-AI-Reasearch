import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import lstsq

PARAMS = {
    'first_fraction': (0.28, 0.15, 0.4),
    'second_fraction': (0.25, 0.1, 0.35),
    'iterations': (220, 80, 400),
    'safety': (2e-14, 3e-15, 1e-11, 'log')
}
I, J = np.triu_indices(21, 1)

class Deadline(Exception):
    pass

def valid(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return False
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return False
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    return (max(u+v for u, v in zip(x, r))-min(u-v for u, v in zip(x, r))
            +max(u+v for u, v in zip(y, r))-min(u-v for u, v in zip(y, r))) <= 2

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    end = started + 0.70*budget_s
    eps = params['safety']
    best = None
    best_a = None
    score = -np.inf

    def offer(a):
        nonlocal best, best_a, score
        if time.monotonic() >= end:
            return
        a = np.array(a, dtype=float, copy=True)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return
        d = np.sqrt((a[I, 0]-a[J, 0])**2+(a[I, 1]-a[J, 1])**2)
        q = min(1.0, float(np.min(d/(a[I, 2]+a[J, 2]))))
        if q <= 0:
            return
        a[:, 2] *= q*(1-eps)
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        f = (2-12*eps)/float(np.sum(hi-lo))
        a[:, :2] = (a[:, :2]-(hi+lo)/2)*f
        a[:, 2] *= f
        s = float(np.sum(a[:, 2]))
        if s > score and valid(a):
            score = s
            best_a = a
            best = {'state': a.T.ravel().tolist()}

    try:
        if init is not None:
            a = np.array(init['circles'], float) if 'circles' in init else np.array(init['state'], float).reshape(3, 21).T
            if valid(a):
                best, best_a, score = init, a.copy(), float(np.sum(a[:, 2]))
            offer(a)
    except (ValueError, KeyError, TypeError):
        pass

    def lattice(counts):
        points = []
        for col, count in enumerate(counts):
            if time.monotonic() >= end:
                return None
            for k in range(count):
                if time.monotonic() >= end:
                    return None
                points.append([col*np.sqrt(3.), 2*k-count+1., 1.])
        a = np.array(points)
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        f = 2/np.sum(hi-lo)
        a[:, :2] = (a[:, :2]-(hi+lo)/2)*f
        a[:, 2] *= f
        return a

    seeds = []
    for counts in ((3, 4, 5, 4, 5), (3, 4, 3, 4, 3, 4)):
        if time.monotonic() >= end:
            break
        a = lattice(counts)
        if a is not None:
            offer(a)
            seeds.append(a)
    if best_a is None:
        a = np.array([[0.2*(i % 3), 0.2*(i//3), 0.09] for i in range(21)])
        offer(a)
    if time.monotonic() >= end-.025:
        return best

    def quotient(a):
        a = a.copy()
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        a[:, 0] -= lo[0]
        a[:, 1] -= (lo[1]+hi[1])/2
        singles = np.flatnonzero(abs(a[:, 1]) < 1e-8)
        positive = np.flatnonzero(a[:, 1] > 1e-8)
        if len(singles) != 3 or len(positive) != 9:
            return None
        return np.r_[a[singles, 0], a[positive, 0], a[positive, 1],
                     a[singles, 2], a[positive, 2], hi[0]-lo[0]]

    def model(symmetric):
        if symmetric:
            T = np.zeros((64, 34))
            ids = np.r_[np.arange(12), np.arange(3, 12)]
            T[np.arange(21), ids] = 1
            T[21+np.arange(3, 12), np.arange(12, 21)] = 1
            T[21+np.arange(12, 21), np.arange(12, 21)] = -1
            T[42+np.arange(21), 21+ids] = 1
            T[63, 33] = 1
            pairs = [(i, j) for i in range(12) for j in range(i+1, 12)]
            pairs += [(i, j+9) for i in range(3, 12) for j in range(i, 12)]
            pi, pj = np.array(pairs).T
            ids_wall = np.arange(12)
            bounds = [(None, None)]*34
            bounds[12:21] = [(0., None)]*9
            bounds[21:33] = [(1e-7, .7)]*12
        else:
            T = np.eye(64)
            pi, pj = I, J
            ids_wall = np.arange(21)
            bounds = [(None, None)]*64
            bounds[42:63] = [(1e-7, .7)]*21
        bounds[-1] = (.15, 1.85)
        X, Y, R, W = T[:21], T[21:42], T[42:63], T[63]
        dx, dy, rs = X[pi]-X[pj], Y[pi]-Y[pj], R[pi]+R[pj]
        xx, yy, rr = X[ids_wall], Y[ids_wall], R[ids_wall]
        walls = np.concatenate((xx-rr, W-xx-rr, -.5*W-yy-rr, -.5*W+yy-rr))
        off = np.r_[np.zeros(2*len(ids_wall)), np.ones(2*len(ids_wall))]
        objective = -R.sum(axis=0)
        def raw(z):
            u, v, r = dx@z, dy@z, rs@z
            c = np.r_[u*u+v*v-r*r, walls@z+off]
            jac = np.concatenate((2*(u[:, None]*dx+v[:, None]*dy-r[:, None]*rs), walls))
            return c, jac
        return T, bounds, objective, raw

    def solve(z, mod, stop):
        T, bounds, objective, raw = mod
        cache = [None, None, None]
        accepted = [z.copy()]
        def calc(v):
            if time.monotonic() >= stop:
                raise Deadline
            if cache[0] is None or not np.array_equal(v, cache[0]):
                c, j = raw(v)
                cache[:] = [v.copy(), c, j]
            return cache[1], cache[2]
        def fun(v):
            if time.monotonic() >= stop:
                raise Deadline
            return float(objective@v), objective
        def callback(v):
            if time.monotonic() >= stop:
                raise Deadline
            accepted[0] = v.copy()
        try:
            result = minimize(fun, z, jac=True, method='SLSQP', bounds=bounds,
                constraints={'type': 'ineq', 'fun': lambda v: calc(v)[0], 'jac': lambda v: calc(v)[1]},
                callback=callback,
                options={'maxiter': int(params['iterations']), 'ftol': 1e-13, 'disp': False})
            z = result.x
        except Deadline:
            z = accepted[0]
        offer((T@z)[:63].reshape(3, 21).T)
        for k in range(4):
            if time.monotonic() >= min(stop+.012, end-.015):
                break
            c, j = raw(z)
            active = np.flatnonzero(abs(c) < 2e-7)
            if len(active) < len(z)-1:
                break
            delta, _, rank, _ = lstsq(j[active], -c[active], lapack_driver='gelsy', check_finite=False)
            if rank < len(z)-1 or np.max(abs(delta)) > 2e-4:
                break
            z = z+delta
            offer((T@z)[:63].reshape(3, 21).T)
        return z

    sym = model(True)
    if score > 2.3 and not seeds:
        return best
    for k, a in enumerate(seeds):
        if time.monotonic() >= end-.08:
            break
        z = quotient(a)
        if z is None:
            continue
        fraction = params['first_fraction'] if k == 0 else params['second_fraction']
        stop = min(end-.13*budget_s, time.monotonic()+fraction*budget_s)
        if stop > time.monotonic()+.015:
            solve(z, sym, stop)
    if best_a is not None and time.monotonic() < end-.035:
        a = best_a
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        z = np.r_[a[:, 0]-lo[0], a[:, 1]-(lo[1]+hi[1])/2, a[:, 2], hi[0]-lo[0]]
        full = model(False)
        if time.monotonic() < end-.02:
            solve(z, full, end-.012)
    return best
