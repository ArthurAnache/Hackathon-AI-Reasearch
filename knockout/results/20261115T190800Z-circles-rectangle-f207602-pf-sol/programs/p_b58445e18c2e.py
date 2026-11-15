import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"iterations": (180, 40, 350), "slack": (3e-13, 1e-14, 1e-9, "log")}


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -float('inf')
    ii, jj = np.triu_indices(21, 1)

    def score(data):
        try:
            c = np.asarray(data['circles'], dtype=float)
            if c.shape != (21, 3) or not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
                return -float('inf')
            d = c[ii, :2] - c[jj, :2]
            if np.any(np.sqrt(d[:, 0]**2 + d[:, 1]**2) < c[ii, 2] + c[jj, 2]):
                return -float('inf')
            x = [Fraction(float(v)) for v in c[:, 0]]
            y = [Fraction(float(v)) for v in c[:, 1]]
            r = [Fraction(float(v)) for v in c[:, 2]]
            wh = max(a+b for a,b in zip(x,r))-min(a-b for a,b in zip(x,r))
            wh += max(a+b for a,b in zip(y,r))-min(a-b for a,b in zip(y,r))
            if wh > 2:
                return -float('inf')
            return float(np.sum(c[:, 2]))
        except (ValueError, TypeError, KeyError):
            return -float('inf')

    best_score = score(init)

    def retain(c):
        nonlocal best, best_score
        if not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return
        c = c.copy()
        delta = c[ii, :2] - c[jj, :2]
        dist = np.sqrt(np.sum(delta*delta, axis=1))
        factor = min(1.0, float(np.min(dist/(c[ii, 2]+c[jj, 2]))))
        if factor <= 0:
            return
        c[:, 2] *= factor * (1-params['slack'])
        lo = np.min(c[:, :2]-c[:, 2,None], axis=0)
        hi = np.max(c[:, :2]+c[:, 2,None], axis=0)
        c[:, :2] -= (lo+hi)/2
        c *= (2/(hi-lo).sum())*(1-params['slack'])
        if c[:, 2].sum() <= best_score:
            return
        data = {'circles': c.tolist()}
        s = score(data)
        if s > best_score:
            best, best_score = data, s

    # Three axis circles, followed by nine independently positioned mirror pairs.
    X = np.zeros((21, 34))
    Y = np.zeros((21, 34))
    R = np.zeros((21, 34))
    for k in range(3):
        if time.monotonic() >= deadline:
            break
        X[k, 2*k] = 1
        R[k, 2*k+1] = 1
    for k in range(9):
        if time.monotonic() >= deadline:
            break
        a, b = 3+2*k, 4+2*k
        t = 6+3*k
        X[a,t] = X[b,t] = 1
        Y[a,t+1], Y[b,t+1] = 1, -1
        R[a,t+2] = R[b,t+2] = 1
    DX, DY, RR = X[ii]-X[jj], Y[ii]-Y[jj], R[ii]+R[jj]
    wall = np.concatenate((X-R, -X-R, -Y-R, Y-R), axis=0)
    wall[21:42,33] += 1
    wall[42:,33] -= .5
    offset = np.r_[np.zeros(42), np.ones(42)]
    objective = -R.sum(axis=0)
    lower = np.zeros(34)
    upper = np.full(34, 2.0)
    lower[1:6:2] = 1e-8
    lower[8:33:3] = 1e-8
    lower[33], upper[33] = .15, 1.85

    cache_z = None
    cache_c = None
    cache_j = None
    def calculate(z):
        nonlocal cache_z, cache_c, cache_j
        if time.monotonic() >= deadline:
            raise TimeoutError
        if cache_z is None or not np.array_equal(cache_z,z):
            dx, dy = DX@z, DY@z
            d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-16)
            cache_c = np.r_[d-RR@z, wall@z+offset]
            cache_j = np.vstack(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RR, wall))
            cache_z = z.copy()
        return cache_c, cache_j

    count = 0
    def callback(z):
        nonlocal count
        if time.monotonic() >= deadline:
            raise TimeoutError
        count += 1
        if count % 8 == 0:
            retain(np.column_stack((X@z,Y@z,R@z)))

    # The three starts differ substantially in aspect ratio and contact structure.
    layouts = [(5,4), (3,3,3), (4,5)]
    for trial, rows in enumerate(layouts):
        if time.monotonic() >= deadline - .035:
            break
        z = np.zeros(34)
        widest = max(3,max(rows))
        r = 1/(2*widest+2+2*len(rows)*np.sqrt(3))
        width = 2*widest*r
        z[33] = width
        z[:6:2] = np.linspace(r,width-r,3)
        z[1:6:2] = r
        p = 0
        for row, n in enumerate(reversed(rows)):
            if time.monotonic() >= deadline:
                break
            xs = (np.arange(n)-(n-1)/2)*2*r+width/2
            for x in xs:
                if time.monotonic() >= deadline:
                    break
                t = 6+3*p
                z[t:t+3] = [x,(row+1)*np.sqrt(3)*r,r]
                p += 1
        retain(np.column_stack((X@z,Y@z,R@z)))
        try:
            result = minimize(lambda v: objective@v, z, jac=lambda v: objective,
                bounds=list(zip(lower,upper)),
                constraints={'type':'ineq','fun':lambda v:calculate(v)[0],
                             'jac':lambda v:calculate(v)[1]},
                method='SLSQP', callback=callback,
                options={'maxiter':int(params['iterations']), 'ftol':2e-11, 'disp':False})
            retain(np.column_stack((X@result.x,Y@result.x,R@result.x)))
        except TimeoutError:
            break
    if best_score == -float('inf'):
        # A conservative valid fallback requiring no optimizer success.
        c = np.array([[.08+.16*(k%7), .08+.16*(k//7), .079] for k in range(21)])
        retain(c)
    return best
