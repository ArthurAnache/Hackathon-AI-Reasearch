import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {'symmetric_fraction': (0.24, 0.15, 0.35), 'warm_fraction': (0.27, 0.15, 0.35), 'ftol': (1e-11, 1e-13, 1e-8, 'log')}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    finish = start + 0.69 * budget_s
    search_end = start + 0.675 * budget_s
    ii, jj = np.triu_indices(21, 1)
    rows = np.arange(len(ii))
    identity = np.eye(21)
    best = init
    best_score = -float('inf')

    def exact(data):
        if time.monotonic() >= finish:
            return -float('inf')
        try:
            a = np.asarray(data['circles'], dtype=float)
            if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
                return -float('inf')
            xs, ys, rs = [], [], []
            for x, y, r in a:
                if time.monotonic() >= finish:
                    return -float('inf')
                xs.append(Fraction(float(x)))
                ys.append(Fraction(float(y)))
                rs.append(Fraction(float(r)))
            xr = [x+r for x, r in zip(xs, rs)]
            xl = [x-r for x, r in zip(xs, rs)]
            yt = [y+r for y, r in zip(ys, rs)]
            yb = [y-r for y, r in zip(ys, rs)]
            if max(xr)-min(xl)+max(yt)-min(yb) > 2:
                return -float('inf')
            for k, l in zip(ii, jj):
                if time.monotonic() >= finish:
                    return -float('inf')
                c, d = a[k], a[l]
                dist = np.sqrt((c[0]-d[0])**2 + (c[1]-d[1])**2)
                if dist < c[2]+d[2]:
                    return -float('inf')
            return float(np.sum(a[:, 2]))
        except (ValueError, TypeError, KeyError, OverflowError):
            return -float('inf')

    if init is not None:
        best_score = exact(init)

    def accept(a):
        nonlocal best, best_score
        if time.monotonic() >= search_end:
            return
        a = np.array(a, dtype=float, copy=True)
        if a.shape != (21, 3) or not np.isfinite(a).all():
            return
        a[:, 2] = np.maximum(a[:, 2], 1e-10)
        dx = a[ii, 0]-a[jj, 0]
        dy = a[ii, 1]-a[jj, 1]
        dist = np.sqrt(dx*dx+dy*dy)
        ratio = min(1.0, float(np.min(dist/(a[ii, 2]+a[jj, 2]))))
        if ratio <= 0:
            return
        a[:, 2] *= ratio * (1.0-3e-14)
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        a[:, :2] -= (hi+lo)/2
        perimeter = float(np.sum(hi-lo))
        if perimeter <= 0:
            return
        a *= (2.0/perimeter)*(1.0-3e-14)
        if float(np.sum(a[:, 2])) <= best_score:
            return
        candidate = {'circles': a.tolist()}
        score = exact(candidate)
        if score > best_score:
            best, best_score = candidate, score

    grid = np.zeros((21, 3))
    grid[:, 0] = np.tile(np.arange(7), 3)*0.2
    grid[:, 1] = np.repeat(np.arange(3), 7)*0.2
    grid[:, 2] = 0.1
    accept(grid)

    def encode(a):
        a = np.array(a, dtype=float, copy=True)
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        a[:, :2] -= (hi+lo)/2
        scale = min(1.0, 1.999999999/float(np.sum(hi-lo)))
        a *= scale
        w = 1.0 + 0.5*scale*((hi[0]-lo[0])-(hi[1]-lo[1]))
        return np.r_[a[:, 0], a[:, 1], a[:, 2], w]

    def expand(z):
        v = np.zeros(64)
        v[:3] = z[:3]
        v[42:45] = z[3:6]
        v[3:12] = z[6:15]
        v[12:21] = z[6:15]
        v[24:33] = z[15:24]
        v[33:42] = -z[15:24]
        v[45:54] = z[24:33]
        v[54:63] = z[24:33]
        v[63] = z[33]
        return v

    def reduce_jac(j):
        return np.concatenate((j[:, :3], j[:, 42:45], j[:, 3:12]+j[:, 12:21], j[:, 24:33]-j[:, 33:42], j[:, 45:54]+j[:, 54:63], j[:, 63:64]), axis=1)

    boundary_jac = np.zeros((84, 64))
    boundary_jac[:21, :21] = identity
    boundary_jac[21:42, :21] = -identity
    boundary_jac[42:63, 21:42] = identity
    boundary_jac[63:84, 21:42] = -identity
    boundary_jac[:21, 42:63] = -identity
    boundary_jac[21:42, 42:63] = -identity
    boundary_jac[42:63, 42:63] = -identity
    boundary_jac[63:84, 42:63] = -identity
    boundary_jac[:42, 63] = 0.5
    boundary_jac[42:, 63] = -0.5

    def optimize(z, symmetric, deadline):
        deadline = min(deadline, search_end)
        gradient = np.zeros(len(z))
        if symmetric:
            gradient[3:6] = -1
            gradient[24:33] = -2
            bounds = [(-1, 1)]*3 + [(1e-7, 0.6)]*3 + [(-1, 1)]*9 + [(1e-7, 1)]*9 + [(1e-7, 0.6)]*9 + [(0.15, 1.85)]
        else:
            gradient[42:63] = -1
            bounds = [(-1, 1)]*42 + [(1e-7, 0.6)]*21 + [(0.15, 1.85)]
        cache_z, cache_c, cache_j = None, None, None

        def check():
            if time.monotonic() >= deadline:
                raise StopSearch()

        def objective(t):
            check()
            return float(gradient @ t)

        def constraints(t):
            nonlocal cache_z, cache_c, cache_j
            check()
            if cache_z is not None and np.array_equal(t, cache_z):
                return cache_c
            v = expand(t) if symmetric else t
            x, y, r, w = v[:21], v[21:42], v[42:63], v[63]
            dx, dy = x[ii]-x[jj], y[ii]-y[jj]
            d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-20)
            c = np.r_[d-r[ii]-r[jj], w/2+x-r, w/2-x-r, 1-w/2+y-r, 1-w/2-y-r]
            j = np.zeros((294, 64))
            j[rows, ii] = dx/d
            j[rows, jj] = -dx/d
            j[rows, 21+ii] = dy/d
            j[rows, 21+jj] = -dy/d
            j[rows, 42+ii] = -1
            j[rows, 42+jj] = -1
            j[210:] = boundary_jac
            cache_z = t.copy()
            cache_c = c
            cache_j = reduce_jac(j) if symmetric else j
            return c

        def jacobian(t):
            constraints(t)
            return cache_j

        def callback(t):
            check()
            v = expand(t) if symmetric else t
            accept(np.column_stack((v[:21], v[21:42], v[42:63])))

        try:
            result = minimize(objective, z, jac=lambda t: gradient, bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                callback=callback, method='SLSQP',
                options={'maxiter': 400, 'ftol': params['ftol'], 'disp': False})
            callback(result.x)
        except (StopSearch, ValueError, FloatingPointError):
            pass

    scale = 2.0/(6.0+2.0*np.sqrt(3.0))
    z = np.zeros(34)
    z[:3] = np.array([-2., 0., 2.])*scale
    z[3:6] = 0.5*scale
    z[6:15] = np.r_[[-1.5, -0.5, 0.5, 1.5], [-2., -1., 0., 1., 2.]]*scale
    z[15:24] = np.r_[np.full(4, np.sqrt(3)/2), np.full(5, np.sqrt(3))]*scale
    z[24:33] = 0.5*scale
    z[33] = 5*scale
    optimize(z, True, start+params['symmetric_fraction']*budget_s)

    if time.monotonic() < search_end:
        try:
            warm = np.asarray(init['circles'], dtype=float) if init is not None else grid
            if warm.shape != (21, 3) or not np.isfinite(warm).all():
                warm = grid
            optimize(encode(warm), False, start+(params['symmetric_fraction']+params['warm_fraction'])*budget_s)
        except (ValueError, TypeError, KeyError):
            pass
    if best is not None and time.monotonic() < search_end:
        optimize(encode(best['circles']), False, search_end)
    return best
