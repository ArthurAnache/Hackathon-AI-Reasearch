import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'excursion': (0.22, 0.05, 0.5),
    'fixed_iterations': (36, 10, 100),
    'free_iterations': (70, 20, 150),
    'branches': (6, 2, 12),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class StopSearch(Exception):
    pass

def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0] - a[J, 0])**2 + (a[I, 1] - a[J, 1])**2)
    if np.any(d < a[I, 2] + a[J, 2]):
        return -np.inf
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    perimeter = (max(u+v for u,v in zip(x,r)) - min(u-v for u,v in zip(x,r))
                 + max(u+v for u,v in zip(y,r)) - min(u-v for u,v in zip(y,r)))
    if perimeter > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))

def unpack(z):
    return np.column_stack((z[:N], z[N:2*N], z[2*N:3*N]))

def pack(a):
    lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
    side = np.max(a[:, :2] + a[:, 2, None], axis=0) - lo
    s = 2.0 / np.sum(side)
    p = (a[:, :2] - lo) * s
    return np.r_[p[:, 0], p[:, 1], a[:, 2]*s, side[0]*s]

def repair(z, deadline):
    a = unpack(z).copy()
    if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return None
    d = np.sqrt((a[I,0]-a[J,0])**2 + (a[I,1]-a[J,1])**2)
    ratio = float(np.min(d / (a[I,2] + a[J,2])))
    if ratio <= 0:
        return None
    a[:,2] *= min(1.0, ratio) * (1.0 - 3e-14)
    lo = np.min(a[:,:2]-a[:,2,None], axis=0)
    side = np.max(a[:,:2]+a[:,2,None], axis=0)-lo
    scale = (2.0 / np.sum(side)) * (1.0-3e-14)
    a[:,:2] = (a[:,:2]-lo)*scale
    a[:,2] *= scale
    for k in range(4):
        if time.monotonic() >= deadline:
            return None
        if exact_score(a) > -np.inf:
            return a
        a[:,2] *= 1.0-2e-13
    return None

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    solve_end = deadline - min(0.04, 0.04*budget_s)
    rng = np.random.default_rng(seed)
    if init is not None:
        original = np.asarray(init['circles'], dtype=float)
        best = init
        best_score = exact_score(original)
    else:
        original = np.array([[0.1+0.2*(k%7), 0.1+0.2*(k//7), 0.1*(1-1e-12)] for k in range(N)])
        best = {'circles': original.tolist()}
        best_score = exact_score(original)
    base = pack(original)
    grad = np.r_[np.zeros(2*N), -np.ones(N), 0.0]
    cached_z = None
    cached_c = None
    cached_j = None
    latest = base.copy()

    def evaluate(z):
        nonlocal cached_z, cached_c, cached_j
        if time.monotonic() >= solve_end:
            raise StopSearch()
        if cached_z is not None and np.array_equal(z, cached_z):
            return cached_c, cached_j
        x, y, r, w = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
        dx, dy = x[I]-x[J], y[I]-y[J]
        d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-16)
        c = np.r_[d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r]
        jac = np.zeros((M+4*N, 3*N+1))
        rows = np.arange(M)
        jac[rows,I] = dx/d
        jac[rows,J] = -dx/d
        jac[rows,N+I] = dy/d
        jac[rows,N+J] = -dy/d
        jac[rows,2*N+I] = -1
        jac[rows,2*N+J] = -1
        q = np.arange(N)
        for b, col, sign in ((0,0,1),(1,0,-1),(2,N,1),(3,N,-1)):
            if time.monotonic() >= solve_end:
                raise StopSearch()
            rows = M+b*N+q
            jac[rows,col+q] = sign
            jac[rows,2*N+q] = -1
        jac[M+N:M+2*N,-1] = 1
        jac[M+3*N:M+4*N,-1] = -1
        cached_z, cached_c, cached_j = z.copy(), c, jac
        return c,jac

    def callback(z):
        nonlocal latest
        latest = z.copy()
        if time.monotonic() >= solve_end:
            raise StopSearch()

    def consider(z):
        nonlocal best, best_score
        a = repair(z, deadline)
        if a is not None:
            score = exact_score(a)
            if score > best_score:
                best_score = score
                best = {'circles': a.tolist()}

    bounds = [(0.0,2.0)]*(2*N)+[(1e-7,0.5)]*N+[(0.08,1.92)]
    cons = {'type':'ineq','fun':lambda z:evaluate(z)[0], 'jac':lambda z:evaluate(z)[1]}
    amplitudes = [1.0,-1.0,1.7,-1.7,0.55,-0.55]
    for branch in range(int(params['branches'])):
        if time.monotonic() >= solve_end:
            break
        # Every branch starts from the verified best, not an accumulated infeasible iterate.
        anchor = pack(np.asarray(best['circles'], dtype=float))
        width = anchor[-1]
        amp = amplitudes[branch % len(amplitudes)]
        target = float(np.clip(width + amp*float(params['excursion']), 0.25, 1.75))
        sx, sy = target/width, (2-target)/(2-width)
        z = anchor.copy()
        z[:N] *= sx
        z[N:2*N] *= sy
        z[2*N:3*N] *= min(sx,sy)
        z[-1] = target
        if branch >= 4:
            z[:2*N] += rng.normal(0,0.003,2*N)
        latest = z.copy()
        fixed_bounds = bounds[:-1]+[(target,target)]
        try:
            result = minimize(lambda v:float(grad@v), z, jac=lambda v:grad,
                              constraints=cons, bounds=fixed_bounds, method='SLSQP',
                              callback=callback,
                              options={'maxiter':int(params['fixed_iterations']), 'ftol':2e-10, 'disp':False})
            z = result.x
            consider(z)
            if time.monotonic() >= solve_end:
                break
            result = minimize(lambda v:float(grad@v), z, jac=lambda v:grad,
                              constraints=cons, bounds=bounds, method='SLSQP',
                              callback=callback,
                              options={'maxiter':int(params['free_iterations']), 'ftol':2e-13, 'disp':False})
            consider(result.x)
        except StopSearch:
            consider(latest)
            break
        except (ValueError, FloatingPointError):
            if time.monotonic() >= solve_end:
                break
    return best
