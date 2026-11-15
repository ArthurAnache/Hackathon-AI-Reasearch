import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'excursion': (0.075, 0.005, 0.25),
    'modes': (7, 2, 15),
    'iterations': (90, 20, 200),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class Deadline(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -float('inf')

    def exact(a):
        if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
        if np.any(d < a[I, 2]+a[J, 2]):
            return -float('inf')
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        per = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        per += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        return float(np.sum(a[:, 2])) if per <= 2 else -float('inf')

    if init is not None:
        a0 = np.asarray(init['circles'], dtype=float)
        best_score = exact(a0)
    else:
        a0 = np.array([[(k % 5)*0.20+0.10, (k//5)*0.20+0.10, 0.04] for k in range(N)])
        best = {'circles': a0.tolist()}
        best_score = exact(a0)

    lo = np.min(a0[:, :2]-a0[:, 2, None], axis=0)
    sides = np.max(a0[:, :2]+a0[:, 2, None], axis=0)-lo
    scale = 2.0/np.sum(sides)
    p = (a0[:, :2]-lo)*scale
    r = a0[:, 2]*scale
    z0 = np.r_[p[:, 0], p[:, 1], r, sides[0]*scale]

    def constraints(z, jac=False):
        x, y, r, w = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
        dx, dy = x[I]-x[J], y[I]-y[J]
        d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-100)
        c = np.r_[d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r]
        if not jac:
            return c
        a = np.zeros((M+4*N, 3*N+1))
        q = np.arange(M)
        a[q,I] = dx/d
        a[q,J] = -dx/d
        a[q,N+I] = dy/d
        a[q,N+J] = -dy/d
        a[q,2*N+I] = -1
        a[q,2*N+J] = -1
        t = np.arange(N)
        a[M+t,t] = 1
        a[M+N+t,t] = -1
        a[M+N+t,-1] = 1
        a[M+2*N+t,N+t] = 1
        a[M+3*N+t,N+t] = -1
        a[M+3*N+t,-1] = -1
        a[M+t,2*N+t] = -1
        a[M+N+t,2*N+t] = -1
        a[M+2*N+t,2*N+t] = -1
        a[M+3*N+t,2*N+t] = -1
        return a

    def consider(z):
        nonlocal best, best_score
        if time.monotonic() >= end:
            return
        p = np.column_stack((z[:N], z[N:2*N]))
        rr = z[2*N:3*N].copy()
        if not np.isfinite(z).all() or np.any(rr <= 0):
            return
        d = np.sqrt(np.sum((p[I]-p[J])**2, axis=1))
        ratio = min(1.0, float(np.min(d/(rr[I]+rr[J]))))
        rr *= ratio*(1-4e-15)
        if np.any(rr <= 0):
            return
        lower = np.min(p-rr[:,None], axis=0)
        upper = np.max(p+rr[:,None], axis=0)
        s = (2.0/(upper-lower).sum())*(1-4e-15)
        a = np.column_stack(((p-lower)*s, rr*s))
        value = exact(a)
        if value > best_score:
            best_score = value
            best = {'circles': a.tolist()}

    gradient = np.r_[np.zeros(2*N), -np.ones(N), 0.0]
    bounds = [(None,None)]*(2*N)+[(1e-7,0.8)]*N+[(0.15,1.85)]
    latest = z0.copy()
    calls = 0

    def objective(z):
        if time.monotonic() >= end:
            raise Deadline()
        return -np.sum(z[2*N:3*N])

    def callback(z):
        nonlocal latest, calls
        latest = z.copy()
        calls += 1
        if time.monotonic() >= end:
            raise Deadline()
        if calls % 12 == 0:
            consider(z)

    def relax(z, iterations):
        nonlocal latest
        latest = z.copy()
        try:
            result = minimize(objective, z, jac=lambda z: gradient, method='SLSQP',
                bounds=bounds,
                constraints={'type':'ineq', 'fun':constraints, 'jac':lambda z: constraints(z, True)},
                callback=callback,
                options={'maxiter':iterations, 'ftol':2e-12, 'disp':False})
            latest = result.x
        except Deadline:
            return latest
        consider(latest)
        return latest.copy()

    # At a rigid optimum these modes encode the least expensive collective
    # contact changes, rather than independently moving individual circles.
    if time.monotonic() < end:
        c = constraints(z0)
        active = constraints(z0, True)[c < 2e-6]
        try:
            _, singular, vt = np.linalg.svd(active, full_matrices=True)
            modes = vt[-int(params['modes']):]
        except np.linalg.LinAlgError:
            modes = rng.normal(size=(int(params['modes']), 3*N+1))
        order = rng.permutation(len(modes))
        attempt = 0
        while time.monotonic() < end:
            k = int(order[(attempt//2) % len(order)])
            direction = modes[k].copy()
            sign = 1.0 if attempt % 2 == 0 else -1.0
            direction /= max(np.max(np.abs(direction[:2*N])), 1e-9)
            amplitude = float(params['excursion']) * (1.0 if attempt < 2*len(order) else 1.8)
            trial = z0 + sign*amplitude*direction
            trial[2*N:3*N] = np.maximum(trial[2*N:3*N], 0.004)
            trial[-1] = np.clip(trial[-1], 0.2, 1.8)
            relax(trial, int(params['iterations']))
            attempt += 1
    return best
