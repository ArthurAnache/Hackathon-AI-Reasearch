import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    "exchanges": (3, 2, 7),
    "fixed_fraction": (0.23, 0.1, 0.4),
    "penalty": (180.0, 30.0, 1500.0, "log"),
    "attempts": (3, 1, 8),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
IDS = np.arange(N)

class Timeout(Exception):
    pass


def score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    dx = a[I, 0] - a[J, 0]
    dy = a[I, 1] - a[J, 1]
    if np.any(np.sqrt(dx*dx + dy*dy) < a[I, 2] + a[J, 2]):
        return -np.inf
    x = list(map(Fraction, map(float, a[:, 0])))
    y = list(map(Fraction, map(float, a[:, 1])))
    r = list(map(Fraction, map(float, a[:, 2])))
    perimeter = (max(u+v for u,v in zip(x,r)) - min(u-v for u,v in zip(x,r))
                 + max(u+v for u,v in zip(y,r)) - min(u-v for u,v in zip(y,r)))
    if perimeter > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a = np.zeros((N, 3))
        a[:, 0] = 0.1 + 0.18 * (IDS % 5)
        a[:, 1] = 0.1 + 0.18 * (IDS // 5)
        a[:, 2] = 0.075
        best = {"circles": a.tolist()}
    else:
        a = np.asarray(init['circles'], dtype=float).copy()
        best = init
    best_score = score(a)
    lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
    sides = np.max(a[:, :2] + a[:, 2, None], axis=0) - lo
    scale = 2.0 / np.sum(sides)
    a[:, :2] = (a[:, :2] - lo) * scale
    a[:, 2] *= scale
    base = np.r_[a[:, 0], a[:, 1], a[:, 2], sides[0]*scale]
    objective_gradient = np.r_[np.zeros(2*N), -np.ones(N), 0.0]

    def save(z):
        nonlocal best, best_score
        if time.monotonic() >= end:
            return
        p = np.column_stack((z[:N], z[N:2*N]))
        r = z[2*N:3*N].copy()
        if not np.isfinite(z).all() or np.min(r) <= 0:
            return
        d = np.sqrt(np.sum((p[I]-p[J])**2, axis=1))
        ratio = min(1.0, float(np.min(d / (r[I]+r[J]))))
        r *= max(0.0, ratio) * (1.0 - 3e-14)
        low = np.min(p-r[:, None], axis=0)
        high = np.max(p+r[:, None], axis=0)
        factor = min(1.0, 2.0 / float(np.sum(high-low))) * (1.0-3e-14)
        candidate = np.column_stack(((p-low)*factor, r*factor))
        if float(np.sum(candidate[:, 2])) <= best_score:
            return
        value = score(candidate)
        if value > best_score:
            best_score = value
            best = {"circles": candidate.tolist()}

    def constraints(z, jac=False):
        x, y, r, w = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
        dx, dy = x[I]-x[J], y[I]-y[J]
        d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-15)
        if not jac:
            return np.r_[d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r]
        A = np.zeros((M+4*N, 3*N+1))
        row = np.arange(M)
        A[row,I] = dx/d
        A[row,J] = -dx/d
        A[row,N+I] = dy/d
        A[row,N+J] = -dy/d
        A[row,2*N+I] = -1
        A[row,2*N+J] = -1
        A[M+IDS,IDS] = 1
        A[M+N+IDS,IDS] = -1
        A[M+2*N+IDS,N+IDS] = 1
        A[M+3*N+IDS,N+IDS] = -1
        A[M+IDS,2*N+IDS] = -1
        A[M+N+IDS,2*N+IDS] = -1
        A[M+2*N+IDS,2*N+IDS] = -1
        A[M+3*N+IDS,2*N+IDS] = -1
        A[M+N+IDS,-1] = 1
        A[M+3*N+IDS,-1] = -1
        return A

    attempts = int(params['attempts'])
    for attempt in range(attempts):
        now = time.monotonic()
        if now >= end - 0.04:
            break
        stop = now + (end-now)/(attempts-attempt)
        fixed_stop = now + float(params['fixed_fraction'])*(stop-now)
        z = base.copy()
        radii = z[2*N:3*N].copy()
        order = np.argsort(radii)
        count = int(params['exchanges'])
        small = rng.choice(order[:10], size=count, replace=False)
        large = rng.choice(order[-10:], size=count, replace=False)
        radii[small], radii[large] = radii[large].copy(), radii[small].copy()
        z[2*N:3*N] = radii
        u = np.r_[z[:2*N], z[-1]]
        latest_u = u.copy()
        mu = float(params['penalty'])

        def fixed_fun(u):
            nonlocal latest_u
            if time.monotonic() >= fixed_stop:
                raise Timeout
            latest_u = u.copy()
            zz = np.r_[u[:2*N], radii, u[-1]]
            c = constraints(zz)
            v = np.minimum(c, 0.0)
            A = constraints(zz, True)
            g = 2*mu*(v @ A)
            # A tiny anchor resolves rigid translations without restraining rearrangements.
            displacement = u[:2*N]-base[:2*N]
            f = mu*float(v@v) + 1e-5*float(displacement@displacement)
            grad = np.r_[g[:2*N]+2e-5*displacement, g[-1]]
            return f, grad

        try:
            result = minimize(fixed_fun, u, jac=True, method='L-BFGS-B',
                              bounds=[(-0.3,2.3)]*(2*N)+[(0.25,1.75)],
                              options={'maxiter':100, 'ftol':1e-12, 'maxls':15})
            latest_u = result.x
        except Timeout:
            pass
        z = np.r_[latest_u[:2*N], radii, latest_u[-1]]
        latest = z.copy()

        def fun(v):
            if time.monotonic() >= stop-0.012:
                raise Timeout
            return -float(np.sum(v[2*N:3*N]))

        def con(v):
            if time.monotonic() >= stop-0.012:
                raise Timeout
            return constraints(v)

        def jac(v):
            if time.monotonic() >= stop-0.012:
                raise Timeout
            return constraints(v, True)

        def callback(v):
            nonlocal latest
            if time.monotonic() >= stop-0.012:
                raise Timeout
            latest = v.copy()
            if np.min(constraints(v)) > -2e-7:
                save(v)

        try:
            result = minimize(fun, z, jac=lambda v: objective_gradient,
                              method='SLSQP',
                              bounds=[(-0.1,2.1)]*(2*N)+[(1e-7,0.5)]*N+[(0.2,1.8)],
                              constraints={'type':'ineq','fun':con,'jac':jac},
                              callback=callback,
                              options={'maxiter':140,'ftol':2e-12,'disp':False})
            latest = result.x
        except Timeout:
            pass
        if time.monotonic() < end:
            save(latest)
    return best
