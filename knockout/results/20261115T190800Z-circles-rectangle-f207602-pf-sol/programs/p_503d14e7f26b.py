import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'exclusion_strength': (0.095, 0.005, 0.4),
    'exclusion_radius': (0.115, 0.02, 0.35),
    'cluster_size': (7, 3, 15),
    'kick': (0.012, 0.001, 0.08),
    'penalty_iterations': (100, 30, 250),
}
N = 21
I, J = np.triu_indices(N, 1)
E = np.eye(N)

class StopSearch(Exception):
    pass


def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    dx = a[I, 0] - a[J, 0]
    dy = a[I, 1] - a[J, 1]
    if np.any(np.sqrt(dx * dx + dy * dy) < a[I, 2] + a[J, 2]):
        return -np.inf
    x = list(map(Fraction, map(float, a[:, 0])))
    y = list(map(Fraction, map(float, a[:, 1])))
    r = list(map(Fraction, map(float, a[:, 2])))
    perimeter = (max(map(sum, zip(x, r))) - min(u-v for u,v in zip(x,r))
                 + max(map(sum, zip(y, r))) - min(u-v for u,v in zip(y,r)))
    if perimeter > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.69 * budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a = np.zeros((N, 3))
        a[:, 0] = (np.arange(N) % 7 + 0.5) / 5
        a[:, 1] = (np.arange(N) // 7 + 0.5) / 5
        a[:, 2] = 0.099
        init = {'circles': a.tolist()}
    a0 = np.array(init['circles'], dtype=float)
    best = init
    best_score = exact_score(a0)
    lo = np.min(a0[:, :2] - a0[:, 2, None], axis=0)
    side = np.max(a0[:, :2] + a0[:, 2, None], axis=0) - lo
    scale = 2.0 / side.sum()
    a = a0.copy()
    a[:, :2] = (a[:, :2] - lo) * scale
    a[:, 2] *= scale
    anchor = np.r_[a[:, 0], a[:, 1], a[:, 2], side[0] * scale]
    bounds = [(0.0, 2.0)] * (2*N) + [(1e-7, 0.5)] * N + [(0.15, 1.85)]
    objective_gradient = np.r_[np.zeros(2*N), -np.ones(N), 0.0]

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def constraints(z, jacobian=False):
        check()
        x, y, r, w = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
        dx, dy = x[I]-x[J], y[I]-y[J]
        d = np.sqrt(dx*dx + dy*dy + 1e-30)
        c = np.r_[d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r]
        if not jacobian:
            return c
        jac = np.zeros((len(I)+4*N, 3*N+1))
        rows = np.arange(len(I))
        jac[rows, I] = dx/d
        jac[rows, J] = -dx/d
        jac[rows, N+I] = dy/d
        jac[rows, N+J] = -dy/d
        jac[rows, 2*N+I] = -1
        jac[rows, 2*N+J] = -1
        m = len(I)
        jac[m:m+N, :N] = E
        jac[m+N:m+2*N, :N] = -E
        jac[m+2*N:m+3*N, N:2*N] = E
        jac[m+3*N:m+4*N, N:2*N] = -E
        jac[m:m+N, 2*N:3*N] = -E
        jac[m+N:m+2*N, 2*N:3*N] = -E
        jac[m+2*N:m+3*N, 2*N:3*N] = -E
        jac[m+3*N:m+4*N, 2*N:3*N] = -E
        jac[m+N:m+2*N, -1] = 1
        jac[m+3*N:m+4*N, -1] = -1
        return jac

    def consider(z):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        p = np.column_stack((z[:N], z[N:2*N]))
        r = np.maximum(z[2*N:3*N], 1e-12).copy()
        d = np.sqrt(np.sum((p[I]-p[J])**2, axis=1))
        factor = min(1.0, float(np.min(d / (r[I]+r[J]))))
        r *= factor * (1.0 - 3e-15)
        out = np.column_stack((p, r))
        sides = np.max(p+r[:,None], axis=0)-np.min(p-r[:,None], axis=0)
        out *= min(1.0, 2.0/float(sides.sum())) * (1.0-3e-15)
        score = exact_score(out)
        if score > best_score:
            best_score = score
            best = {'circles': out.tolist()}

    try:
        attempt = 0
        while time.monotonic() < deadline - 0.12:
            check()
            # Pick a connected spatial patch, so the exclusion requires a
            # coordinated rearrangement rather than a single-circle release.
            pivot = int(rng.integers(N))
            distances = np.sum((a[:, :2]-a[pivot, :2])**2, axis=1)
            group = np.argsort(distances)[:int(params['cluster_size'])]
            mask = np.zeros(3*N+1)
            mask[group] = 1
            mask[N+group] = 1
            mask[2*N+group] = 0.6
            z = anchor.copy()
            kick = rng.normal(size=(len(group), 2)) * float(params['kick'])
            z[group] += kick[:, 0]
            z[N+group] += kick[:, 1]
            radius2 = float(params['exclusion_radius'])**2
            latest = [z.copy()]
            for mu, strength in ((65.0, 1.0), (600.0, 0.7), (6000.0, 0.0)):
                check()
                stage_end = min(deadline - 0.16, time.monotonic() + 0.19*budget_s)
                def penalty(v):
                    check()
                    if time.monotonic() >= stage_end:
                        raise StopSearch()
                    latest[0] = v.copy()
                    x,y,r,w = v[:N],v[N:2*N],v[2*N:3*N],v[-1]
                    dx,dy = x[I]-x[J],y[I]-y[J]
                    d = np.sqrt(dx*dx+dy*dy+1e-30)
                    pair = np.maximum(r[I]+r[J]-d,0)
                    b0 = np.maximum(r-x,0)
                    b1 = np.maximum(r+x-w,0)
                    b2 = np.maximum(r-y,0)
                    b3 = np.maximum(r+y+w-2,0)
                    c = 2*mu*pair
                    gx = np.bincount(I,weights=-c*dx/d,minlength=N)+np.bincount(J,weights=c*dx/d,minlength=N)
                    gy = np.bincount(I,weights=-c*dy/d,minlength=N)+np.bincount(J,weights=c*dy/d,minlength=N)
                    gr = -np.ones(N)+np.bincount(I,weights=c,minlength=N)+np.bincount(J,weights=c,minlength=N)
                    gx += 2*mu*(-b0+b1)
                    gy += 2*mu*(-b2+b3)
                    gr += 2*mu*(b0+b1+b2+b3)
                    g = np.r_[gx,gy,gr,2*mu*(-b1.sum()+b3.sum())]
                    f = -r.sum()+mu*(pair@pair+b0@b0+b1@b1+b2@b2+b3@b3)
                    delta = (v-anchor)*mask
                    rep = float(params['exclusion_strength'])*strength*np.exp(-np.dot(delta,delta)/(2*radius2))
                    return f+rep, g-rep*delta*mask/radius2
                try:
                    z = minimize(penalty,z,jac=True,bounds=bounds,method='L-BFGS-B',options={'maxiter':int(params['penalty_iterations']),'ftol':1e-12,'maxls':20}).x
                except StopSearch:
                    z = latest[0]
                    if time.monotonic() >= deadline-0.13:
                        break
            check()
            latest[0] = z.copy()
            def fun(v):
                check()
                latest[0] = v.copy()
                return -np.sum(v[2*N:3*N])
            try:
                res = minimize(fun,z,jac=lambda v: objective_gradient,bounds=bounds,method='SLSQP',constraints={'type':'ineq','fun':constraints,'jac':lambda v:constraints(v,True)},options={'maxiter':100,'ftol':2e-12,'disp':False})
                z = res.x
            except StopSearch:
                z = latest[0]
            consider(z)
            attempt += 1
    except StopSearch:
        pass
    return best
