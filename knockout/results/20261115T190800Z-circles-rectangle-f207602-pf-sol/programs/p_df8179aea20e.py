import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'low_norm': (1.3, 1.05, 1.9),
    'high_norm': (5.0, 2.2, 10.0),
    'penalty': (120.0, 20.0, 1000.0, 'log'),
    'iterations': (105, 30, 250),
    'rotation': (0.19, 0.0, 0.7853981633974483)
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class Timeout(Exception):
    pass


def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    x, y, r = ([Fraction(float(v)) for v in a[:, k]] for k in range(3))
    perimeter = (max(u + v for u, v in zip(x, r)) - min(u - v for u, v in zip(x, r))
                 + max(u + v for u, v in zip(y, r)) - min(u - v for u, v in zip(y, r)))
    if perimeter > 2:
        return -np.inf
    d = np.sqrt((a[I, 0] - a[J, 0]) ** 2 + (a[I, 1] - a[J, 1]) ** 2)
    if np.any(d < a[I, 2] + a[J, 2]):
        return -np.inf
    return float(np.sum(a[:, 2]))


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf
    if init is not None:
        a = np.asarray(init['circles'], dtype=float).copy()
        best_score = exact_score(a)
    else:
        a = np.zeros((N, 3))
        for k in range(N):
            if time.monotonic() >= deadline:
                break
            a[k] = [(k % 5 + .5) * .2, (k // 5 + .5) * .2, .045]
        best = {'circles': a.tolist()}
        best_score = exact_score(a)
    lo = (a[:, :2] - a[:, 2, None]).min(axis=0)
    sides = (a[:, :2] + a[:, 2, None]).max(axis=0) - lo
    scale = 2.0 / sides.sum()
    p0 = (a[:, :2] - lo) * scale
    r0 = a[:, 2] * scale
    base = np.r_[p0[:, 0], p0[:, 1], r0, sides[0] * scale]

    def check():
        if time.monotonic() >= deadline:
            raise Timeout()

    def save(z):
        nonlocal best, best_score
        check()
        p = np.column_stack((z[:N], z[N:2*N]))
        r = np.maximum(z[2*N:3*N], 1e-14).copy()
        if not np.isfinite(p).all() or not np.isfinite(r).all():
            return
        # A common radius contraction repairs all Euclidean pair constraints.
        dist = np.sqrt((p[I, 0] - p[J, 0]) ** 2 + (p[I, 1] - p[J, 1]) ** 2)
        ratio = np.min(dist / (r[I] + r[J]))
        r *= min(1.0, float(ratio)) * (1.0 - 3e-15)
        # Uniform scaling preserves the repaired separation constraints.
        low = (p - r[:, None]).min(axis=0)
        high = (p + r[:, None]).max(axis=0)
        factor = min(1.0, 2.0 / float(np.sum(high - low))) * (1.0 - 3e-15)
        p = (p - low) * factor
        r *= factor
        for _ in range(4):
            check()
            candidate = np.column_stack((p, r))
            score = exact_score(candidate)
            if np.isfinite(score):
                if score > best_score:
                    best_score = score
                    best = {'circles': candidate.tolist()}
                return
            r *= 1.0 - 2e-14

    # Euclidean constraints, including a variable width and complementary height.
    def constraints(z):
        check()
        x, y, r, w = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
        dx, dy = x[I] - x[J], y[I] - y[J]
        d = np.sqrt(dx*dx + dy*dy + 1e-30)
        return np.r_[d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r]

    def jacobian(z):
        check()
        x, y = z[:N], z[N:2*N]
        dx, dy = x[I]-x[J], y[I]-y[J]
        d = np.sqrt(dx*dx + dy*dy + 1e-30)
        jac = np.zeros((M+4*N, 3*N+1))
        rows = np.arange(M)
        jac[rows, I] = dx/d
        jac[rows, J] = -dx/d
        jac[rows, N+I] = dy/d
        jac[rows, N+J] = -dy/d
        jac[rows, 2*N+I] = -1
        jac[rows, 2*N+J] = -1
        q = np.arange(N)
        for k in range(4):
            check()
            rr = M+k*N+q
            jac[rr, 2*N+q] = -1
            jac[rr, (0 if k < 2 else N)+q] = 1 if k % 2 == 0 else -1
        jac[M+N:M+2*N, -1] = 1
        jac[M+3*N:M+4*N, -1] = -1
        return jac

    linear = np.r_[np.zeros(2*N), -np.ones(N), 0.0]
    bounds = [(None, None)]*(2*N) + [(1e-7, .6)]*N + [(.15, 1.85)]

    def polish(z, maxiter):
        check()
        result = minimize(lambda v: float(linear @ v), z, jac=lambda v: linear,
                          method='SLSQP', bounds=bounds,
                          constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                          options={'maxiter': maxiter, 'ftol': 3e-13})
        save(result.x)
        return result.x

    def penalty(z, norm, mu, theta):
        check()
        x, y, r, w = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
        dx, dy = x[I]-x[J], y[I]-y[J]
        co, si = np.cos(theta), np.sin(theta)
        u, v = co*dx+si*dy, -si*dx+co*dy
        au, av = np.maximum(np.abs(u), 1e-16), np.maximum(np.abs(v), 1e-16)
        d = (au**norm+av**norm)**(1.0/norm)
        du = np.sign(u)*(au/d)**(norm-1)
        dv = np.sign(v)*(av/d)**(norm-1)
        ddx, ddy = co*du-si*dv, si*du+co*dv
        overlap = np.maximum(r[I]+r[J]-d, 0)
        walls = np.array([np.maximum(r-x, 0), np.maximum(r+x-w, 0),
                          np.maximum(r-y, 0), np.maximum(r+y-(2-w), 0)])
        val = -np.sum(r) + mu*(np.dot(overlap, overlap)+np.sum(walls*walls))
        gx, gy, gr = np.zeros(N), np.zeros(N), -np.ones(N)
        t = 2*mu*overlap
        gx += np.bincount(I, -t*ddx, minlength=N)+np.bincount(J, t*ddx, minlength=N)
        gy += np.bincount(I, -t*ddy, minlength=N)+np.bincount(J, t*ddy, minlength=N)
        gr += np.bincount(I, t, minlength=N)+np.bincount(J, t, minlength=N)
        gx += 2*mu*(-walls[0]+walls[1])
        gy += 2*mu*(-walls[2]+walls[3])
        gr += 2*mu*np.sum(walls, axis=0)
        gw = 2*mu*(-walls[1].sum()+walls[3].sum())
        return float(val), np.r_[gx, gy, gr, gw]

    try:
        for branch, target in enumerate([float(params['low_norm']), float(params['high_norm'])]):
            check()
            z = base.copy()
            theta = float(params['rotation']) * (1 if branch == 0 else -1)
            # Change the metric substantially, then restore it gradually.
            for norm, multiple in [(target, 1.0), ((target+2)/2, 2.0), (2.0, 8.0)]:
                check()
                result = minimize(penalty, z,
                                  args=(norm, float(params['penalty'])*multiple, theta),
                                  jac=True, method='L-BFGS-B', bounds=bounds,
                                  options={'maxiter': int(params['iterations']),
                                           'ftol': 1e-11, 'gtol': 1e-7, 'maxls': 18})
                z = result.x
            z = polish(z, 85)
    except Timeout:
        pass
    return best
