import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {
    'first_fraction': (0.39, 0.20, 0.55),
    'symmetry_fraction': (0.18, 0.08, 0.30),
    'noise': (0.012, 0.001, 0.06),
    'maxiter': (190, 50, 400),
    'ftol': (2e-12, 1e-13, 1e-7, 'log')
}
N = 21
I, J = np.triu_indices(N, 1)
K = np.arange(N)
R = np.arange(len(I))

class Deadline(Exception):
    pass

def exact_score(a):
    try:
        if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -np.inf
        d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
        if np.any(d < a[I, 2]+a[J, 2]):
            return -np.inf
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        wh = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        wh += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        if wh > 2:
            return -np.inf
        return float(np.sum(a[:, 2]))
    except Exception:
        return -np.inf

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    best = None
    best_array = None
    best_score = -np.inf

    def retain(a, original=None):
        nonlocal best, best_array, best_score
        score = exact_score(a)
        if score > best_score:
            best_score = score
            best_array = a.copy()
            best = original if original is not None else {'circles': a.tolist()}

    def pack(a):
        p = a[:, :2].copy()
        r = np.maximum(a[:, 2], 1e-10).copy()
        lo = np.min(p-r[:, None], axis=0)
        side = np.max(p+r[:, None], axis=0)-lo
        scale = 2.0/float(side.sum())
        p = (p-lo)*scale
        return np.r_[p[:, 0], p[:, 1], r*scale, side[0]*scale]

    def repair(z):
        if not np.isfinite(z).all():
            return
        p = np.column_stack((z[:N], z[N:2*N])).copy()
        r = np.maximum(z[2*N:3*N], 1e-10).copy()
        d = np.sqrt((p[I, 0]-p[J, 0])**2+(p[I, 1]-p[J, 1])**2)
        ratio = min(1.0, float(np.min(d/(r[I]+r[J]))))
        if ratio <= 0:
            return
        r *= ratio*(1-4e-14)
        lo = np.min(p-r[:, None], axis=0)
        side = np.max(p+r[:, None], axis=0)-lo
        scale = (2-2e-13)/float(side.sum())
        p = (p-lo)*scale
        r *= scale*(1-4e-14)
        retain(np.column_stack((p, r)))

    initial = None
    if init is not None:
        try:
            a = np.asarray(init['circles'], dtype=float)
            retain(a, init)
            if a.shape == (N, 3) and np.isfinite(a).all():
                initial = pack(a)
        except Exception:
            pass

    pp = []
    for row in range(5):
        if time.monotonic() >= end:
            break
        for col in range(5):
            if time.monotonic() >= end:
                break
            if row in (0, 4) and col in (0, 4):
                continue
            pp.append((float(col), float(row)))
    if len(pp) == N:
        pp = np.asarray(pp)
        fallback = np.r_[pp[:, 0], pp[:, 1], np.full(N, 0.5), 1.0]
        repair(fallback)
    if initial is None:
        initial = pack(best_array)
    else:
        repair(initial)

    A = np.zeros((4*N, 3*N+1))
    b = np.zeros(4*N)
    A[K, K] = 1
    A[K, 2*N+K] = -1
    A[N+K, K] = -1
    A[N+K, 2*N+K] = -1
    A[N+K, -1] = 1
    A[2*N+K, N+K] = 1
    A[2*N+K, 2*N+K] = -1
    A[3*N+K, N+K] = -1
    A[3*N+K, 2*N+K] = -1
    A[3*N+K, -1] = -1
    b[3*N:] = 2
    g = np.r_[np.zeros(2*N), -np.ones(N), 0.0]

    def optimize(z0, until, T=None, offset=None):
        until = min(until, end)
        if time.monotonic() >= until:
            return z0
        def check():
            if time.monotonic() >= until:
                raise Deadline()
        if T is None:
            q0 = z0.copy()
            AA, bb, gg = A, b, g
            bounds = [(-1, 3)]*(2*N)+[(1e-9, 0.8)]*N+[(0.08, 1.92)]
            def expand(q):
                return q
        else:
            q0 = np.linalg.lstsq(T, z0-offset, rcond=None)[0]
            AA, bb, gg = A@T, A@offset+b, T.T@g
            bounds = None
            def expand(q):
                return offset+T@q
        last = [expand(q0).copy()]
        ticks = [0]
        def fun(q):
            check()
            return float(gg@q)
        def con(q):
            check()
            z = expand(q)
            dx, dy = z[I]-z[J], z[N+I]-z[N+J]
            return np.sqrt(dx*dx+dy*dy+1e-30)-z[2*N+I]-z[2*N+J]
        def jac(q):
            check()
            z = expand(q)
            dx, dy = z[I]-z[J], z[N+I]-z[N+J]
            d = np.sqrt(dx*dx+dy*dy+1e-30)
            C = np.zeros((len(I), 3*N+1))
            C[R, I] = dx/d
            C[R, J] = -dx/d
            C[R, N+I] = dy/d
            C[R, N+J] = -dy/d
            C[R, 2*N+I] = -1
            C[R, 2*N+J] = -1
            return C if T is None else C@T
        def cb(q):
            last[0] = expand(q).copy()
            check()
            ticks[0] += 1
            if ticks[0] % 10 == 0:
                repair(last[0])
        try:
            result = minimize(fun, q0, jac=lambda q:gg, method='SLSQP',
                bounds=bounds,
                constraints=[{'type':'ineq', 'fun':lambda q:AA@q+bb, 'jac':lambda q:AA},
                             {'type':'ineq', 'fun':con, 'jac':jac}],
                callback=cb,
                options={'maxiter':int(params['maxiter']), 'ftol':float(params['ftol']), 'disp':False})
            last[0] = expand(result.x)
        except Deadline:
            pass
        repair(last[0])
        return last[0]

    z = optimize(initial, start+float(params['first_fraction'])*budget_s)

    def symmetry(a, transpose):
        if time.monotonic() >= end:
            raise Deadline()
        a = a.copy()
        if transpose:
            a[:, [0, 1]] = a[:, [1, 0]]
        z0 = pack(a)
        order = np.argsort(z0[N:2*N])
        lower, singles, upper = order[:9], order[9:12], order[12:]
        h = 2-z0[-1]
        cost = ((z0[lower, None]-z0[upper][None, :])**2
                +(z0[N+lower, None]+z0[N+upper][None, :]-h)**2
                +2*(z0[2*N+lower, None]-z0[2*N+upper][None, :])**2)
        rows, match = linear_sum_assignment(cost)
        quality = float(cost[rows, match].sum()+np.sum((z0[N+singles]-h/2)**2))
        upper = upper[match]
        T = np.zeros((64, 34))
        off = np.zeros(64)
        for k, s in enumerate(singles):
            if time.monotonic() >= end:
                raise Deadline()
            T[s, k] = 1
            off[N+s] = 1
            T[N+s, 33] = -0.5
            T[2*N+s, 21+k] = 1
        for k, (l, u) in enumerate(zip(lower, upper)):
            if time.monotonic() >= end:
                raise Deadline()
            T[l, 3+k] = T[u, 3+k] = 1
            off[N+l] = off[N+u] = 1
            T[N+l, 33] = T[N+u, 33] = -0.5
            T[N+l, 12+k] = -1
            T[N+u, 12+k] = 1
            T[2*N+l, 24+k] = T[2*N+u, 24+k] = 1
        T[-1, 33] = 1
        return quality, z0, T, off

    try:
        options = [symmetry(best_array, False), symmetry(best_array, True)]
        _, z0, T, off = min(options, key=lambda t:t[0])
        zs = optimize(z0, time.monotonic()+float(params['symmetry_fraction'])*budget_s, T, off)
        if time.monotonic() < end:
            optimize(pack(best_array), min(end, time.monotonic()+0.075*budget_s))
        attempt = 0
        while time.monotonic() < end:
            base = pack(best_array)
            amp = float(params['noise'])*(1+0.6*(attempt % 3))
            base[:2*N] += rng.normal(0, amp, 2*N)
            base[2*N:3*N] *= rng.uniform(0.98, 1.02, N)
            optimize(base, min(end, time.monotonic()+0.12*budget_s))
            attempt += 1
    except Deadline:
        pass
    return best
