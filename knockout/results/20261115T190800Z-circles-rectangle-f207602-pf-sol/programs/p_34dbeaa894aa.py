import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import lstsq

PARAMS = {
    'deformation': (0.085, 0.02, 0.25),
    'screen_iterations': (38, 15, 90),
    'polish_iterations': (65, 20, 130),
    'search_fraction': (0.88, 0.6, 0.95),
}

II, JJ = np.triu_indices(21, 1)

class Stop(Exception):
    pass


def exact(data):
    try:
        a = np.asarray(data['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        d = a[II, :2] - a[JJ, :2]
        if np.any(np.sqrt(d[:, 0]**2 + d[:, 1]**2) < a[II, 2] + a[JJ, 2]):
            return -float('inf')
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        p = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        p += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        if p > 2:
            return -float('inf')
        return float(np.sum(a[:, 2]))
    except (ValueError, TypeError, KeyError):
        return -float('inf')


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + .70 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    score = exact(init) if init is not None else -float('inf')
    swapped = False

    def check(deadline=end):
        if time.monotonic() >= deadline:
            raise Stop()

    def retain(v):
        nonlocal best, score
        if time.monotonic() >= end:
            return -float('inf')
        a = v[:63].reshape(21, 3).copy()
        if not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return -float('inf')
        a[:, :2] -= np.mean(a[:, :2], axis=0)
        d = a[II, :2]-a[JJ, :2]
        dist = np.sqrt(d[:, 0]**2+d[:, 1]**2)
        f = min(1., float(np.min(dist/(a[II, 2]+a[JJ, 2]))))
        if f <= 0:
            return -float('inf')
        a[:, 2] *= f*(1-6e-15)
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        a *= (2-3e-14)/float(np.sum(hi-lo))
        if swapped:
            a[:, [0, 1]] = a[:, [1, 0]]
        data = {'circles': a.tolist()}
        s = exact(data)
        if s > score:
            best, score = data, s
        return s

    if init is None:
        a = np.array([[(i+.5)*.2, (j+.5)*.2, .099999999999] for j in range(7) for i in range(3)])
        best = {'circles': a.tolist()}
        score = exact(best)
    else:
        a = np.asarray(init['circles'], dtype=float).copy()
    lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
    hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
    mid = (hi+lo)/2
    dev = np.sort(np.abs(a[:, :2]-mid), axis=0)[:3].sum(axis=0)
    swapped = bool(dev[0] < dev[1])
    if swapped:
        a[:, [0, 1]] = a[:, [1, 0]]
    lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
    hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
    a[:, 0] -= lo[0]
    a[:, 1] -= (lo[1]+hi[1])/2
    initial = np.r_[a.ravel(), hi[0]-lo[0]]

    X = np.zeros((21, 64)); Y = X.copy(); R = X.copy()
    k = np.arange(21)
    X[k, 3*k] = 1
    Y[k, 3*k+1] = 1
    R[k, 3*k+2] = 1
    W = np.zeros(64); W[63] = 1
    walls = np.concatenate((X-R, W-X-R, Y-R-.5*W, -Y-R-.5*W))
    offsets = np.r_[np.zeros(42), np.ones(42)]
    objective = -R.sum(axis=0)

    def make_model(B):
        return (B, (X[II]-X[JJ])@B, (Y[II]-Y[JJ])@B,
                (R[II]+R[JJ])@B, walls@B, objective@B)

    full = make_model(np.eye(64))

    def values(z, m):
        B, DX, DY, RS, L, obj = m
        dx, dy, rr = DX@z, DY@z, RS@z
        d = np.sqrt(dx*dx+dy*dy+1e-32)
        g = np.r_[d-rr, L@z+offsets]
        J = np.concatenate(((dx/d)[:, None]*DX+(dy/d)[:, None]*DY-RS, L))
        return g, J

    def polish_contacts(v, m, deadline):
        B = m[0]
        z = (B.T@v)/np.sum(B*B, axis=0)
        g, J = values(z, m)
        active = np.flatnonzero(g < 1e-7)
        if len(active) < len(z)-1:
            return B@z
        for _ in range(5):
            check(deadline)
            g, J = values(z, m)
            step = lstsq(J[active], -g[active], lapack_driver='gelsy')[0]
            if np.max(np.abs(step)) > .003:
                break
            z += step
            retain(B@z)
            if np.max(np.abs(step)) < 2e-15:
                break
        return B@z

    def optimize(v, m, deadline, iterations):
        B = m[0]
        z = (B.T@v)/np.sum(B*B, axis=0)
        obj = m[-1]
        cached = None
        cg = cj = None
        last = z.copy()
        bestz = z.copy()
        local = -float('inf')
        calls = 0
        def get(t):
            nonlocal cached, cg, cj, last
            check(deadline)
            if cached is None or not np.array_equal(cached, t):
                cg, cj = values(t, m)
                cached = t.copy()
                last = t.copy()
            return cg, cj
        def fun(t):
            check(deadline)
            return float(obj@t), obj
        def callback(t):
            nonlocal calls, bestz, local
            check(deadline)
            calls += 1
            if calls % 6 == 0:
                s = retain(B@t)
                if s > local:
                    local, bestz = s, t.copy()
        try:
            result = minimize(fun, z, jac=True, method='SLSQP',
                constraints={'type':'ineq', 'fun':lambda t:get(t)[0], 'jac':lambda t:get(t)[1]},
                callback=callback,
                options={'maxiter':int(iterations), 'ftol':8e-14, 'disp':False})
            last = result.x
        except Stop:
            pass
        if time.monotonic() < end:
            s = retain(B@last)
            if s >= local:
                bestz = last.copy()
        return B@bestz

    try:
        check()
        u = polish_contacts(initial, full, min(end, start+.12*budget_s))
        aa = u[:63].reshape(21, 3)
        singles = list(np.argsort(np.abs(aa[:, 1]))[:3])
        unused = set(range(21))-set(singles)
        pairs = []
        while unused:
            check()
            i = min(unused)
            unused.remove(i)
            target = aa[i].copy(); target[1] *= -1
            j = min(unused, key=lambda t:float(np.sum((aa[t]-target)**2)))
            unused.remove(j)
            if aa[i, 1] < aa[j, 1]:
                i, j = j, i
            pairs.append((i, j))
        B = np.zeros((64, 34))
        col = 0
        for i in singles:
            check()
            B[3*i, col] = 1
            B[3*i+2, col+1] = 1
            col += 2
        for i, j in pairs:
            check()
            B[3*i, col] = B[3*j, col] = 1
            B[3*i+1, col+1] = 1
            B[3*j+1, col+1] = -1
            B[3*i+2, col+2] = B[3*j+2, col+2] = 1
            col += 3
        B[63, 33] = 1
        sym = make_model(B)
        u = optimize(u, sym, min(end-.025, start+.23*budget_s), params['polish_iterations'])
        if time.monotonic() < start+.28*budget_s:
            u = polish_contacts(u, sym, start+.28*budget_s)
        base = (B.T@u)/np.sum(B*B, axis=0)
        archive = [(float(objective@u), u.copy())]
        search_end = start + .70*budget_s*params['search_fraction']
        attempt = 0
        while time.monotonic() < search_end-.035:
            check()
            z = base.copy()
            amp = params['deformation']*(1.0, 1.6, .65, 2.1)[attempt % 4]
            stretch = np.exp(rng.normal(0, amp*2))
            w = np.clip(z[33]*stretch, .55, 1.45)
            sx = w/z[33]
            sy = (2-w)/(2-z[33])
            z[33] = w
            for t in range(3):
                check()
                z[2*t] = z[2*t]*sx+rng.normal(0, amp*.6)
                z[2*t+1] *= np.exp(rng.normal(0, amp*.3))
            for t in range(9):
                check()
                q = 6+3*t
                z[q] = z[q]*sx+rng.normal(0, amp)
                z[q+1] = max(.012, z[q+1]*sy+rng.normal(0, amp))
                z[q+2] *= np.exp(rng.normal(0, amp*.5))
            local_end = min(search_end, time.monotonic()+.19*budget_s)
            candidate = optimize(B@z, sym, local_end, params['screen_iterations'])
            g, _ = values((B.T@candidate)/np.sum(B*B, axis=0), sym)
            if np.min(g) > -2e-7:
                archive.append((float(objective@candidate), candidate.copy()))
                archive.sort(key=lambda t:t[0])
                archive = archive[:4]
                if archive[0][0] < float(objective@(B@base))-1e-10:
                    base = (B.T@archive[0][1])/np.sum(B*B, axis=0)
            attempt += 1
        if time.monotonic() < end-.025:
            target = archive[0][1]
            target = optimize(target, full, end-.008, params['polish_iterations'])
            if time.monotonic() < end-.005:
                polish_contacts(target, full, end-.002)
    except (Stop, ValueError, np.linalg.LinAlgError):
        pass
    return best
