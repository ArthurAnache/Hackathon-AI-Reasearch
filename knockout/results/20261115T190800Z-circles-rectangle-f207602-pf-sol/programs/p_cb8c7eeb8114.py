import time
import math
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'initial_fraction': (0.46, 0.25, 0.58),
    'contact_margin': (0.11, 0.04, 0.25),
    'restart_fraction': (0.15, 0.08, 0.3),
    'jitter': (0.018, 0.003, 0.06)
}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.69 * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    ids = np.arange(21)
    best = init
    best_score = -float('inf')
    best_a = None

    def exact(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        dx = a[ii, 0] - a[jj, 0]
        dy = a[ii, 1] - a[jj, 1]
        if np.any(np.sqrt(dx*dx + dy*dy) < a[ii, 2] + a[jj, 2]):
            return -float('inf')
        xs = [Fraction(float(v)) for v in a[:, 0]]
        ys = [Fraction(float(v)) for v in a[:, 1]]
        rs = [Fraction(float(v)) for v in a[:, 2]]
        w = max(x+r for x,r in zip(xs,rs)) - min(x-r for x,r in zip(xs,rs))
        h = max(y+r for y,r in zip(ys,rs)) - min(y-r for y,r in zip(ys,rs))
        if w+h > 2:
            return -float('inf')
        return float(np.sum(a[:, 2]))

    def vector(a):
        a = a.copy()
        a[:, :2] -= np.min(a[:, :2]-a[:, 2, None], axis=0)
        spans = np.max(a[:, :2]+a[:, 2, None], axis=0)
        scale = 2.0 / spans.sum()
        a *= scale
        return np.r_[a[:, 0], a[:, 1], a[:, 2], spans[0]*scale]

    def consider(v):
        nonlocal best, best_score, best_a
        if time.monotonic() >= deadline:
            return
        a = np.column_stack((v[:21], v[21:42], np.maximum(v[42:63], 1e-12)))
        if not np.isfinite(a).all():
            return
        d = np.hypot(a[ii, 0]-a[jj, 0], a[ii, 1]-a[jj, 1])
        factor = min(1.0, float(np.min(d/(a[ii, 2]+a[jj, 2]))))
        if factor <= 0:
            return
        a[:, 2] *= factor*(1.0-4e-14)
        a[:, :2] -= np.min(a[:, :2]-a[:, 2, None], axis=0)
        span = np.max(a[:, :2]+a[:, 2, None], axis=0)-np.min(a[:, :2]-a[:, 2, None], axis=0)
        a *= (2.0/span.sum())*(1.0-4e-14)
        if float(np.sum(a[:, 2])) <= best_score:
            return
        score = exact(a)
        if score > best_score:
            best_score = score
            best_a = a.copy()
            best = {'circles': a.tolist()}

    raw = None
    if init is not None:
        try:
            raw = np.asarray(init['circles'], dtype=float)
            score = exact(raw)
            if np.isfinite(score):
                best_score = score
                best_a = raw.copy()
            if raw.shape != (21, 3) or not np.isfinite(raw).all() or np.any(raw[:, 2] <= 0):
                raw = None
        except Exception:
            raw = None
    if raw is None:
        raw = np.array([(2.0*c, 2.0*r, 1.0) for r in range(5) for c in range(4)] + [(3.0, 3.0, math.sqrt(2)-1)])
    z = vector(raw)
    consider(z)
    grad = np.zeros(64)
    grad[42:63] = -1.0
    bounds = [(None, None)]*42 + [(1e-7, 0.8)]*21 + [(0.08, 1.92)]
    forced = np.zeros(len(ii), dtype=bool)
    run = 0
    continuing = False

    while time.monotonic() < deadline - 0.025:
        now = time.monotonic()
        if run == 0:
            local_end = min(deadline-0.008, start + params['initial_fraction']*budget_s)
        else:
            local_end = min(deadline-0.008, now + params['restart_fraction']*budget_s)
            if not continuing:
                if best_a is not None:
                    z = vector(best_a)
                amp = params['jitter']*(1.0 if run % 3 else 2.2)
                displacement = rng.normal(size=(21, 2))
                if run % 2 == 0:
                    p = rng.integers(21)
                    xy = np.column_stack((z[:21], z[21:42]))
                    weight = np.exp(-np.sum((xy-xy[p])**2, axis=1)/0.07)
                    displacement += 2.0*weight[:, None]*rng.normal(size=(1, 2))
                z[:21] += amp*displacement[:, 0]
                z[21:42] += amp*displacement[:, 1]
                z[42:63] *= 0.99
                forced[:] = False
        gaps = np.hypot(z[ii]-z[jj], z[21+ii]-z[21+jj])-z[42+ii]-z[42+jj]
        selected = (gaps < params['contact_margin']) | forced
        si, sj = ii[selected], jj[selected]
        m = len(si)
        rows = np.arange(m)
        template = np.zeros((m+84, 64))
        template[rows, 42+si] = -1
        template[rows, 42+sj] = -1
        q = m+ids
        template[q, ids] = 1
        template[q, 42+ids] = -1
        q = m+21+ids
        template[q, ids] = -1
        template[q, 42+ids] = -1
        template[q, 63] = 1
        q = m+42+ids
        template[q, 21+ids] = 1
        template[q, 42+ids] = -1
        q = m+63+ids
        template[q, 21+ids] = -1
        template[q, 42+ids] = -1
        template[q, 63] = -1
        last = [z.copy()]
        count = [0]

        def check():
            if time.monotonic() >= local_end:
                raise StopSearch()

        def objective(v):
            check()
            return -float(np.sum(v[42:63]))

        def constraints(v):
            check()
            x, y, r, w = v[:21], v[21:42], v[42:63], v[63]
            return np.r_[np.hypot(x[si]-x[sj], y[si]-y[sj])-r[si]-r[sj], x-r, w-x-r, y-r, 2-w-y-r]

        def jacobian(v):
            check()
            dx = v[si]-v[sj]
            dy = v[21+si]-v[21+sj]
            d = np.maximum(np.hypot(dx, dy), 1e-20)
            j = template.copy()
            j[rows, si] = dx/d
            j[rows, sj] = -dx/d
            j[rows, 21+si] = dy/d
            j[rows, 21+sj] = -dy/d
            return j

        def callback(v):
            last[0] = v.copy()
            count[0] += 1
            if count[0] % 2 == 0:
                consider(v)
            check()

        try:
            result = minimize(objective, z, jac=lambda v: grad, method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                              callback=callback, options={'maxiter': 250, 'ftol': 8e-12, 'disp': False})
            z = result.x.copy()
        except (StopSearch, ValueError, FloatingPointError):
            z = last[0]
        consider(z)
        gaps = np.hypot(z[ii]-z[jj], z[21+ii]-z[21+jj])-z[42+ii]-z[42+jj]
        missed = (~selected) & (gaps < 1e-6)
        continuing = bool(np.any(missed))
        forced |= selected | missed
        run += 1
    return best
