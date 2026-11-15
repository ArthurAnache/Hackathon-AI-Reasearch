import time
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, OptimizeWarning

PARAMS = {
    'time_fraction': (0.70, 0.50, 0.75),
    'minimum_radius': (0.00001, 0.000001, 0.025, 'log'),
    'stage_iterations': (45, 15, 100),
    'jitter': (0.025, 0.0, 0.15)
}


def construct(params, seed, budget_s, init=None):
    begin = time.monotonic()
    deadline = begin + float(params['time_fraction']) * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    best_obj = None
    best_score = -float('inf')
    best_array = None

    def array_of(obj):
        if obj is None:
            return None
        try:
            if 'circles' in obj:
                return np.asarray(obj['circles'], dtype=float)
            return np.asarray(obj['coordinate_rows'], dtype=float).T
        except Exception:
            return None

    def exact_score(a):
        if a is None or a.shape != (21, 3):
            return -float('inf')
        if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        dx = a[ii, 0] - a[jj, 0]
        dy = a[ii, 1] - a[jj, 1]
        if np.any(np.sqrt(dx**2 + dy**2) < a[ii, 2] + a[jj, 2]):
            return -float('inf')
        xmin = ymin = None
        xmax = ymax = None
        for x, y, r in a:
            if time.monotonic() >= deadline:
                return -float('inf')
            x, y, r = Fraction(float(x)), Fraction(float(y)), Fraction(float(r))
            xmin = x-r if xmin is None else min(xmin, x-r)
            xmax = x+r if xmax is None else max(xmax, x+r)
            ymin = y-r if ymin is None else min(ymin, y-r)
            ymax = y+r if ymax is None else max(ymax, y+r)
        if xmax-xmin+ymax-ymin > 2:
            return -float('inf')
        return float(np.sum(a[:, 2]))

    original = array_of(init)
    original_score = exact_score(original)
    if np.isfinite(original_score):
        best_obj, best_score, best_array = init, original_score, original.copy()

    def consider(a):
        nonlocal best_obj, best_score, best_array
        if time.monotonic() >= deadline or not np.isfinite(a).all():
            return
        a = a.copy()
        if np.any(a[:, 2] <= 0):
            return
        a[:, :2] -= np.mean(a[:, :2], axis=0)
        delta = a[ii, :2] - a[jj, :2]
        d = np.sqrt(delta[:, 0]**2 + delta[:, 1]**2)
        ratio = float(np.min(d / (a[ii, 2] + a[jj, 2])))
        a[:, 2] *= min(1.0, ratio) * (1.0 - 3e-12)
        lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
        hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
        total = float(np.sum(hi-lo))
        if total <= 0:
            return
        a *= 2.0 / total * (1.0 - 3e-12)
        if float(np.sum(a[:, 2])) <= best_score:
            return
        score = exact_score(a)
        if score > best_score:
            best_score = score
            best_array = a.copy()
            best_obj = {'coordinate_rows': a.T.tolist()}

    def layout(counts):
        rows = []
        for k, count in enumerate(counts):
            if time.monotonic() >= deadline:
                return None
            xs = 2.0 * (np.arange(count) - (count-1)/2.0)
            for x in xs:
                if time.monotonic() >= deadline:
                    return None
                rows.append((x, k*np.sqrt(3.0), 1.0))
        a = np.asarray(rows, dtype=float)
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        a *= 2.0 / np.sum(hi-lo)
        return a

    first = layout([4, 3, 4, 3, 4, 3])
    if first is not None:
        consider(first)
    if time.monotonic() >= deadline:
        return best_obj if best_obj is not None else init

    A = np.zeros((m+4*n, 3*n+1), dtype=float)
    b = np.zeros(m+4*n, dtype=float)
    idx = np.arange(n)
    pair_rows = np.arange(m)
    A[pair_rows, 2*n+ii] = 1.0
    A[pair_rows, 2*n+jj] = 1.0
    base = m
    A[base+idx, idx] = -1.0
    A[base+idx, 2*n+idx] = 1.0
    A[base+n+idx, idx] = 1.0
    A[base+n+idx, 2*n+idx] = 1.0
    A[base+n+idx, -1] = -1.0
    A[base+2*n+idx, n+idx] = -1.0
    A[base+2*n+idx, 2*n+idx] = 1.0
    A[base+3*n+idx, n+idx] = 1.0
    A[base+3*n+idx, 2*n+idx] = 1.0
    A[base+3*n+idx, -1] = 1.0
    b[base+3*n+idx] = 2.0
    objective = np.zeros(3*n+1)
    objective[2*n:3*n] = -1.0
    bounds = [(0.0, 2.0)]*(2*n) + [(float(params['minimum_radius']), 1.0)]*n + [(0.02, 1.98)]

    def improve(start, until, max_steps):
        if start is None:
            return
        a = start.copy()
        previous = -1.0
        stalled = 0
        for iteration in range(max_steps):
            now = time.monotonic()
            if now >= until or deadline-now < 0.015:
                break
            delta = a[ii, :2] - a[jj, :2]
            norms = np.sqrt(np.sum(delta*delta, axis=1))
            zero = norms < 1e-14
            if np.any(zero):
                angles = rng.uniform(0, 2*np.pi, int(np.sum(zero)))
                delta[zero, 0] = np.cos(angles)
                delta[zero, 1] = np.sin(angles)
                norms[zero] = 1.0
            u = delta / norms[:, None]
            A[pair_rows, ii] = -u[:, 0]
            A[pair_rows, jj] = u[:, 0]
            A[pair_rows, n+ii] = -u[:, 1]
            A[pair_rows, n+jj] = u[:, 1]
            remaining = min(until, deadline)-time.monotonic()
            if remaining < 0.012:
                break
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', OptimizeWarning)
                result = linprog(objective, A_ub=A, b_ub=b, bounds=bounds,
                    method='highs-ds', options={
                        'threads': 1,
                        'primal_feasibility_tolerance': 1e-9,
                        'dual_feasibility_tolerance': 1e-9,
                        'time_limit': max(0.005, remaining-0.005)
                    })
            if not result.success or result.x is None:
                break
            z = result.x
            a = np.column_stack((z[:n], z[n:2*n], z[2*n:3*n]))
            consider(a)
            value = float(np.sum(a[:, 2]))
            if abs(value-previous) < 2e-11:
                stalled += 1
            else:
                stalled = 0
            previous = value
            if stalled >= 3:
                break

    available = max(0.0, deadline-time.monotonic())
    stage_start = time.monotonic()
    steps = int(params['stage_iterations'])
    improve(first, min(deadline, stage_start+available*0.25), steps)
    second = layout([5, 4, 3, 4, 5])
    improve(second, min(deadline, stage_start+available*0.50), steps)
    if original is not None and original.shape == (21, 3) and np.isfinite(original).all():
        third = original.copy()
    else:
        third = layout([4, 5, 3, 5, 4])
    if third is not None:
        third[:, :2] += rng.normal(0.0, float(params['jitter']), (n, 2))
    improve(third, min(deadline, stage_start+available*0.75), steps)
    improve(best_array, deadline, 200)
    return best_obj if best_obj is not None else init
