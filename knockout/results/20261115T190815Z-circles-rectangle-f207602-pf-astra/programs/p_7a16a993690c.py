import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'equalizing_power': (0.18, 0.05, 0.8),
    'concentrating_power': (2.4, 1.2, 4.0),
    'excursion_fraction': (0.36, 0.15, 0.6),
    'iterations': (100, 20, 300),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class StopSearch(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -float('inf')

    def exact_score(a):
        if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
        if np.any(d < a[I, 2] + a[J, 2]):
            return -float('inf')
        xs, ys, rs = [], [], []
        for x, y, r in a:
            if time.monotonic() >= end:
                raise StopSearch()
            xs.append(Fraction(float(x)))
            ys.append(Fraction(float(y)))
            rs.append(Fraction(float(r)))
        perimeter = (max(x+r for x, r in zip(xs, rs)) - min(x-r for x, r in zip(xs, rs))
                     + max(y+r for y, r in zip(ys, rs)) - min(y-r for y, r in zip(ys, rs)))
        if perimeter > 2:
            return -float('inf')
        return float(np.sum(a[:, 2]))

    if init is not None:
        a0 = np.array(init['circles'], dtype=float)
        try:
            best_score = exact_score(a0)
        except StopSearch:
            return init
    else:
        # A conservative valid fallback, independent of the optimizer.
        a0 = np.zeros((N, 3))
        a0[:, 0] = (np.arange(N) % 5 + 0.5) / 5
        a0[:, 1] = (np.arange(N) // 5 + 0.5) / 5
        a0[:, 2] = 0.09
        best = {'circles': a0.tolist()}
        best_score = float(np.sum(a0[:, 2]))

    lo = np.min(a0[:, :2] - a0[:, 2, None], axis=0)
    sides = np.max(a0[:, :2] + a0[:, 2, None], axis=0) - lo
    scale = 2.0 / np.sum(sides)
    p = (a0[:, :2] - lo) * scale
    r = a0[:, 2] * scale
    base = np.r_[p[:, 0], p[:, 1], r, sides[0] * scale]
    if init is None:
        base[:42] += rng.normal(0, 0.006, 42)

    rows = np.arange(M)
    ids = np.arange(N)
    constant_jac = np.zeros((M + 4*N, 64))
    constant_jac[rows, 42+I] = -1
    constant_jac[rows, 42+J] = -1
    constant_jac[M+ids, ids] = 1
    constant_jac[M+ids, 42+ids] = -1
    constant_jac[M+N+ids, ids] = -1
    constant_jac[M+N+ids, 42+ids] = -1
    constant_jac[M+N+ids, 63] = 1
    constant_jac[M+2*N+ids, N+ids] = 1
    constant_jac[M+2*N+ids, 42+ids] = -1
    constant_jac[M+3*N+ids, N+ids] = -1
    constant_jac[M+3*N+ids, 42+ids] = -1
    constant_jac[M+3*N+ids, 63] = -1

    def consider(z):
        nonlocal best, best_score
        if time.monotonic() >= end:
            raise StopSearch()
        rr = z[42:63].copy()
        if np.any(rr <= 0):
            return
        pp = np.column_stack((z[:21], z[21:42]))
        dd = np.sqrt((pp[I, 0]-pp[J, 0])**2 + (pp[I, 1]-pp[J, 1])**2)
        factor = min(1.0, float(np.min(dd/(rr[I]+rr[J]))))
        rr *= max(0.0, factor) * (1-8e-15)
        lower = np.min(pp-rr[:, None], axis=0)
        upper = np.max(pp+rr[:, None], axis=0)
        s = (2.0 / np.sum(upper-lower)) * (1-8e-15)
        rr *= s
        if float(np.sum(rr)) <= best_score:
            return
        pp = (pp-lower)*s
        a = np.column_stack((pp, rr))
        score = exact_score(a)
        if score > best_score:
            best, best_score = {'circles': a.tolist()}, score

    bounds = [(0.0, 2.0)]*42 + [(0.0001, 0.7)]*21 + [(0.15, 1.85)]
    powers = [float(params['equalizing_power']), float(params['concentrating_power'])]
    reference = float(np.mean(r))

    try:
        for attempt, power in enumerate(powers):
            if time.monotonic() >= end:
                break
            remaining = end-time.monotonic()
            slot_end = time.monotonic() + remaining/(len(powers)-attempt)
            excursion_end = time.monotonic() + (slot_end-time.monotonic())*float(params['excursion_fraction'])
            current = base.copy()
            for phase in range(2):
                if time.monotonic() >= end:
                    raise StopSearch()
                deadline = excursion_end if phase == 0 else slot_end
                exponent = power if phase == 0 else 1.0
                last = [current.copy()]
                cache = [None, None, None]

                def check():
                    if time.monotonic() >= deadline:
                        raise StopSearch()

                def objective(z):
                    check()
                    q = np.maximum(z[42:63]/reference, 1e-12)
                    value = -reference*np.sum(q**exponent)/exponent
                    grad = np.zeros(64)
                    grad[42:63] = -q**(exponent-1)
                    return value, grad

                def constraints(z):
                    check()
                    if cache[0] is not None and np.array_equal(z, cache[0]):
                        return cache[1], cache[2]
                    x, y, rr, w = z[:21], z[21:42], z[42:63], z[63]
                    dx, dy = x[I]-x[J], y[I]-y[J]
                    d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-15)
                    c = np.r_[d-rr[I]-rr[J], x-rr, w-x-rr, y-rr, 2-w-y-rr]
                    jac = constant_jac.copy()
                    jac[rows, I] = dx/d
                    jac[rows, J] = -dx/d
                    jac[rows, N+I] = dy/d
                    jac[rows, N+J] = -dy/d
                    cache[:] = [z.copy(), c, jac]
                    return c, jac

                def callback(z):
                    check()
                    last[0] = z.copy()
                    consider(z)

                try:
                    result = minimize(objective, current, jac=True, method='SLSQP', bounds=bounds,
                                      constraints={'type': 'ineq', 'fun': lambda z: constraints(z)[0],
                                                   'jac': lambda z: constraints(z)[1]},
                                      callback=callback,
                                      options={'maxiter': int(params['iterations']), 'ftol': 3e-12, 'disp': False})
                    current = result.x
                except StopSearch:
                    current = last[0]
                if time.monotonic() < end:
                    consider(current)
    except StopSearch:
        pass
    return best
