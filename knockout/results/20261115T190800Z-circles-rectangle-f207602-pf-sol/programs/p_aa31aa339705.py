import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'remove': (2, 1, 3),
    'iterations': (85, 20, 160),
    'grid': (27, 15, 45),
    'hole_starts': (3, 1, 6),
    'attempts': (8, 1, 20),
}

class Deadline(Exception):
    pass


def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    i, j = np.triu_indices(21, 1)
    d = np.sqrt((a[i, 0]-a[j, 0])**2 + (a[i, 1]-a[j, 1])**2)
    if np.any(d < a[i, 2] + a[j, 2]):
        return -np.inf
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    perimeter = (max(u+v for u, v in zip(x, r))-min(u-v for u, v in zip(x, r))
                 + max(u+v for u, v in zip(y, r))-min(u-v for u, v in zip(y, r)))
    return float(np.sum(a[:, 2])) if perimeter <= 2 else -np.inf


def repair(a, end):
    a = np.array(a, dtype=float, copy=True)
    if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return None
    a[:, :2] -= np.min(a[:, :2]-a[:, 2, None], axis=0)
    i, j = np.triu_indices(len(a), 1)
    d = np.sqrt(np.sum((a[i, :2]-a[j, :2])**2, axis=1))
    factor = min(1.0, float(np.min(d/(a[i, 2]+a[j, 2]))))
    a[:, 2] *= factor * (1.0-3e-15)
    lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
    hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
    a[:, :2] -= lo
    a *= min(1.0, 2.0/float(np.sum(hi-lo))) * (1.0-3e-15)
    for k in range(8):
        if time.monotonic() >= end:
            return None
        if exact_score(a) > -np.inf:
            return a
        a[:, 2] *= 1.0-2e-13
    return None


def optimize_layout(a, w, end, iterations):
    n = len(a)
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    rows = np.arange(m)
    ids = np.arange(n)
    size = 3*n+1
    z = np.r_[a[:, 0], a[:, 1], a[:, 2], w]
    gradient = np.r_[np.zeros(2*n), -np.ones(n), 0.0]
    last = [z.copy()]
    cache = [None, None, None]

    def check():
        if time.monotonic() >= end:
            raise Deadline()

    def objective(v):
        check()
        return -float(np.sum(v[2*n:3*n]))

    def calculate(v):
        check()
        if cache[0] is not None and np.array_equal(v, cache[0]):
            return cache[1], cache[2]
        x, y, r, width = v[:n], v[n:2*n], v[2*n:3*n], v[-1]
        dx, dy = x[ii]-x[jj], y[ii]-y[jj]
        d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-15)
        c = np.r_[d-r[ii]-r[jj], x-r, width-x-r, y-r, 2-width-y-r]
        jac = np.zeros((m+4*n, size))
        jac[rows, ii] = dx/d
        jac[rows, jj] = -dx/d
        jac[rows, n+ii] = dy/d
        jac[rows, n+jj] = -dy/d
        jac[rows, 2*n+ii] = -1
        jac[rows, 2*n+jj] = -1
        rr = m+ids
        jac[rr, ids] = 1
        jac[rr, 2*n+ids] = -1
        rr = m+n+ids
        jac[rr, ids] = -1
        jac[rr, 2*n+ids] = -1
        jac[rr, -1] = 1
        rr = m+2*n+ids
        jac[rr, n+ids] = 1
        jac[rr, 2*n+ids] = -1
        rr = m+3*n+ids
        jac[rr, n+ids] = -1
        jac[rr, 2*n+ids] = -1
        jac[rr, -1] = -1
        cache[:] = [v.copy(), c, jac]
        return c, jac

    def callback(v):
        last[0] = v.copy()
        check()

    try:
        result = minimize(objective, z, jac=lambda v: gradient,
                          constraints={'type': 'ineq', 'fun': lambda v: calculate(v)[0],
                                       'jac': lambda v: calculate(v)[1]},
                          bounds=[(0, 2)]*(2*n)+[(1e-7, 0.95)]*n+[(0.12, 1.88)],
                          method='SLSQP', callback=callback,
                          options={'maxiter': iterations, 'ftol': 3e-11, 'disp': False})
        v = result.x
    except Deadline:
        v = last[0]
    return np.column_stack((v[:n], v[n:2*n], v[2*n:3*n])), float(v[-1])


def insert_hole(a, w, params, end):
    h = 2-w
    g = int(params['grid'])
    xx, yy = np.meshgrid(np.linspace(0, w, g), np.linspace(0, h, g))
    p = np.column_stack((xx.ravel(), yy.ravel()))
    clearance = np.minimum.reduce((p[:, 0], w-p[:, 0], p[:, 1], h-p[:, 1]))
    d = np.sqrt(np.sum((p[:, None, :]-a[None, :, :2])**2, axis=2))-a[None, :, 2]
    clearance = np.minimum(clearance, np.min(d, axis=1))
    order = np.argsort(clearance)[::-1]
    best = np.r_[p[order[0]], max(1e-7, clearance[order[0]])]
    starts = []
    for index in order:
        if time.monotonic() >= end:
            raise Deadline()
        point = p[index]
        if all(np.linalg.norm(point-q) > 0.07 for q in starts):
            starts.append(point)
        if len(starts) >= int(params['hole_starts']):
            break

    def con(v):
        if time.monotonic() >= end:
            raise Deadline()
        delta = v[:2]-a[:, :2]
        dist = np.maximum(np.sqrt(np.sum(delta*delta, axis=1)), 1e-15)
        c = np.r_[dist-a[:, 2]-v[2], v[0]-v[2], w-v[0]-v[2],
                  v[1]-v[2], h-v[1]-v[2]]
        j = np.empty((len(a)+4, 3))
        j[:len(a), :2] = delta/dist[:, None]
        j[:len(a), 2] = -1
        j[len(a):] = [[1, 0, -1], [-1, 0, -1], [0, 1, -1], [0, -1, -1]]
        return c, j

    for point in starts:
        if time.monotonic() >= end:
            raise Deadline()
        v = np.r_[point, 1e-7]
        result = minimize(lambda v: -v[2], v, jac=lambda v: np.array([0., 0., -1.]),
                          constraints={'type': 'ineq', 'fun': lambda v: con(v)[0],
                                       'jac': lambda v: con(v)[1]},
                          bounds=[(0, w), (0, h), (1e-8, min(w, h)/2)],
                          method='SLSQP', options={'maxiter': 30, 'ftol': 1e-11})
        if result.x[2] > best[2] and np.min(con(result.x)[0]) > -1e-8:
            best = result.x
    return np.vstack((a, best))


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70*budget_s
    search_end = start + 0.65*budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a = np.array([[(k % 5+0.5)/5, (k//5+0.5)/5, 0.09] for k in range(21)])
        init = {'circles': a.tolist()}
    original = np.asarray(init['circles'], dtype=float)
    best = init
    score = exact_score(original)
    if original.shape != (21, 3) or not np.isfinite(original).all():
        return best
    lo = np.min(original[:, :2]-original[:, 2, None], axis=0)
    hi = np.max(original[:, :2]+original[:, 2, None], axis=0)
    scale = 2/np.sum(hi-lo)
    base = original.copy()
    base[:, :2] -= lo
    base *= scale
    width = float((hi[0]-lo[0])*scale)
    anchors = np.argsort(base[:, 2])
    for attempt in range(int(params['attempts'])):
        if time.monotonic() >= search_end:
            break
        anchor = int(anchors[(attempt*3) % 21])
        distances = np.linalg.norm(base[:, :2]-base[anchor, :2], axis=1)
        if attempt % 2:
            distances += rng.uniform(0, 0.12, 21)
            distances[anchor] = -1
        remove = np.argsort(distances)[:int(params['remove'])]
        a = np.delete(base, remove, axis=0)
        try:
            local_end = min(search_end, time.monotonic()+0.34)
            a, w = optimize_layout(a, width, local_end, int(params['iterations']))
            if time.monotonic() >= search_end:
                break
            for unused in range(len(remove)):
                if time.monotonic() >= search_end:
                    raise Deadline()
                a = insert_hole(a, w, params, search_end)
            a, w = optimize_layout(a, w, min(search_end, time.monotonic()+0.40),
                                   int(params['iterations'])+30)
            candidate = repair(a, end)
            if candidate is not None:
                value = exact_score(candidate)
                if value > score:
                    best = {'circles': candidate.tolist()}
                    score = value
        except Deadline:
            break
    return best
