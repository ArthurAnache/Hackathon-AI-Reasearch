import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'poles': (480, 40, 2000),
    'starts': (5, 1, 12),
    'iterations': (105, 20, 300),
    'diversity': (0.065, 0.01, 0.3),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class Deadline(Exception):
    pass


def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0] - a[J, 0]) ** 2 + (a[I, 1] - a[J, 1]) ** 2)
    if np.any(d < a[I, 2] + a[J, 2]):
        return -np.inf
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    perimeter = (max(u + v for u, v in zip(x, r)) - min(u - v for u, v in zip(x, r))
                 + max(u + v for u, v in zip(y, r)) - min(u - v for u, v in zip(y, r)))
    if perimeter > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))


def normalize(a):
    lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
    hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
    scale = 2.0 / np.sum(hi - lo)
    b = a.copy()
    b[:, :2] = (b[:, :2] - lo) * scale
    b[:, 2] *= scale
    return b, float((hi[0] - lo[0]) * scale)


def construct(params, seed, budget_s, init=None):
    begin = time.monotonic()
    deadline = begin + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf
    if init is not None:
        a = np.asarray(init['circles'], dtype=float)
        best_score = exact_score(a)
    else:
        a = np.array([[(k % 5 + .5) / 5, (k // 5 + .5) / 5, .045] for k in range(N)])
        best = {'circles': a.tolist()}
        best_score = exact_score(a)
    base, w0 = normalize(a)

    def check():
        if time.monotonic() >= deadline:
            raise Deadline()

    def submit(b):
        nonlocal best, best_score
        check()
        if not np.isfinite(b).all() or np.any(b[:, 2] <= 0):
            return
        b, unused = normalize(b)
        d = np.sqrt((b[I, 0] - b[J, 0]) ** 2 + (b[I, 1] - b[J, 1]) ** 2)
        ratios = d / (b[I, 2] + b[J, 2])
        factor = min(1.0, float(np.min(ratios))) * (1.0 - 3e-14)
        b[:, 2] *= factor
        s = exact_score(b)
        if s > best_score:
            best_score = s
            best = {'circles': b.tolist()}

    objective_gradient = np.r_[np.zeros(2*N), -np.ones(N), 0.0]
    idx = np.arange(N)

    def objective(z):
        check()
        return -float(np.sum(z[2*N:3*N])), objective_gradient

    def constraints(z):
        check()
        x, y, r, w = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
        dx, dy = x[I] - x[J], y[I] - y[J]
        d = np.sqrt(dx*dx + dy*dy + 1e-30)
        return np.r_[d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r]

    def jacobian(z):
        check()
        x, y = z[:N], z[N:2*N]
        dx, dy = x[I] - x[J], y[I] - y[J]
        d = np.sqrt(dx*dx + dy*dy + 1e-30)
        q = np.zeros((M+4*N, 3*N+1))
        rows = np.arange(M)
        q[rows, I] = dx/d
        q[rows, J] = -dx/d
        q[rows, N+I] = dy/d
        q[rows, N+J] = -dy/d
        q[rows, 2*N+I] = -1
        q[rows, 2*N+J] = -1
        q[M+idx, idx] = 1
        q[M+N+idx, idx] = -1
        q[M+2*N+idx, N+idx] = 1
        q[M+3*N+idx, N+idx] = -1
        q[M+idx, 2*N+idx] = -1
        q[M+N+idx, 2*N+idx] = -1
        q[M+2*N+idx, 2*N+idx] = -1
        q[M+3*N+idx, 2*N+idx] = -1
        q[M+N+idx, -1] = 1
        q[M+3*N+idx, -1] = -1
        return q

    pool = []
    try:
        center = np.array([w0/2, (2-w0)/2])
        for k in range(int(params['poles'])):
            check()
            if k % 3 == 0:
                pole = rng.uniform([-.15*w0, -.15*(2-w0)], [1.15*w0, 1.15*(2-w0)])
            else:
                angle = rng.uniform(0, 2*np.pi)
                radius = np.exp(rng.uniform(np.log(.35), np.log(3.5)))
                pole = center + radius*np.array([np.cos(angle), np.sin(angle)])
            delta = base[:, :2] - pole
            den = np.sum(delta*delta, axis=1) - base[:, 2]**2
            if np.min(den) <= 1e-7:
                continue
            b = np.column_stack((delta / den[:, None], base[:, 2]/den))
            b, w = normalize(b)
            score = float(b[:, 2].sum())
            if score < 1.85:
                continue
            feature = np.r_[b[:, 2], w*.15]
            interior = bool(0 < pole[0] < w0 and 0 < pole[1] < 2-w0)
            pool.append((score, interior, b, w, feature))
        pool.sort(key=lambda item: item[0], reverse=True)
        chosen = []
        inner = [item for item in pool if item[1]]
        if inner:
            chosen.append(inner[0])
        for item in pool:
            check()
            if all(np.linalg.norm(item[4]-other[4]) > float(params['diversity']) for other in chosen):
                chosen.append(item)
            if len(chosen) >= int(params['starts']):
                break
        if not chosen:
            chosen = [(0, False, base, w0, None)]
        # Exterior inversions usually retain more of the original objective.
        chosen.sort(key=lambda item: item[0], reverse=True)
        for number, item in enumerate(chosen):
            check()
            b, w = item[2], item[3]
            submit(b.copy())
            z = np.r_[b[:, 0], b[:, 1], b[:, 2], w]
            local_end = min(deadline, time.monotonic() + max(.12, (deadline-time.monotonic()) / max(1, len(chosen)-number)))
            last = [z.copy()]
            def callback(v):
                check()
                last[0] = v.copy()
                if time.monotonic() > local_end:
                    raise StopIteration()
            try:
                result = minimize(objective, z, jac=True, method='SLSQP',
                    bounds=[(None, None)]*(2*N)+[(1e-7, 1)]*N+[(.05, 1.95)],
                    constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                    callback=callback,
                    options={'maxiter': int(params['iterations']), 'ftol': 2e-12, 'disp': False})
                last[0] = result.x
            except StopIteration:
                pass
            v = last[0]
            submit(np.column_stack((v[:N], v[N:2*N], v[2*N:3*N])))
    except Deadline:
        pass
    return best
