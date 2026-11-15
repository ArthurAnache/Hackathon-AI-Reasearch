import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {
    'perturbation': (0.055, 0.015, 0.15),
    'restart_seconds': (0.245, 0.10, 0.40),
    'safety': (8e-15, 3e-15, 1e-11, 'log'),
    'iterations': (150, 50, 350)
}
I, J = np.triu_indices(21, 1)

class Stop(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    safety = float(params['safety'])
    best = init
    best_score = -float('inf')
    best_array = None

    def read(obj):
        if 'circles' in obj:
            return np.asarray(obj['circles'], dtype=float).copy()
        return np.asarray(obj['state'], dtype=float).reshape(3, 21).T.copy()

    def exact(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
        if np.any(d < a[I, 2]+a[J, 2]):
            return -float('inf')
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        perimeter = (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
                     +max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)))
        if perimeter > 2:
            return -float('inf')
        return float(np.sum(a[:, 2]))

    if init is not None:
        try:
            a = read(init)
            best_score = exact(a)
            if np.isfinite(best_score):
                best_array = a.copy()
        except (ValueError, TypeError, KeyError):
            pass

    def offer(a):
        nonlocal best, best_score, best_array
        if time.monotonic() >= deadline:
            return
        a = np.asarray(a, dtype=float).copy()
        if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return
        d = np.sqrt((a[I, 0]-a[J, 0])**2+(a[I, 1]-a[J, 1])**2)
        ratio = float(np.min(d/(a[I, 2]+a[J, 2])))
        if ratio <= 0:
            return
        a[:, 2] *= min(1.0, ratio)*(1-safety)
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        perimeter = float(np.sum(hi-lo))
        if perimeter <= 0:
            return
        a[:, :2] -= (lo+hi)/2
        a *= (2/perimeter)*(1-4*safety)
        if float(np.sum(a[:, 2])) <= best_score:
            return
        score = exact(a)
        if score > best_score:
            best_score = score
            best_array = a.copy()
            best = {'circles': a.tolist()}

    if best_array is None:
        fallback = np.array([[(i-2)*2/9, (j-1.5)*2/9, 1/9] for j in range(4) for i in range(5)] + [[-1/9, -2/9, (np.sqrt(2)-1)/9]])
        offer(fallback)
    if best_array is None:
        return best

    # Three axial representatives and nine representatives of reflected pairs.
    A = np.zeros((64, 34))
    for k in range(12):
        if time.monotonic() >= deadline:
            return best
        A[k, k] = 1
        A[42+k, 21+k] = 1
        if k >= 3:
            A[21+k, 12+k-3] = 1
            A[12+k-3, k] = 1
            A[21+12+k-3, 12+k-3] = -1
            A[42+12+k-3, 21+k] = 1
    A[63, 33] = 1
    pairs = [(i,j) for i in range(12) for j in range(i+1,12)]
    pairs += [(i,12+j-3) for i in range(3,12) for j in range(i,12)]
    pi, pj = np.asarray(pairs).T
    X, Y, R = A[:21], A[21:42], A[42:63]
    DX, DY, SR = X[pi]-X[pj], Y[pi]-Y[pj], R[pi]+R[pj]
    W = A[63]
    wall = np.concatenate((W[None,:]/2-X[:12]-R[:12],
                           W[None,:]/2+X[:12]-R[:12],
                           -W[None,:]/2-Y[:12]-R[:12],
                           -W[None,:]/2+Y[:12]-R[:12]))
    offsets = np.r_[np.zeros(24), np.ones(24)]
    objective = -np.sum(R, axis=0)
    bounds = [(-1.,1.)]*12 + [(0.,.95)]*9 + [(1e-7,.49)]*12 + [(.15,1.85)]

    def unpack(q):
        z = A@q
        return np.column_stack((z[:21], z[21:42], z[42:63]))

    def optimize(q, stop):
        stop = min(stop, deadline-.008)
        if time.monotonic() >= stop:
            return q
        weights = 1/np.maximum(2*(SR@q), .08)
        cache = [None, None, None]
        last = [q.copy()]
        calls = [0]
        def check():
            if time.monotonic() >= stop:
                raise Stop()
        def calc(v):
            check()
            if cache[0] is not None and np.array_equal(cache[0], v):
                return cache[1], cache[2]
            dx, dy, rr = DX@v, DY@v, SR@v
            c = np.r_[(dx*dx+dy*dy-rr*rr)*weights, wall@v+offsets]
            jac = np.vstack((2*weights[:,None]*(dx[:,None]*DX+dy[:,None]*DY-rr[:,None]*SR), wall))
            cache[:] = [v.copy(), c, jac]
            last[0] = v.copy()
            return c, jac
        def fun(v):
            check()
            return float(objective@v), objective
        def callback(v):
            check()
            calls[0] += 1
            if calls[0] % 12 == 0:
                offer(unpack(v))
        try:
            result = minimize(fun, q, jac=True, method='SLSQP', bounds=bounds,
                              constraints={'type':'ineq', 'fun':lambda v:calc(v)[0], 'jac':lambda v:calc(v)[1]},
                              callback=callback, options={'maxiter':int(params['iterations']), 'ftol':2e-14, 'disp':False})
            q = result.x
        except Stop:
            q = last[0]
        except (ValueError, FloatingPointError):
            q = last[0]
        offer(unpack(q))
        return q

    # Recover whichever rectangle orientation has the horizontal reflection.
    choices = []
    for swap in (False, True):
        if time.monotonic() >= deadline-.02:
            return best
        a = best_array.copy()
        if swap:
            a[:, :2] = a[:, [1, 0]]
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        a[:, :2] -= (lo+hi)/2
        reflected = a.copy()
        reflected[:, 1] *= -1
        cost = np.sum((a[:, None, :]-reflected[None, :, :])**2, axis=2)
        rows, cols = linear_sum_assignment(cost)
        choices.append((float(cost[rows, cols].sum()), a, float(hi[0]-lo[0])))
    _, a, width = min(choices, key=lambda item:item[0])
    singles = np.argsort(np.abs(a[:, 1]))[:3]
    remaining = np.setdiff1d(np.arange(21), singles)
    order = remaining[np.argsort(a[remaining, 1])]
    negative, positive = order[:9], order[9:]
    cost = ((a[positive, None, 0]-a[None, negative, 0])**2
            +(a[positive, None, 1]+a[None, negative, 1])**2
            +(a[positive, None, 2]-a[None, negative, 2])**2)
    _, cols = linear_sum_assignment(cost)
    negative = negative[cols]
    base = np.r_[a[singles, 0], .5*(a[positive, 0]+a[negative, 0]),
                 .5*(a[positive, 1]-a[negative, 1]), a[singles, 2],
                 .5*(a[positive, 2]+a[negative, 2]), width]
    base = optimize(base, min(deadline, start+.12*budget_s))
    anchor = base.copy()

    attempt = 0
    while time.monotonic() < deadline-.045:
        now = time.monotonic()
        if attempt == 1:
            # Complementary structured seed from the first parent's lattice family.
            near, far = (5, 4) if seed % 2 else (4, 5)
            trial = np.zeros(34)
            trial[:3] = [-.28, 0., .28]
            trial[21:33] = .10
            trial[33] = 1.
            k = 3
            for count, height in ((near, .21), (far, .41)):
                if time.monotonic() >= deadline:
                    return best
                trial[k:k+count] = np.linspace(-.105*(count-1), .105*(count-1), count)
                trial[12+k-3:12+k-3+count] = height
                k += count
        else:
            trial = anchor.copy()
            count = 4 + attempt % 3
            selected = rng.choice(12, count, replace=False)
            amplitude = float(params['perturbation'])*(1+.3*(attempt % 3))
            trial[selected] += rng.normal(0, amplitude, count)
            paired = selected[selected >= 3]
            trial[12+paired-3] = np.abs(trial[12+paired-3]+rng.normal(0, amplitude, len(paired)))
            changes = rng.normal(0, .22, count)
            trial[21+selected] *= np.exp(changes)
            trial[33] = np.clip(trial[33]+rng.normal(0, amplitude*.6), .3, 1.7)
        trial[21:33] = np.clip(trial[21:33], 1e-5, .45)
        trial[12:21] = np.clip(trial[12:21], 0., .9)
        result = optimize(trial, min(deadline-.01, now+float(params['restart_seconds'])))
        if objective@result < objective@base and np.min(wall@result+offsets) > -1e-8:
            dx, dy, rr = DX@result, DY@result, SR@result
            if np.min(dx*dx+dy*dy-rr*rr) > -1e-8:
                base = result.copy()
                anchor = base.copy()
        attempt += 1
    return best
