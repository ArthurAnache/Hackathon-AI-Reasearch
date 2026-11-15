import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'iterations': (110, 30, 200),
    'contact_margin': (0.11, 0.04, 0.3),
    'first_fraction': (0.48, 0.25, 0.65),
    'perturbation': (0.018, 0.002, 0.08, 'log'),
    'safety': (1e-14, 2e-15, 1e-11, 'log')
}

class Halt(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    begin = time.monotonic()
    deadline = begin + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    ident = np.eye(21)
    best = init
    best_score = -float('inf')
    best_array = None
    safety = float(params['safety'])
    gradient = np.r_[np.zeros(42), -np.ones(21), 0.]

    def exact(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        q = [[Fraction(float(v)) for v in row] for row in a]
        extent = Fraction(0)
        for axis in range(2):
            if time.monotonic() >= begin + 0.91 * budget_s:
                return -float('inf')
            extent += max(c[axis] + c[2] for c in q) - min(c[axis] - c[2] for c in q)
        if extent > 2:
            return -float('inf')
        for i, j in zip(ii, jj):
            if time.monotonic() >= begin + 0.93 * budget_s:
                return -float('inf')
            d = np.sqrt((a[i, 0]-a[j, 0])**2 + (a[i, 1]-a[j, 1])**2)
            if d < a[i, 2] + a[j, 2]:
                return -float('inf')
        return float(np.sum(a[:, 2]))

    original = None
    if init is not None:
        original = np.asarray(init['circles'], dtype=float).copy()
        best_score = exact(original)
        if np.isfinite(best_score):
            best_array = original.copy()

    base = np.array([[2*c+1., 2*r+1., 1.] for r in range(4) for c in range(5)] + [[4., 4., np.sqrt(2.)-1.]])/9.

    def retain(a):
        nonlocal best, best_score, best_array
        if time.monotonic() >= deadline:
            raise Halt()
        a = np.asarray(a, dtype=float).copy()
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return
        delta = a[ii, :2]-a[jj, :2]
        distances = np.sqrt(np.sum(delta*delta, axis=1))
        shrink = min(1., float(np.min(distances/(a[ii, 2]+a[jj, 2]))))
        if shrink <= 0:
            return
        a[:, 2] *= shrink*(1.-safety)
        lower = np.min(a[:, :2]-a[:, 2, None], axis=0)
        a[:, :2] -= lower
        wh = np.max(a[:, :2]+a[:, 2, None], axis=0)
        a *= (2./float(np.sum(wh)))*(1.-safety)
        if float(np.sum(a[:, 2])) <= best_score:
            return
        score = exact(a)
        if score > best_score:
            best_score = score
            best_array = a.copy()
            best = {'circles': a.tolist()}

    def optimize(a, duration, margin):
        end = min(deadline, time.monotonic()+duration)
        def check():
            if time.monotonic() >= end:
                raise Halt()
        a = np.array(a, dtype=float, copy=True)
        a[:, 2] = np.maximum(a[:, 2], 1e-5)
        a[:, :2] -= np.min(a[:, :2]-a[:, 2, None], axis=0)
        wh = np.max(a[:, :2]+a[:, 2, None], axis=0)
        scale = 2./float(np.sum(wh))
        a *= scale
        z = np.r_[a[:, 0], a[:, 1], a[:, 2], wh[0]*scale]
        d = np.sqrt(np.sum((a[ii, :2]-a[jj, :2])**2, axis=1))
        mask = d-a[ii, 2]-a[jj, 2] < margin
        pi, pj = ii[mask], jj[mask]
        m = len(pi)
        rows = np.arange(m)
        fixed = np.zeros((m+84, 64))
        fixed[rows, 42+pi] = -1.
        fixed[rows, 42+pj] = -1.
        fixed[m:m+21, :21] = ident
        fixed[m:m+21, 42:63] = -ident
        fixed[m+21:m+42, :21] = -ident
        fixed[m+21:m+42, 42:63] = -ident
        fixed[m+21:m+42, 63] = 1.
        fixed[m+42:m+63, 21:42] = ident
        fixed[m+42:m+63, 42:63] = -ident
        fixed[m+63:m+84, 21:42] = -ident
        fixed[m+63:m+84, 42:63] = -ident
        fixed[m+63:m+84, 63] = -1.

        def constraints(v):
            check()
            x, y, r, w = v[:21], v[21:42], v[42:63], v[63]
            dx, dy = x[pi]-x[pj], y[pi]-y[pj]
            return np.r_[np.sqrt(dx*dx+dy*dy+1e-30)-r[pi]-r[pj], x-r, w-x-r, y-r, 2.-w-y-r]

        def jacobian(v):
            check()
            dx, dy = v[pi]-v[pj], v[21+pi]-v[21+pj]
            d = np.sqrt(dx*dx+dy*dy+1e-30)
            j = fixed.copy()
            j[rows, pi], j[rows, pj] = dx/d, -dx/d
            j[rows, 21+pi], j[rows, 21+pj] = dy/d, -dy/d
            return j

        last = a.copy()
        def callback(v):
            nonlocal last
            check()
            last = np.column_stack((v[:21], v[21:42], v[42:63]))
            retain(last)

        try:
            result = minimize(lambda v: -float(np.sum(v[42:63])), z,
                jac=lambda v: gradient, method='SLSQP',
                bounds=[(None, None)]*42+[(1e-6, 0.85)]*21+[(0.12, 1.88)],
                constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                callback=callback,
                options={'maxiter': int(params['iterations']), 'ftol': 3e-13, 'disp': False})
            callback(result.x)
        except (Halt, ValueError, FloatingPointError):
            pass
        return last

    try:
        retain(base)
        origin = original if original is not None and original.shape == (21, 3) and np.isfinite(original).all() else base
        # Keep the original trajectory even if the fallback temporarily scores higher.
        trajectory = optimize(origin, float(params['first_fraction'])*budget_s, float(params['contact_margin']))
        if time.monotonic() < deadline-0.06:
            trajectory = optimize(trajectory, min(0.22, deadline-time.monotonic()-0.015), 0.25)
        attempt = 0
        while time.monotonic() < deadline-0.055:
            if best_score < 2.30 and attempt == 0:
                trial = base.copy()
                allowance = 0.28
            else:
                trial = best_array.copy() if best_array is not None else trajectory.copy()
                amplitude = float(params['perturbation'])*(1.6**(attempt % 3))
                pivot = int(rng.integers(21))
                order = np.argsort(np.sum((trial[:, :2]-trial[pivot, :2])**2, axis=1))
                group = order[:int(rng.integers(4, 10))]
                direction = rng.normal(size=2)
                direction /= max(1e-12, float(np.linalg.norm(direction)))
                trial[group, :2] += amplitude*direction
                trial[group, 2] *= np.exp(rng.normal(0., amplitude, len(group)))
                allowance = 0.24
            trajectory = optimize(trial, min(allowance, deadline-time.monotonic()-0.008), 0.16)
            attempt += 1
    except (Halt, ValueError, FloatingPointError):
        pass
    if best is not None and np.isfinite(best_score):
        return best
    base[:, 2] *= 1.-1e-10
    base *= 1.-1e-10
    return {'circles': base.tolist()}
