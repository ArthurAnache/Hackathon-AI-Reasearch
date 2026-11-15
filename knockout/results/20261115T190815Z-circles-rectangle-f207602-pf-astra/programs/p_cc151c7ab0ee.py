import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import lstsq

PARAMS = {
    'restart_scale': (0.018, 0.001, 0.09),
    'maxiter': (65, 15, 150),
    'contact_tol': (2e-7, 1e-9, 1e-5, 'log'),
    'time_fraction': (0.70, 0.5, 0.8)
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class StopSearch(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + float(params['time_fraction']) * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def exact(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -np.inf
        d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
        if np.any(d < a[I, 2]+a[J, 2]):
            return -np.inf
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        perimeter = (max(t+q for t, q in zip(x, r))-min(t-q for t, q in zip(x, r))
                     + max(t+q for t, q in zip(y, r))-min(t-q for t, q in zip(y, r)))
        return float(np.sum(a[:, 2])) if perimeter <= 2 else -np.inf

    if init is not None:
        a0 = np.asarray(init['circles'], dtype=float).copy()
        best_score = exact(a0)
    else:
        a0 = np.zeros((21, 3))
        a0[:, 0] = (np.arange(21) % 5 + 0.5) / 5
        a0[:, 1] = (np.arange(21) // 5 + 0.5) / 5
        a0[:, 2] = 0.09
        best = {'circles': a0.tolist()}
        best_score = exact(a0)

    lo = np.min(a0[:, :2]-a0[:, 2, None], axis=0)
    sides = np.max(a0[:, :2]+a0[:, 2, None], axis=0)-lo
    factor = 2.0 / sides.sum()
    a0[:, :2] = (a0[:, :2]-lo)*factor
    a0[:, 2] *= factor
    sides *= factor

    # Choose the orientation with the cleaner horizontal reflection.
    def reflection_error(a, h):
        target = a.copy()
        target[:, 1] = h-target[:, 1]
        dist = np.sum((target[:, None, :]-a[None, :, :])**2, axis=2)
        return float(np.max(np.min(dist, axis=1)))

    swapped = reflection_error(a0[:, [1, 0, 2]], sides[0]) < reflection_error(a0, sides[1])
    if swapped:
        a0 = a0[:, [1, 0, 2]]
        sides = sides[::-1]
    z0 = np.r_[a0[:, 0], a0[:, 1], a0[:, 2], sides[0]]

    # Boundary constraints are linear; contact constraints use true distances.
    base = np.zeros((M+84, 64))
    ix = np.arange(21)
    base[np.arange(M), 42+I] = -1
    base[np.arange(M), 42+J] = -1
    base[M+ix, ix] = 1
    base[M+ix, 42+ix] = -1
    base[M+21+ix, ix] = -1
    base[M+21+ix, 42+ix] = -1
    base[M+21+ix, 63] = 1
    base[M+42+ix, 21+ix] = 1
    base[M+42+ix, 42+ix] = -1
    base[M+63+ix, 21+ix] = -1
    base[M+63+ix, 42+ix] = -1
    base[M+63+ix, 63] = -1

    def geometry(z, jac=False):
        check()
        x, y, r, w = z[:21], z[21:42], z[42:63], z[63]
        dx, dy = x[I]-x[J], y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy)
        if not jac:
            return np.r_[d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r]
        g = base.copy()
        d = np.maximum(d, 1e-100)
        rows = np.arange(M)
        g[rows, I] = dx/d
        g[rows, J] = -dx/d
        g[rows, 21+I] = dy/d
        g[rows, 21+J] = -dy/d
        return g

    def consider(z):
        nonlocal best, best_score
        check()
        p = np.column_stack((z[:21], z[21:42]))
        r = z[42:63].copy()
        if not np.isfinite(z).all() or np.any(r <= 0):
            return
        d = np.sqrt((p[I, 0]-p[J, 0])**2+(p[I, 1]-p[J, 1])**2)
        shrink = min(1.0, float(np.min(d/(r[I]+r[J]))))
        r *= shrink*(1-3e-15)
        low = np.min(p-r[:, None], axis=0)
        extent = np.max(p+r[:, None], axis=0)-low
        scale = (2.0/extent.sum())*(1-3e-15)
        p = (p-low)*scale
        r *= scale
        if float(r.sum()) <= best_score:
            return
        a = np.column_stack((p, r))
        if swapped:
            a = a[:, [1, 0, 2]]
        score = exact(a)
        if score > best_score:
            best_score = score
            best = {'circles': a.tolist()}

    # Build an affine reflection quotient: axis circles use two variables,
    # and each reflected pair uses three, plus the rectangle width.
    T = np.eye(64)
    offset = np.zeros(64)
    if reflection_error(a0, sides[1]) < 1e-7:
        columns = []
        offset = np.zeros(64)
        widthcol = np.zeros(64)
        widthcol[63] = 1
        used = set()
        for i in range(21):
            if time.monotonic() >= deadline:
                return best
            if i in used:
                continue
            target = a0[i].copy()
            target[1] = sides[1]-target[1]
            j = int(np.argmin(np.sum((a0-target)**2, axis=1)))
            if j in used:
                columns = []
                break
            used.add(i)
            used.add(j)
            cx = np.zeros(64)
            cr = np.zeros(64)
            cx[i] = cx[j] = 1
            cr[42+i] = cr[42+j] = 1
            columns.extend([cx, cr])
            if i == j:
                offset[21+i] = 1
                widthcol[21+i] = -0.5
            else:
                cy = np.zeros(64)
                cy[21+i], cy[21+j] = 1, -1
                offset[21+j] = 2
                widthcol[21+j] = -1
                columns.append(cy)
        if columns and len(used) == 21:
            columns.append(widthcol)
            T = np.column_stack(columns)
        else:
            T, offset = np.eye(64), np.zeros(64)

    objective = np.r_[np.zeros(42), -np.ones(21), 0.0]
    latest = z0.copy()

    def polish(z, mapping, shift):
        u = lstsq(mapping, z-shift, lapack_driver='gelsy')[0]
        z = mapping@u+shift
        active = np.flatnonzero(geometry(z) < float(params['contact_tol']))
        if len(active) < len(u):
            return z
        for _ in range(5):
            check()
            residual = geometry(z)[active]
            if np.max(np.abs(residual)) < 2e-16:
                break
            mat = geometry(z, True)[active]@mapping
            delta = lstsq(mat, -residual, lapack_driver='gelsy')[0]
            if np.linalg.norm(delta) > 0.02:
                break
            u += delta
            z = mapping@u+shift
        consider(z)
        return z

    try:
        consider(z0)
        for attempt in range(20):
            check()
            mapping = np.eye(64) if attempt == 0 else T
            shift = np.zeros(64) if attempt == 0 else offset
            center = latest if attempt < 3 else z0
            u = lstsq(mapping, center-shift, lapack_driver='gelsy')[0]
            if attempt:
                amplitude = float(params['restart_scale']) * (0.55, 1.0, 1.8, 2.5)[(attempt-1) % 4]
                u += rng.normal(0, amplitude, len(u))
            grad = objective@mapping
            saved = [mapping@u+shift]

            def fun(v):
                check()
                return float(objective@(mapping@v+shift))

            def con(v):
                z = mapping@v+shift
                return np.r_[geometry(z), z[42:63]-1e-8]

            def cjac(v):
                z = mapping@v+shift
                return np.vstack((geometry(z, True)@mapping, mapping[42:63]))

            def callback(v):
                check()
                z = mapping@v+shift
                saved[0] = z.copy()
                if np.min(geometry(z)) > -2e-8:
                    consider(z)

            result = minimize(fun, u, jac=lambda v: grad, method='SLSQP',
                              constraints={'type': 'ineq', 'fun': con, 'jac': cjac},
                              callback=callback,
                              options={'maxiter': int(params['maxiter']), 'ftol': 3e-14, 'disp': False})
            z = mapping@result.x+shift
            consider(z)
            z = polish(z, mapping, shift)
            if attempt == 0 or (np.min(geometry(z)) > -1e-8 and z[42:63].sum() > latest[42:63].sum()):
                latest = z.copy()
    except StopSearch:
        pass
    return best
