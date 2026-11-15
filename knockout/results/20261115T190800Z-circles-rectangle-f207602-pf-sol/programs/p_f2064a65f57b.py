import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'iterations': (95, 30, 200),
    'warm_fraction': (0.24, 0.1, 0.5),
    'family_fraction': (0.15, 0.06, 0.3),
    'tolerance': (2e-13, 1e-14, 1e-9, 'log'),
}

class Stop(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.695 * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    best = init
    best_score = -float('inf')
    archive = []

    def check(until=None):
        if time.monotonic() >= (deadline if until is None else min(until, deadline)):
            raise Stop()

    def exact_score(data):
        try:
            a = np.asarray(data['circles'], dtype=np.float64)
            if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
                return -float('inf')
            dx = a[ii, 0] - a[jj, 0]
            dy = a[ii, 1] - a[jj, 1]
            if np.any(np.sqrt(dx**2 + dy**2) < a[ii, 2] + a[jj, 2]):
                return -float('inf')
            xs = [Fraction(float(x)) for x in a[:, 0]]
            ys = [Fraction(float(y)) for y in a[:, 1]]
            rs = [Fraction(float(r)) for r in a[:, 2]]
            width = max(x+r for x, r in zip(xs, rs)) - min(x-r for x, r in zip(xs, rs))
            height = max(y+r for y, r in zip(ys, rs)) - min(y-r for y, r in zip(ys, rs))
            if width + height > 2:
                return -float('inf')
            return float(np.sum(a[:, 2]))
        except Exception:
            return -float('inf')

    if init is not None:
        best_score = exact_score(init)

    def retain(c):
        nonlocal best, best_score
        check()
        c = np.array(c, dtype=float, copy=True)
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return
        dx = c[ii, 0] - c[jj, 0]
        dy = c[ii, 1] - c[jj, 1]
        dist = np.sqrt(dx**2 + dy**2)
        f = min(1., float(np.min(dist / (c[ii, 2] + c[jj, 2]))))
        if f <= 0:
            return
        c[:, 2] *= f * (1. - 7e-15)
        lo = np.min(c[:, :2] - c[:, 2, None], axis=0)
        hi = np.max(c[:, :2] + c[:, 2, None], axis=0)
        c[:, :2] -= (hi + lo) * 0.5
        c *= (2. / float(np.sum(hi - lo))) * (1. - 7e-15)
        if float(np.sum(c[:, 2])) <= best_score:
            return
        candidate = {'circles': c.tolist()}
        score = exact_score(candidate)
        if score > best_score:
            best, best_score = candidate, score

    # Each representative is (positive horizontal offset, height, radius, paired).
    def model(reps, width):
        check()
        nv = sum(3 if q[3] else 2 for q in reps) + 1
        X, Y, R, walls, constants, bounds = [], [], [], [], [], []
        z = np.zeros(nv)
        p = 0
        mirror = []
        for xx, yy, rr, paired in reps:
            check()
            x, y, r = np.zeros(nv), np.zeros(nv), np.zeros(nv)
            if paired:
                x[p] = 1.; z[p] = abs(xx)
                bounds.append((0., 1.)); p += 1
            y[p] = 1.; z[p] = yy
            bounds.append((0., 2.)); p += 1
            r[p] = 1.; z[p] = rr
            bounds.append((1e-7, 0.8)); p += 1
            k = len(X)
            X.append(x); Y.append(y); R.append(r)
            if paired:
                X.append(-x); Y.append(y); R.append(r)
                mirror.extend((k+1, k))
            else:
                mirror.append(k)
            wall = -x-r; wall[-1] += .5
            walls.append(wall); constants.append(0.)
            walls.append(y-r); constants.append(0.)
            wall = -y-r; wall[-1] -= 1.
            walls.append(wall); constants.append(2.)
        if len(X) != 21:
            return None, None
        z[-1] = width
        bounds.append((.12, 1.88))
        X, Y, R = np.asarray(X), np.asarray(Y), np.asarray(R)
        keep = []
        for k, (i, j) in enumerate(zip(ii, jj)):
            check()
            if (int(i), int(j)) <= tuple(sorted((mirror[i], mirror[j]))):
                keep.append(k)
        a, b = ii[keep], jj[keep]
        DX, DY, SR = X[a]-X[b], Y[a]-Y[b], R[a]+R[b]
        W, wc = np.asarray(walls), np.asarray(constants)
        obj = -R.sum(axis=0)
        return (X, Y, R, DX, DY, SR, W, wc, obj, bounds), z

    def unpack(m, z):
        return np.column_stack((m[0] @ z, m[1] @ z, m[2] @ z))

    def calc(m, z):
        dx, dy, sr, w, wc = m[3:8]
        xx, yy = dx @ z, dy @ z
        dd = np.sqrt(xx*xx + yy*yy + 1e-32)
        values = np.concatenate((dd - sr @ z, w @ z + wc))
        jac = np.vstack(((xx/dd)[:, None]*dx + (yy/dd)[:, None]*dy - sr, w))
        return values, jac

    def polish(m, z, until):
        # A determined active contact system is faster and more accurate than
        # asking SQP to resolve its last few floating-point residuals.
        check(until)
        vals, jac = calc(m, z)
        active = np.flatnonzero(vals < 2e-7)
        if len(active) < len(z) or np.min(vals) < -1e-6:
            return z
        v = z.copy()
        for iteration in range(7):
            check(until)
            values, J = calc(m, v)
            A, residual = J[active], values[active]
            try:
                step = np.linalg.lstsq(A, -residual, rcond=1e-11)[0]
            except np.linalg.LinAlgError:
                break
            if np.linalg.norm(step) > .025:
                break
            v += step
            retain(unpack(m, v))
            if np.max(np.abs(residual)) < 2e-16:
                break
        return v

    def solve(m, z, until, iterations):
        last = z.copy()
        cache = [None, None, None]
        count = 0
        def cached(v):
            check(until)
            if cache[0] is None or not np.array_equal(v, cache[0]):
                values, jac = calc(m, v)
                cache[:] = [v.copy(), values, jac]
            return cache[1], cache[2]
        def objective(v):
            check(until)
            return float(m[8] @ v)
        def callback(v):
            nonlocal last, count
            check(until)
            last = v.copy(); count += 1
            if count % 4 == 0 and -m[8] @ v > best_score:
                retain(unpack(m, v))
        try:
            check(until)
            res = minimize(objective, z, jac=lambda v: m[8], method='SLSQP',
                           bounds=m[9], constraints={'type': 'ineq',
                           'fun': lambda v: cached(v)[0], 'jac': lambda v: cached(v)[1]},
                           callback=callback,
                           options={'maxiter': int(iterations), 'ftol': params['tolerance'], 'disp': False})
            last = res.x
        except Stop:
            pass
        check()
        retain(unpack(m, last))
        vals, _ = calc(m, last)
        if np.min(vals) > -1e-7:
            archive.append((float(-m[8] @ last), m, last.copy()))
            if time.monotonic() < min(until, deadline)-.008:
                try:
                    last = polish(m, last, until)
                except Stop:
                    pass
        return last

    def warm_model(data):
        raw = np.asarray(data['circles'], dtype=float)
        choices = []
        for swap in (False, True):
            check()
            c = raw.copy()
            if swap:
                c[:, [0, 1]] = c[:, [1, 0]]
            lo = np.min(c[:, :2]-c[:, 2, None], axis=0)
            hi = np.max(c[:, :2]+c[:, 2, None], axis=0)
            c[:, 0] -= (lo[0]+hi[0])*.5
            c[:, 1] -= lo[1]
            used = set(); reps = []; error = 0.
            for k in np.argsort(np.abs(c[:, 0])):
                check()
                k = int(k)
                if k in used:
                    continue
                if abs(c[k, 0]) < 2e-6:
                    reps.append((0., c[k, 1], c[k, 2], False))
                    error = max(error, abs(c[k, 0])); used.add(k)
                else:
                    target = c[k].copy(); target[0] *= -1
                    costs = np.sum((c-target)**2, axis=1)
                    costs[k] = np.inf
                    for j in used:
                        check(); costs[j] = np.inf
                    j = int(np.argmin(costs))
                    error = max(error, float(np.sqrt(costs[j])))
                    reps.append(((abs(c[k, 0])+abs(c[j, 0]))*.5,
                                 (c[k, 1]+c[j, 1])*.5,
                                 (c[k, 2]+c[j, 2])*.5, True))
                    used.update((k, j))
            if sum(2 if q[3] else 1 for q in reps) == 21:
                choices.append((error, reps, float(hi[0]-lo[0])))
        if not choices:
            return None, None
        error, reps, width = min(choices, key=lambda q: q[0])
        if error > 2e-5:
            return None, None
        return model(reps, width)

    def lattice(rows):
        reps = []; y = 1.; previous = None
        for count in rows:
            check()
            if previous is not None:
                y += np.sqrt(3.) if count % 2 != previous % 2 else 2.
            if count % 2:
                reps.append((0., y, 1., False))
            for k in range(count//2):
                check()
                reps.append((float(2*k+1 if count % 2 == 0 else 2*k+2), y, 1., True))
            previous = count
        width = 2.*max(rows)
        scale = 2./(width+y+1.)
        reps = [(x*scale, yy*scale, r*scale, paired) for x, yy, r, paired in reps]
        return model(reps, width*scale)

    try:
        if init is not None:
            m, z = warm_model(init)
            if m is not None:
                until = min(deadline, start + params['warm_fraction']*budget_s)
                solve(m, z, until, 65)
        # Different axis-circle counts explore different reflection quotients.
        families = [(4,4,5,4,4), (5,3,5,3,5), (4,3,4,3,4,3),
                    (6,5,6,4), (5,4,5,4,3), (4,5,3,5,4), (7,7,7)]
        # Keep the first two reproducible; diversify subsequent family order.
        tail = list(rng.permutation(len(families)-2)+2)
        order = [0, 1] + tail
        for index in order:
            check()
            if deadline-time.monotonic() < .055:
                break
            m, z = lattice(families[index])
            until = min(deadline-.015, time.monotonic()+params['family_fraction']*budget_s)
            solve(m, z, until, params['iterations'])
        if archive and time.monotonic() < deadline-.025:
            _, m, z = max(archive, key=lambda q: q[0])
            solve(m, z, deadline-.005, 100)
    except Stop:
        pass
    if best is None:
        best = {'circles': [[(k % 7)*.14, (k//7)*.14, .069] for k in range(21)]}
    return best
