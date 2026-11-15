import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import qr

PARAMS = {
    'release_trials': (100, 20, 300),
    'restarts': (5, 1, 12),
    'step': (0.12, 0.02, 0.35),
    'time_fraction': (0.70, 0.50, 0.75)
}


def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + params['time_fraction'] * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    rows = np.arange(m)
    ids = np.arange(n)
    objective = np.zeros(64)
    objective[42:63] = -1.0

    class Stop(Exception):
        pass

    def check():
        if time.monotonic() >= deadline:
            raise Stop()

    def exact(data):
        a = np.asarray(data['circles'], dtype=np.float64)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        xs = [Fraction(float(v)) for v in a[:, 0]]
        ys = [Fraction(float(v)) for v in a[:, 1]]
        rs = [Fraction(float(v)) for v in a[:, 2]]
        width = max(x+r for x, r in zip(xs, rs)) - min(x-r for x, r in zip(xs, rs))
        height = max(y+r for y, r in zip(ys, rs)) - min(y-r for y, r in zip(ys, rs))
        if width + height > 2:
            return -float('inf')
        for a1, a2 in itertools.combinations(a, 2):
            if time.monotonic() >= deadline:
                raise Stop()
            dist = np.sqrt((a1[0]-a2[0])**2 + (a1[1]-a2[1])**2)
            if dist < a1[2] + a2[2]:
                return -float('inf')
        return float(np.sum(a[:, 2]))

    best = init
    best_score = -float('inf')
    try:
        if init is not None:
            best_score = exact(init)
    except Stop:
        return init

    if init is None:
        a = np.array([[j * 0.08, i * 0.08, 0.039] for i in range(3) for j in range(7)], dtype=float)
        best = {'circles': a.tolist()}
        best_score = float(a[:, 2].sum())
    else:
        a = np.array(init['circles'], dtype=float)

    def encode(a):
        x = a[:, 0] - np.min(a[:, 0] - a[:, 2])
        y = a[:, 1] - np.min(a[:, 1] - a[:, 2])
        w = np.max(x + a[:, 2])
        h = np.max(y + a[:, 2])
        s = 2.0 / (w + h)
        return np.r_[x*s, y*s, a[:, 2]*s, w*s]

    def consider(z):
        nonlocal best, best_score
        check()
        if not np.isfinite(z).all() or np.min(z[42:63]) <= 0:
            return
        q = np.column_stack((z[:21], z[21:42], z[42:63])).copy()
        dist = np.sqrt((q[ii, 0]-q[jj, 0])**2 + (q[ii, 1]-q[jj, 1])**2)
        ratio = float(np.min(dist / (q[ii, 2] + q[jj, 2])))
        q[:, 2] *= min(1.0, ratio) * (1.0 - 8e-15)
        w = np.max(q[:, 0]+q[:, 2])-np.min(q[:, 0]-q[:, 2])
        h = np.max(q[:, 1]+q[:, 2])-np.min(q[:, 1]-q[:, 2])
        q *= (2.0 / (w+h)) * (1.0 - 8e-15)
        if float(q[:, 2].sum()) <= best_score:
            return
        data = {'circles': q.tolist()}
        score = exact(data)
        if score > best_score:
            best, best_score = data, score

    cache_z = None
    cache_g = None
    cache_j = None

    def geometry(z):
        nonlocal cache_z, cache_g, cache_j
        check()
        if cache_z is not None and np.array_equal(z, cache_z):
            return cache_g, cache_j
        x, y, r, w = z[:21], z[21:42], z[42:63], z[63]
        dx, dy = x[ii]-x[jj], y[ii]-y[jj]
        d = np.sqrt(dx*dx + dy*dy)
        safe = np.maximum(d, 1e-30)
        g = np.concatenate((d-r[ii]-r[jj], x-r, w-x-r, y-r, 2-w-y-r))
        j = np.zeros((m+84, 64))
        j[rows, ii] = dx/safe
        j[rows, jj] = -dx/safe
        j[rows, 21+ii] = dy/safe
        j[rows, 21+jj] = -dy/safe
        j[rows, 42+ii] = -1
        j[rows, 42+jj] = -1
        j[m+ids, ids] = 1
        j[m+ids, 42+ids] = -1
        j[m+21+ids, ids] = -1
        j[m+21+ids, 42+ids] = -1
        j[m+21+ids, 63] = 1
        j[m+42+ids, 21+ids] = 1
        j[m+42+ids, 42+ids] = -1
        j[m+63+ids, 21+ids] = -1
        j[m+63+ids, 42+ids] = -1
        j[m+63+ids, 63] = -1
        cache_z, cache_g, cache_j = z.copy(), g, j
        return g, j

    bounds = [(None, None)] * 42 + [(1e-7, 0.8)] * 21 + [(0.1, 1.9)]
    last = encode(a)

    def polish(z, allowance, iterations):
        nonlocal last
        local_end = min(deadline, time.monotonic()+allowance)
        last = z.copy()
        count = 0
        def fun(v):
            check()
            if time.monotonic() >= local_end:
                raise Stop()
            return -float(np.sum(v[42:63]))
        def callback(v):
            nonlocal last, count
            check()
            last = v.copy()
            count += 1
            if count % 3 == 0:
                consider(v)
            if time.monotonic() >= local_end:
                raise Stop()
        try:
            result = minimize(fun, z, jac=lambda v: objective, method='SLSQP',
                              bounds=bounds,
                              constraints={'type': 'ineq', 'fun': lambda v: geometry(v)[0],
                                           'jac': lambda v: geometry(v)[1]},
                              callback=callback,
                              options={'maxiter': iterations, 'ftol': 2e-13, 'disp': False})
            last = result.x.copy()
        except Stop:
            pass
        if time.monotonic() < deadline:
            consider(last)
        return last.copy()

    try:
        z = polish(last, min(0.30, budget_s*0.17), 65)
        check()
        # Select a full-rank set of nearly active contacts and walls.
        g, jac = geometry(z)
        tight = np.flatnonzero(g < 2e-6)
        if len(tight) < 64:
            tight = np.argsort(g)[:64]
        _, _, piv = qr(jac[tight].T, pivoting=True, mode='economic')
        active = tight[piv[:64]]
        if len(active) != 64:
            return best
        # Newton refinement removes optimization and feasibility slack.
        for _ in range(5):
            check()
            g, jac = geometry(z)
            if np.max(np.abs(g[active])) < 3e-15:
                break
            try:
                delta = np.linalg.solve(jac[active], -g[active])
            except np.linalg.LinAlgError:
                break
            if np.max(np.abs(delta)) > 0.02:
                break
            z = z + delta
            consider(z)
        g, jac = geometry(z)
        if np.min(g) < -1e-7:
            z = encode(np.asarray(best['circles']))
            g, jac = geometry(z)
        try:
            inv = np.linalg.solve(jac[active], np.eye(64))
        except np.linalg.LinAlgError:
            return best
        multipliers = objective @ inv
        if not np.isfinite(inv).all() or np.max(np.abs(inv)) > 1e10:
            return best
        d0 = np.sqrt((z[ii]-z[jj])**2 + (z[21+ii]-z[21+jj])**2)
        nx = (z[ii]-z[jj])/np.maximum(d0, 1e-30)
        ny = (z[21+ii]-z[21+jj])/np.maximum(d0, 1e-30)
        probability = 1.0 / np.sqrt(np.maximum(multipliers, 0) + 0.03)
        probability /= probability.sum()
        proposals = []
        for trial in range(params['release_trials']):
            check()
            k = 3 + trial % 3
            selected = rng.choice(64, size=k, replace=False, p=probability)
            weights = rng.uniform(0.5, 1.5, size=k)
            direction = inv[:, selected] @ weights
            magnitude = np.max(np.sqrt(direction[:21]**2 + direction[21:42]**2))
            if magnitude < 1e-15:
                continue
            direction *= params['step']/magnitude
            vx = direction[ii]-direction[jj]
            vy = direction[21+ii]-direction[21+jj]
            bend = np.zeros(m+84)
            bend[:m] = np.maximum(0, vx*vx+vy*vy-(nx*vx+ny*vy)**2)/np.maximum(d0, 1e-30)
            acceleration = -inv @ bend[active]
            loss = float(objective @ direction)
            gain_curve = -float(objective @ acceleration)
            if loss <= 0 or gain_curve <= 1e-12:
                continue
            barrier = loss*loss/gain_curve
            distance = 2*loss/gain_curve
            # Prefer barriers reachable before the quadratic model becomes extreme.
            rank = barrier*(1 + max(0.0, distance-2.0)**2)
            proposals.append((rank, direction, acceleration, distance))
        proposals.sort(key=lambda v: v[0])
        used = []
        for _, direction, acceleration, crossing in proposals:
            check()
            if len(used) >= params['restarts']:
                break
            if any(abs(float(direction @ old))/(np.linalg.norm(direction)*np.linalg.norm(old)) > 0.94 for old in used):
                continue
            used.append(direction)
            t = min(1.7, max(0.65, 1.15*crossing))
            target = z + t*direction + 0.5*t*t*acceleration
            # Keep the topology-changing displacement local enough to polish.
            displacement = target-z
            size = np.max(np.abs(displacement[:42]))
            if size > 0.24:
                target = z + displacement*(0.24/size)
            target[42:63] = np.clip(target[42:63], 0.015, 0.3)
            target[63] = np.clip(target[63], 0.4, 1.6)
            remaining = deadline-time.monotonic()
            if remaining < 0.035:
                break
            polish(target, min(0.25, remaining), 70)
    except (Stop, np.linalg.LinAlgError, ValueError, FloatingPointError):
        pass
    return best
