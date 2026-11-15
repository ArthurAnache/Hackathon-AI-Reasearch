import time
import math
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog
from scipy.linalg import lstsq

PARAMS = {
    "restart_scale": (0.037, 0.002, 0.2, "log"),
    "steps": (14, 4, 40),
    "time_fraction": (0.69, 0.4, 0.75)
}


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + float(params["time_fraction"]) * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    ids = np.arange(n)

    def exact_score(obj):
        try:
            c = np.asarray(obj['circles'], dtype=float)
            if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
                return -math.inf
            xs, ys, rs = [], [], []
            for x, y, r in c:
                time.monotonic()
                xs.append(Fraction(float(x)))
                ys.append(Fraction(float(y)))
                rs.append(Fraction(float(r)))
            w = max(x+r for x, r in zip(xs, rs)) - min(x-r for x, r in zip(xs, rs))
            h = max(y+r for y, r in zip(ys, rs)) - min(y-r for y, r in zip(ys, rs))
            if w+h > 2:
                return -math.inf
            for i, j in zip(ii, jj):
                time.monotonic()
                d = np.sqrt((c[i, 0]-c[j, 0])**2 + (c[i, 1]-c[j, 1])**2)
                if d < c[i, 2]+c[j, 2]:
                    return -math.inf
            return float(np.sum(c[:, 2]))
        except (ValueError, TypeError, KeyError, OverflowError):
            return -math.inf

    if init is None:
        init = {'circles': [[0.1+0.2*(k % 5), 0.1+0.2*(k//5), 0.1-2e-15] for k in range(21)]}
    best = init
    bestscore = exact_score(init)
    initial = np.asarray(init['circles'], dtype=float).copy()
    lo = np.min(initial[:, :2]-initial[:, 2, None], axis=0)
    initial[:, :2] -= lo
    extent = np.max(initial[:, :2]+initial[:, 2, None], axis=0)
    base = np.r_[initial[:, 0], initial[:, 1], initial[:, 2], extent[0]]

    def save(z):
        nonlocal best, bestscore
        if time.monotonic() >= deadline:
            return
        c = np.column_stack((z[:n], z[n:2*n], z[2*n:3*n]))
        if not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return
        delta = c[ii, :2]-c[jj, :2]
        distances = np.sqrt(np.sum(delta*delta, axis=1))
        ratio = float(np.min(distances/(c[ii, 2]+c[jj, 2])))
        c[:, 2] *= min(1.0, ratio)*(1.0-3e-15)
        extent = np.max(c[:, :2]+c[:, 2, None], axis=0)-np.min(c[:, :2]-c[:, 2, None], axis=0)
        factor = min(1.0, 2.0/float(np.sum(extent)))*(1.0-3e-15)
        c *= factor
        if float(np.sum(c[:, 2])) <= bestscore:
            return
        obj = {'circles': c.tolist()}
        score = exact_score(obj)
        if score > bestscore:
            bestscore, best = score, obj

    boundary = np.zeros((4*n, 3*n+1))
    boundary[ids, ids] = -1
    boundary[ids, 2*n+ids] = 1
    boundary[n+ids, ids] = 1
    boundary[n+ids, 2*n+ids] = 1
    boundary[n+ids, -1] = -1
    boundary[2*n+ids, n+ids] = -1
    boundary[2*n+ids, 2*n+ids] = 1
    boundary[3*n+ids, n+ids] = 1
    boundary[3*n+ids, 2*n+ids] = 1
    boundary[3*n+ids, -1] = 1
    rhs = np.r_[np.zeros(m+3*n), np.full(n, 2.0)]
    objective = np.r_[np.zeros(2*n), -np.ones(n), 0.0]
    bounds = [(None, None)]*(2*n) + [(1e-7, None)]*n + [(0.1, 1.9)]
    rows = np.arange(m)

    def constraints(z):
        dx, dy = z[ii]-z[jj], z[n+ii]-z[n+jj]
        d = np.maximum(np.hypot(dx, dy), 1e-30)
        a = np.zeros((m+4*n, 3*n+1))
        a[rows, ii] = -dx/d
        a[rows, jj] = dx/d
        a[rows, n+ii] = -dy/d
        a[rows, n+jj] = dy/d
        a[rows, 2*n+ii] = 1
        a[rows, 2*n+jj] = 1
        a[m:] = boundary
        residual = np.r_[d-z[2*n+ii]-z[2*n+jj], rhs[m:]-boundary@z]
        return a, residual

    def polish(z):
        if time.monotonic() >= deadline:
            return z
        a, residual = constraints(z)
        active = np.where(residual < 2e-7)[0]
        if len(active) < 3*n+1:
            return z
        out = z.copy()
        for iteration in range(5):
            if time.monotonic() >= deadline:
                break
            a, residual = constraints(out)
            step, _, rank, _ = lstsq(a[active], residual[active], cond=1e-11, check_finite=False, lapack_driver='gelsy')
            if rank < 3*n+1 or np.max(np.abs(step)) > 0.02:
                break
            out += step
            if np.max(np.abs(step)) < 2e-15:
                break
        save(out)
        return out

    save(base)
    polish(base)
    restart = 0
    while time.monotonic() < deadline:
        z = base.copy()
        if restart:
            scale = float(params['restart_scale']) * (1.0 if restart % 3 else 2.3)
            perturb = rng.normal(size=(n, 2))*scale
            if restart % 2:
                center = initial[rng.integers(n), :2]
                weight = np.exp(-np.sum((initial[:, :2]-center)**2, axis=1)/0.075)
                perturb *= weight[:, None]
            z[:n] += perturb[:, 0]
            z[n:2*n] += perturb[:, 1]
        prev = -math.inf
        for iteration in range(int(params['steps'])):
            if time.monotonic() >= deadline:
                break
            a, _ = constraints(z)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result = linprog(objective, A_ub=a, b_ub=rhs, bounds=bounds, method='highs', options={'threads': 1, 'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9, 'time_limit': max(0.001, deadline-time.monotonic())})
            if result.x is None or not result.success:
                break
            z = result.x
            score = float(np.sum(z[2*n:3*n]))
            if score > bestscore-1e-8:
                save(z)
            if abs(score-prev) < 3e-11:
                polish(z)
                break
            prev = score
        if time.monotonic() < deadline:
            polish(z)
        restart += 1
    return best
