import time
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog
from scipy.linalg import qr, solve, lstsq

PARAMS = {
    'perturbation': (0.045, 0.001, 0.2, 'log'),
    'trajectory_steps': (13, 4, 30),
    'active_tolerance': (2e-7, 1e-10, 1e-4, 'log')
}


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.69 * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    npair = len(ii)
    best = init
    best_score = -float('inf')

    def exact_score(c):
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return -float('inf')
        d = c[ii, :2] - c[jj, :2]
        dist = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
        if np.any(dist < c[ii, 2] + c[jj, 2]):
            return -float('inf')
        x = [Fraction(float(v)) for v in c[:, 0]]
        y = [Fraction(float(v)) for v in c[:, 1]]
        r = [Fraction(float(v)) for v in c[:, 2]]
        w = max(a+b for a,b in zip(x,r)) - min(a-b for a,b in zip(x,r))
        h = max(a+b for a,b in zip(y,r)) - min(a-b for a,b in zip(y,r))
        if w+h > 2:
            return -float('inf')
        return float(np.sum(c[:, 2]))

    if init is not None:
        original = np.array(init['circles'], dtype=float)
        best_score = exact_score(original)
    else:
        original = np.array([[(k % 5 + .5)*.2, (k // 5 + .5)*.2, .085] for k in range(21)])

    def encode(c):
        c = c.copy()
        lo = np.min(c[:, :2]-c[:, 2,None], axis=0)
        hi = np.max(c[:, :2]+c[:, 2,None], axis=0)
        wh = hi-lo
        s = 2.0 / np.sum(wh)
        c[:, :2] = (c[:, :2]-lo)*s
        c[:, 2] *= s
        return np.r_[c[:, 0], c[:, 1], c[:, 2], wh[0]*s]

    def consider(z):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        c = np.column_stack((z[:21], z[21:42], z[42:63]))
        if not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return
        d = c[ii,:2]-c[jj,:2]
        dist = np.sqrt(d[:,0]**2+d[:,1]**2)
        ratio = float(np.min(dist/(c[ii,2]+c[jj,2])))
        c[:,2] *= min(1.0, ratio)*(1.0-7e-16)
        lo = np.min(c[:,:2]-c[:,2,None], axis=0)
        c[:,:2] -= lo
        for attempt in range(3):
            if time.monotonic() >= deadline:
                return
            xs = [Fraction(float(v)) for v in c[:,0]]
            ys = [Fraction(float(v)) for v in c[:,1]]
            rs = [Fraction(float(v)) for v in c[:,2]]
            total = (max(x+r for x,r in zip(xs,rs))-min(x-r for x,r in zip(xs,rs)) +
                     max(y+r for y,r in zip(ys,rs))-min(y-r for y,r in zip(ys,rs)))
            if total > 2:
                c *= float(Fraction(2)/total)*(1.0-5e-16)
            d = c[ii,:2]-c[jj,:2]
            ratio = np.min(np.sqrt(d[:,0]**2+d[:,1]**2)/(c[ii,2]+c[jj,2]))
            if ratio < 1:
                c[:,2] *= ratio*(1.0-7e-16)
            value = exact_score(c)
            if value > best_score:
                best_score = value
                best = {'circles': c.tolist()}
                return
            if value != -float('inf'):
                return
            c[:,2] *= 1.0-2e-15

    # Rows represent nonnegative wall and separation margins.
    wall = np.zeros((84,64))
    wb = np.zeros(84)
    for k in range(21):
        if time.monotonic() >= deadline:
            return best if best is not None else {'circles': original.tolist()}
        wall[4*k,k] = 1
        wall[4*k,42+k] = -1
        wall[4*k+1,21+k] = 1
        wall[4*k+1,42+k] = -1
        wall[4*k+2,k] = -1
        wall[4*k+2,42+k] = -1
        wall[4*k+2,63] = 1
        wall[4*k+3,21+k] = -1
        wall[4*k+3,42+k] = -1
        wall[4*k+3,63] = -1
        wb[4*k+3] = 2
    rows = np.arange(npair)

    def geometry(z):
        dx = z[ii]-z[jj]
        dy = z[21+ii]-z[21+jj]
        ds = np.maximum(np.hypot(dx,dy), 1e-15)
        mat = np.zeros((npair,64))
        mat[rows,ii] = dx/ds
        mat[rows,jj] = -dx/ds
        mat[rows,21+ii] = dy/ds
        mat[rows,21+jj] = -dy/ds
        mat[rows,42+ii] = -1
        mat[rows,42+jj] = -1
        return np.vstack((mat,wall)), np.r_[ds-z[42+ii]-z[42+jj], wall@z+wb]

    def polish(z):
        if time.monotonic() >= deadline:
            return z
        mat,g = geometry(z)
        active = np.flatnonzero(g < params['active_tolerance'])
        if len(active) < 64:
            return z
        _,_,pivot = qr(mat[active].T, pivoting=True, mode='economic', check_finite=False)
        active = active[pivot[:64]]
        out = z.copy()
        for _ in range(7):
            if time.monotonic() >= deadline:
                break
            mat,g = geometry(out)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    step = solve(mat[active], -g[active], check_finite=False)
            except Exception:
                break
            if not np.isfinite(step).all() or np.max(np.abs(step)) > .02:
                break
            out += step
            if np.max(np.abs(step)) < 3e-15:
                break
        consider(out)
        return out

    z0 = encode(original)
    consider(z0)
    z0 = polish(z0)
    objective = np.zeros(64)
    objective[42:63] = -1
    bounds = [(0,2)]*42 + [(1e-7,1)]*21 + [(.1,1.9)]
    trajectory = 0
    while time.monotonic() < deadline-.035:
        z = z0.copy()
        if trajectory:
            scale = params['perturbation'] * (0.45,1.0,1.8,2.8)[(trajectory-1)%4]
            noise = rng.normal(size=(21,2))
            noise -= np.mean(noise,axis=0)
            z[:21] += scale*noise[:,0]
            z[21:42] += scale*noise[:,1]
        previous = -1e9
        for iteration in range(params['trajectory_steps']):
            remaining = deadline-time.monotonic()
            if remaining < .025:
                break
            mat,g = geometry(z)
            rhs = np.r_[np.zeros(npair),wb]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    result = linprog(objective, A_ub=-mat, b_ub=rhs, bounds=bounds,
                        method='highs-ds', options={'threads':1, 'primal_feasibility_tolerance':1e-9,
                        'dual_feasibility_tolerance':1e-9, 'time_limit':max(.001,remaining*.75)})
            except Exception:
                break
            if not result.success:
                break
            z = result.x
            score = float(np.sum(z[42:63]))
            if score > best_score-2e-8:
                consider(z)
                if abs(score-previous) < 1e-8:
                    z = polish(z)
            if abs(score-previous) < 2e-12:
                break
            previous = score
        if time.monotonic() < deadline-.012:
            polish(z)
        trajectory += 1
    return best if best is not None else {'circles': original.tolist()}
