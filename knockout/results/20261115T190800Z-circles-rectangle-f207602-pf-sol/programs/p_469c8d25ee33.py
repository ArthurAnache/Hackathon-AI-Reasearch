import time
import math
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize

PARAMS = {
    'lp_steps': (9, 1, 25),
    'trust': (0.025, 0.001, 0.15, 'log'),
    'perturb': (0.009, 0.0001, 0.08, 'log'),
    'polish_iterations': (110, 20, 250)
}


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    best = init
    best_score = -float('inf')

    def score(obj):
        try:
            c = np.asarray(obj['circles'], dtype=np.float64)
            if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
                return -float('inf')
            xx = [Fraction(float(v)) for v in c[:, 0]]
            yy = [Fraction(float(v)) for v in c[:, 1]]
            rr = [Fraction(float(v)) for v in c[:, 2]]
            w = max(x+r for x, r in zip(xx, rr)) - min(x-r for x, r in zip(xx, rr))
            h = max(y+r for y, r in zip(yy, rr)) - min(y-r for y, r in zip(yy, rr))
            if w+h > 2:
                return -float('inf')
            for i in range(n):
                if time.monotonic() >= deadline:
                    return -float('inf')
                for j in range(i):
                    if time.monotonic() >= deadline:
                        return -float('inf')
                    d = np.sqrt((c[i, 0]-c[j, 0])**2 + (c[i, 1]-c[j, 1])**2)
                    if d < c[i, 2]+c[j, 2]:
                        return -float('inf')
            return float(np.sum(c[:, 2]))
        except Exception:
            return -float('inf')

    if init is not None:
        best_score = score(init)
        c = np.array(init['circles'], dtype=float)
    else:
        c = np.array([[0.12+0.25*(k % 5), 0.12+0.25*(k // 5), 0.11] for k in range(n)])
    lo = np.min(c[:, :2]-c[:, 2, None], axis=0)
    c[:, :2] -= lo
    wh = np.max(c[:, :2]+c[:, 2, None], axis=0)
    c *= 2.0 / np.sum(wh)
    z = np.empty(64)
    z[:42] = c[:, :2].ravel()
    z[42:63] = c[:, 2]
    z[63] = wh[0] * 2.0 / np.sum(wh)

    # A z >= b expresses the four rectangle walls.
    A = np.zeros((84, 64))
    b = np.zeros(84)
    for k in range(n):
        if time.monotonic() >= deadline:
            return best
        x, y, r = 2*k, 2*k+1, 42+k
        A[4*k, x] = 1
        A[4*k, r] = -1
        A[4*k+1, x] = -1
        A[4*k+1, r] = -1
        A[4*k+1, 63] = 1
        A[4*k+2, y] = 1
        A[4*k+2, r] = -1
        A[4*k+3, y] = -1
        A[4*k+3, r] = -1
        A[4*k+3, 63] = -1
        b[4*k+3] = -2
    objective = np.zeros(64)
    objective[42:63] = -1
    bounds = [(None, None)]*42 + [(1e-7, 0.7)]*21 + [(0.15, 1.85)]
    rows = np.arange(len(ii))

    def geometry(v):
        xy = v[:42].reshape(n, 2)
        delta = xy[ii]-xy[jj]
        dist = np.sqrt(np.sum(delta*delta, axis=1))
        return delta, np.maximum(dist, 1e-30)

    def con(v):
        _, d = geometry(v)
        return d-v[42+ii]-v[42+jj]

    def jac(v):
        delta, d = geometry(v)
        u = delta/d[:, None]
        J = np.zeros((len(ii), 64))
        J[rows, 2*ii] = u[:, 0]
        J[rows, 2*ii+1] = u[:, 1]
        J[rows, 2*jj] = -u[:, 0]
        J[rows, 2*jj+1] = -u[:, 1]
        J[rows, 42+ii] = -1
        J[rows, 42+jj] = -1
        return J

    def consider(v):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        cc = np.column_stack((v[:42].reshape(n, 2), v[42:63])).copy()
        if not np.isfinite(cc).all() or np.min(cc[:, 2]) <= 0:
            return
        # Uniformly reduce radii only as much as floating-point feasibility requires.
        xy = cc[:, :2]
        d = np.sqrt((xy[ii, 0]-xy[jj, 0])**2+(xy[ii, 1]-xy[jj, 1])**2)
        factor = min(1.0, float(np.min(d/(cc[ii, 2]+cc[jj, 2]))))
        cc[:, 2] *= factor*(1.0-3e-15)
        cc[:, :2] -= np.min(cc[:, :2]-cc[:, 2, None], axis=0)
        extent = np.max(cc[:, :2]+cc[:, 2, None], axis=0)
        cc *= min(1.0, 2.0/float(np.sum(extent)))*(1.0-3e-15)
        obj = {'circles': cc.tolist()}
        s = score(obj)
        if s > best_score:
            best_score, best = s, obj

    class Stop(Exception):
        pass

    def callback(v):
        if time.monotonic() >= deadline-0.025:
            raise Stop()
        if -float(objective @ v) > best_score+1e-12:
            consider(v)

    constraints = [
        {'type': 'ineq', 'fun': lambda v: A@v-b, 'jac': lambda v: A},
        {'type': 'ineq', 'fun': con, 'jac': jac}
    ]

    consider(z)
    # First polish the incumbent before spending time on graph-changing trials.
    try:
        if time.monotonic() < deadline-0.10:
            res = minimize(lambda v: float(objective@v), z, jac=lambda v: objective,
                           bounds=bounds, constraints=constraints, method='SLSQP',
                           callback=callback,
                           options={'maxiter': int(params['polish_iterations']), 'ftol': 2e-14})
            z = res.x
            consider(z)
    except Stop:
        return best

    anchor = z.copy()
    for step in range(int(params['lp_steps'])):
        if time.monotonic() >= deadline-0.12:
            break
        J = jac(z)
        trust = float(params['trust'])
        lb = z-trust
        ub = z+trust
        lb[42:63] = np.maximum(lb[42:63], 1e-7)
        lb[63] = max(lb[63], 0.15)
        ub[63] = min(ub[63], 1.85)
        try:
            lp = linprog(objective, A_ub=-np.vstack((A, J)),
                         b_ub=-np.concatenate((b, np.zeros(len(ii)))),
                         bounds=list(zip(lb, ub)), method='highs',
                         options={'primal_feasibility_tolerance': 1e-9,
                                  'dual_feasibility_tolerance': 1e-9,
                                  'time_limit': max(0.001, deadline-time.monotonic()-0.07)})
            if lp.success:
                z = lp.x
                consider(z)
            else:
                break
        except Exception:
            break

    attempt = 0
    while time.monotonic() < deadline-0.10:
        trial = z.copy() if attempt == 0 else anchor.copy()
        if attempt:
            scale = float(params['perturb'])*(1.0+(attempt % 4))
            trial[:42] += rng.normal(0, scale, 42)
            trial[42:63] *= np.exp(rng.normal(0, scale*1.5, 21))
            trial[63] += rng.normal(0, scale)
        try:
            res = minimize(lambda v: float(objective@v), trial, jac=lambda v: objective,
                           bounds=bounds, constraints=constraints, method='SLSQP',
                           callback=callback,
                           options={'maxiter': int(params['polish_iterations']), 'ftol': 2e-14})
            consider(res.x)
        except Stop:
            break
        except Exception:
            break
        attempt += 1
    return best
