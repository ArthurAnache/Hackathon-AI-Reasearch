import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import time
import math
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize

PARAMS = {
    'excursion_weight': (1.8, 0.1, 5.0),
    'excursion_steps': (3, 1, 8),
    'recovery_steps': (12, 3, 30),
    'polish_iterations': (45, 5, 100)
}

class TimeLimit(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.69 * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    rows = np.arange(m)
    best = init
    best_score = -math.inf

    def exact_score(obj):
        a = np.asarray(obj['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -math.inf
        d = a[ii, :2] - a[jj, :2]
        if np.any(np.sqrt(d[:, 0]**2 + d[:, 1]**2) < a[ii, 2] + a[jj, 2]):
            return -math.inf
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        p = max(x[k]+r[k] for k in range(n))-min(x[k]-r[k] for k in range(n))
        p += max(y[k]+r[k] for k in range(n))-min(y[k]-r[k] for k in range(n))
        if p > 2:
            return -math.inf
        return float(np.sum(a[:, 2]))

    if init is not None:
        best_score = exact_score(init)
        a = np.array(init['circles'], dtype=float)
    else:
        a = np.array([[(k % 5)*0.19+0.095, (k // 5)*0.19+0.095, 0.094] for k in range(n)])

    def consider(z):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        a = np.column_stack((z[:n], z[n:2*n], z[2*n:3*n])).copy()
        if not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return
        a[:, :2] -= np.min(a[:, :2]-a[:, 2, None], axis=0)
        d = a[ii, :2]-a[jj, :2]
        dist = np.sqrt(d[:, 0]**2+d[:, 1]**2)
        ratio = float(np.min(dist/(a[ii, 2]+a[jj, 2])))
        a[:, 2] *= min(1.0, ratio)*(1.0-3e-15)
        wh = np.max(a[:, :2]+a[:, 2, None], axis=0)-np.min(a[:, :2]-a[:, 2, None], axis=0)
        a *= (2.0/float(np.sum(wh)))*(1.0-3e-15)
        for _ in range(2):
            if time.monotonic() >= deadline:
                return
            d = a[ii, :2]-a[jj, :2]
            ratio = float(np.min(np.sqrt(d[:, 0]**2+d[:, 1]**2)/(a[ii, 2]+a[jj, 2])))
            if ratio < 1.0:
                a[:, 2] *= ratio*(1.0-3e-15)
        obj = {'circles': a.tolist()}
        score = exact_score(obj)
        if score > best_score:
            best, best_score = obj, score

    def encode(a):
        a = a.copy()
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        a[:, :2] -= lo
        wh = np.max(a[:, :2]+a[:, 2, None], axis=0)
        return np.r_[a[:, 0], a[:, 1], a[:, 2], wh[0]]

    z = encode(a)
    boundary = np.zeros((4*n, 3*n+1))
    rhs = np.zeros(4*n)
    for k in range(n):
        if time.monotonic() >= deadline:
            return best
        boundary[k, k] = -1
        boundary[k, 2*n+k] = 1
        boundary[n+k, k] = 1
        boundary[n+k, 2*n+k] = 1
        boundary[n+k, -1] = -1
        boundary[2*n+k, n+k] = -1
        boundary[2*n+k, 2*n+k] = 1
        boundary[3*n+k, n+k] = 1
        boundary[3*n+k, 2*n+k] = 1
        boundary[3*n+k, -1] = 1
        rhs[3*n+k] = 2
    objective = np.zeros(3*n+1)
    objective[2*n:3*n] = -1

    def polish(v, maxiter):
        def fun(q):
            if time.monotonic() >= deadline:
                raise TimeLimit()
            dx = q[ii]-q[jj]
            dy = q[n+ii]-q[n+jj]
            return np.r_[rhs-boundary@q, np.sqrt(dx*dx+dy*dy)-q[2*n+ii]-q[2*n+jj]]
        def jac(q):
            if time.monotonic() >= deadline:
                raise TimeLimit()
            dx = q[ii]-q[jj]
            dy = q[n+ii]-q[n+jj]
            dd = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-20)
            j = np.zeros((m, 3*n+1))
            j[rows, ii] = dx/dd
            j[rows, jj] = -dx/dd
            j[rows, n+ii] = dy/dd
            j[rows, n+jj] = -dy/dd
            j[rows, 2*n+ii] = -1
            j[rows, 2*n+jj] = -1
            return np.vstack((-boundary, j))
        try:
            result = minimize(lambda q: float(objective@q), v, jac=lambda q: objective,
                method='SLSQP', bounds=[(0, 2)]*(2*n)+[(1e-7, 0.8)]*n+[(0.15, 1.85)],
                constraints=[{'type':'ineq', 'fun':fun, 'jac':jac}],
                options={'ftol':2e-14, 'maxiter':maxiter, 'disp':False})
            consider(result.x)
            return result.x
        except TimeLimit:
            return v

    z = polish(z, int(params['polish_iterations']))
    if time.monotonic() >= deadline:
        return best
    consider(z)
    base = encode(np.array(best['circles'], dtype=float)) if best is not None else z.copy()
    iteration = 0
    cycle = int(params['excursion_steps']) + int(params['recovery_steps'])
    weights = objective.copy()
    while time.monotonic() < deadline - 0.025:
        phase = iteration % cycle
        if phase == 0:
            z = base.copy()
            weights = objective.copy()
            weights[2*n:3*n] = -np.exp(np.clip(rng.normal(0, float(params['excursion_weight']), n), -3, 3))
        c = weights if phase < int(params['excursion_steps']) else objective
        dx = z[ii]-z[jj]
        dy = z[n+ii]-z[n+jj]
        dd = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-20)
        mat = np.zeros((m, 3*n+1))
        mat[rows, ii] = -dx/dd
        mat[rows, jj] = dx/dd
        mat[rows, n+ii] = -dy/dd
        mat[rows, n+jj] = dy/dd
        mat[rows, 2*n+ii] = 1
        mat[rows, 2*n+jj] = 1
        if time.monotonic() >= deadline - 0.02:
            break
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = linprog(c, A_ub=np.vstack((boundary, mat)), b_ub=np.r_[rhs, np.zeros(m)],
                bounds=[(0, 2)]*(2*n)+[(1e-7, 0.8)]*n+[(0.15, 1.85)], method='highs',
                options={'primal_feasibility_tolerance':1e-9, 'dual_feasibility_tolerance':1e-9,
                         'threads':1, 'time_limit':max(0.001, deadline-time.monotonic()-0.015)})
        if res.x is not None:
            delta = float(np.max(np.abs(res.x-z)))
            z = res.x
            consider(z)
            if phase >= int(params['excursion_steps']) and (delta < 2e-7 or phase == cycle-1):
                if time.monotonic() < deadline-0.08:
                    z = polish(z, min(20, int(params['polish_iterations'])))
                if best is not None:
                    base = encode(np.array(best['circles'], dtype=float))
                iteration += cycle-1-phase
        iteration += 1
    return best if best is not None else {'circles': a.tolist()}
