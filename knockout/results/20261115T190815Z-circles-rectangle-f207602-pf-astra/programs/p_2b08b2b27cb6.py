import time
import math
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"first_share": (0.40, 0.20, 0.55), "ftol": (1e-10, 1e-12, 1e-7, "log")}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.69 * budget_s
    stage_end = deadline
    ii, jj = np.triu_indices(21, 1)
    best = init
    best_score = -float('inf')
    best_z = None

    def guard():
        if time.monotonic() >= stage_end:
            raise StopSearch()

    def exact(c):
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return -float('inf')
        d = c[ii, :2] - c[jj, :2]
        distances = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
        if np.any(distances < c[ii, 2] + c[jj, 2]):
            return -float('inf')
        left = []; right = []; bottom = []; top = []
        for x, y, r in c:
            if time.monotonic() >= deadline:
                return -float('inf')
            x, y, r = Fraction(float(x)), Fraction(float(y)), Fraction(float(r))
            left.append(x-r); right.append(x+r)
            bottom.append(y-r); top.append(y+r)
        if max(right)-min(left)+max(top)-min(bottom) > 2:
            return -float('inf')
        return float(np.sum(c[:, 2]))

    if init is not None:
        try:
            best_score = exact(np.asarray(init['circles'], dtype=float))
        except (KeyError, ValueError, TypeError):
            pass

    def consider(z):
        nonlocal best, best_score, best_z
        if time.monotonic() >= deadline:
            return
        c = np.array(z[:63], dtype=float).reshape(21, 3).copy()
        if not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return
        d = c[ii, :2] - c[jj, :2]
        ds = np.sqrt(d[:, 0]**2 + d[:, 1]**2)
        factor = min(1.0, float(np.min(ds / (c[ii, 2] + c[jj, 2]))))
        if factor <= 0:
            return
        c[:, 2] *= factor * (1.0 - 2e-14)
        lo = np.min(c[:, :2] - c[:, 2, None], axis=0)
        hi = np.max(c[:, :2] + c[:, 2, None], axis=0)
        span = float(np.sum(hi-lo))
        if span <= 0:
            return
        c[:, :2] -= lo
        scale = (2.0 / span) * (1.0 - 2e-14)
        c *= scale
        if float(np.sum(c[:, 2])) <= best_score:
            return
        score = exact(c)
        if score > best_score:
            best_score = score
            best = {'circles': c.tolist()}
            width = float(np.max(c[:, 0]+c[:, 2])-np.min(c[:, 0]-c[:, 2]))
            best_z = np.r_[c.ravel(), width]

    def make_seed(kind):
        circles = []
        if kind < 2:
            for y in range(4):
                guard()
                for x in range(4):
                    guard()
                    circles.append([2.0*x+1, 2.0*y+1, 1.0])
            holes = [(0,1),(1,0),(1,1),(1,2),(2,1)] if kind == 0 else [(0,0),(0,2),(1,1),(2,0),(2,2)]
            for x, y in holes:
                guard()
                circles.append([2.0*x+2, 2.0*y+2, math.sqrt(2)-1])
        else:
            for row, count in enumerate([4,5,4,4,4]):
                guard()
                for col in range(count):
                    guard()
                    circles.append([2.0*col + (1.0 if row % 2 == 0 else 0.0), math.sqrt(3)*row, 1.0])
        c = np.array(circles)
        lo = np.min(c[:, :2]-c[:, 2,None], axis=0)
        hi = np.max(c[:, :2]+c[:, 2,None], axis=0)
        c[:, :2] -= lo
        scale = 2.0 / np.sum(hi-lo)
        c *= scale
        return np.r_[c.ravel(), (hi[0]-lo[0])*scale]

    objective_gradient = np.zeros(64)
    objective_gradient[2:63:3] = -1
    ar = np.arange(21)
    bounds = [(None, None), (None, None), (1e-7, 0.7)] * 21 + [(0.1, 1.9)]

    def objective(z):
        guard()
        return -float(np.sum(z[2:63:3]))

    def constraints(z):
        guard()
        c = z[:63].reshape(21, 3)
        x, y, r = c.T
        dx = x[ii]-x[jj]; dy = y[ii]-y[jj]
        rr = r[ii]+r[jj]
        return np.r_[dx*dx+dy*dy-rr*rr, x-r, z[63]-x-r, y-r, 2-z[63]-y-r]

    def jacobian(z):
        guard()
        c = z[:63].reshape(21, 3)
        x, y, r = c.T
        dx = 2*(x[ii]-x[jj]); dy = 2*(y[ii]-y[jj]); rr = -2*(r[ii]+r[jj])
        j = np.zeros((294, 64))
        rows = np.arange(210)
        j[rows,3*ii] = dx; j[rows,3*jj] = -dx
        j[rows,3*ii+1] = dy; j[rows,3*jj+1] = -dy
        j[rows,3*ii+2] = rr; j[rows,3*jj+2] = rr
        j[210+ar,3*ar] = 1; j[210+ar,3*ar+2] = -1
        j[231+ar,3*ar] = -1; j[231+ar,3*ar+2] = -1; j[231+ar,63] = 1
        j[252+ar,3*ar+1] = 1; j[252+ar,3*ar+2] = -1
        j[273+ar,3*ar+1] = -1; j[273+ar,3*ar+2] = -1; j[273+ar,63] = -1
        return j

    def callback(z):
        guard()
        consider(z)

    def optimize(z):
        try:
            consider(z)
            result = minimize(objective, z, jac=lambda z: objective_gradient,
                method='SLSQP', bounds=bounds,
                constraints={'type':'ineq', 'fun':constraints, 'jac':jacobian},
                callback=callback,
                options={'maxiter':160, 'ftol':params['ftol'], 'disp':False})
            consider(result.x)
        except StopSearch:
            pass

    for kind in (0, 2, 1):
        if time.monotonic() >= deadline - 0.015:
            break
        stage_end = deadline
        try:
            z = make_seed(kind)
        except StopSearch:
            break
        if kind == 0:
            stage_end = min(deadline, start + params['first_share']*budget_s)
        elif kind == 2:
            stage_end = min(deadline, time.monotonic() + 0.23*budget_s)
        optimize(z)
    if time.monotonic() < deadline - 0.04 and best_z is not None:
        stage_end = deadline
        optimize(best_z)
    return best
