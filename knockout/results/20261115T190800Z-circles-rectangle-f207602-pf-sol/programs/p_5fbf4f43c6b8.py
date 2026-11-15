import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    "iterations": (160, 40, 300),
    "jitter": (0.006, 0.0, 0.03),
    "time_fraction": (0.68, 0.4, 0.72)
}

N = 21
II, JJ = np.triu_indices(N, 1)
ROWS = np.arange(len(II))
IX = 3 * np.arange(N)
IR = IX + 2


def exact_score(candidate):
    try:
        c = np.asarray(candidate['circles'], dtype=np.float64)
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return -float('inf')
        xs = [Fraction(float(v)) for v in c[:, 0]]
        ys = [Fraction(float(v)) for v in c[:, 1]]
        rs = [Fraction(float(v)) for v in c[:, 2]]
        w = max(x+r for x,r in zip(xs,rs)) - min(x-r for x,r in zip(xs,rs))
        h = max(y+r for y,r in zip(ys,rs)) - min(y-r for y,r in zip(ys,rs))
        if w+h > 2:
            return -float('inf')
        for a,b in itertools.combinations(c, 2):
            d = np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
            if d < a[2]+b[2]:
                return -float('inf')
        return float(np.sum(c[:, 2]))
    except Exception:
        return -float('inf')


def repair(c):
    c = np.asarray(c, dtype=float).reshape(21,3).copy()
    if not np.isfinite(c).all():
        return None
    c[:,2] = np.maximum(c[:,2], 1e-10)
    delta = c[II,:2]-c[JJ,:2]
    d = np.sqrt(np.sum(delta*delta, axis=1))
    ratio = float(np.min(d/(c[II,2]+c[JJ,2])))
    if ratio <= 0:
        return None
    c[:,2] *= min(1.0, ratio)*(1.0-2e-12)
    lo = np.min(c[:,:2]-c[:,2,None], axis=0)
    hi = np.max(c[:,:2]+c[:,2,None], axis=0)
    scale = (2.0-4e-12)/float(np.sum(hi-lo))
    c[:,:2] -= lo
    c *= scale
    return {'circles': c.tolist()}


class Finished(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + float(params['time_fraction'])*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = exact_score(init)
    approximate_best = best_score
    pending = None

    def check():
        if time.monotonic() >= deadline:
            raise Finished()

    def consider(c):
        nonlocal pending, approximate_best
        cand = repair(c)
        if cand is not None:
            score = float(np.sum(np.asarray(cand['circles'])[:,2]))
            if score > approximate_best:
                approximate_best = score
                pending = cand

    # All three patterns have 16 main circles and five gap circles.
    patterns = [
        [(0,0),(0,2),(1,1),(2,0),(2,2)],
        [(0,0),(0,1),(1,2),(2,0),(2,2)],
        [(0,0),(0,2),(1,0),(1,2),(2,1)]
    ]

    # Establish a good feasible answer before starting the numerical solver.
    base = [[1+2*x,1+2*y,1.0] for y in range(4) for x in range(4)]
    base += [[2+2*x,2+2*y,np.sqrt(2)-1] for x,y in patterns[0]]
    consider(np.array(base)/8)

    objective_grad = np.zeros(64)
    objective_grad[IR] = -1.0
    bounds = [(0.0,2.0),(0.0,2.0),(1e-7,0.7)]*21 + [(0.2,1.8)]

    def objective(z):
        check()
        return -float(np.sum(z[IR]))

    def gradient(z):
        check()
        return objective_grad

    def constraints(z):
        check()
        c = z[:63].reshape(21,3)
        x,y,r = c.T
        dx = x[II]-x[JJ]
        dy = y[II]-y[JJ]
        distances = np.sqrt(dx*dx+dy*dy)
        w = z[63]
        return np.concatenate((distances-r[II]-r[JJ], x-r, w-x-r, y-r, 2-w-y-r))

    def jacobian(z):
        check()
        c = z[:63].reshape(21,3)
        dx = c[II,0]-c[JJ,0]
        dy = c[II,1]-c[JJ,1]
        d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-20)
        j = np.zeros((294,64))
        j[ROWS,3*II] = dx/d
        j[ROWS,3*JJ] = -dx/d
        j[ROWS,3*II+1] = dy/d
        j[ROWS,3*JJ+1] = -dy/d
        j[ROWS,3*II+2] = -1
        j[ROWS,3*JJ+2] = -1
        t = 210 + np.arange(21)
        j[t,IX] = 1
        j[t,IR] = -1
        j[t+21,IX] = -1
        j[t+21,IR] = -1
        j[t+21,63] = 1
        j[t+42,IX+1] = 1
        j[t+42,IR] = -1
        j[t+63,IX+1] = -1
        j[t+63,IR] = -1
        j[t+63,63] = -1
        return j

    def callback(z):
        check()
        consider(z[:63].reshape(21,3))

    attempt = 0
    while time.monotonic() < deadline:
        try:
            check()
            if attempt < len(patterns):
                circles = [[1+2*x,1+2*y,1.0] for y in range(4) for x in range(4)]
                circles += [[2+2*x,2+2*y,np.sqrt(2)-1] for x,y in patterns[attempt]]
                c = np.array(circles,dtype=float)/8
                c[:,:2] += rng.normal(0,float(params['jitter']),size=(21,2))
            elif attempt == 3 and init is not None:
                c = np.asarray(init['circles'],dtype=float).copy()
            else:
                anchor = pending if pending is not None else best
                c = np.asarray(anchor['circles'],dtype=float).copy()
                c[:,:2] += rng.normal(0,0.018,size=(21,2))
            lo = np.min(c[:,:2]-c[:,2,None],axis=0)
            hi = np.max(c[:,:2]+c[:,2,None],axis=0)
            factor = 2/float(np.sum(hi-lo))
            c[:,:2] -= lo
            c *= factor
            width = float((hi[0]-lo[0])*factor)
            z = np.r_[c.ravel(),width]
            result = minimize(objective,z,jac=gradient,method='SLSQP',bounds=bounds,
                constraints=[{'type':'ineq','fun':constraints,'jac':jacobian}],
                callback=callback,options={'maxiter':int(params['iterations']),'ftol':2e-11,'disp':False})
            consider(result.x[:63].reshape(21,3))
            attempt += 1
        except Finished:
            break
        except (ValueError, FloatingPointError):
            attempt += 1
    if pending is not None:
        score = exact_score(pending)
        if score > best_score:
            best = pending
            best_score = score
    return best
