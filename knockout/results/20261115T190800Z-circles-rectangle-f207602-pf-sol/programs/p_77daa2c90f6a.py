import time
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize

PARAMS = {
    'excursion': (0.105, 0.02, 0.3),
    'tilt': (0.65, 0.1, 2.0),
    'polish_steps': (48, 15, 120),
    'safety': (8e-15, 1e-15, 1e-11, 'log')
}

N = 21
I, J = np.triu_indices(N, 1)
ROWS = np.arange(len(I))
D = 64
C = np.zeros(D)
C[42:63] = -1.0
A = np.zeros((84, D))
B = np.zeros(84)
K = np.arange(N)
A[K, K] = -1
A[K, 42+K] = 1
A[21+K, K] = 1
A[21+K, 42+K] = 1
A[21+K, 63] = -1
A[42+K, 21+K] = -1
A[42+K, 42+K] = 1
A[63+K, 21+K] = 1
A[63+K, 42+K] = 1
A[63+K, 63] = 1
B[63+K] = 2
LOW = np.r_[np.zeros(42), np.full(21, 1e-7), .05]
HIGH = np.r_[np.full(42, 2.), np.full(21, 1.), 1.95]

class StopSearch(Exception):
    pass

def perimeter(p, r):
    x = list(map(lambda v: Fraction(float(v)), p[:, 0]))
    y = list(map(lambda v: Fraction(float(v)), p[:, 1]))
    q = list(map(lambda v: Fraction(float(v)), r))
    return (max(map(lambda t: t[0]+t[1], zip(x,q)))
            - min(map(lambda t: t[0]-t[1], zip(x,q)))
            + max(map(lambda t: t[0]+t[1], zip(y,q)))
            - min(map(lambda t: t[0]-t[1], zip(y,q))))

def valid(p, r):
    if not np.isfinite(p).all() or not np.isfinite(r).all() or np.min(r) <= 0:
        return False
    d = np.sqrt((p[I,0]-p[J,0])**2 + (p[I,1]-p[J,1])**2)
    return not np.any(d < r[I]+r[J]) and perimeter(p,r) <= 2

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + .70*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf
    safety = float(params['safety'])

    def check():
        if time.monotonic() >= end:
            raise StopSearch()

    def pack(p,r):
        lower = np.min(p-r[:,None], axis=0)
        sides = np.max(p+r[:,None], axis=0)-lower
        p = p-lower
        return np.r_[p[:,0], p[:,1], r, sides[0]]

    def save(z):
        nonlocal best, best_score
        check()
        p = np.column_stack((z[:21],z[21:42]))
        r = z[42:63].copy()
        if not np.isfinite(z).all() or np.min(r) <= 0:
            return
        d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
        rat = min(1., float(np.min(d/(r[I]+r[J]))))
        r *= rat*(1-safety)
        p -= np.min(p,axis=0)
        per = perimeter(p,r)
        if per > 2:
            scale = (2/float(per))*(1-safety)
            p *= scale
            r *= scale
        score = float(np.sum(r))
        if score > best_score and valid(p,r):
            best_score = score
            best = {'circles':np.column_stack((p,r)).tolist()}

    if init is None:
        p = np.column_stack((.1+.2*(K%7), .1+.2*(K//7)))
        r = np.full(21,.085)
        best = {'circles':np.column_stack((p,r)).tolist()}
        best_score = float(np.sum(r))
    else:
        arr = np.asarray(init['circles'], dtype=float)
        p,r = arr[:,:2].copy(),arr[:,2].copy()
        if valid(p,r):
            best_score = float(np.sum(r))
    base = pack(p,r)

    def fun(z):
        check()
        dx,dy = z[I]-z[J], z[21+I]-z[21+J]
        return np.r_[np.sqrt(dx*dx+dy*dy+1e-30)-z[42+I]-z[42+J], B-A@z]

    def jac(z):
        check()
        dx,dy = z[I]-z[J],z[21+I]-z[21+J]
        dd = np.sqrt(dx*dx+dy*dy+1e-30)
        out = np.zeros((294,D))
        out[ROWS,I] = dx/dd
        out[ROWS,J] = -dx/dd
        out[ROWS,21+I] = dy/dd
        out[ROWS,21+J] = -dy/dd
        out[ROWS,42+I] = -1
        out[ROWS,42+J] = -1
        out[210:] = -A
        return out

    def callback(z):
        check()
        if np.sum(z[42:63]) > best_score:
            save(z)

    def polish(z, steps):
        check()
        result = minimize(lambda v:(float(C@v), C), z, jac=True,
                          method='SLSQP', bounds=list(zip(LOW,HIGH)),
                          constraints={'type':'ineq','fun':fun,'jac':jac},
                          callback=callback,
                          options={'maxiter':steps,'ftol':3e-14,'disp':False})
        check()
        save(result.x)
        return result.x

    def plane_step(z, objective, radius):
        check()
        dx,dy = z[I]-z[J], z[21+I]-z[21+J]
        dist = np.maximum(np.hypot(dx,dy),1e-20)
        mat = np.zeros((294,D))
        mat[ROWS,I] = -dx/dist
        mat[ROWS,J] = dx/dist
        mat[ROWS,21+I] = -dy/dist
        mat[ROWS,21+J] = dy/dist
        mat[ROWS,42+I] = 1
        mat[ROWS,42+J] = 1
        mat[210:] = A
        lo = np.maximum(LOW,z-radius)
        hi = np.minimum(HIGH,z+radius)
        if np.any(lo > hi):
            return z
        remaining = end-time.monotonic()
        if remaining < .04:
            raise StopSearch()
        result = linprog(objective, A_ub=mat, b_ub=np.r_[np.zeros(210),B],
                         bounds=list(zip(lo,hi)), method='highs',
                         options={'time_limit':min(.10,remaining*.5),
                                  'dual_feasibility_tolerance':1e-8,
                                  'primal_feasibility_tolerance':1e-8})
        check()
        if result.x is not None:
            save(result.x)
            return result.x
        return z

    try:
        save(base)
        polish(base, 18)
        attempt = 0
        while time.monotonic() < end:
            check()
            arr = np.asarray(best['circles'], dtype=float)
            base = pack(arr[:,:2],arr[:,2])
            objective = C.copy()
            tilt = float(params['tilt'])*(1+.25*(attempt%3))
            # A coherent force on several circles releases coordinated contacts.
            chosen = rng.choice(21, size=5+(attempt%4), replace=False)
            objective[42:63] += rng.normal(0,tilt,21)
            force = rng.normal(0,tilt*.35,(21,2))
            force -= np.mean(force,axis=0)
            objective[:21] = force[:,0]
            objective[21:42] = force[:,1]
            objective[42+chosen] += tilt*.65
            radius = float(params['excursion'])*(0.65+.35*(attempt%3))
            z = plane_step(base,objective,radius)
            # Refresh the supporting directions before removing the tilt.
            if attempt%2:
                z = plane_step(z, .35*objective+.65*C, radius*.65)
            z = polish(z,int(params['polish_steps']))
            attempt += 1
    except StopSearch:
        pass
    return best
