import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {'first_fraction': (0.36, 0.2, 0.5), 'second_fraction': (0.23, 0.1, 0.35), 'tolerance': (1e-11, 1e-13, 1e-8, 'log')}

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + 0.69 * budget_s
    ii, jj = np.triu_indices(21, 1)
    best = init
    best_score = -float('inf')
    best_c = None

    def canonical(obj):
        if 'circles' in obj:
            return np.asarray(obj['circles'], dtype=float)
        z = np.asarray(obj['state'], dtype=float)
        return np.column_stack((z[:21], z[21:42], z[42:63]))

    def exact(c):
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return -float('inf')
        dx = c[ii, 0] - c[jj, 0]
        dy = c[ii, 1] - c[jj, 1]
        if np.any(np.sqrt(dx*dx + dy*dy) < c[ii, 2] + c[jj, 2]):
            return -float('inf')
        xs = [Fraction(float(x)) for x in c[:, 0]]
        ys = [Fraction(float(x)) for x in c[:, 1]]
        rs = [Fraction(float(x)) for x in c[:, 2]]
        w = max(x+r for x,r in zip(xs,rs))-min(x-r for x,r in zip(xs,rs))
        h = max(y+r for y,r in zip(ys,rs))-min(y-r for y,r in zip(ys,rs))
        if w+h > 2:
            return -float('inf')
        return float(np.sum(c[:, 2]))

    if init is not None:
        try:
            c = canonical(init)
            best_score = exact(c)
            if np.isfinite(best_score):
                best_c = c.copy()
        except Exception:
            pass

    def consider(c):
        nonlocal best, best_score, best_c
        if time.monotonic() >= deadline or not np.isfinite(c).all():
            return
        c = c.copy()
        c[:, 2] = np.maximum(c[:, 2], 1e-10)
        dx = c[ii, 0]-c[jj, 0]
        dy = c[ii, 1]-c[jj, 1]
        ratio = np.min(np.sqrt(dx*dx+dy*dy)/(c[ii, 2]+c[jj, 2]))
        c[:, 2] *= min(1.0, float(ratio)) * (1-3e-13)
        lo = np.min(c[:, :2]-c[:, 2, None], axis=0)
        hi = np.max(c[:, :2]+c[:, 2, None], axis=0)
        perimeter = float(np.sum(hi-lo))
        if perimeter <= 0:
            return
        c[:, :2] -= (hi+lo)*0.5
        c *= (2/perimeter)*(1-3e-13)
        if float(np.sum(c[:, 2])) <= best_score:
            return
        score = exact(c)
        if score > best_score:
            best_score = score
            best_c = c.copy()
            best = {'circles': c.tolist()}

    # An explicit feasible fallback: twenty equal grid circles and one gap circle.
    grid = np.array([[(i-2)*2/9, (j-1.5)*2/9, 1/9] for j in range(4) for i in range(5)] + [[-1/9, -2/9, (2**0.5-1)/9]], dtype=float)
    consider(grid)

    class Stop(Exception):
        pass

    def solve(A, v, bounds, until):
        if time.monotonic() >= min(until, deadline):
            return
        X, Y, R = A[:21], A[21:42], A[42:63]
        DX, DY, SR = X[ii]-X[jj], Y[ii]-Y[jj], R[ii]+R[jj]
        width = A[63]
        wall = np.concatenate((width[None,:]/2-X-R, width[None,:]/2+X-R,
                               -width[None,:]/2-Y-R, -width[None,:]/2+Y-R), axis=0)
        offsets = np.r_[np.zeros(42), np.ones(42)]
        objective = -np.sum(R, axis=0)
        last = [v.copy()]
        def check():
            if time.monotonic() >= min(until, deadline):
                raise Stop()
        def fun(q):
            check()
            return float(objective @ q)
        def con(q):
            check()
            dx, dy, sr = DX@q, DY@q, SR@q
            return np.r_[dx*dx+dy*dy-sr*sr, wall@q+offsets]
        def jac(q):
            check()
            dx, dy, sr = DX@q, DY@q, SR@q
            return np.vstack((2*(dx[:,None]*DX+dy[:,None]*DY-sr[:,None]*SR), wall))
        def callback(q):
            check()
            last[0] = q.copy()
            z = A@q
            consider(np.column_stack((z[:21],z[21:42],z[42:63])))
        try:
            res = minimize(fun, v, jac=lambda q: objective, bounds=bounds,
                           constraints={'type':'ineq','fun':con,'jac':jac}, method='SLSQP',
                           callback=callback, options={'maxiter':240,'ftol':params['tolerance'],'disp':False})
            callback(res.x)
        except Stop:
            pass
        except (ValueError, FloatingPointError):
            pass

    def symmetric_seed(near, far):
        A = np.zeros((64,34))
        v = np.zeros(34)
        bounds = []
        circle = 0
        variable = 0
        for x in (-0.28,0.0,0.28):
            if time.monotonic() >= deadline:
                raise Stop()
            A[circle,variable] = 1
            v[variable] = x
            bounds.append((-0.95,0.95))
            variable += 1
            A[42+circle,variable] = 1
            v[variable] = 0.10
            bounds.append((1e-6,0.49))
            variable += 1
            circle += 1
        for count, height in ((near,0.21),(far,0.41)):
            if time.monotonic() >= deadline:
                raise Stop()
            for x in np.linspace(-0.105*(count-1),0.105*(count-1),count):
                if time.monotonic() >= deadline:
                    raise Stop()
                a,b = circle,circle+1
                A[a,variable] = A[b,variable] = 1
                v[variable] = x
                bounds.append((-0.95,0.95))
                variable += 1
                A[21+a,variable] = 1
                A[21+b,variable] = -1
                v[variable] = height
                bounds.append((0.001,0.95))
                variable += 1
                A[42+a,variable] = A[42+b,variable] = 1
                v[variable] = 0.095
                bounds.append((1e-6,0.49))
                variable += 1
                circle += 2
        A[63,33] = 1
        v[33] = 1.0
        bounds.append((0.2,1.8))
        return A,v,bounds

    try:
        A,v,bounds = symmetric_seed(4,5)
        solve(A,v,bounds,started+params['first_fraction']*budget_s)
        if time.monotonic() < deadline:
            A,v,bounds = symmetric_seed(5,4)
            solve(A,v,bounds,started+(params['first_fraction']+params['second_fraction'])*budget_s)
    except Stop:
        pass
    if best_c is not None and time.monotonic() < deadline:
        c = best_c.copy()
        lo = np.min(c[:,:2]-c[:,2,None],axis=0)
        hi = np.max(c[:,:2]+c[:,2,None],axis=0)
        c[:,:2] -= (lo+hi)/2
        w = float(hi[0]-lo[0])
        v = np.r_[c[:,0],c[:,1],c[:,2],w]
        solve(np.eye(64),v,[(-1,1)]*42+[(1e-7,0.5)]*21+[(0.1,1.9)],deadline)
    return best
