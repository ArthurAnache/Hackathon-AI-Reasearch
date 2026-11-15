import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    "iterations": (140, 40, 240),
    "restart_noise": (0.012, 0.001, 0.05),
    "time_fraction": (0.70, 0.50, 0.75)
}

class Deadline(Exception):
    pass

def evaluate(data):
    try:
        c = np.asarray(data['circles'], dtype=np.float64)
        if c.shape != (21, 3) or not np.isfinite(c).all() or not (c[:, 2] > 0).all():
            return -float('inf')
        xs = [Fraction(float(x)) for x in c[:, 0]]
        ys = [Fraction(float(y)) for y in c[:, 1]]
        rs = [Fraction(float(r)) for r in c[:, 2]]
        w = max(x+r for x,r in zip(xs,rs))-min(x-r for x,r in zip(xs,rs))
        h = max(y+r for y,r in zip(ys,rs))-min(y-r for y,r in zip(ys,rs))
        if w+h > 2:
            return -float('inf')
        for a,b in itertools.combinations(c,2):
            d = np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)
            if d < a[2]+b[2]:
                return -float('inf')
        return float(np.sum(c[:,2]))
    except (ValueError, TypeError, KeyError, OverflowError):
        return -float('inf')

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + float(params['time_fraction'])*budget_s
    rng = np.random.default_rng(seed)
    ii,jj = np.triu_indices(21,1)
    pair_rows = np.arange(len(ii))
    best = init
    best_score = evaluate(init) if init is not None else -float('inf')
    best_array = None
    if np.isfinite(best_score):
        best_array = np.asarray(init['circles'],dtype=float).copy()

    def check():
        if time.monotonic() >= deadline:
            raise Deadline()

    def recover(c):
        nonlocal best, best_score, best_array
        if not np.isfinite(c).all() or np.min(c[:,2]) <= 0:
            return
        c = c.copy()
        dx = c[ii,0]-c[jj,0]
        dy = c[ii,1]-c[jj,1]
        distances = np.sqrt(dx*dx+dy*dy)
        ratio = min(1.0, float(np.min(distances/(c[ii,2]+c[jj,2]))))
        if ratio <= 0:
            return
        c[:,2] *= ratio*(1.0-4e-14)
        lo = np.min(c[:,:2]-c[:,2,None],axis=0)
        hi = np.max(c[:,:2]+c[:,2,None],axis=0)
        extent = float(np.sum(hi-lo))
        if extent <= 0:
            return
        c[:,:2] -= lo
        c *= (2.0/extent)*(1.0-4e-14)
        if float(np.sum(c[:,2])) <= best_score:
            return
        candidate = {'circles':c.tolist()}
        score = evaluate(candidate)
        if score > best_score:
            best, best_score, best_array = candidate, score, c.copy()

    if init is not None:
        try:
            initial = np.asarray(init['circles'],dtype=float).copy()
            if initial.shape != (21,3) or not np.isfinite(initial).all():
                initial = None
        except (ValueError,TypeError,KeyError):
            initial = None
    else:
        initial = None
    if initial is None or np.min(initial[:,2]) <= 0:
        rows = []
        for k,n in enumerate((5,4,5,4,3)):
            if time.monotonic() >= deadline:
                break
            for j in range(n):
                if time.monotonic() >= deadline:
                    break
                rows.append([2*j+(k%2),np.sqrt(3)*k,1.0])
        if len(rows) != 21:
            rows = [[float(k%7)*2,float(k//7)*2,1.0] for k in range(21)]
        initial = np.array(rows,dtype=float)
    recover(initial)

    # Variables are x[21], y[21], r[21], and rectangle width.
    # The height is exactly 2-width in the optimization model.
    A = np.zeros((84,64))
    q = np.arange(21)
    A[q,q] = 1
    A[q,42+q] = -1
    A[21+q,q] = -1
    A[21+q,42+q] = -1
    A[21+q,63] = 1
    A[42+q,21+q] = 1
    A[42+q,42+q] = -1
    A[63+q,21+q] = -1
    A[63+q,42+q] = -1
    A[63+q,63] = -1
    offset = np.zeros(84)
    offset[63:] = 2.0
    gradient = np.zeros(64)
    gradient[42:63] = -1.0
    bounds = [(0.0,2.0)]*42+[(1e-8,0.7)]*21+[(0.15,1.85)]

    def unpack(z):
        return np.column_stack((z[:21],z[21:42],z[42:63]))

    def pack(c):
        c = c.copy()
        lo = np.min(c[:,:2]-c[:,2,None],axis=0)
        hi = np.max(c[:,:2]+c[:,2,None],axis=0)
        extent = np.sum(hi-lo)
        c[:,:2] -= lo
        c *= 2.0/extent
        return np.r_[c[:,0],c[:,1],c[:,2],2.0*(hi[0]-lo[0])/extent]

    def objective(z):
        check()
        return -float(np.sum(z[42:63]))

    def constraints(z):
        check()
        dx = z[ii]-z[jj]
        dy = z[21+ii]-z[21+jj]
        dist = np.sqrt(dx*dx+dy*dy)
        return np.r_[dist-z[42+ii]-z[42+jj], A@z+offset]

    def jacobian(z):
        check()
        dx = z[ii]-z[jj]
        dy = z[21+ii]-z[21+jj]
        dist = np.maximum(np.sqrt(dx*dx+dy*dy),1e-20)
        J = np.zeros((294,64))
        J[pair_rows,ii] = dx/dist
        J[pair_rows,jj] = -dx/dist
        J[pair_rows,21+ii] = dy/dist
        J[pair_rows,21+jj] = -dy/dist
        J[pair_rows,42+ii] = -1
        J[pair_rows,42+jj] = -1
        J[210:] = A
        return J

    count = 0
    latest = None
    def callback(z):
        nonlocal count,latest
        check()
        latest = z.copy()
        count += 1
        if count % 5 == 0:
            recover(unpack(z))

    round_number = 0
    try:
        while time.monotonic() < deadline:
            check()
            if round_number == 0:
                z = pack(initial)
            else:
                if best_array is None:
                    break
                z = pack(best_array)
                noise = float(params['restart_noise'])
                z[:42] += rng.normal(0,noise,42)
                z[:42] = np.clip(z[:42],0.0,2.0)
                z[42:63] *= 0.98
            latest = z.copy()
            result = minimize(objective,z,jac=lambda z:gradient,
                constraints={'type':'ineq','fun':constraints,'jac':jacobian},
                bounds=bounds,method='SLSQP',callback=callback,
                options={'maxiter':int(params['iterations']), 'ftol':2e-12,'disp':False})
            recover(unpack(result.x))
            round_number += 1
            if deadline-time.monotonic() < 0.12:
                break
    except Deadline:
        pass
    if latest is not None and time.monotonic() < start+0.84*budget_s:
        recover(unpack(latest))
    return best
