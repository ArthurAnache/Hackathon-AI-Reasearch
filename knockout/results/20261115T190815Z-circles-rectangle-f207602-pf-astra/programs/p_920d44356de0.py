import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'excursion': (0.065, 0.01, 0.18),
    'cluster': (5, 3, 9),
    'penalty': (90.0, 15.0, 500.0, 'log'),
    'explore_iterations': (65, 15, 150),
    'polish_iterations': (95, 20, 200),
    'trials': (8, 1, 20)
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
K = np.arange(N)

class Deadline(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    hard = start + 0.94 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf

    def check():
        if time.monotonic() >= deadline:
            raise Deadline()

    def exact_score(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -np.inf
        d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
        if np.any(d < a[I, 2] + a[J, 2]):
            return -np.inf
        xs, ys, rs = [], [], []
        for x, y, r in a:
            if time.monotonic() >= hard:
                return -np.inf
            xs.append(Fraction(float(x)))
            ys.append(Fraction(float(y)))
            rs.append(Fraction(float(r)))
        perimeter = (max(x+r for x,r in zip(xs,rs)) - min(x-r for x,r in zip(xs,rs))
                     + max(y+r for y,r in zip(ys,rs)) - min(y-r for y,r in zip(ys,rs)))
        if perimeter > 2:
            return -np.inf
        return float(np.sum(a[:, 2]))

    if init is not None:
        a0 = np.asarray(init['circles'], dtype=float).copy()
        best_score = exact_score(a0)
    else:
        a0 = np.zeros((21, 3))
        a0[:, 0] = (np.arange(21) % 5 + 0.5) / 5
        a0[:, 1] = (np.arange(21) // 5 + 0.5) / 5
        a0[:, 2] = 0.095
        best = {'circles': a0.tolist()}
        best_score = exact_score(a0)

    lo = np.min(a0[:, :2] - a0[:, 2, None], axis=0)
    sides = np.max(a0[:, :2] + a0[:, 2, None], axis=0) - lo
    scale = 2.0 / np.sum(sides)
    base = np.r_[(a0[:, 0]-lo[0])*scale,
                 (a0[:, 1]-lo[1])*scale,
                 a0[:, 2]*scale, sides[0]*scale]
    objective_gradient = np.r_[np.zeros(42), -np.ones(21), 0.0]

    def geometry(z, jacobian=False):
        check()
        x, y, r, w = z[:21], z[21:42], z[42:63], z[63]
        dx, dy = x[I]-x[J], y[I]-y[J]
        d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-30)
        c = np.r_[d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r]
        if not jacobian:
            return c
        A = np.zeros((M+84, 64))
        rows = np.arange(M)
        A[rows, I], A[rows, J] = dx/d, -dx/d
        A[rows, 21+I], A[rows, 21+J] = dy/d, -dy/d
        A[rows, 42+I], A[rows, 42+J] = -1, -1
        A[M+K, K], A[M+K, 42+K] = 1, -1
        A[M+21+K, K], A[M+21+K, 42+K] = -1, -1
        A[M+21+K, 63] = 1
        A[M+42+K, 21+K], A[M+42+K, 42+K] = 1, -1
        A[M+63+K, 21+K], A[M+63+K, 42+K] = -1, -1
        A[M+63+K, 63] = -1
        return A

    def consider(z):
        nonlocal best, best_score
        check()
        r = z[42:63].copy()
        if not np.isfinite(z).all() or np.any(r <= 0):
            return
        p = np.column_stack((z[:21], z[21:42]))
        d = np.sqrt((p[I, 0]-p[J, 0])**2 + (p[I, 1]-p[J, 1])**2)
        factor = min(1.0, float(np.min(d/(r[I]+r[J]))))
        r *= factor * (1.0-2e-14)
        sides = np.max(p+r[:, None], axis=0)-np.min(p-r[:, None], axis=0)
        s = min(1.0, 2.0/float(np.sum(sides))) * (1.0-2e-14)
        p *= s
        r *= s
        if float(np.sum(r)) <= best_score:
            return
        a = np.column_stack((p, r))
        score = exact_score(a)
        if score > best_score:
            best_score, best = score, {'circles': a.tolist()}

    bounds = [(None, None)]*42 + [(1e-7, 0.5)]*21 + [(0.3, 1.7)]

    def polish(z, iterations):
        check()
        result = minimize(lambda q: -float(np.sum(q[42:63])), z,
                          jac=lambda q: objective_gradient,
                          method='SLSQP', bounds=bounds,
                          constraints={'type': 'ineq', 'fun': geometry,
                                       'jac': lambda q: geometry(q, True)},
                          callback=consider,
                          options={'maxiter': iterations, 'ftol': 3e-14, 'disp': False})
        consider(result.x)
        return result.x

    def penalty(z, mu):
        check()
        x,y,r,w = z[:21],z[21:42],z[42:63],z[63]
        dx,dy = x[I]-x[J],y[I]-y[J]
        d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-20)
        v = np.maximum(r[I]+r[J]-d,0)
        left = np.maximum(r-x,0)
        right = np.maximum(r+x-w,0)
        bottom = np.maximum(r-y,0)
        top = np.maximum(r+y+w-2,0)
        c = 2*mu*v
        gx = np.bincount(I,weights=-c*dx/d,minlength=N)+np.bincount(J,weights=c*dx/d,minlength=N)
        gy = np.bincount(I,weights=-c*dy/d,minlength=N)+np.bincount(J,weights=c*dy/d,minlength=N)
        gr = -np.ones(N)+np.bincount(I,weights=c,minlength=N)+np.bincount(J,weights=c,minlength=N)
        gx += 2*mu*(right-left)
        gy += 2*mu*(top-bottom)
        gr += 2*mu*(left+right+bottom+top)
        gw = 2*mu*(top.sum()-right.sum())
        value = -r.sum()+mu*(v@v+left@left+right@right+bottom@bottom+top@top)
        return value,np.r_[gx,gy,gr,gw]

    try:
        polished = polish(base, min(35, int(params['polish_iterations'])))
        if np.max(np.abs(polished-base)) < 0.03:
            base = polished
        for trial in range(int(params['trials'])):
            check()
            z = base.copy()
            p = np.column_stack((z[:21],z[21:42]))
            anchor = int(rng.integers(21))
            distance = np.sum((p-p[anchor])**2,axis=1)
            cluster = np.argsort(distance)[:int(params['cluster'])]
            amplitude = float(params['excursion']) * (1.0+0.22*(trial%3))
            displacement = rng.normal(size=(len(cluster),2))
            displacement -= 0.45*displacement.mean(axis=0)
            z[cluster] += amplitude*displacement[:,0]
            z[21+cluster] += amplitude*displacement[:,1]
            z[42+cluster] *= rng.uniform(0.78,1.04,len(cluster))
            mu = float(params['penalty'])
            result = minimize(penalty,z,args=(mu,),jac=True,method='L-BFGS-B',
                              bounds=bounds,
                              options={'maxiter':int(params['explore_iterations']),
                                       'ftol':1e-10,'gtol':1e-7,'maxls':15})
            check()
            polish(result.x,int(params['polish_iterations']))
    except Deadline:
        pass
    return best
