import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'patch_size': (3, 2, 6),
    'iterations': (65, 20, 150),
    'trials': (12, 2, 30),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class StopSearch(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2] + a[J, 2]):
        return -np.inf
    x = list(map(lambda v: Fraction(float(v)), a[:, 0]))
    y = list(map(lambda v: Fraction(float(v)), a[:, 1]))
    r = list(map(lambda v: Fraction(float(v)), a[:, 2]))
    perimeter = (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
                 + max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)))
    return float(np.sum(a[:, 2])) if perimeter <= 2 else -np.inf

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a = np.array([[(k%7)*0.12, (k//7)*0.12, 0.045] for k in range(N)], float)
        init = {'circles': a.tolist()}
    original = np.asarray(init['circles'], float)
    best = init
    best_score = exact_score(original)
    lo = (original[:, :2]-original[:, 2, None]).min(axis=0)
    sides = (original[:, :2]+original[:, 2, None]).max(axis=0)-lo
    scale = 2.0 / sides.sum()
    base = original.copy()
    base[:, :2] = (base[:, :2]-lo)*scale
    base[:, 2] *= scale
    w = sides[0]*scale
    zbase = np.r_[base[:, 0], base[:, 1], base[:, 2], w]
    bounds = [(0.0, 2.0)]*42 + [(1e-7, 0.7)]*21 + [(0.15, 1.85)]
    objective_gradient = np.r_[np.zeros(42), -np.ones(21), 0.0]

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def objective(z):
        check()
        return -float(z[42:63].sum()), objective_gradient

    def constraints(z):
        check()
        x, y, r, width = z[:21], z[21:42], z[42:63], z[63]
        dx, dy = x[I]-x[J], y[I]-y[J]
        distance = np.sqrt(dx*dx+dy*dy+1e-30)
        return np.r_[distance-r[I]-r[J], x-r, width-x-r, y-r, 2-width-y-r]

    def jacobian(z):
        check()
        x, y = z[:21], z[21:42]
        dx, dy = x[I]-x[J], y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        out = np.zeros((M+84, 64))
        rows = np.arange(M)
        out[rows,I] = dx/d
        out[rows,J] = -dx/d
        out[rows,21+I] = dy/d
        out[rows,21+J] = -dy/d
        out[rows,42+I] = -1
        out[rows,42+J] = -1
        k = np.arange(21)
        out[M+k,k] = 1
        out[M+21+k,k] = -1
        out[M+42+k,21+k] = 1
        out[M+63+k,21+k] = -1
        out[M+21+k,63] = 1
        out[M+63+k,63] = -1
        out[M+k,42+k] = -1
        out[M+21+k,42+k] = -1
        out[M+42+k,42+k] = -1
        out[M+63+k,42+k] = -1
        return out

    def retain(z):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        p = np.column_stack((z[:21], z[21:42]))
        r = np.maximum(z[42:63], 1e-12).copy()
        d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
        ratio = min(1.0, float(np.min(d/(r[I]+r[J]))))
        r *= max(0.0, ratio)*(1-3e-14)
        lower = (p-r[:,None]).min(axis=0)
        side = (p+r[:,None]).max(axis=0)-lower
        s = (2.0/side.sum())*(1-3e-14)
        a = np.column_stack(((p-lower)*s, r*s))
        score = exact_score(a)
        if score > best_score:
            best_score = score
            best = {'circles': a.tolist()}

    # Nearby pairs serve as geometric hinges; rank folds by introduced overlap.
    pairs = np.argsort(np.sqrt(((base[I,:2]-base[J,:2])**2).sum(axis=1))
                       - base[I,2]-base[J,2])[:90]
    proposals = []
    for attempt in range(100):
        if time.monotonic() >= deadline:
            break
        edge = int(pairs[int(rng.integers(len(pairs)))])
        i, j = int(I[edge]), int(J[edge])
        midpoint = (base[i,:2]+base[j,:2])*0.5
        tangent = base[j,:2]-base[i,:2]
        tangent /= max(np.linalg.norm(tangent), 1e-15)
        normal = np.array([-tangent[1], tangent[0]])
        signed = (base[:,:2]-midpoint)@normal
        sign = 1 if rng.random() < 0.5 else -1
        eligible = np.flatnonzero(signed*sign > 1e-5)
        eligible = eligible[(eligible != i)&(eligible != j)]
        if len(eligible) < 2:
            continue
        metric = np.linalg.norm(base[eligible,:2]-midpoint,axis=1)
        count = min(int(params['patch_size']), len(eligible))
        patch = eligible[np.argsort(metric)[:count]]
        p = base[:,:2].copy()
        p[patch] -= 2*signed[patch,None]*normal
        movement = float(np.linalg.norm(p-base[:,:2]))
        if movement < 0.04:
            continue
        # Recenter the folded configuration inside its new bounding rectangle.
        lower = (p-base[:,2,None]).min(axis=0)
        upper = (p+base[:,2,None]).max(axis=0)
        s = 2.0/(upper-lower).sum()
        p = (p-lower)*s
        z = np.r_[p[:,0], p[:,1], base[:,2]*s, (upper[0]-lower[0])*s]
        gaps = constraints(z)
        badness = float(np.square(np.minimum(gaps,0)).sum())
        proposals.append((badness+0.002*abs(s-1), z))
    proposals.sort(key=lambda item:item[0])
    # Include substantial folds rather than spending the budget polishing init.
    latest = [zbase]
    def callback(z):
        check()
        latest[0] = z.copy()
    try:
        for _, z in proposals[:int(params['trials'])]:
            check()
            latest[0] = z.copy()
            result = minimize(objective, z, jac=True, method='SLSQP', bounds=bounds,
                              constraints={'type':'ineq','fun':constraints,'jac':jacobian},
                              callback=callback,
                              options={'maxiter':int(params['iterations']), 'ftol':2e-11, 'disp':False})
            retain(result.x)
    except StopSearch:
        pass
    return best
