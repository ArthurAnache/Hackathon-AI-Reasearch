import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'poles': (240, 40, 1200),
    'iterations': (95, 20, 250),
    'separation': (0.08, 0.01, 0.3),
}
N = 21
I, J = np.triu_indices(N, 1)

class StopSearch(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    r = a[:, 2]
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < r[I]+r[J]):
        return -np.inf
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    q = [Fraction(float(v)) for v in r]
    perimeter = (max(v+s for v,s in zip(x,q))-min(v-s for v,s in zip(x,q))
                 + max(v+s for v,s in zip(y,q))-min(v-s for v,s in zip(y,q)))
    if perimeter > 2:
        return -np.inf
    return float(np.sum(r))

def normalize(a):
    a = a.copy()
    lo = np.min(a[:, :2]-a[:, 2,None], axis=0)
    hi = np.max(a[:, :2]+a[:, 2,None], axis=0)
    scale = 2.0 / np.sum(hi-lo)
    a[:, :2] = (a[:, :2]-lo)*scale
    a[:, 2] *= scale
    return a, float((hi[0]-lo[0])*scale)

def repair(a):
    a = a.copy()
    if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return None
    d = np.sqrt((a[I, 0]-a[J, 0])**2+(a[I, 1]-a[J, 1])**2)
    factor = min(1.0, float(np.min(d/(a[I, 2]+a[J, 2]))))
    a[:, 2] *= factor*(1-3e-14)
    a, _ = normalize(a)
    a *= 1-3e-14
    return a

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a = np.array([[(i%7+0.5)/4.0, (i//7+0.5)/4.0, 0.12] for i in range(21)])
        a, _ = normalize(a)
        a *= 1-1e-12
        best = {'circles': a.tolist()}
    else:
        best = init
        a = np.asarray(init['circles'], dtype=float).copy()
    best_score = exact_score(a)
    base, width = normalize(a)
    p = base[:, :2]
    r = base[:, 2]
    height = 2-width

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def accept(candidate):
        nonlocal best, best_score
        if candidate is None:
            return
        score = exact_score(candidate)
        if score > best_score:
            best_score = score
            best = {'circles': candidate.tolist()}

    # Gap centers and external poles both produce globally coordinated moves.
    pool = []
    count = int(params['poles'])
    try:
        for k in range(count):
            check()
            if k < count//2:
                pole = rng.uniform([0.0, 0.0], [width, height])
            elif k < 3*count//4:
                ids = rng.choice(N, 3, replace=False)
                pole = p[ids].mean(axis=0)
            else:
                angle = rng.uniform(0, 2*np.pi)
                distance = rng.uniform(0.55, 1.6)*max(width,height)
                pole = np.array([width/2,height/2])+distance*np.array([np.cos(angle),np.sin(angle)])
            v = p-pole
            den = np.sum(v*v, axis=1)-r*r
            # The inversion pole must be exterior to every disk.
            if np.min(den) <= 1e-7:
                continue
            transformed = np.column_stack((v/den[:,None], r/den))
            transformed, w = normalize(transformed)
            score = float(transformed[:,2].sum())
            pool.append((score, transformed, w, pole))
        pool.sort(key=lambda t: -t[0])
        selected = []
        for score, transformed, w, pole in pool:
            check()
            if any(np.linalg.norm(pole-old) < float(params['separation']) for old in selected):
                continue
            selected.append(pole)
            accept(repair(transformed))
            z0 = np.r_[transformed[:,0],transformed[:,1],transformed[:,2]*0.998,w]
            objective_gradient = np.r_[np.zeros(42),-np.ones(21),0.0]
            rows = np.arange(len(I))
            eye = np.eye(N)

            def objective(z):
                check()
                return -float(np.sum(z[42:63])), objective_gradient

            def constraints(z):
                check()
                x,y,q,w = z[:21],z[21:42],z[42:63],z[63]
                dx,dy = x[I]-x[J],y[I]-y[J]
                dist = np.sqrt(dx*dx+dy*dy+1e-30)
                return np.r_[dist-q[I]-q[J], x-q,w-x-q,y-q,2-w-y-q]

            def jacobian(z):
                check()
                x,y = z[:21],z[21:42]
                dx,dy = x[I]-x[J],y[I]-y[J]
                d = np.sqrt(dx*dx+dy*dy+1e-30)
                jac = np.zeros((len(I)+84,64))
                jac[rows,I] = dx/d
                jac[rows,J] = -dx/d
                jac[rows,I+21] = dy/d
                jac[rows,J+21] = -dy/d
                jac[rows,I+42] = -1
                jac[rows,J+42] = -1
                m = len(I)
                jac[m:m+21,:21] = eye
                jac[m:m+21,42:63] = -eye
                jac[m+21:m+42,:21] = -eye
                jac[m+21:m+42,42:63] = -eye
                jac[m+21:m+42,63] = 1
                jac[m+42:m+63,21:42] = eye
                jac[m+42:m+63,42:63] = -eye
                jac[m+63:m+84,21:42] = -eye
                jac[m+63:m+84,42:63] = -eye
                jac[m+63:m+84,63] = -1
                return jac

            last = [z0.copy()]
            def callback(z):
                check()
                last[0] = z.copy()

            try:
                result = minimize(objective,z0,jac=True,method='SLSQP',
                    bounds=[(None,None)]*42+[(1e-7,0.7)]*21+[(0.05,1.95)],
                    constraints={'type':'ineq','fun':constraints,'jac':jacobian},
                    callback=callback,
                    options={'maxiter':int(params['iterations']),'ftol':2e-12,'disp':False})
                z = result.x
            except StopSearch:
                z = last[0]
                candidate = np.column_stack((z[:21],z[21:42],z[42:63]))
                accept(repair(candidate))
                break
            candidate = np.column_stack((z[:21],z[21:42],z[42:63]))
            accept(repair(candidate))
    except StopSearch:
        pass
    return best
