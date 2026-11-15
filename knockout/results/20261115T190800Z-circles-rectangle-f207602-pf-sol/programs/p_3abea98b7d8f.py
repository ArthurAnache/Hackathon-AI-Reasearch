import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'time_fraction': (0.69, 0.4, 0.72),
    'soft_mu': (22.0, 4.0, 100.0, 'log'),
    'jitter': (0.012, 0.0, 0.06),
    'polish_iterations': (90, 20, 200)
}

N = 21
I, J = np.triu_indices(N, 1)
K = np.arange(len(I))
T = np.arange(N)
X = 3*T
R = X+2

class StopSearch(Exception):
    pass


def exact_score(candidate):
    try:
        c = np.asarray(candidate['circles'], dtype=np.float64)
        if c.shape != (N, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return -np.inf
        xs, ys, rs = [[Fraction(float(v)) for v in c[:, j]] for j in range(3)]
        perimeter = (max(x+r for x,r in zip(xs,rs))-min(x-r for x,r in zip(xs,rs))
                     +max(y+r for y,r in zip(ys,rs))-min(y-r for y,r in zip(ys,rs)))
        if perimeter > 2:
            return -np.inf
        d = np.sqrt((c[I,0]-c[J,0])**2+(c[I,1]-c[J,1])**2)
        if np.any(d < c[I,2]+c[J,2]):
            return -np.inf
        return float(np.sum(c[:,2]))
    except Exception:
        return -np.inf


def repair(c):
    c = np.asarray(c, dtype=float).reshape(N,3).copy()
    if not np.isfinite(c).all():
        return None
    c[:,2] = np.maximum(c[:,2], 1e-10)
    d = np.sqrt((c[I,0]-c[J,0])**2+(c[I,1]-c[J,1])**2)
    ratio = float(np.min(d/(c[I,2]+c[J,2])))
    if ratio <= 0:
        return None
    c[:,2] *= min(1.0, ratio)*(1-8e-13)
    lo = np.min(c[:,:2]-c[:,2,None], axis=0)
    hi = np.max(c[:,:2]+c[:,2,None], axis=0)
    c[:,:2] -= lo
    c *= (2-3e-12)/float(np.sum(hi-lo))
    return {'circles': c.tolist()}


def normalize(c):
    c = np.array(c, dtype=float, copy=True)
    c[:,2] = np.maximum(np.abs(c[:,2]), 1e-7)
    lo = np.min(c[:,:2]-c[:,2,None], axis=0)
    hi = np.max(c[:,:2]+c[:,2,None], axis=0)
    factor = 2/float(np.sum(hi-lo))
    c[:,:2] -= lo
    c *= factor
    return np.r_[c.ravel(), (hi[0]-lo[0])*factor]


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start+float(params['time_fraction'])*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = exact_score(init)
    stage_end = end

    def check():
        if time.monotonic() >= min(end, stage_end):
            raise StopSearch()

    def consider(c):
        nonlocal best, best_score
        cand = repair(c)
        if cand is None:
            return
        if float(np.sum(np.asarray(cand['circles'])[:,2])) <= best_score:
            return
        score = exact_score(cand)
        if score > best_score:
            best, best_score = cand, score

    fallback = np.array([[1+2*x,1+2*y,1.] for y in range(4) for x in range(4)]
                        + [[2+2*x,2+2*y,np.sqrt(2)-1] for x,y in [(0,0),(0,2),(1,1),(2,0),(2,2)]])/8
    consider(fallback)
    try:
        initial = np.asarray(init['circles'], dtype=float)
        if initial.shape != (21,3) or not np.isfinite(initial).all():
            initial = fallback
    except Exception:
        initial = fallback
    consider(initial)

    grad = np.zeros(64)
    grad[R] = -1
    bounds = [(0.,2.),(0.,2.),(1e-8,.8)]*N+[(.15,1.85)]

    def objective(z):
        check()
        return -float(np.sum(z[R]))

    def gradient(z):
        check()
        return grad

    def constraints(z):
        check()
        c = z[:63].reshape(N,3)
        x,y,r = c.T
        d = np.sqrt((x[I]-x[J])**2+(y[I]-y[J])**2)
        w = z[63]
        return np.r_[d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r]

    def jacobian(z):
        check()
        c = z[:63].reshape(N,3)
        dx = c[I,0]-c[J,0]
        dy = c[I,1]-c[J,1]
        d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-20)
        a = np.zeros((294,64))
        a[K,3*I] = dx/d
        a[K,3*J] = -dx/d
        a[K,3*I+1] = dy/d
        a[K,3*J+1] = -dy/d
        a[K,3*I+2] = -1
        a[K,3*J+2] = -1
        t = 210+T
        a[t,X] = 1
        a[t,R] = -1
        a[t+21,X] = -1
        a[t+21,R] = -1
        a[t+21,63] = 1
        a[t+42,X+1] = 1
        a[t+42,R] = -1
        a[t+63,X+1] = -1
        a[t+63,R] = -1
        a[t+63,63] = -1
        return a

    def callback(z):
        check()
        consider(z[:63].reshape(N,3))

    def polish(z, allowance):
        nonlocal stage_end
        stage_end = min(end, time.monotonic()+allowance)
        try:
            result = minimize(objective, z, jac=gradient, method='SLSQP', bounds=bounds,
                              constraints={'type':'ineq','fun':constraints,'jac':jacobian},
                              callback=callback,
                              options={'maxiter':int(params['polish_iterations']), 'ftol':2e-12, 'disp':False})
            consider(result.x[:63].reshape(N,3))
        except (StopSearch, ValueError, FloatingPointError):
            pass

    def penalty(z, mu):
        check()
        c = z[:63].reshape(N,3)
        x,y,r = c.T
        dx,dy = x[I]-x[J], y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-24)
        v = np.maximum(r[I]+r[J]-d, 0)
        b0 = np.maximum(r-x,0)
        b1 = np.maximum(r+x-z[63],0)
        b2 = np.maximum(r-y,0)
        b3 = np.maximum(r+y-2+z[63],0)
        f = -r.sum()+mu*(v@v+b0@b0+b1@b1+b2@b2+b3@b3)
        q = 2*mu*v
        gx = np.bincount(I, -q*dx/d, minlength=N)+np.bincount(J, q*dx/d, minlength=N)
        gy = np.bincount(I, -q*dy/d, minlength=N)+np.bincount(J, q*dy/d, minlength=N)
        gr = -np.ones(N)+np.bincount(I,q,minlength=N)+np.bincount(J,q,minlength=N)
        gx += 2*mu*(-b0+b1)
        gy += 2*mu*(-b2+b3)
        gr += 2*mu*(b0+b1+b2+b3)
        g = np.column_stack((gx,gy,gr)).ravel()
        return f, np.r_[g, 2*mu*(-b1.sum()+b3.sum())]

    polish(normalize(initial), min(.32*budget_s, .45))
    attempt = 0
    while time.monotonic() < end-.025:
        anchor = np.asarray(best['circles'],dtype=float).copy() if best is not None else fallback.copy()
        if attempt:
            amplitude = float(params['jitter'])*(1+(attempt%3)*.65)
            anchor[:,:2] += rng.normal(0,amplitude,(N,2))
        z = normalize(anchor)
        stage_end = min(end-.03, time.monotonic()+.10*budget_s)
        last = [z.copy()]
        def remember(zz):
            check()
            last[0] = zz.copy()
        try:
            result = minimize(penalty,z,args=(float(params['soft_mu'])*(1+attempt%2),),
                              jac=True,method='L-BFGS-B',bounds=bounds,callback=remember,
                              options={'maxiter':65,'ftol':1e-10,'gtol':1e-7,'maxls':20})
            last[0] = result.x
        except (StopSearch,ValueError,FloatingPointError):
            pass
        if time.monotonic() >= end-.015:
            break
        polish(last[0], .22*budget_s)
        attempt += 1
    return best
