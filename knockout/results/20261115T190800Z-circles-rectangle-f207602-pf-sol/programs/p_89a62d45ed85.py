import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'iterations': (65, 20, 150),
    'jitter': (0.014, 0.001, 0.06),
    'time_fraction': (0.69, 0.4, 0.72)
}

N = 21
I, J = np.triu_indices(N, 1)
ROW = np.arange(len(I))
X = 3*np.arange(N)
R = X+2


def exact_score(data):
    try:
        c = np.asarray(data['circles'], dtype=np.float64)
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return -float('inf')
        xs = [Fraction(float(v)) for v in c[:, 0]]
        ys = [Fraction(float(v)) for v in c[:, 1]]
        rs = [Fraction(float(v)) for v in c[:, 2]]
        w = max(x+r for x,r in zip(xs,rs))-min(x-r for x,r in zip(xs,rs))
        h = max(y+r for y,r in zip(ys,rs))-min(y-r for y,r in zip(ys,rs))
        if w+h > 2:
            return -float('inf')
        for a,b in itertools.combinations(c, 2):
            if np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2) < a[2]+b[2]:
                return -float('inf')
        return float(np.sum(c[:, 2]))
    except Exception:
        return -float('inf')


def repair(c):
    c = np.asarray(c, dtype=float).reshape(21, 3).copy()
    if not np.isfinite(c).all():
        return None
    c[:, 2] = np.maximum(c[:, 2], 1e-9)
    d = c[I, :2]-c[J, :2]
    ratio = np.min(np.sqrt(np.sum(d*d, axis=1))/(c[I, 2]+c[J, 2]))
    if ratio <= 0:
        return None
    c[:, 2] *= min(1., float(ratio))*(1.-3e-13)
    lo = np.min(c[:, :2]-c[:, 2,None], axis=0)
    hi = np.max(c[:, :2]+c[:, 2,None], axis=0)
    c[:, :2] -= (lo+hi)/2
    c *= (2.-2e-12)/float(np.sum(hi-lo))
    return {'circles': c.tolist()}


class StopSearch(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start+float(params['time_fraction'])*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = exact_score(init)

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def consider(c):
        nonlocal best, best_score
        candidate = repair(c)
        if candidate is None:
            return
        rough = float(np.sum(np.asarray(candidate['circles'])[:, 2]))
        if rough > best_score:
            score = exact_score(candidate)
            if score > best_score:
                best, best_score = candidate, score

    grid = np.array([[1+2*x, 1+2*y, 1.] for y in range(4) for x in range(4)] +
                    [[2+2*x, 2+2*y, np.sqrt(2)-1] for x,y in [(0,0),(0,2),(1,1),(2,0),(2,2)]])/8
    consider(grid)
    try:
        initial = np.asarray(init['circles'], dtype=float).copy()
        if initial.shape != (21,3) or not np.isfinite(initial).all():
            initial = grid.copy()
    except Exception:
        initial = grid.copy()

    grad = np.zeros(64)
    grad[R] = -1
    bounds = [(0.,2.), (0.,2.), (1e-7,0.7)]*21+[(0.15,1.85)]
    base_jac = np.zeros((294,64))
    base_jac[ROW,3*I+2] = -1
    base_jac[ROW,3*J+2] = -1
    t = 210+np.arange(21)
    base_jac[t,X] = 1
    base_jac[t,R] = -1
    base_jac[t+21,X] = -1
    base_jac[t+21,R] = -1
    base_jac[t+21,63] = 1
    base_jac[t+42,X+1] = 1
    base_jac[t+42,R] = -1
    base_jac[t+63,X+1] = -1
    base_jac[t+63,R] = -1
    base_jac[t+63,63] = -1

    def objective(z):
        check()
        return -float(np.sum(z[R]))

    def gradient(z):
        check()
        return grad

    def constraints(z):
        check()
        c = z[:63].reshape(21,3)
        x,y,r = c.T
        d = c[I,:2]-c[J,:2]
        return np.concatenate((np.sqrt(np.sum(d*d,axis=1))-r[I]-r[J],
                               x-r,z[63]-x-r,y-r,2-z[63]-y-r))

    def jacobian(z):
        check()
        c = z[:63].reshape(21,3)
        delta = c[I,:2]-c[J,:2]
        dist = np.maximum(np.sqrt(np.sum(delta*delta,axis=1)),1e-20)
        u = delta/dist[:,None]
        j = base_jac.copy()
        j[ROW,3*I] = u[:,0]
        j[ROW,3*J] = -u[:,0]
        j[ROW,3*I+1] = u[:,1]
        j[ROW,3*J+1] = -u[:,1]
        return j

    def callback(z):
        check()
        consider(z[:63].reshape(21,3))

    attempt = 0
    while time.monotonic() < deadline:
        try:
            check()
            if attempt == 0:
                c = initial.copy()
            else:
                c = np.asarray(best['circles'],dtype=float).copy()
                amplitude = float(params['jitter'])*[0.5,1.,2.,1.5][(attempt-1)%4]
                if attempt % 4 == 3:
                    # Move one small disk to a promising empty pocket, then polish jointly.
                    k = int(np.argsort(c[:,2])[int(rng.integers(0,4))])
                    lo = np.min(c[:,:2]-c[:,2,None],axis=0)
                    hi = np.max(c[:,:2]+c[:,2,None],axis=0)
                    points = rng.uniform(lo,hi,size=(180,2))
                    mask = np.arange(21) != k
                    delta = points[:,None,:]-c[None,mask,:2]
                    free = np.min(np.sqrt(np.sum(delta*delta,axis=2))-c[None,mask,2],axis=1)
                    free = np.minimum(free,np.min(np.minimum(points-lo,hi-points),axis=1))
                    p = int(np.argmax(free))
                    c[k,:2] = points[p]
                    c[k,2] = max(0.015,float(free[p]))
                    amplitude *= 0.25
                c[:,:2] += rng.normal(0,amplitude,size=(21,2))
                c[:,2] *= np.exp(rng.normal(0,amplitude*0.7,size=21))
            lo = np.min(c[:,:2]-c[:,2,None],axis=0)
            hi = np.max(c[:,:2]+c[:,2,None],axis=0)
            factor = 2./float(np.sum(hi-lo))
            c[:,:2] -= lo
            c *= factor
            width = float((hi[0]-lo[0])*factor)
            z = np.r_[c.ravel(),width]
            result = minimize(objective,z,jac=gradient,method='SLSQP',bounds=bounds,
                              constraints=[{'type':'ineq','fun':constraints,'jac':jacobian}],
                              callback=callback,
                              options={'maxiter':int(params['iterations']), 'ftol':3e-12,'disp':False})
            consider(result.x[:63].reshape(21,3))
            attempt += 1
        except StopSearch:
            break
        except (ValueError,FloatingPointError):
            attempt += 1
    return best
