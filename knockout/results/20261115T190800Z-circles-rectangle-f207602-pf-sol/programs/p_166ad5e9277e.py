import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'symmetric_fraction': (0.53, 0.3, 0.6),
    'maxiter': (180, 40, 400),
    'repair_margin': (2e-13, 1e-14, 1e-10, 'log')
}
N = 21
I, J = np.triu_indices(N, 1)
K = np.arange(N)
M = len(I)

class StopSearch(Exception):
    pass

def exact_score(data):
    try:
        a = np.asarray(data['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -np.inf
        dx = a[I, 0] - a[J, 0]
        dy = a[I, 1] - a[J, 1]
        if np.any(np.sqrt(dx*dx + dy*dy) < a[I, 2] + a[J, 2]):
            return -np.inf
        x, y, r = [[Fraction(float(v)) for v in a[:, c]] for c in range(3)]
        perimeter = (max(u+v for u,v in zip(x,r)) - min(u-v for u,v in zip(x,r))
                     + max(u+v for u,v in zip(y,r)) - min(u-v for u,v in zip(y,r)))
        return float(a[:, 2].sum()) if perimeter <= 2 else -np.inf
    except (TypeError, ValueError, KeyError, OverflowError):
        return -np.inf

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = exact_score(init)
    margin = float(params['repair_margin'])

    def retain(a):
        nonlocal best, best_score
        if time.monotonic() >= end:
            return
        a = np.asarray(a, dtype=float).copy()
        if a.shape != (N,3) or not np.isfinite(a).all():
            return
        p, r = a[:, :2], np.maximum(a[:, 2], 1e-11)
        d = np.sqrt(np.sum((p[I]-p[J])**2, axis=1))
        factor = min(1.0, float(np.min(d/(r[I]+r[J]))))
        if factor <= 0:
            return
        r = r * factor * (1-margin)
        lo = np.min(p-r[:,None], axis=0)
        sides = np.max(p+r[:,None],axis=0)-lo
        scale = 2.0/sides.sum()*(1-margin)
        out = {'circles': np.column_stack(((p-lo)*scale, r*scale)).tolist()}
        if float(np.sum(r)*scale) > best_score:
            s = exact_score(out)
            if s > best_score:
                best, best_score = out, s

    def optimize(z, X, Y, R, deadline):
        dim = len(z)
        W = np.zeros(dim)
        W[-1] = 1.0
        # Each row is a linear map from optimization variables to a coordinate.
        objgrad = -R.sum(axis=0)
        boundaries = np.vstack((X-R+0.5*W, -X-R+0.5*W,
                                Y-R-0.5*W, -Y-R-0.5*W))
        offsets = np.r_[np.zeros(2*N), np.ones(2*N)]
        DX, DY, RR = X[I]-X[J], Y[I]-Y[J], R[I]+R[J]
        latest = z.copy()
        count = 0
        def check():
            if time.monotonic() >= deadline:
                raise StopSearch
        def fun(v):
            check()
            return float(objgrad@v), objgrad
        def con(v):
            check()
            dx, dy = DX@v, DY@v
            return np.r_[np.sqrt(dx*dx+dy*dy+1e-30)-RR@v, boundaries@v+offsets]
        def jac(v):
            check()
            dx,dy = DX@v,DY@v
            d = np.sqrt(dx*dx+dy*dy+1e-30)
            return np.vstack(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RR, boundaries))
        def callback(v):
            nonlocal latest, count
            latest = v.copy()
            count += 1
            check()
            if count % 4 == 0:
                retain(np.column_stack((X@v,Y@v,R@v)))
        bounds = [(None,None)]*dim
        bounds[-1] = (0.15,1.85)
        # Radius positivity is imposed explicitly for both parameterizations.
        def allcon(v):
            return np.r_[con(v), R@v-1e-9]
        def alljac(v):
            return np.vstack((jac(v),R))
        try:
            res = minimize(fun,z,jac=True,method='SLSQP',bounds=bounds,
                           constraints={'type':'ineq','fun':allcon,'jac':alljac},
                           callback=callback,
                           options={'maxiter':int(params['maxiter']), 'ftol':2e-12})
            latest = res.x
        except StopSearch:
            pass
        retain(np.column_stack((X@latest,Y@latest,R@latest)))

    def full_search(data, deadline):
        if time.monotonic() >= deadline:
            return
        a = np.asarray(data['circles'],dtype=float)
        p,r = a[:,:2].copy(),a[:,2].copy()
        lo = np.min(p-r[:,None],axis=0)
        hi = np.max(p+r[:,None],axis=0)
        scale = 2/(hi-lo).sum()
        p = (p-(lo+hi)/2)*scale
        r *= scale
        z = np.r_[p[:,0],p[:,1],r,(hi[0]-lo[0])*scale]
        X,Y,R = np.zeros((N,64)),np.zeros((N,64)),np.zeros((N,64))
        X[K,K]=1
        Y[K,N+K]=1
        R[K,2*N+K]=1
        optimize(z,X,Y,R,deadline)

    # Always provide a valid candidate, independently of the supplied start.
    fallback = np.array([[0.1+0.2*(k%5),0.1+0.2*(k//5),0.09] for k in range(N)])
    retain(fallback)
    if best_score > 2.25:
        full_search(best, start+0.55*budget_s)
    else:
        layouts = ([4,5,3,5,4], [5,4,5,4,3], [5,4,3,4,5])
        sym_end = start+float(params['symmetric_fraction'])*budget_s
        for run, rows in enumerate(layouts):
            now = time.monotonic()
            if now >= sym_end:
                break
            deadline = now+(sym_end-now)/(len(layouts)-run)
            X,Y,R = np.zeros((N,34)),np.zeros((N,34)),np.zeros((N,34))
            z = np.zeros(34)
            z[-1] = 1.02
            circle = pair = orbit = 0
            for row, number in enumerate(rows):
                if time.monotonic() >= deadline:
                    break
                yy = (row-2)*0.185
                for j in range(number//2):
                    if time.monotonic() >= deadline:
                        break
                    xx = (j+0.5 if number%2 == 0 else j+1)*0.205
                    X[circle,pair]=1
                    X[circle+1,pair]=-1
                    Y[circle:circle+2,9+orbit]=1
                    R[circle:circle+2,21+orbit]=1
                    z[pair]=xx
                    z[9+orbit]=yy
                    z[21+orbit]=0.09
                    circle+=2
                    pair+=1
                    orbit+=1
                if number%2:
                    Y[circle,9+orbit]=1
                    R[circle,21+orbit]=1
                    z[9+orbit]=yy
                    z[21+orbit]=0.09
                    circle+=1
                    orbit+=1
            if circle == N and time.monotonic() < deadline:
                retain(np.column_stack((X@z,Y@z,R@z)))
                optimize(z,X,Y,R,deadline)
    if time.monotonic() < end and best is not None:
        full_search(best,end)
    return best
