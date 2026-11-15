import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"iterations": (180, 50, 500), "polish_share": (0.25, 0.1, 0.5)}


def construct(params, seed, budget_s, init=None):
    begin = time.monotonic()
    deadline = begin + 0.69 * budget_s
    n = 21
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    rows = np.arange(m)
    k = np.arange(n)
    best = None
    best_score = -np.inf
    original = None

    def score(a):
        if a is None or a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -np.inf
        d = a[ii, :2] - a[jj, :2]
        if np.any(np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2) < a[ii, 2] + a[jj, 2]):
            return -np.inf
        x, y, r = [[Fraction(float(v)) for v in a[:, j]] for j in range(3)]
        p = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        p += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        return float(np.sum(a[:, 2])) if p <= 2 else -np.inf

    if init is not None:
        try:
            original = np.asarray(init['circles'], float) if 'circles' in init else np.asarray(init['state'], float).reshape(3,21).T
            s = score(original)
            if s > best_score:
                best, best_score = original.copy(), s
        except Exception:
            original = None

    def repair(a):
        a = np.array(a, dtype=float, copy=True)
        if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return None
        lo = np.min(a[:, :2] - a[:, 2,None], axis=0)
        hi = np.max(a[:, :2] + a[:, 2,None], axis=0)
        a[:, :2] -= (lo+hi)/2
        d = a[ii,:2]-a[jj,:2]
        rat = np.min(np.sqrt(np.sum(d*d, axis=1))/(a[ii,2]+a[jj,2]))
        a[:,2] *= min(1., float(rat))*(1.-5e-14)
        if np.any(a[:,2] <= 0):
            return None
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        a *= 2./np.sum(hi-lo)*(1.-5e-14)
        return a

    def consider(a):
        nonlocal best, best_score
        a = repair(a)
        s = score(a)
        if s > best_score:
            best, best_score = a, s

    # Three axis circles and nine reflected pairs.
    T = np.zeros((64,34))
    for j in range(9):
        if time.monotonic() >= deadline:
            break
        T[3+j,j] = 1
        T[12+j,j] = -1
    for j in range(3):
        if time.monotonic() >= deadline:
            break
        T[21+j,9+j] = 1
        T[42+j,21+j] = 1
    for j in range(9):
        if time.monotonic() >= deadline:
            break
        T[24+j,12+j] = T[33+j,12+j] = 1
        T[45+j,24+j] = T[54+j,24+j] = 1
    T[63,33] = 1

    fixed = np.zeros((m+84,64))
    fixed[m+k,k] = -1
    fixed[m+k,42+k] = -1
    fixed[m+k,63] = .5
    fixed[m+21+k,k] = 1
    fixed[m+21+k,42+k] = -1
    fixed[m+21+k,63] = .5
    fixed[m+42+k,21+k] = -1
    fixed[m+42+k,42+k] = -1
    fixed[m+42+k,63] = -.5
    fixed[m+63+k,21+k] = 1
    fixed[m+63+k,42+k] = -1
    fixed[m+63+k,63] = -.5
    grad = np.zeros(64)
    grad[42:63] = -1

    class Stop(Exception):
        pass

    def optimize(a, symmetric, end):
        if time.monotonic() >= end:
            return
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        a = a.copy()
        a[:,:2] -= (lo+hi)/2
        z = np.r_[a.T.ravel(), np.clip(hi[0]-lo[0],.02,1.98)]
        if symmetric:
            z0 = (T.T @ z)/np.sum(T*T,axis=0)
            g = T.T @ grad
            bounds = [(0.,1.)]*9+[(-1.,1.)]*12+[(1e-7,.5)]*12+[(.02,1.98)]
        else:
            z0 = z
            g = grad
            bounds = [(-1.,1.)]*42+[(1e-7,.5)]*21+[(.02,1.98)]
        last = None
        count = 0

        def check():
            if time.monotonic() >= end:
                raise Stop()

        def expand(q):
            return T @ q if symmetric else q

        def fun(q):
            check()
            return float(g @ q)

        def con(q):
            check()
            z = expand(q)
            x,y,r,w = z[:21],z[21:42],z[42:63],z[63]
            dx,dy,rr = x[ii]-x[jj],y[ii]-y[jj],r[ii]+r[jj]
            return np.r_[4*(dx*dx+dy*dy-rr*rr),w/2-x-r,w/2+x-r,1-w/2-y-r,1-w/2+y-r]

        def jac(q):
            check()
            z = expand(q)
            x,y,r = z[:21],z[21:42],z[42:63]
            dx,dy,rr = 8*(x[ii]-x[jj]),8*(y[ii]-y[jj]),-8*(r[ii]+r[jj])
            J = fixed.copy()
            J[rows,ii],J[rows,jj] = dx,-dx
            J[rows,21+ii],J[rows,21+jj] = dy,-dy
            J[rows,42+ii],J[rows,42+jj] = rr,rr
            if not symmetric:
                return J
            return np.column_stack((J[:,3:12]-J[:,12:21],J[:,21:24],J[:,24:33]+J[:,33:42],J[:,42:45],J[:,45:54]+J[:,54:63],J[:,63]))

        def callback(q):
            nonlocal last, count
            last = expand(q).copy()
            count += 1
            check()
            if count % 15 == 0:
                consider(last[:63].reshape(3,21).T)

        try:
            result = minimize(fun,z0,jac=lambda q:g,method='SLSQP',bounds=bounds,
                              constraints={'type':'ineq','fun':con,'jac':jac},callback=callback,
                              options={'maxiter':int(params['iterations']),'ftol':1e-11,'disp':False})
            last = expand(result.x)
        except Stop:
            pass
        except (ValueError, FloatingPointError):
            pass
        if last is not None:
            consider(last[:63].reshape(3,21).T)

    patterns = [(4,3,4,3,4,3),(5,4,3,4,5),(3,4,5,4,5),(4,5,4,5,3)]
    search_end = begin + (deadline-begin)*(1-params['polish_share'])
    for t, pattern in enumerate(patterns):
        if time.monotonic() >= search_end:
            break
        singles, pairs = [], []
        for j,c in enumerate(pattern):
            if time.monotonic() >= deadline:
                break
            y = (j-(len(pattern)-1)/2)*np.sqrt(3.)
            if c % 2:
                singles.append([0.,y,1.])
            for h in range(c//2):
                if time.monotonic() >= deadline:
                    break
                x = 2*(h+1) if c%2 else 2*h+1
                pairs.append([float(x),y,1.])
        if len(singles) != 3 or len(pairs) != 9:
            continue
        a = np.array(singles+pairs+[[-x,y,r] for x,y,r in pairs])
        a = repair(a)
        consider(a)
        remaining = len(patterns)-t
        end = min(search_end,time.monotonic()+max(0.,search_end-time.monotonic())/remaining)
        optimize(a,True,end)
    if best is not None and time.monotonic() < deadline:
        optimize(best,False,deadline)
    if best is None:
        a = np.array([[(i-1)*.2,(j-3)*.2,.099] for j in range(7) for i in range(3)])
        consider(a)
    if original is not None and score(original) >= best_score:
        return init
    return {'state':best.T.ravel().tolist()}
