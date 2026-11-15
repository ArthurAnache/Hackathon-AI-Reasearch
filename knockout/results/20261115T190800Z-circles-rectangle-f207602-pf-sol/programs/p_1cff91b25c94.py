import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"iterations": (110, 30, 250), "safety": (2e-13, 1e-14, 1e-9, "log")}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.69 * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    safety = float(params['safety'])
    best_array = None
    best_score = -float('inf')

    def exact(data):
        try:
            a = np.asarray(data['circles'], dtype=float)
            if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
                return -float('inf')
            x = [Fraction(float(t)) for t in a[:, 0]]
            y = [Fraction(float(t)) for t in a[:, 1]]
            r = [Fraction(float(t)) for t in a[:, 2]]
            if max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r)) + max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)) > 2:
                return -float('inf')
            for i,j in zip(ii,jj):
                if time.monotonic() > start + 0.91 * budget_s:
                    return -float('inf')
                d = np.sqrt((a[i,0]-a[j,0])**2 + (a[i,1]-a[j,1])**2)
                if d < a[i,2]+a[j,2]:
                    return -float('inf')
            return float(np.sum(a[:,2]))
        except Exception:
            return -float('inf')

    original_score = exact(init) if init is not None else -float('inf')

    def retain(a):
        nonlocal best_array, best_score
        if time.monotonic() >= deadline:
            raise StopSearch()
        a = np.asarray(a, dtype=float).copy()
        if not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        delta = a[ii,:2] - a[jj,:2]
        dist = np.sqrt(np.sum(delta*delta, axis=1))
        factor = min(1.0, float(np.min(dist/(a[ii,2]+a[jj,2]))))
        if factor <= 0:
            return
        a[:,2] *= factor * (1-safety)
        lo = np.min(a[:,:2]-a[:,2,None], axis=0)
        hi = np.max(a[:,:2]+a[:,2,None], axis=0)
        a[:,:2] -= lo
        a *= (2.0/(float(np.sum(hi-lo))))*(1-safety)
        score = float(np.sum(a[:,2]))
        if score > best_score:
            best_score = score
            best_array = a

    # Twenty equal circles plus one circle inscribed in a square gap.
    base = np.array([[2*c+1., 2*r+1., 1.] for r in range(4) for c in range(5)] + [[4.,4.,np.sqrt(2.)-1.]]) / 9.
    try:
        retain(base)
    except StopSearch:
        pass

    grad = np.zeros(64)
    grad[42:63] = -1
    rows = np.arange(len(ii))
    eye = np.eye(21)

    def run(a, allowance):
        local_end = min(deadline, time.monotonic()+allowance)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        a = a.copy()
        a[:,:2] -= lo
        wh = np.max(a[:,:2]+a[:,2,None],axis=0)
        scale = 2/np.sum(wh)
        a *= scale
        z = np.r_[a[:,0],a[:,1],a[:,2],wh[0]*scale]

        def check():
            if time.monotonic() >= local_end:
                raise StopSearch()

        def fun(v):
            check()
            return -float(np.sum(v[42:63]))

        def con(v):
            check()
            x,y,r,w = v[:21],v[21:42],v[42:63],v[63]
            dx=x[ii]-x[jj]
            dy=y[ii]-y[jj]
            ds=np.sqrt(dx*dx+dy*dy+1e-30)
            return np.r_[ds-r[ii]-r[jj], x-r, w-x-r, y-r, 2-w-y-r]

        def jac(v):
            check()
            x,y = v[:21],v[21:42]
            dx=x[ii]-x[jj]
            dy=y[ii]-y[jj]
            ds=np.sqrt(dx*dx+dy*dy+1e-30)
            j=np.zeros((294,64))
            j[rows,ii]=dx/ds
            j[rows,jj]=-dx/ds
            j[rows,21+ii]=dy/ds
            j[rows,21+jj]=-dy/ds
            j[rows,42+ii]=-1
            j[rows,42+jj]=-1
            j[210:231,:21]=eye
            j[210:231,42:63]=-eye
            j[231:252,:21]=-eye
            j[231:252,42:63]=-eye
            j[231:252,63]=1
            j[252:273,21:42]=eye
            j[252:273,42:63]=-eye
            j[273:294,21:42]=-eye
            j[273:294,42:63]=-eye
            j[273:294,63]=-1
            return j

        def callback(v):
            check()
            retain(np.column_stack((v[:21],v[21:42],v[42:63])))

        try:
            res=minimize(fun,z,jac=lambda v:grad,method='SLSQP',bounds=[(None,None)]*42+[(1e-7,1.)]*21+[(0.05,1.95)],constraints={'type':'ineq','fun':con,'jac':jac},callback=callback,options={'maxiter':int(params['iterations']),'ftol':1e-11,'disp':False})
            callback(res.x)
        except StopSearch:
            pass

    try:
        run(base, max(0.05,0.62*(deadline-time.monotonic())))
        if time.monotonic() < deadline-0.04:
            second=base.copy()
            second[-1,:2]=[4/9,2/9]
            second[:,:2] += rng.normal(0,0.001, (21,2))
            run(second,deadline-time.monotonic())
    except (StopSearch, ValueError, FloatingPointError):
        pass

    if best_array is not None:
        candidate={'circles':best_array.tolist()}
        score=exact(candidate)
        if score > original_score:
            return candidate
    if original_score > -float('inf'):
        return init
    # A fixed, strictly feasible fallback independent of the optimizer.
    fallback=base.copy()
    fallback[:,2] *= 1-1e-10
    fallback *= 1-1e-10
    return {'circles':fallback.tolist()}
