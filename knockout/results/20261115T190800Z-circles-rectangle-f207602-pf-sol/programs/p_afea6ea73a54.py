import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import lstsq

PARAMS = {
    'iterations': (85, 20, 180),
    'contact_margin': (0.045, 0.015, 0.12),
    'perturbation': (0.012, 0.001, 0.08, 'log'),
    'safety': (4e-15, 1e-15, 1e-11, 'log')
}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    eye = np.eye(n)
    gradient = np.r_[np.zeros(42), -np.ones(21), 0.0]
    safety = float(params['safety'])
    best = init
    best_score = -float('inf')
    best_a = None

    def exact(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        perimeter = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        perimeter += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        if perimeter > 2:
            return -float('inf')
        for i,j in zip(ii,jj):
            if time.monotonic() > start + 0.94 * budget_s:
                return -float('inf')
            d = np.sqrt((a[i,0]-a[j,0])**2 + (a[i,1]-a[j,1])**2)
            if d < a[i,2]+a[j,2]:
                return -float('inf')
        return float(np.sum(a[:,2]))

    if init is not None:
        best_a = np.asarray(init['circles'], dtype=float).copy()
        best_score = exact(best_a)

    base = np.array([[2*c+1., 2*r+1., 1.] for r in range(4) for c in range(5)] + [[4.,4.,np.sqrt(2.)-1.]])/9.

    def retain(a):
        nonlocal best, best_a, best_score
        if time.monotonic() >= deadline:
            raise StopSearch()
        a = np.asarray(a, dtype=float).copy()
        if not np.isfinite(a).all() or np.min(a[:,2]) <= 0:
            return
        dx = a[ii,0]-a[jj,0]
        dy = a[ii,1]-a[jj,1]
        dist = np.sqrt(dx*dx+dy*dy)
        factor = min(1., float(np.min(dist/(a[ii,2]+a[jj,2]))))
        a[:,2] *= factor*(1.-safety)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        a[:,:2] -= lo
        extent = np.max(a[:,:2]+a[:,2,None],axis=0)
        a *= (2./float(np.sum(extent)))*(1.-safety)
        if float(np.sum(a[:,2])) <= best_score:
            return
        score = exact(a)
        if score > best_score:
            best_score = score
            best_a = a.copy()
            best = {'circles': a.tolist()}

    def run(a, allowance, polish=False):
        local_end = min(deadline, time.monotonic()+allowance)
        def check():
            if time.monotonic() >= local_end:
                raise StopSearch()
        a = a.copy()
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        a[:,:2] -= lo
        wh = np.max(a[:,:2]+a[:,2,None],axis=0)
        scale = 2./float(np.sum(wh))
        a *= scale
        z = np.r_[a[:,0],a[:,1],a[:,2],wh[0]*scale]
        dd = np.sqrt(np.sum((a[ii,:2]-a[jj,:2])**2,axis=1))-a[ii,2]-a[jj,2]
        mask = dd < float(params['contact_margin'])
        pi,pj = ii[mask],jj[mask]
        m = len(pi)
        rows = np.arange(m)
        constant = np.zeros((m+84,64))
        constant[rows,42+pi] = -1.
        constant[rows,42+pj] = -1.
        constant[m:m+21,:21] = eye
        constant[m:m+21,42:63] = -eye
        constant[m+21:m+42,:21] = -eye
        constant[m+21:m+42,42:63] = -eye
        constant[m+21:m+42,63] = 1.
        constant[m+42:m+63,21:42] = eye
        constant[m+42:m+63,42:63] = -eye
        constant[m+63:m+84,21:42] = -eye
        constant[m+63:m+84,42:63] = -eye
        constant[m+63:m+84,63] = -1.

        def con(v):
            check()
            x,y,r,w = v[:21],v[21:42],v[42:63],v[63]
            dx,dy = x[pi]-x[pj],y[pi]-y[pj]
            d = np.sqrt(dx*dx+dy*dy+1e-30)
            return np.r_[d-r[pi]-r[pj],x-r,w-x-r,y-r,2.-w-y-r]

        def jac(v):
            check()
            dx = v[pi]-v[pj]
            dy = v[21+pi]-v[21+pj]
            d = np.sqrt(dx*dx+dy*dy+1e-30)
            j = constant.copy()
            j[rows,pi] = dx/d
            j[rows,pj] = -dx/d
            j[rows,21+pi] = dy/d
            j[rows,21+pj] = -dy/d
            return j

        last = z.copy()
        def callback(v):
            nonlocal last
            check()
            last = v.copy()
            retain(np.column_stack((v[:21],v[21:42],v[42:63])))

        try:
            res = minimize(lambda v: -float(np.sum(v[42:63])),z,jac=lambda v:gradient,
                           method='SLSQP',bounds=[(None,None)]*42+[(1e-6,0.9)]*21+[(0.1,1.9)],
                           constraints={'type':'ineq','fun':con,'jac':jac},callback=callback,
                           options={'maxiter':int(params['iterations']),'ftol':2e-14,'disp':False})
            callback(res.x)
            last = res.x.copy()
            if polish:
                active = np.flatnonzero(np.abs(con(last)) < 2e-7)
                if len(active) >= 64:
                    for unused in range(5):
                        check()
                        values = con(last)[active]
                        matrix = jac(last)[active]
                        step, residual, rank, singular = lstsq(matrix,-values,check_finite=False,lapack_driver='gelsy')
                        if rank < 64 or np.max(np.abs(step)) > 0.002:
                            break
                        last += step
                        callback(last)
                        if np.max(np.abs(values)) < 2e-16:
                            break
        except (StopSearch, ValueError, FloatingPointError):
            pass
        return last

    try:
        retain(base)
        origin = best_a.copy() if best_a is not None else base.copy()
        run(origin, min(0.48,0.32*budget_s), True)
        attempt = 0
        while time.monotonic() < deadline-0.035:
            anchor = best_a.copy() if best_a is not None else origin.copy()
            amplitude = float(params['perturbation'])*(1.55**(attempt % 4))
            pivot = int(rng.integers(21))
            distance = np.sum((anchor[:,:2]-anchor[pivot,:2])**2,axis=1)
            group = np.argsort(distance)[:int(rng.integers(4,9))]
            direction = rng.normal(size=2)
            direction /= max(1e-12,float(np.linalg.norm(direction)))
            anchor[group,:2] += amplitude*direction
            anchor[group,2] *= np.exp(rng.normal(0.,amplitude*1.4,len(group)))
            anchor[:,:2] += rng.normal(0.,amplitude*0.12,(21,2))
            run(anchor,min(0.27,deadline-time.monotonic()),True)
            attempt += 1
    except (StopSearch, ValueError, FloatingPointError):
        pass
    if best is not None:
        return best
    base[:,2] *= 1.-1e-10
    base *= 1.-1e-10
    return {'circles':base.tolist()}
