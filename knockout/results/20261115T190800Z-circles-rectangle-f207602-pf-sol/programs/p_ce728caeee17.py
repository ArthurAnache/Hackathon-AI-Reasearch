import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"warm_fraction": (0.35, 0.15, 0.55), "iterations": (180, 50, 350), "safety": (3e-13, 1e-14, 1e-10, "log")}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    npair = len(ii)
    rr = np.arange(n)
    pr = np.arange(npair)
    safety = params['safety']
    best = init
    best_score = -float('inf')

    def exact(c):
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return -float('inf')
        d = c[ii, :2] - c[jj, :2]
        if np.any(np.sqrt(d[:, 0]**2 + d[:, 1]**2) < c[ii, 2] + c[jj, 2]):
            return -float('inf')
        xs = [Fraction(float(v)) for v in c[:, 0]]
        ys = [Fraction(float(v)) for v in c[:, 1]]
        rs = [Fraction(float(v)) for v in c[:, 2]]
        w = max(x+r for x,r in zip(xs,rs))-min(x-r for x,r in zip(xs,rs))
        h = max(y+r for y,r in zip(ys,rs))-min(y-r for y,r in zip(ys,rs))
        if w+h > 2:
            return -float('inf')
        return float(np.sum(c[:, 2]))

    def consider(c, repair=True):
        nonlocal best, best_score
        c = np.array(c, dtype=float, copy=True)
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return
        if repair:
            d = c[ii, :2]-c[jj, :2]
            ratio = np.min(np.sqrt(np.sum(d*d, axis=1))/(c[ii, 2]+c[jj, 2]))
            c[:, 2] *= min(1.0, float(ratio))*(1-safety)
            lo = np.min(c[:, :2]-c[:, 2,None], axis=0)
            hi = np.max(c[:, :2]+c[:, 2,None], axis=0)
            perimeter = float(np.sum(hi-lo))
            if perimeter <= 0:
                return
            c[:, :2] -= (lo+hi)/2
            c *= (2/perimeter)*(1-safety)
        if float(np.sum(c[:, 2])) > best_score:
            score = exact(c)
            if score > best_score:
                best_score = score
                best = {'circles': c.tolist()}

    if init is not None:
        try:
            initial = np.asarray(init['circles'], dtype=float)
            best_score = exact(initial)
            consider(initial)
        except (KeyError, ValueError, TypeError):
            initial = None
    else:
        initial = None

    obj = np.zeros(64)
    obj[42:63] = -1
    base_jac = np.zeros((npair+84, 64))
    base_jac[pr, 42+ii] = -1
    base_jac[pr, 42+jj] = -1
    k = npair
    base_jac[k+rr, rr] = 1
    base_jac[k+rr, 42+rr] = -1
    base_jac[k+21+rr, rr] = -1
    base_jac[k+21+rr, 42+rr] = -1
    base_jac[k+21+rr, 63] = 1
    base_jac[k+42+rr, 21+rr] = 1
    base_jac[k+42+rr, 42+rr] = -1
    base_jac[k+63+rr, 21+rr] = -1
    base_jac[k+63+rr, 42+rr] = -1
    base_jac[k+63+rr, 63] = -1
    bounds = [(0,2)]*42 + [(1e-7,0.8)]*21 + [(0.1,1.9)]

    def pack(c):
        c = np.array(c, copy=True)
        lo = np.min(c[:,:2]-c[:,2,None], axis=0)
        hi = np.max(c[:,:2]+c[:,2,None], axis=0)
        scale = 2/np.sum(hi-lo)
        c[:,:2] -= lo
        c *= scale
        return np.r_[c[:,0], c[:,1], c[:,2], (hi[0]-lo[0])*scale]

    def unpack(z):
        return np.column_stack((z[:21],z[21:42],z[42:63]))

    def optimize(c, until):
        if time.monotonic() >= until:
            return
        z0 = pack(c)
        last = [z0.copy()]
        cache = [None, None, None]
        def calc(z):
            if time.monotonic() >= until:
                raise StopSearch()
            if cache[0] is not None and np.array_equal(z, cache[0]):
                return cache[1], cache[2]
            x,y,r,w = z[:21],z[21:42],z[42:63],z[63]
            dx,dy = x[ii]-x[jj],y[ii]-y[jj]
            ds = np.sqrt(dx*dx+dy*dy)
            inv = 1/np.maximum(ds,1e-15)
            values = np.r_[ds-r[ii]-r[jj], x-r, w-x-r, y-r, 2-w-y-r]
            jac = base_jac.copy()
            jac[pr,ii] = dx*inv
            jac[pr,jj] = -dx*inv
            jac[pr,21+ii] = dy*inv
            jac[pr,21+jj] = -dy*inv
            cache[:] = [z.copy(),values,jac]
            return values,jac
        counter = [0]
        def callback(z):
            last[0] = z.copy()
            counter[0] += 1
            if time.monotonic() >= until:
                raise StopSearch()
            if counter[0] % 8 == 0:
                consider(unpack(z))
        try:
            result = minimize(lambda z: float(obj@z), z0, jac=lambda z: obj,
                method='SLSQP', bounds=bounds,
                constraints={'type':'ineq','fun':lambda z:calc(z)[0], 'jac':lambda z:calc(z)[1]},
                callback=callback,
                options={'maxiter':int(params['iterations']), 'ftol':2e-12, 'disp':False})
            last[0] = result.x
        except StopSearch:
            pass
        consider(unpack(last[0]))

    def rows(counts):
        c=[]
        for row,count in enumerate(counts):
            if time.monotonic() >= deadline:
                break
            for j in range(count):
                if time.monotonic() >= deadline:
                    break
                c.append([2*j-(count-1), (row-2)*np.sqrt(3), 1.0])
        return np.array(c)

    if initial is not None and initial.shape == (21,3):
        optimize(initial, min(deadline, started + params['warm_fraction']*0.70*budget_s))
    families = [[5,4,3,4,5], [4,5,3,5,4], [4,4,5,4,4]]
    for index, family in enumerate(families):
        now = time.monotonic()
        if now >= deadline-0.025:
            break
        c = rows(family)
        if c.shape != (21,3):
            break
        consider(c)
        remaining = deadline-now
        allocation = remaining*(0.55 if index == 0 else 0.65 if index == 1 else 0.95)
        optimize(c, min(deadline-0.008, now+allocation))
    if best is not None and time.monotonic() < deadline-0.04:
        optimize(np.asarray(best['circles']), deadline-0.008)
    if best is None:
        # A simple strictly feasible fallback.
        c = np.array([[0.1*(i%7),0.1*(i//7),0.04] for i in range(21)])
        consider(c)
    return best
