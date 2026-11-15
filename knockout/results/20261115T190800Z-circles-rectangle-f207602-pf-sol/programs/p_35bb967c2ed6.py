import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"iterations": (190, 50, 400), "restart_threshold": (2.345, 2.2, 2.365)}


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.69 * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    rows = np.arange(len(ii))
    best = None
    best_score = -float('inf')

    def canonical(obj):
        if obj is None:
            return None
        if 'circles' in obj:
            return obj
        out = []
        for x, y, r, m in obj['orbits']:
            if time.monotonic() >= deadline:
                break
            out.append([x, y, r])
            if m == 2:
                out.append([x, -y, r])
        return {'circles': out}

    def evaluate(obj):
        try:
            a = np.asarray(obj['circles'], dtype=np.float64)
            if a.shape != (21, 3) or not np.isfinite(a).all() or not (a[:, 2] > 0).all():
                return -float('inf')
            xs = [Fraction(float(v)) for v in a[:, 0]]
            ys = [Fraction(float(v)) for v in a[:, 1]]
            rs = [Fraction(float(v)) for v in a[:, 2]]
            w = max(x+r for x,r in zip(xs,rs))-min(x-r for x,r in zip(xs,rs))
            h = max(y+r for y,r in zip(ys,rs))-min(y-r for y,r in zip(ys,rs))
            if w+h > 2:
                return -float('inf')
            d = np.sqrt((a[ii,0]-a[jj,0])**2 + (a[ii,1]-a[jj,1])**2)
            if np.any(d < a[ii,2]+a[jj,2]):
                return -float('inf')
            return float(np.sum(a[:,2]))
        except (ValueError, TypeError, KeyError, OverflowError):
            return -float('inf')

    original = canonical(init)
    if original is not None:
        sc = evaluate(original)
        if sc > best_score:
            best_score, best = sc, original

    def consider(a):
        nonlocal best, best_score
        if time.monotonic() >= deadline or a.shape != (21,3) or not np.isfinite(a).all():
            return
        a = a.copy()
        if np.any(a[:,2] <= 0):
            return
        d = np.sqrt((a[ii,0]-a[jj,0])**2 + (a[ii,1]-a[jj,1])**2)
        shrink = min(1.0, float(np.min(d/(a[ii,2]+a[jj,2]))))
        if shrink <= 0:
            return
        a[:,2] *= shrink * (1.0-3e-14)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        scale = (2.0-8e-14)/float(np.sum(hi-lo))
        a[:,:2] -= lo
        a *= scale
        if float(np.sum(a[:,2])) <= best_score:
            return
        obj = {'circles': a.tolist()}
        sc = evaluate(obj)
        if sc > best_score:
            best_score, best = sc, obj

    # A compact fallback is available even if the supplied candidate is malformed.
    fallback = []
    for k in range(21):
        if time.monotonic() >= deadline:
            break
        fallback.append([(k%5)*0.2+0.1, (k//5)*0.2+0.1, 0.099999999999])
    if len(fallback) == 21:
        consider(np.array(fallback))

    # Variables are the 63 circle entries followed by rectangle width.
    # Height is eliminated using width + height = 2.
    constant = np.zeros((84,64))
    q = np.arange(21)
    constant[q,3*q] = 1
    constant[q,3*q+2] = -1
    constant[21+q,3*q] = -1
    constant[21+q,3*q+2] = -1
    constant[21+q,63] = 1
    constant[42+q,3*q+1] = 1
    constant[42+q,3*q+2] = -1
    constant[63+q,3*q+1] = -1
    constant[63+q,3*q+2] = -1
    constant[63+q,63] = -1
    offset = np.zeros(84)
    offset[63:] = 2
    grad = np.zeros(64)
    grad[2:63:3] = -1
    bounds = [(None,None),(None,None),(1e-7,None)]*21 + [(0.15,1.85)]

    class Stop(Exception):
        pass

    def run(a, stop, maxiter):
        if time.monotonic() >= stop:
            return
        a = np.asarray(a,dtype=float).copy()
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        wh = hi-lo
        a[:,:2] -= lo
        a *= 2/float(np.sum(wh))
        z = np.r_[a.ravel(), np.clip(2*wh[0]/np.sum(wh),0.15,1.85)]
        last = [z.copy()]
        counter = [0]
        cache = [None,None,None]

        def calc(v):
            if time.monotonic() >= stop:
                raise Stop()
            if cache[0] is not None and np.array_equal(v,cache[0]):
                return cache[1],cache[2]
            c = v[:63].reshape(21,3)
            dx = c[ii,0]-c[jj,0]
            dy = c[ii,1]-c[jj,1]
            dist = np.sqrt(dx*dx+dy*dy)
            safe = np.maximum(dist,1e-15)
            values = np.r_[constant@v+offset, dist-c[ii,2]-c[jj,2]]
            jac = np.zeros((294,64))
            jac[:84] = constant
            rr = rows+84
            jac[rr,3*ii] = dx/safe
            jac[rr,3*jj] = -dx/safe
            jac[rr,3*ii+1] = dy/safe
            jac[rr,3*jj+1] = -dy/safe
            jac[rr,3*ii+2] = -1
            jac[rr,3*jj+2] = -1
            cache[:] = [v.copy(),values,jac]
            return values,jac

        def callback(v):
            if time.monotonic() >= stop:
                raise Stop()
            last[0] = v.copy()
            counter[0] += 1
            if counter[0] % 4 == 0:
                consider(v[:63].reshape(21,3))

        try:
            result = minimize(lambda v: float(grad@v), z, jac=lambda v: grad,
                bounds=bounds, constraints={'type':'ineq','fun':lambda v:calc(v)[0],
                'jac':lambda v:calc(v)[1]}, method='SLSQP', callback=callback,
                options={'maxiter':maxiter,'ftol':2e-12,'disp':False})
            last[0] = result.x
        except Stop:
            pass
        consider(last[0][:63].reshape(21,3))

    a0 = None
    try:
        aa = np.asarray(original['circles'],dtype=float)
        if aa.shape == (21,3) and np.isfinite(aa).all() and (aa[:,2]>0).all():
            a0 = aa
    except (TypeError,KeyError,ValueError):
        pass
    if a0 is None:
        a0 = np.asarray(fallback)
    # Reserve a short final polishing window, rather than interrupt a good basin early.
    run(a0, start+0.56*budget_s, int(params['iterations']))
    if time.monotonic() < deadline-0.035:
        if best is not None and best_score >= params['restart_threshold']:
            run(np.asarray(best['circles']), deadline-0.008, 80)
        else:
            # Five reflected rows: three singles and nine mirrored pairs.
            lattice = []
            for row,count in enumerate([4,5,3,5,4]):
                if time.monotonic() >= deadline:
                    break
                yy = (row-2)*np.sqrt(3.0)
                for j in range(count):
                    if time.monotonic() >= deadline:
                        break
                    lattice.append([2*j-(count-1),yy,0.96])
            if len(lattice)==21:
                run(np.asarray(lattice), deadline-0.008, 110)
    if best is not None:
        return best
    return original if original is not None else {'circles': fallback}
