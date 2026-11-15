import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'first_fraction': (0.44, 0.25, 0.58),
    'perturbation': (0.025, 0.005, 0.08),
    'ftol': (2e-11, 1e-13, 1e-7, 'log'),
}

I, J = np.triu_indices(21, 1)
K = np.arange(21)
P = np.arange(len(I))

class Deadline(Exception):
    pass

def exact_score(data):
    try:
        a = np.asarray(data['circles'], dtype=np.float64)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        w = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        h = max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        if w+h > 2:
            return -float('inf')
        for c, d in itertools.combinations(a, 2):
            dist = np.sqrt((c[0]-d[0])**2+(c[1]-d[1])**2)
            if dist < c[2]+d[2]:
                return -float('inf')
        return float(np.sum(a[:, 2]))
    except Exception:
        return -float('inf')

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = exact_score(init) if init is not None else -float('inf')

    def consider(a):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        a = np.asarray(a, dtype=float).copy()
        if a.shape != (21,3) or not np.isfinite(a).all():
            return
        a[:,2] = np.maximum(a[:,2], 1e-10)
        delta = a[I,:2]-a[J,:2]
        distances = np.sqrt(np.sum(delta*delta, axis=1))
        f = min(1.0, float(np.min(distances/(a[I,2]+a[J,2]))))
        if f <= 0:
            return
        a[:,2] *= f * (1.0-2e-13)
        lo = np.min(a[:,:2]-a[:,2,None], axis=0)
        a[:,:2] -= lo
        ext = np.max(a[:,:2]+a[:,2,None], axis=0)
        a *= min(1.0, 2.0/float(np.sum(ext))) * (1.0-2e-13)
        if float(np.sum(a[:,2])) <= best_score:
            return
        candidate = {'circles': a.tolist()}
        score = exact_score(candidate)
        if score > best_score:
            best_score, best = score, candidate

    fallback = np.array([[(i % 7 + .5)*.2, (i // 7 + .5)*.2, .1] for i in range(21)])
    consider(fallback)
    try:
        initial = np.asarray(init['circles'], dtype=float).copy()
        if initial.shape != (21,3) or not np.isfinite(initial).all():
            initial = fallback.copy()
    except Exception:
        initial = fallback.copy()
    consider(initial)

    grad = np.zeros(64)
    grad[42:63] = -1.0
    bounds = [(0.,2.)]*42 + [(1e-8,.7)]*21 + [(.2,1.8)]

    def pack(a):
        a = a.copy()
        a[:,2] = np.maximum(a[:,2], 1e-7)
        lo = np.min(a[:,:2]-a[:,2,None], axis=0)
        a[:,:2] -= lo
        wh = np.max(a[:,:2]+a[:,2,None], axis=0)
        scale = 2.0/max(float(np.sum(wh)), 1e-10)
        a *= scale
        return np.r_[a[:,0], a[:,1], a[:,2], np.clip(wh[0]*scale,.2,1.8)]

    def unpack(z):
        return np.column_stack((z[:21], z[21:42], z[42:63]))

    def optimize(a, until):
        z0 = pack(a)
        cache_z = None
        cache_c = None
        cache_j = None
        last_record = 0

        def check():
            if time.monotonic() >= until:
                raise Deadline()

        def calculate(z):
            nonlocal cache_z, cache_c, cache_j
            check()
            if cache_z is not None and np.array_equal(cache_z,z):
                return
            x,y,r,w = z[:21],z[21:42],z[42:63],z[63]
            dx,dy = x[I]-x[J],y[I]-y[J]
            d = np.sqrt(dx*dx+dy*dy)
            safe = np.maximum(d,1e-15)
            c = np.concatenate((d-r[I]-r[J],x-r,w-x-r,y-r,2-w-y-r))
            jac = np.zeros((294,64))
            jac[P,I] = dx/safe
            jac[P,J] = -dx/safe
            jac[P,21+I] = dy/safe
            jac[P,21+J] = -dy/safe
            jac[P,42+I] = -1
            jac[P,42+J] = -1
            jac[210+K,K] = 1
            jac[210+K,42+K] = -1
            jac[231+K,K] = -1
            jac[231+K,42+K] = -1
            jac[231+K,63] = 1
            jac[252+K,21+K] = 1
            jac[252+K,42+K] = -1
            jac[273+K,21+K] = -1
            jac[273+K,42+K] = -1
            jac[273+K,63] = -1
            cache_z,cache_c,cache_j = z.copy(),c,jac

        def fun(z):
            check()
            return -float(np.sum(z[42:63]))

        def con(z):
            calculate(z)
            return cache_c

        def jac(z):
            calculate(z)
            return cache_j

        def callback(z):
            nonlocal last_record
            check()
            last_record += 1
            if last_record % 2 == 0:
                consider(unpack(z))

        try:
            result = minimize(fun,z0,jac=lambda z:grad,method='SLSQP',bounds=bounds,
                              constraints={'type':'ineq','fun':con,'jac':jac},
                              callback=callback,
                              options={'maxiter':180,'ftol':float(params['ftol']),'disp':False})
            consider(unpack(result.x))
        except Deadline:
            if cache_z is not None:
                consider(unpack(cache_z))
        except (ValueError, FloatingPointError):
            pass

    first_end = min(deadline, start + float(params['first_fraction'])*budget_s)
    if time.monotonic() < first_end:
        optimize(initial, first_end)
    attempt = 0
    while time.monotonic() < deadline - 0.035:
        attempt += 1
        a = np.asarray(best['circles'],dtype=float).copy()
        if best_score < 2.28 and attempt == 1:
            rows = [4,5,4,5,3]
            points = []
            for row,count in enumerate(rows):
                if time.monotonic() >= deadline:
                    break
                for col in range(count):
                    if time.monotonic() >= deadline:
                        break
                    points.append([2*col + (5-count),row*np.sqrt(3),.98])
            if len(points)==21:
                a=np.asarray(points,dtype=float)
        else:
            count = min(21, 4+2*attempt)
            chosen = rng.choice(21,size=count,replace=False)
            amplitude = float(params['perturbation'])*(1.0+.35*(attempt%3))
            a[chosen,:2] += rng.normal(0,amplitude,size=(count,2))
            a[chosen,2] *= rng.uniform(.87,1.03,size=count)
        now=time.monotonic()
        optimize(a, min(deadline,now+max(.12,min(.38*budget_s,deadline-now))))
    return best
