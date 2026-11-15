import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"iterations": (160, 40, 400), "restarts": (8, 1, 20), "jitter": (0.025, 0.0, 0.1)}


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.69 * budget_s
    rng = np.random.default_rng(seed)
    ids = np.array([0, 1, 2] + [k for k in range(3, 12) for _ in range(2)])
    signs = np.array([0, 0, 0] + [s for _ in range(9) for s in (1, -1)])
    T = np.zeros((21, 3, 34))
    T[np.arange(21), 0, ids] = 1
    T[np.arange(21), 2, 21 + ids] = 1
    for a in range(3, 21):
        if time.monotonic() >= deadline:
            break
        T[a, 1, 12 + ids[a] - 3] = signs[a]
    ii, jj = np.triu_indices(21, 1)
    DX = T[ii, 0] - T[jj, 0]
    DY = T[ii, 1] - T[jj, 1]
    DR = T[ii, 2] + T[jj, 2]
    B = np.concatenate((T[:, 0] - T[:, 2], -T[:, 0] - T[:, 2], -T[:, 1] - T[:, 2], T[:, 1] - T[:, 2]))
    B[21:42, 33] += 1
    B[42:, 33] -= 0.5
    offset = np.r_[np.zeros(42), np.ones(42)]
    weights = np.array([1., 1., 1.] + [2.] * 9)
    grad = np.zeros(34)
    grad[21:33] = -weights

    def exact_score(obj):
        try:
            c = np.asarray(obj['circles'], dtype=float)
            if c.shape != (21, 3) or not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
                return -float('inf')
            dd = c[ii, :2] - c[jj, :2]
            if np.any(np.sqrt(dd[:, 0] ** 2 + dd[:, 1] ** 2) < c[ii, 2] + c[jj, 2]):
                return -float('inf')
            xs = [Fraction(float(v)) for v in c[:, 0]]
            ys = [Fraction(float(v)) for v in c[:, 1]]
            rs = [Fraction(float(v)) for v in c[:, 2]]
            perimeter = max(x+r for x,r in zip(xs,rs))-min(x-r for x,r in zip(xs,rs))
            perimeter += max(y+r for y,r in zip(ys,rs))-min(y-r for y,r in zip(ys,rs))
            if perimeter > 2:
                return -float('inf')
            return float(np.sum(c[:, 2]))
        except Exception:
            return -float('inf')

    best = init
    best_score = exact_score(init) if init is not None and 'circles' in init else -float('inf')
    if init is not None and 'orbits' in init:
        cs = []
        for x,y,r,m in init['orbits']:
            if time.monotonic() >= deadline:
                break
            cs.append([x,y,r])
            if m == 2:
                cs.append([x,-y,r])
        ob = {'circles': cs}
        sc = exact_score(ob)
        if sc > best_score:
            best, best_score = init, sc

    def repair(c):
        c = np.array(c, dtype=float, copy=True)
        if not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return None
        d = c[ii, :2] - c[jj, :2]
        distances = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
        ratio = np.min(distances / (c[ii, 2] + c[jj, 2]))
        c[:, 2] *= min(1., ratio) * (1. - 2e-13)
        w = np.max(c[:, 0]+c[:, 2])-np.min(c[:, 0]-c[:, 2])
        h = np.max(c[:, 1]+c[:, 2])-np.min(c[:, 1]-c[:, 2])
        c *= (2. / (w+h)) * (1.-2e-13)
        return {'circles': c.tolist()}

    # A guaranteed valid fallback is available even for malformed warm starts.
    fallback = np.array([[0.1 + 0.3*(k%3), -0.75 + 0.25*(k//3), 0.12] for k in range(21)])
    fb = repair(fallback)
    fs = exact_score(fb)
    if fs > best_score:
        best, best_score = fb, fs

    cache_z = None
    cache_v = None
    cache_j = None
    def calculate(z):
        nonlocal cache_z, cache_v, cache_j
        if cache_z is not None and np.array_equal(z, cache_z):
            return
        dx, dy, rr = DX @ z, DY @ z, DR @ z
        cache_v = np.r_[dx*dx + dy*dy - rr*rr, B @ z + offset]
        cache_j = np.concatenate((2*(dx[:,None]*DX + dy[:,None]*DY - rr[:,None]*DR), B))
        cache_z = z.copy()

    def fun(z):
        return float(grad @ z)
    def con(z):
        calculate(z)
        return cache_v
    def jac(z):
        calculate(z)
        return cache_j

    promising = []
    class Stop(Exception):
        pass
    def callback(z):
        if time.monotonic() >= deadline:
            raise Stop()
        calculate(z)
        if np.min(cache_v) >= -2e-7:
            promising.append((float(-grad @ z), z.copy()))
            if len(promising) > 12:
                promising.sort(key=lambda a: a[0], reverse=True)
                del promising[8:]

    bounds = [(0.001, 1.999)]*12 + [(0.00001, 0.999)]*9 + [(0.000001, 0.6)]*12 + [(0.25,1.75)]
    patterns = [(4,5), (5,4), (3,3,3), (6,3), (3,6), (4,5), (5,4), (3,3,3)]
    for trial in range(int(params['restarts'])):
        if time.monotonic() >= deadline:
            break
        counts = patterns[trial % len(patterns)]
        width = 1.02 if len(counts)==2 else 0.66
        height = 2-width
        z = np.zeros(34)
        z[33] = width
        z[:3] = np.linspace(width/6, 5*width/6, 3)
        z[21:24] = min(width/6, height/(4*len(counts)+2))
        at = 3
        for row, count in enumerate(counts):
            if time.monotonic() >= deadline:
                break
            xs = (np.arange(count)+0.5)*width/count
            for k in range(count):
                if time.monotonic() >= deadline:
                    break
                z[at] = xs[k]
                z[12+at-3] = (row+1)*height/(2*len(counts)+1)
                z[21+at] = min(width/(2*count),height/(4*len(counts)+2))*0.9
                at += 1
        if trial >= 4:
            z[:12] += rng.normal(0, float(params['jitter']), 12)
            z[12:21] += rng.normal(0, float(params['jitter']), 9)
        z = np.clip(z, np.array(bounds)[:,0], np.array(bounds)[:,1])
        try:
            result = minimize(fun, z, jac=lambda z: grad, bounds=bounds,
                constraints={'type':'ineq','fun':con,'jac':jac}, method='SLSQP',
                callback=callback, options={'maxiter':int(params['iterations']), 'ftol':2e-12, 'disp':False})
            promising.append((-float(grad @ result.x), result.x.copy()))
        except Stop:
            break
        except Exception:
            pass
    promising.sort(key=lambda a: a[0], reverse=True)
    for _, z in promising:
        if time.monotonic() > start + 0.91*budget_s:
            break
        candidate = repair(np.einsum('ijk,k->ij', T, z))
        score = exact_score(candidate)
        if score > best_score:
            best, best_score = candidate, score
    return best
