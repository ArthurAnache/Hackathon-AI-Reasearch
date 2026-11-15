import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"iterations": (180, 60, 350), "restart_noise": (0.025, 0.0, 0.15)}


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -float('inf')
    iu, ju = np.triu_indices(21, 1)

    def verify(obj):
        try:
            a = np.asarray(obj['circles'], dtype=float)
            if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
                return -float('inf')
            d = a[iu, :2] - a[ju, :2]
            dist = np.sqrt(d[:, 0]**2 + d[:, 1]**2)
            if np.any(dist < a[iu, 2] + a[ju, 2]):
                return -float('inf')
            xs = list(map(Fraction, map(float, a[:, 0])))
            ys = list(map(Fraction, map(float, a[:, 1])))
            rs = list(map(Fraction, map(float, a[:, 2])))
            w = max(x+r for x, r in zip(xs, rs)) - min(x-r for x, r in zip(xs, rs))
            h = max(y+r for y, r in zip(ys, rs)) - min(y-r for y, r in zip(ys, rs))
            if w+h > 2:
                return -float('inf')
            return float(np.sum(a[:, 2]))
        except Exception:
            return -float('inf')

    if init is not None:
        if 'circles' not in init and 'orbits' in init:
            cs = []
            for x, y, r, m in init['orbits']:
                if time.monotonic() >= deadline:
                    break
                cs.append([x, y, r])
                if m == 2:
                    cs.append([x, -y, r])
            candidate = {'circles': cs}
            s = verify(candidate)
            if np.isfinite(s):
                best, best_score = candidate, s
        else:
            best_score = verify(init)

    # Orbit coordinates: twelve x's, nine positive y's, twelve radii,
    # and the rectangle's half-width. Its half-height is 1-halfwidth.
    mult = np.array([1.0]*3 + [2.0]*9)
    oi = np.concatenate((np.arange(12), np.arange(3, 12)))
    sg = np.concatenate((np.ones(12), -np.ones(9)))

    def expand(z):
        yy = np.concatenate((np.zeros(3), z[12:21]))
        return np.column_stack((z[oi], yy[oi]*sg, z[21+oi]))

    def repair(z):
        nonlocal best, best_score
        a = expand(z)
        if not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return
        # Uniformly shrink radii enough to remove every floating overlap.
        delta = a[iu, :2] - a[ju, :2]
        dd = np.sqrt(delta[:, 0]**2 + delta[:, 1]**2)
        ratio = np.min(dd/(a[iu, 2]+a[ju, 2]))
        if ratio <= 0:
            return
        a[:, 2] *= min(1.0, float(ratio)) * (1.0-3e-14)
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        perimeter = float(np.sum(hi-lo))
        a[:, :2] -= (hi+lo)/2
        a *= min(1.0, 2.0/perimeter)*(1.0-3e-14)
        if float(np.sum(a[:, 2])) <= best_score:
            return
        obj = {'circles': a.tolist()}
        score = verify(obj)
        if score > best_score:
            best, best_score = obj, score

    # Distinct nonoverlap constraints modulo horizontal reflection.
    aa, bb = np.triu_indices(12, 1)
    pp, qq = np.triu_indices(9, 1)
    A = np.concatenate((aa, pp+3))
    B = np.concatenate((bb, qq+3))
    signs = np.concatenate((np.ones(len(aa)), -np.ones(len(pp))))
    ncon = len(A)
    row = np.arange(ncon)
    Y = np.zeros((12, 34))
    Y[np.arange(3, 12), np.arange(12, 21)] = 1
    DX = np.zeros((ncon, 34))
    DX[row, A] = 1
    DX[row, B] = -1
    DY = Y[A] - signs[:, None]*Y[B]
    RR = np.zeros((ncon, 34))
    RR[row, 21+A] = 1
    RR[row, 21+B] = 1

    # Linear containment and reflected self-contact constraints.
    L = np.zeros((45, 34))
    offset = np.zeros(45)
    k = np.arange(12)
    L[k, k] = -1
    L[k, 21+k] = -1
    L[k, 33] = 1
    L[12+k, k] = 1
    L[12+k, 21+k] = -1
    L[12+k, 33] = 1
    L[24+k] = -Y
    L[24+k, 21+k] = -1
    L[24+k, 33] = -1
    offset[24:36] = 1
    q = np.arange(9)
    L[36+q, 12+q] = 1
    L[36+q, 24+q] = -1
    gradient = np.zeros(34)
    gradient[21:33] = -mult
    cached_z = None
    cached_c = None
    cached_j = None

    class StopSearch(Exception):
        pass

    def constraints(z):
        nonlocal cached_z, cached_c, cached_j
        if time.monotonic() >= deadline:
            raise StopSearch()
        if cached_z is None or not np.array_equal(z, cached_z):
            dx = z[A]-z[B]
            yy = np.concatenate((np.zeros(3), z[12:21]))
            dy = yy[A]-signs*yy[B]
            rr = z[21+A]+z[21+B]
            vals = dx*dx+dy*dy-rr*rr
            jac = 2*(dx[:, None]*DX + dy[:, None]*DY - rr[:, None]*RR)
            cached_c = np.concatenate((vals, L@z+offset))
            cached_j = np.vstack((jac, L))
            cached_z = z.copy()
        return cached_c

    def jacobian(z):
        constraints(z)
        return cached_j

    last = None
    count = 0
    def callback(z):
        nonlocal last, count
        last = z.copy()
        count += 1
        if time.monotonic() >= deadline:
            raise StopSearch()
        if count % 8 == 0:
            repair(z)

    bounds = [(-0.96, 0.96)]*12 + [(0.00001, 0.98)]*9 + [(0.00001, 0.5)]*12 + [(0.12, 0.88)]
    incumbent_z = None
    attempt = 0
    while time.monotonic() < deadline-0.035:
        if attempt == 0:
            # Three circles on the axis, then four and five in each half.
            z = np.zeros(34)
            r = 2/(18+4*np.sqrt(3))
            z[:3] = np.array([-2, 0, 2])*r
            z[3:7] = np.array([-3, -1, 1, 3])*r
            z[7:12] = np.array([-4, -2, 0, 2, 4])*r
            z[12:16] = np.sqrt(3)*r
            z[16:21] = 2*np.sqrt(3)*r
            z[21:33] = r*(1-1e-6)
            z[33] = 5*r
        elif attempt == 1:
            z = np.zeros(34)
            r = 2/(8+6*np.sqrt(3))
            z[:3] = np.array([-2, 0, 2])*r
            z[3:12] = np.tile(np.array([-2, 0, 2])*r, 3)
            z[3:6] += r
            z[9:12] += r
            z[12:21] = np.repeat(np.arange(1, 4)*np.sqrt(3)*r, 3)
            z[:12] -= 0.5*r
            z[21:33] = r*0.98
            z[33] = 3.5*r
        else:
            if incumbent_z is None:
                break
            z = incumbent_z.copy()
            noise = float(params['restart_noise'])
            z[:21] += rng.normal(0, noise, 21)
            z[12:21] = np.maximum(z[12:21], 0.005)
            z[21:33] *= rng.uniform(0.80, 0.98, 12)
        repair(z)
        last = z.copy()
        count = 0
        cached_z = None
        try:
            result = minimize(lambda v: float(gradient@v), z, jac=lambda v: gradient,
                              method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                              callback=callback,
                              options={'maxiter': int(params['iterations']), 'ftol': 2e-12, 'disp': False})
            last = result.x
        except StopSearch:
            pass
        except (ValueError, FloatingPointError):
            pass
        if last is not None:
            repair(last)
            if incumbent_z is None or float(gradient@last) < float(gradient@incumbent_z):
                incumbent_z = last.copy()
        attempt += 1
    if not np.isfinite(best_score):
        # Always-valid deterministic fallback.
        r = 0.08
        circles = [[(i % 7-3)*0.17, (i//7-1)*0.17, r] for i in range(21)]
        return {'circles': circles}
    return best
