import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {
    'initial_fraction': (0.39, 0.2, 0.5),
    'symmetric_fraction': (0.19, 0.08, 0.3),
    'iterations': (180, 60, 300),
    'noise': (0.014, 0.001, 0.06),
    'ftol': (2e-12, 1e-14, 1e-8, 'log')
}
N = 21
I, J = np.triu_indices(N, 1)
K = np.arange(N)

class Deadline(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best = None
    best_a = None
    best_score = -np.inf

    def retain(a, original=None):
        nonlocal best, best_a, best_score
        if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return
        score = float(np.sum(a[:, 2]))
        if score <= best_score:
            return
        d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
        if np.any(d < a[I, 2]+a[J, 2]):
            return
        x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
        perimeter = (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
                     + max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)))
        if perimeter > 2:
            return
        best_score = score
        best_a = a.copy()
        best = original if original is not None else {'circles': a.tolist()}

    def pack(a):
        p = a[:, :2].copy()
        r = np.maximum(a[:, 2], 1e-9)
        lo = np.min(p-r[:, None], axis=0)
        side = np.max(p+r[:, None], axis=0)-lo
        scale = 2.0 / float(side.sum())
        p = (p-lo)*scale
        return np.r_[p[:, 0], p[:, 1], r*scale, side[0]*scale]

    def repair(z):
        if time.monotonic() > end + 0.025*budget_s or not np.isfinite(z).all():
            return
        p = np.column_stack((z[:N], z[N:2*N]))
        r = np.maximum(z[2*N:3*N], 1e-10).copy()
        d = np.sqrt((p[I, 0]-p[J, 0])**2+(p[I, 1]-p[J, 1])**2)
        ratio = min(1.0, float(np.min(d/(r[I]+r[J]))))
        if ratio <= 0:
            return
        r *= ratio*(1-2e-13)
        lo = np.min(p-r[:, None], axis=0)
        side = np.max(p+r[:, None], axis=0)-lo
        scale = (2-8e-13)/float(side.sum())
        retain(np.column_stack(((p-lo)*scale, r*scale)))

    a0 = None
    if init is not None:
        try:
            a0 = np.asarray(init['circles'], dtype=float)
            retain(a0, init)
            if a0.shape != (N, 3) or not np.isfinite(a0).all():
                a0 = None
        except (KeyError, TypeError, ValueError):
            a0 = None
    grid = np.array([(x, y, .49) for y in range(5) for x in range(5)
                     if not (x in (0,4) and y in (0,4))], dtype=float)
    repair(pack(grid))
    if a0 is None:
        a0 = best_a.copy()
    repair(pack(a0))

    A = np.zeros((4*N, 64))
    b = np.zeros(4*N)
    A[K, K] = 1; A[K, 2*N+K] = -1
    A[N+K, K] = -1; A[N+K, 2*N+K] = -1; A[N+K, -1] = 1
    A[2*N+K, N+K] = 1; A[2*N+K, 2*N+K] = -1
    A[3*N+K, N+K] = -1; A[3*N+K, 2*N+K] = -1; A[3*N+K, -1] = -1
    b[3*N:] = 2
    g = np.r_[np.zeros(2*N), -np.ones(N), 0.0]

    def optimize(z0, until, T=None, off=None, mirror=None):
        until = min(until, end)
        if time.monotonic() >= until:
            return z0
        def check():
            if time.monotonic() >= until:
                raise Deadline()
        if T is None:
            expand = lambda q: q
            q0 = z0.copy()
            AA, bb, gg = A, b, g
            ii, jj = I, J
            bounds = [(0,2)]*(2*N)+[(1e-8,.6)]*N+[(.1,1.9)]
        else:
            expand = lambda q: off+T@q
            q0 = np.linalg.lstsq(T,z0-off,rcond=None)[0]
            AA, bb, gg = A@T, A@off+b, T.T@g
            combined = np.column_stack((AA,bb))
            _, indices = np.unique(combined,axis=0,return_index=True)
            AA, bb = AA[indices], bb[indices]
            mi, mj = np.minimum(mirror[I],mirror[J]), np.maximum(mirror[I],mirror[J])
            keep = (I < mi) | ((I == mi) & (J <= mj))
            ii, jj = I[keep], J[keep]
            bounds = [(0,2)]*12+[(0,1)]*9+[(1e-8,.6)]*12+[(.1,1.9)]
        rr = np.arange(len(ii))
        last = [expand(q0).copy()]
        ticks = [0]
        def fun(q):
            check()
            return float(gg@q)
        def con(q):
            check()
            z = expand(q)
            dx, dy = z[ii]-z[jj], z[N+ii]-z[N+jj]
            return np.r_[np.sqrt(dx*dx+dy*dy+1e-30)-z[2*N+ii]-z[2*N+jj], AA@q+bb]
        def jac(q):
            check()
            z = expand(q)
            dx, dy = z[ii]-z[jj], z[N+ii]-z[N+jj]
            d = np.sqrt(dx*dx+dy*dy+1e-30)
            C = np.zeros((len(ii),64))
            C[rr,ii] = dx/d; C[rr,jj] = -dx/d
            C[rr,N+ii] = dy/d; C[rr,N+jj] = -dy/d
            C[rr,2*N+ii] = -1; C[rr,2*N+jj] = -1
            return np.vstack((C if T is None else C@T,AA))
        def callback(q):
            last[0] = expand(q).copy()
            check()
            ticks[0] += 1
            if ticks[0] % 10 == 0:
                repair(last[0])
        try:
            res = minimize(fun,q0,jac=lambda q:gg,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':con,'jac':jac},callback=callback,
                options={'maxiter':int(params['iterations']),'ftol':float(params['ftol']),'disp':False})
            last[0] = expand(res.x)
        except Deadline:
            pass
        repair(last[0])
        return last[0]

    optimize(pack(a0), start+float(params['initial_fraction'])*budget_s)

    def symmetric(a, transpose=False):
        a = a.copy()
        if transpose:
            a[:,[0,1]] = a[:,[1,0]]
        z = pack(a)
        order = np.argsort(z[N:2*N])
        lower, singles, upper = order[:9], order[9:12], order[12:]
        h = 2-z[-1]
        cost = ((z[lower,None]-z[upper][None,:])**2
                +(z[N+lower,None]+z[N+upper][None,:]-h)**2
                +2*(z[2*N+lower,None]-z[2*N+upper][None,:])**2)
        _, match = linear_sum_assignment(cost)
        upper = upper[match]
        T = np.zeros((64,34)); off = np.zeros(64)
        mirror = np.arange(N)
        for k,s in enumerate(singles):
            if time.monotonic() >= end:
                raise Deadline()
            T[s,k] = 1; off[N+s] = 1; T[N+s,33] = -.5
            T[2*N+s,21+k] = 1
        for k,(l,u) in enumerate(zip(lower,upper)):
            if time.monotonic() >= end:
                raise Deadline()
            mirror[l] = u; mirror[u] = l
            T[l,3+k] = T[u,3+k] = 1
            off[N+l] = off[N+u] = 1
            T[N+l,33] = T[N+u,33] = -.5
            T[N+l,12+k] = -1; T[N+u,12+k] = 1
            T[2*N+l,24+k] = T[2*N+u,24+k] = 1
        T[-1,33] = 1
        return z,T,off,mirror

    try:
        z,T,off,mirror = symmetric(best_a)
        zs = optimize(z,time.monotonic()+float(params['symmetric_fraction'])*budget_s,T,off,mirror)
        optimize(zs,min(end,time.monotonic()+.10*budget_s))
        attempt = 0
        while time.monotonic() < end:
            if attempt % 3 == 2:
                z,T,off,mirror = symmetric(best_a,True)
                optimize(z,min(end,time.monotonic()+.12*budget_s),T,off,mirror)
            else:
                z = pack(best_a)
                amp = float(params['noise'])*(1+.5*(attempt%3))
                selected = rng.choice(N, size=7, replace=False)
                z[selected] += rng.normal(0,amp,len(selected))
                z[N+selected] += rng.normal(0,amp,len(selected))
                optimize(z,min(end,time.monotonic()+.14*budget_s))
            attempt += 1
    except Deadline:
        pass
    return best
