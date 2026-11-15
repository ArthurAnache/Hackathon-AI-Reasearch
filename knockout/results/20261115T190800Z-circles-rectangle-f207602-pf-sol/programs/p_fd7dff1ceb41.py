import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {
    'search_fraction': (0.70, 0.50, 0.75),
    'first_fraction': (0.34, 0.15, 0.45),
    'symmetric_fraction': (0.17, 0.05, 0.25),
    'noise': (0.009, 0.001, 0.04),
    'iterations': (160, 30, 300),
}
N = 21
I, J = np.triu_indices(N, 1)
K = np.arange(N)
RR = np.arange(len(I))

class Deadline(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + float(params['search_fraction']) * budget_s
    rng = np.random.default_rng(seed)
    best = None
    best_score = -np.inf
    best_array = None

    def exact_valid(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return False
        d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
        if np.any(d < a[I, 2]+a[J, 2]):
            return False
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        return (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
                +max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))) <= 2

    def retain(a, original=None):
        nonlocal best, best_score, best_array
        if exact_valid(a):
            score = float(np.sum(a[:, 2]))
            if score > best_score:
                best_score = score
                best_array = a.copy()
                best = original if original is not None else {'circles': a.tolist()}

    def repair(z):
        if not np.isfinite(z).all():
            return
        p = np.column_stack((z[:N], z[N:2*N])).copy()
        r = np.maximum(z[2*N:3*N], 1e-12).copy()
        d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
        f = min(1.0, float(np.min(d/(r[I]+r[J]))))
        if f <= 0:
            return
        r *= f*(1-3e-13)
        lo = np.min(p-r[:,None], axis=0)
        hi = np.max(p+r[:,None], axis=0)
        perimeter = float(np.sum(hi-lo))
        if perimeter <= 0:
            return
        scale = (2-8e-13)/perimeter
        p = (p-lo)*scale
        r *= scale*(1-3e-13)
        retain(np.column_stack((p,r)))

    def pack(a):
        p = a[:,:2].copy()
        r = a[:,2].copy()
        lo = np.min(p-r[:,None],axis=0)
        hi = np.max(p+r[:,None],axis=0)
        scale = 2/float(np.sum(hi-lo))
        p = (p-lo)*scale
        return np.r_[p[:,0],p[:,1],r*scale,(hi[0]-lo[0])*scale]

    a0 = None
    if init is not None:
        try:
            a0 = np.asarray(init['circles'], dtype=float)
            retain(a0, init)
            if a0.shape != (N,3) or not np.isfinite(a0).all():
                a0 = None
        except (ValueError, KeyError, TypeError):
            a0 = None
    xx, yy = np.meshgrid(np.arange(5),np.arange(5))
    pp = np.column_stack((xx.ravel(),yy.ravel()))[:21]
    repair(np.r_[pp[:,0],pp[:,1],np.full(N,0.49),1.0])
    if a0 is None:
        a0 = best_array.copy()
    else:
        repair(pack(a0))

    # Linear box constraints include radius positivity and a nondegenerate box.
    A = np.zeros((4*N+N+2,3*N+1))
    b = np.zeros(len(A))
    A[K,K] = 1; A[K,2*N+K] = -1
    A[N+K,K] = -1; A[N+K,2*N+K] = -1; A[N+K,-1] = 1
    A[2*N+K,N+K] = 1; A[2*N+K,2*N+K] = -1
    A[3*N+K,N+K] = -1; A[3*N+K,2*N+K] = -1; A[3*N+K,-1] = -1
    b[3*N:4*N] = 2
    A[4*N+K,2*N+K] = 1
    b[4*N:5*N] = -1e-9
    A[-2,-1] = 1; b[-2] = -0.05
    A[-1,-1] = -1; b[-1] = 1.95
    g = np.r_[np.zeros(2*N),-np.ones(N),0.0]

    def optimize(z0, until, T=None, offset=None):
        until = min(until,end)
        if time.monotonic() >= until:
            return z0
        def check():
            if time.monotonic() >= until:
                raise Deadline()
        if T is None:
            def expand(q):
                return q
            q0 = z0.copy()
            AA = A
            bb = b
            gg = g
        else:
            def expand(q):
                return offset+T@q
            q0 = np.linalg.lstsq(T,z0-offset,rcond=None)[0]
            AA = A@T
            bb = A@offset+b
            gg = T.T@g
        last = [expand(q0).copy()]
        ticks = [0]
        def objective(q):
            check()
            return float(gg@q)
        def con(q):
            check()
            z = expand(q)
            dx = z[I]-z[J]
            dy = z[N+I]-z[N+J]
            return np.sqrt(dx*dx+dy*dy+1e-30)-z[2*N+I]-z[2*N+J]
        def jac(q):
            check()
            z = expand(q)
            dx = z[I]-z[J]; dy = z[N+I]-z[N+J]
            d = np.sqrt(dx*dx+dy*dy+1e-30)
            C = np.zeros((len(I),3*N+1))
            C[RR,I] = dx/d; C[RR,J] = -dx/d
            C[RR,N+I] = dy/d; C[RR,N+J] = -dy/d
            C[RR,2*N+I] = -1; C[RR,2*N+J] = -1
            return C if T is None else C@T
        def callback(q):
            check()
            last[0] = expand(q).copy()
            ticks[0] += 1
            if ticks[0] % 6 == 0:
                repair(last[0])
        try:
            result = minimize(objective,q0,jac=lambda q:gg,method='SLSQP',
                constraints=[{'type':'ineq','fun':lambda q:AA@q+bb,'jac':lambda q:AA},
                             {'type':'ineq','fun':con,'jac':jac}],
                callback=callback,
                options={'maxiter':int(params['iterations']),'ftol':3e-12,'disp':False})
            last[0] = expand(result.x)
        except Deadline:
            pass
        repair(last[0])
        return last[0]

    z = optimize(pack(a0),start+float(params['first_fraction'])*budget_s)

    def symmetric_seed(a, transpose=False):
        a = a.copy()
        if transpose:
            a[:,[0,1]] = a[:,[1,0]]
        z0 = pack(a)
        order = np.argsort(z0[N:2*N])
        lower, singles, upper = order[:9], order[9:12], order[12:]
        h = 2-z0[-1]
        cost = ((z0[lower,None]-z0[upper][None,:])**2
                +(z0[N+lower,None]+z0[N+upper][None,:]-h)**2
                +2*(z0[2*N+lower,None]-z0[2*N+upper][None,:])**2)
        _, match = linear_sum_assignment(cost)
        upper = upper[match]
        # 12 independent x coordinates, 9 offsets, 12 radii, box width.
        T = np.zeros((64,34))
        off = np.zeros(64)
        for k,s in enumerate(singles):
            if time.monotonic() >= end:
                raise Deadline()
            T[s,k] = 1
            off[N+s] = 1; T[N+s,33] = -0.5
            T[2*N+s,21+k] = 1
        for k,(l,u) in enumerate(zip(lower,upper)):
            if time.monotonic() >= end:
                raise Deadline()
            T[l,3+k] = T[u,3+k] = 1
            off[N+l] = off[N+u] = 1
            T[N+l,33] = T[N+u,33] = -0.5
            T[N+l,12+k] = -1; T[N+u,12+k] = 1
            T[2*N+l,24+k] = T[2*N+u,24+k] = 1
        T[-1,33] = 1
        return z0,T,off

    try:
        sym_end = min(end,time.monotonic()+float(params['symmetric_fraction'])*budget_s)
        z0,T,off = symmetric_seed(best_array)
        zs = optimize(z0,sym_end,T,off)
        if time.monotonic() < end:
            optimize(zs,min(end,time.monotonic()+0.13*budget_s))
        attempt = 0
        while time.monotonic() < end:
            base = pack(best_array)
            amp = float(params['noise'])*(1.0+0.45*(attempt%3))
            base[:2*N] += rng.normal(0,amp,2*N)
            base[2*N:3*N] *= rng.uniform(0.97,1.03,N)
            optimize(base,min(end,time.monotonic()+0.18*budget_s))
            attempt += 1
    except Deadline:
        pass
    return best
