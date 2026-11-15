import time
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize

PARAMS = {
    'trust': (0.18, 0.03, 0.4),
    'first_fraction': (0.29, 0.15, 0.35),
    'second_fraction': (0.23, 0.10, 0.30),
    'lp_share': (0.65, 0.3, 0.9),
    'iterations': (110, 20, 200)
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
K = np.arange(N)
P = np.arange(M)

class Deadline(Exception):
    pass

def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    r = a[:, 2]
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < r[I]+r[J]):
        return -np.inf
    x = list(map(Fraction, map(float, a[:, 0])))
    y = list(map(Fraction, map(float, a[:, 1])))
    q = list(map(Fraction, map(float, r)))
    width = max(u+v for u,v in zip(x,q))-min(u-v for u,v in zip(x,q))
    height = max(u+v for u,v in zip(y,q))-min(u-v for u,v in zip(y,q))
    return float(np.sum(r)) if width+height <= 2 else -np.inf

def encode(a):
    return {'coordinate_rows': a.T.tolist()}

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    end = started + 0.70*budget_s
    best = None
    best_score = -np.inf
    best_array = None
    raw = None
    if init is not None:
        try:
            raw = np.asarray(init['circles'] if 'circles' in init else np.asarray(init['coordinate_rows']).T, dtype=float)
            score = exact_score(raw)
            if score > best_score:
                best, best_score, best_array = init, score, raw.copy()
        except (KeyError, ValueError, TypeError, OverflowError):
            raw = None

    # Sixteen large circles and a symmetric cross of five hole circles.
    grid = np.zeros((21,3))
    grid[:16,0] = (np.arange(16)%4 + 0.5)*0.25
    grid[:16,1] = (np.arange(16)//4 + 0.5)*0.25
    grid[:16,2] = 0.125
    grid[16:,:2] = np.array([[0.25,0.5],[0.5,0.5],[0.75,0.5],[0.5,0.25],[0.5,0.75]])
    grid[16:,2] = (np.sqrt(2.0)-1.0)*0.125
    grid[:,2] *= 1-1e-12
    score = exact_score(grid)
    if score > best_score:
        best, best_score, best_array = encode(grid), score, grid.copy()
    if raw is None or raw.shape != (N,3) or not np.isfinite(raw).all():
        raw = grid.copy()

    objective = np.r_[np.zeros(42),-np.ones(21),0.0]
    B = np.zeros((84,64))
    B[K,K] = -1
    B[K,42+K] = 1
    B[21+K,K] = 1
    B[21+K,42+K] = 1
    B[21+K,-1] = -1
    B[42+K,21+K] = -1
    B[42+K,42+K] = 1
    B[63+K,21+K] = 1
    B[63+K,42+K] = 1
    B[63+K,-1] = 1
    bb = np.r_[np.zeros(63),np.full(21,2.0)]
    A = np.zeros((M+84,64))
    A[M:] = B
    A[P,42+I] = 1
    A[P,42+J] = 1
    rhs = np.r_[np.zeros(M),bb]
    local_end = end

    def guard():
        if time.monotonic() >= min(end,local_end):
            raise Deadline()

    def normalized(a):
        c = a.copy()
        c[:,2] = np.maximum(np.abs(c[:,2]),1e-8)
        low = np.min(c[:,:2]-c[:,2,None],axis=0)
        high = np.max(c[:,:2]+c[:,2,None],axis=0)
        sides = high-low
        scale = 2/max(float(sides.sum()),1e-10)
        c[:,:2] = (c[:,:2]-low)*scale
        c[:,2] *= scale
        return np.r_[c[:,0],c[:,1],c[:,2],sides[0]*scale]

    def consider(v):
        nonlocal best, best_score, best_array
        guard()
        c = np.column_stack((v[:21],v[21:42],np.maximum(v[42:63],1e-10)))
        d = np.sqrt((c[I,0]-c[J,0])**2+(c[I,1]-c[J,1])**2)
        factor = min(1.0,float(np.min(d/(c[I,2]+c[J,2]))))
        if not np.isfinite(factor) or factor <= 0:
            return
        c[:,2] *= factor*(1-3e-12)
        lo = np.min(c[:,:2]-c[:,2,None],axis=0)
        hi = np.max(c[:,:2]+c[:,2,None],axis=0)
        c[:,:2] -= (hi+lo)*0.5
        c *= (2/float((hi-lo).sum()))*(1-3e-12)
        if float(np.sum(c[:,2])) <= best_score:
            return
        sc = exact_score(c)
        if sc > best_score:
            best, best_score, best_array = encode(c), sc, c.copy()

    def con(v):
        guard()
        return np.r_[np.hypot(v[I]-v[J],v[21+I]-v[21+J])-v[42+I]-v[42+J],bb-B@v]

    def jac(v):
        guard()
        dx = v[I]-v[J]
        dy = v[21+I]-v[21+J]
        d = np.maximum(np.hypot(dx,dy),1e-16)
        C = np.zeros((M+84,64))
        C[P,I] = dx/d
        C[P,J] = -dx/d
        C[P,21+I] = dy/d
        C[P,21+J] = -dy/d
        C[P,42+I] = -1
        C[P,42+J] = -1
        C[M:] = -B
        return C

    def fun(v):
        guard()
        return float(objective@v)

    def polish(z):
        guard()
        result = minimize(fun,z,jac=lambda v:objective,method='SLSQP',
            bounds=[(None,None)]*42+[(1e-8,0.6)]*21+[(0.08,1.92)],
            constraints={'type':'ineq','fun':con,'jac':jac},callback=consider,
            options={'maxiter':int(params['iterations']),'ftol':3e-12,'disp':False})
        consider(result.x)
        return result.x

    def trajectory(a, lp_end):
        z = normalized(a)
        consider(z)
        previous = -np.inf
        stale = 0
        for iteration in range(int(params['iterations'])):
            guard()
            if time.monotonic() >= lp_end:
                break
            dx = z[I]-z[J]
            dy = z[21+I]-z[21+J]
            d = np.maximum(np.hypot(dx,dy),1e-16)
            A[P,I] = -dx/d
            A[P,J] = dx/d
            A[P,21+I] = -dy/d
            A[P,21+J] = dy/d
            trust = float(params['trust'])
            bounds = np.empty((64,2))
            bounds[:42,0] = z[:42]-trust
            bounds[:42,1] = z[:42]+trust
            bounds[42:63] = [1e-8,0.6]
            bounds[-1] = [0.08,1.92]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result = linprog(objective,A_ub=A,b_ub=rhs,bounds=bounds,method='highs',
                    options={'threads':1,'time_limit':max(0.001,lp_end-time.monotonic()),
                             'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
            guard()
            if not result.success:
                break
            z = result.x
            consider(z)
            value = float(z[42:63].sum())
            stale = stale+1 if value-previous < 2e-8 else 0
            previous = value
            if stale >= 2:
                break
        if time.monotonic() < min(local_end,end)-0.012:
            polish(z)

    # Protect strong warm starts; cold starts instead compete with a structured seed.
    strong = exact_score(raw) > 2.30
    first = raw if strong else grid
    first_end = min(end,started+float(params['first_fraction'])*budget_s)
    local_end = first_end
    try:
        trajectory(first, started+float(params['first_fraction'])*float(params['lp_share'])*budget_s)
    except Deadline:
        pass

    if time.monotonic() < end-0.04:
        branch_start = time.monotonic()
        local_end = min(end,branch_start+float(params['second_fraction'])*budget_s)
        if strong:
            second = best_array.copy()
            rng = np.random.default_rng(seed)
            second[:,:2] += rng.normal(0,0.007,(21,2))
        else:
            second = raw
        try:
            trajectory(second,branch_start+(local_end-branch_start)*float(params['lp_share']))
        except Deadline:
            pass

    local_end = end
    if best_array is not None and time.monotonic() < end-0.025:
        try:
            polish(normalized(best_array))
        except Deadline:
            pass
    return best
