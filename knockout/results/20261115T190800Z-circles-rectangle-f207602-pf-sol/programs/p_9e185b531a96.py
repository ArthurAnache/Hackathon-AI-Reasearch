import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment
from scipy.linalg import lstsq

PARAMS = {
    'perturbation': (0.028, 0.002, 0.10, 'log'),
    'iterations': (65, 20, 160),
    'restart_seconds': (0.145, 0.05, 0.4),
    'safety': (2.5e-15, 8e-16, 1e-12, 'log')
}
I, J = np.triu_indices(21, 1)

class StopSearch(Exception):
    pass

def valid(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return False
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return False
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    return (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r)) +
            max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))) <= 2

def repair(a, eps):
    a = np.asarray(a, float).copy()
    if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
        return None
    d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
    ratio = min(1., float(np.min(d/(a[I,2]+a[J,2]))))
    if ratio <= 0:
        return None
    a[:,2] *= ratio*(1-eps)
    lo = np.min(a[:,:2]-a[:,2,None], axis=0)
    hi = np.max(a[:,:2]+a[:,2,None], axis=0)
    scale = (2-4*eps)/np.sum(hi-lo)
    a[:,:2] = (a[:,:2]-(hi+lo)/2)*scale
    a[:,2] *= scale
    return a if valid(a) else None

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    eps = float(params['safety'])
    best = init
    bestscore = -np.inf
    original = None
    if init is not None:
        try:
            original = (np.asarray(init['circles'], float).copy() if 'circles' in init else
                        np.asarray(init['state'], float).reshape(3,21).T.copy())
            if valid(original):
                bestscore = float(np.sum(original[:,2]))
            else:
                best = None
        except (ValueError, KeyError, TypeError):
            original = None
            best = None

    def offer(a):
        nonlocal best, bestscore
        if time.monotonic() >= deadline:
            return
        b = repair(a, eps)
        if b is not None:
            score = float(np.sum(b[:,2]))
            if score > bestscore:
                bestscore = score
                best = {'state': b.T.reshape(-1).tolist()}

    if original is None:
        original = np.array([[2*(k+0.5*(row%2)), np.sqrt(3)*row, 1.]
                             for row in range(6) for k in range(3+(row%2))], float)
        original = repair(original, eps)
    if original is not None:
        offer(original)
    if original is None or time.monotonic() >= deadline-.03:
        if best is not None:
            return best
        return {'circles': [[.2*(i%3), .2*(i//3), .09] for i in range(21)]}

    # A 34-dimensional quotient: three axial circles and nine mirror pairs.
    ids = np.r_[np.arange(12), np.arange(3,12)]
    signs = np.r_[np.ones(12), -np.ones(9)]
    X = np.zeros((21,34)); Y = X.copy(); R = X.copy()
    X[np.arange(21), ids] = 1
    mask = ids >= 3
    Y[np.arange(21)[mask], 12+ids[mask]-3] = signs[mask]
    R[np.arange(21),21+ids] = 1
    pairs = [(i,j) for i in range(12) for j in range(i+1,12)]
    pairs += [(i,12+j-3) for i in range(3,12) for j in range(i,12)]
    pi,pj = np.array(pairs).T
    DX, DY, RS = X[pi]-X[pj], Y[pi]-Y[pj], R[pi]+R[pj]
    w = np.zeros(34); w[-1] = 1
    B = np.concatenate((X[:12]-R[:12], w-X[:12]-R[:12],
                        -.5*w-Y[:12]-R[:12], -.5*w+Y[:12]-R[:12]))
    off = np.r_[np.zeros(24), np.ones(24)]
    obj = -R.sum(axis=0)
    bounds = [(None,None)]*12 + [(0.,1.)]*9 + [(1e-6,.5)]*12 + [(.2,1.8)]

    def unpack(q):
        return np.column_stack((X@q, Y@q, R@q))

    choices = []
    for swap in (False, True):
        if time.monotonic() >= deadline-.03:
            return best
        a = original.copy()
        if swap:
            a[:,:2] = a[:,[1,0]]
        lo = np.min(a[:,:2]-a[:,2,None], axis=0)
        hi = np.max(a[:,:2]+a[:,2,None], axis=0)
        a[:,0] -= lo[0]
        a[:,1] -= (lo[1]+hi[1])/2
        reflected = a.copy(); reflected[:,1] *= -1
        cost = np.sum((a[:,None,:]-reflected[None,:,:])**2, axis=2)
        rr,cc = linear_sum_assignment(cost)
        choices.append((float(cost[rr,cc].sum()),a,hi[0]-lo[0]))
    _,a,width = min(choices,key=lambda t:t[0])
    singles = np.argsort(np.abs(a[:,1]))[:3]
    remaining = np.setdiff1d(np.arange(21), singles)
    ordered = remaining[np.argsort(a[remaining,1])]
    neg,pos = ordered[:9],ordered[9:]
    cost = ((a[pos,None,0]-a[None,neg,0])**2 +
            (a[pos,None,1]+a[None,neg,1])**2 +
            (a[pos,None,2]-a[None,neg,2])**2)
    _,cols = linear_sum_assignment(cost)
    neg = neg[cols]
    base = np.r_[a[singles,0], .5*(a[pos,0]+a[neg,0]),
                 .5*(a[pos,1]-a[neg,1]), a[singles,2],
                 .5*(a[pos,2]+a[neg,2]), width]

    def equations(q):
        dx,dy,rs = DX@q,DY@q,RS@q
        c = np.r_[dx*dx+dy*dy-rs*rs, B@q+off]
        jac = np.concatenate((2*(dx[:,None]*DX+dy[:,None]*DY-rs[:,None]*RS), B))
        return c,jac

    def polish(q, stop):
        if time.monotonic() >= stop:
            return q
        c,jac = equations(q)
        active = np.flatnonzero(np.abs(c) < 3e-7)
        if len(active) < 34:
            return q
        z = q.copy()
        for _ in range(5):
            if time.monotonic() >= stop:
                break
            c,jac = equations(z)
            try:
                step,_,rank,_ = lstsq(jac[active], -c[active], lapack_driver='gelsy')
            except Exception:
                break
            if rank < 34 or np.linalg.norm(step) > .005:
                break
            z += step
            if np.max(np.abs(step)) < 3e-16:
                break
        offer(unpack(z))
        return z

    def solve(q, stop):
        cache = [None,None,None]
        accepted = [q.copy()]
        def calc(z):
            if time.monotonic() >= stop:
                raise StopSearch
            if cache[0] is None or not np.array_equal(cache[0],z):
                c,j = equations(z)
                cache[:] = [z.copy(),c,j]
            return cache[1],cache[2]
        def objective(z):
            if time.monotonic() >= stop:
                raise StopSearch
            return float(obj@z),obj
        def callback(z):
            if time.monotonic() >= stop:
                raise StopSearch
            accepted[0] = z.copy()
        try:
            res = minimize(objective,q,jac=True,method='SLSQP',bounds=bounds,
                           constraints={'type':'ineq','fun':lambda z:calc(z)[0],
                                        'jac':lambda z:calc(z)[1]},
                           callback=callback,options={'maxiter':int(params['iterations']),
                                                       'ftol':2e-14,'disp':False})
            q = res.x
        except StopSearch:
            q = accepted[0]
        offer(unpack(q))
        return q

    base = solve(base, min(deadline-.04, time.monotonic()+.24))
    base = polish(base, min(deadline-.02,time.monotonic()+.025))
    incumbent_q = base.copy()
    incumbent_value = float(obj@base)
    trial = 0
    while time.monotonic() < deadline-.055:
        trial += 1
        q = incumbent_q.copy()
        amp = float(params['perturbation'])*[.35,1.,1.8,.7,2.5][(trial-1)%5]
        if trial%3 == 0:
            q[:21] += rng.normal(0,amp*.5,21)
            q[21:33] *= np.exp(rng.normal(0,amp,12))
        else:
            centers = np.column_stack((q[:12],np.r_[np.zeros(3),q[12:21]]))
            anchor = int(rng.integers(12))
            distances = np.sum((centers-centers[anchor])**2,axis=1)
            group = np.argsort(distances)[:int(rng.integers(3,6))]
            dx,dy = rng.normal(0,amp,2)
            q[group] += dx + rng.normal(0,amp*.22,len(group))
            paired = group[group>=3]
            q[12+paired-3] += dy + rng.normal(0,amp*.22,len(paired))
            q[21+group] *= np.exp(rng.normal(0,amp*1.8,len(group)))
        q[12:21] = np.clip(q[12:21],1e-5,.95)
        q[21:33] = np.clip(q[21:33],1e-5,.45)
        q[-1] = np.clip(q[-1]+rng.normal(0,amp*.4),.3,1.7)
        stop = min(deadline-.025,time.monotonic()+float(params['restart_seconds']))
        q = solve(q, stop)
        if time.monotonic() < deadline-.02:
            c,_ = equations(q)
            if np.min(c) > -2e-7:
                q = polish(q,min(deadline-.015,time.monotonic()+.012))
                if float(obj@q) < incumbent_value-1e-12:
                    incumbent_q = q.copy()
                    incumbent_value = float(obj@q)
    return best
