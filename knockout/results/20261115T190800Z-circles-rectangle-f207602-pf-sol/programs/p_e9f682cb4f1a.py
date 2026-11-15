import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment
from scipy.linalg import lstsq, svd

PARAMS = {
    'kick': (0.065, 0.015, 0.18),
    'release': (6, 3, 12),
    'solve_seconds': (0.17, 0.06, 0.4),
    'safety': (3e-15, 1e-15, 1e-11, 'log'),
    'pool_gap': (0.003, 0.000001, 0.02)
}
I, J = np.triu_indices(21, 1)

class SearchStop(Exception):
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

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    eps = float(params['safety'])
    best = init
    bestscore = -np.inf
    original = None
    if init is not None:
        try:
            original = (np.asarray(init['circles'], dtype=float) if 'circles' in init
                        else np.asarray(init['state'], dtype=float).reshape(3,21).T.copy())
            if valid(original):
                bestscore = float(np.sum(original[:,2]))
            else:
                original = None
                best = None
        except (ValueError, KeyError, TypeError):
            original = None
            best = None

    def offer(a):
        nonlocal best, bestscore
        if time.monotonic() >= end:
            return -np.inf
        a = np.asarray(a, dtype=float).copy()
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return -np.inf
        d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
        shrink = min(1.0, float(np.min(d/(a[I,2]+a[J,2]))))
        if shrink <= 0:
            return -np.inf
        a[:,2] *= shrink*(1-eps)
        lo = np.min(a[:,:2]-a[:,2,None], axis=0)
        hi = np.max(a[:,:2]+a[:,2,None], axis=0)
        scale = (2-8*eps)/float(np.sum(hi-lo))
        a[:,:2] = (a[:,:2]-(lo+hi)*0.5)*scale
        a[:,2] *= scale
        score = float(np.sum(a[:,2]))
        if score > bestscore and valid(a):
            bestscore = score
            best = {'state': a.T.reshape(-1).tolist()}
        return score

    if original is None:
        rows = []
        for k, count in enumerate((3,4,3,4,3,4)):
            if time.monotonic() >= end:
                break
            for j in range(count):
                if time.monotonic() >= end:
                    break
                rows.append([2*j+(1 if count == 3 else 0), np.sqrt(3)*k, 1.0])
        if len(rows) == 21:
            original = np.asarray(rows)
            offer(original)
        else:
            return {'circles': [[0.2*(k%3), 0.2*(k//3), 0.09] for k in range(21)]}

    # Identify the reflection axis and assign three singles and nine pairs.
    choices = []
    for swap in (False, True):
        if time.monotonic() >= end-0.02:
            return best
        a = original.copy()
        if swap:
            a[:,:2] = a[:,[1,0]]
        lo = np.min(a[:,:2]-a[:,2,None], axis=0)
        hi = np.max(a[:,:2]+a[:,2,None], axis=0)
        a[:,:2] -= (lo+hi)*0.5
        reflected = a.copy()
        reflected[:,1] *= -1
        cost = np.sum((a[:,None,:]-reflected[None,:,:])**2,axis=2)
        rr, cc = linear_sum_assignment(cost)
        choices.append((float(cost[rr,cc].sum()), a, float(hi[0]-lo[0])))
    _, a, width = min(choices, key=lambda t:t[0])
    singles = np.argsort(np.abs(a[:,1]))[:3]
    remaining = np.setdiff1d(np.arange(21), singles)
    order = remaining[np.argsort(a[remaining,1])]
    neg, pos = order[:9], order[9:]
    cost = ((a[pos,None,0]-a[None,neg,0])**2 +
            (a[pos,None,1]+a[None,neg,1])**2 +
            (a[pos,None,2]-a[None,neg,2])**2)
    _, cc = linear_sum_assignment(cost)
    neg = neg[cc]
    z0 = np.r_[a[singles,0], .5*(a[pos,0]+a[neg,0]),
               .5*(a[pos,1]-a[neg,1]), a[singles,2],
               .5*(a[pos,2]+a[neg,2]), width]

    n = 34
    ids = np.r_[np.arange(12), np.arange(3,12)]
    signs = np.r_[np.ones(12), -np.ones(9)]
    X = np.zeros((21,n)); Y = np.zeros((21,n)); R = np.zeros((21,n))
    X[np.arange(21),ids] = 1
    mask = ids >= 3
    Y[np.flatnonzero(mask),12+ids[mask]-3] = signs[mask]
    R[np.arange(21),21+ids] = 1
    pairs = [(i,j) for i in range(12) for j in range(i+1,12)]
    pairs += [(i,12+j-3) for i in range(3,12) for j in range(i,12)]
    pi,pj = np.asarray(pairs).T
    DX = X[pi]-X[pj]; DY = Y[pi]-Y[pj]; RS = R[pi]+R[pj]
    W = np.zeros(n); W[-1] = 1
    B = np.concatenate((.5*W+X[:12]-R[:12], .5*W-X[:12]-R[:12],
                        -.5*W+Y[:12]-R[:12], -.5*W-Y[:12]-R[:12]))
    offset = np.r_[np.zeros(24), np.ones(24)]
    objective = -R.sum(axis=0)
    npairs = len(pi)
    bounds = [(None,None)]*21 + [(1e-6,.48)]*12 + [(.3,1.7)]

    def unpack(z):
        return np.column_stack((X@z,Y@z,R@z))

    def calc(z):
        dx = DX@z; dy = DY@z; rs = RS@z
        # Fixed scaling keeps contact and wall derivatives comparable.
        values = np.r_[(dx*dx+dy*dy-rs*rs)*2.3, B@z+offset]
        jac = np.concatenate((4.6*(dx[:,None]*DX+dy[:,None]*DY-rs[:,None]*RS),B))
        return values,jac

    pool = []
    def remember(z, score):
        if not np.isfinite(score) or score < bestscore-float(params['pool_gap']):
            return
        c,_ = calc(z)
        if np.min(c) < -2e-6:
            return
        signature = tuple(np.flatnonzero(c[:npairs] < 1e-6))
        for k, entry in enumerate(pool):
            if time.monotonic() >= end:
                return
            if entry[2] == signature:
                if score > entry[0]:
                    pool[k] = (score,z.copy(),signature)
                return
        pool.append((score,z.copy(),signature))
        pool.sort(key=lambda e:-e[0])
        del pool[6:]

    def optimize(z, stop, iterations=110):
        last = [z.copy()]
        cache = [None,None,None]
        def evaluate(v):
            if time.monotonic() >= stop:
                raise SearchStop()
            if cache[0] is None or not np.array_equal(v,cache[0]):
                c,j = calc(v)
                cache[:] = [v.copy(),c,j]
                last[0] = v.copy()
            return cache[1],cache[2]
        def fun(v):
            if time.monotonic() >= stop:
                raise SearchStop()
            return float(objective@v),objective
        try:
            result = minimize(fun,z,jac=True,bounds=bounds,method='SLSQP',
                constraints={'type':'ineq','fun':lambda v:evaluate(v)[0],
                             'jac':lambda v:evaluate(v)[1]},
                options={'maxiter':iterations,'ftol':3e-13,'disp':False})
            z = result.x
        except SearchStop:
            z = last[0]
        if time.monotonic() < end-.006:
            score = offer(unpack(z))
            remember(z,score)
        return z

    base = optimize(z0,min(end-.02,time.monotonic()+.15),90)
    # Refine a square active system without spending nonlinear iterations.
    for _ in range(3):
        if time.monotonic() >= end-.03:
            return best
        c,jac = calc(base)
        active = np.flatnonzero(np.abs(c)<2e-7)
        if len(active)<n:
            break
        delta,_,rank,_ = lstsq(jac[active],-c[active],lapack_driver='gelsy',check_finite=False)
        if rank<n or np.max(np.abs(delta))>1e-4:
            break
        base += delta
        remember(base,offer(unpack(base)))

    attempt = 0
    while time.monotonic() < end-.045:
        now = time.monotonic()
        if pool:
            # Alternate the best basin with the nearby, topologically distinct ones.
            which = 0 if attempt%2 == 0 else int(rng.integers(len(pool)))
            anchor = pool[which][1].copy()
        else:
            anchor = base.copy()
        c,jac = calc(anchor)
        active = np.flatnonzero(np.abs(c)<3e-6)
        if len(active):
            # Remove duplicate active rows before computing release directions.
            norm = np.maximum(np.linalg.norm(jac[active],axis=1),1e-12)
            _, unique = np.unique(np.round(jac[active]/norm[:,None],9),axis=0,return_index=True)
            active = active[np.sort(unique)]
        release = min(len(active), int(params['release'])+(attempt%3)*2)
        trial = anchor.copy()
        kept = np.empty(0,dtype=int)
        if len(active)>release and release>0:
            multipliers = lstsq(jac[active].T,objective,lapack_driver='gelsy',check_finite=False)[0]
            positive = np.maximum(multipliers,0)
            typical = max(float(np.median(positive)),.02)
            weights = 1/(positive+.2*typical)
            weights *= np.where(active<npairs,1.0,.35)
            weights /= weights.sum()
            removed = rng.choice(len(active),release,replace=False,p=weights)
            keepmask = np.ones(len(active),dtype=bool); keepmask[removed] = False
            kept = active[keepmask]
            _, singular, vh = svd(jac[kept],full_matrices=True,check_finite=False)
            rank = int(np.sum(singular>1e-9))
            null = vh[rank:].T
            if null.shape[1]:
                direction = null@rng.normal(size=null.shape[1])
                movement = np.sqrt(np.mean((X@direction)**2+(Y@direction)**2)+.5*np.mean((R@direction)**2))
                amplitude = float(params['kick'])*(0.7+.45*(attempt%4))
                trial += direction*(amplitude/max(movement,1e-12))
            else:
                trial[:21] += rng.normal(0,float(params['kick']),21)
        else:
            trial[:21] += rng.normal(0,float(params['kick']),21)
        # A few manifold corrections preserve the unreleased contact framework.
        for _ in range(2):
            if time.monotonic() >= end-.035:
                return best
            if not len(kept):
                break
            v,j = calc(trial)
            correction = lstsq(j[kept],-v[kept],lapack_driver='gelsy',check_finite=False)[0]
            if np.max(np.abs(correction))>.12:
                break
            trial += correction
        trial[12:21] = np.abs(trial[12:21])
        trial[21:33] = np.clip(trial[21:33],.025,.25)
        trial[-1] = np.clip(trial[-1],.45,1.55)
        stop = min(end-.012,time.monotonic()+float(params['solve_seconds']))
        optimize(trial,stop)
        attempt += 1
    return best
