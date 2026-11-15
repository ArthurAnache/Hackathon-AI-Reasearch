import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {
    "release_size": (3, 3, 6),
    "step": (0.065, 0.005, 0.2),
    "attempts": (8, 1, 30),
    "maxiter": (85, 20, 200),
}

class StopSearch(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def perimeter(a):
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        return (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
                +max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)))

    def score(a):
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return -np.inf
        d = np.sqrt((a[ii,0]-a[jj,0])**2 + (a[ii,1]-a[jj,1])**2)
        if np.any(d < a[ii,2]+a[jj,2]) or perimeter(a) > 2:
            return -np.inf
        return float(np.sum(a[:,2]))

    if init is None:
        a = np.array([[(k % 7)*0.24, (k//7)*0.24, 0.02] for k in range(21)])
        init = {"circles": a.tolist()}
    raw = np.asarray(init['circles'], dtype=float)
    best = init
    best_score = score(raw)

    def consider(a):
        nonlocal best, best_score
        check()
        a = a.copy()
        if not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        d = np.sqrt((a[ii,0]-a[jj,0])**2 + (a[ii,1]-a[jj,1])**2)
        factor = min(1.0, float(np.min(d/(a[ii,2]+a[jj,2]))))
        a[:,2] *= factor * (1.0-4e-15)
        hp = perimeter(a)
        if hp > 2:
            a *= (2.0/float(hp))*(1.0-4e-15)
        value = score(a)
        if value > best_score:
            best_score = value
            best = {'circles': a.tolist()}

    try:
        # Choose the reflection axis from the supplied configuration.
        options = []
        for swap in (False, True):
            check()
            a = raw.copy()
            if swap:
                a[:, :2] = a[:, [1,0]]
            lo = np.min(a[:,:2]-a[:,2,None], axis=0)
            hi = np.max(a[:,:2]+a[:,2,None], axis=0)
            a[:,:2] -= (lo+hi)/2
            axis = np.argsort(np.abs(a[:,1]))[:3]
            rest = np.setdiff1d(np.arange(21), axis)
            order = rest[np.argsort(a[rest,1])]
            low, high = order[:9], order[9:]
            target = a[low].copy()
            target[:,1] *= -1
            cost = np.sum((a[high,None,:]-target[None,:,:])**2, axis=2)
            rows, cols = linear_sum_assignment(cost)
            low = low[cols]
            high = high[rows]
            err = float(cost[rows,cols].sum() + (a[axis,1]**2).sum())
            options.append((err,a,axis,high,low,hi-lo,swap))
        _, a, axis, high, low, sides, swap = min(options, key=lambda q:q[0])

        # q = 12 horizontal coordinates, 9 positive vertical coordinates,
        #     12 radii, and rectangle width.
        nvar = 34
        X = np.zeros((21,nvar))
        Y = np.zeros((21,nvar))
        R = np.zeros((21,nvar))
        for k in range(21):
            check()
            group = k if k < 12 else k-9
            X[k,group] = 1
            R[k,21+group] = 1
            if k >= 3:
                Y[k,12+(k-3)%9] = 1 if k < 12 else -1
        q = np.zeros(nvar)
        q[:3] = a[axis,0]
        q[3:12] = (a[high,0]+a[low,0])/2
        q[12:21] = (a[high,1]-a[low,1])/2
        q[21:24] = a[axis,2]
        q[24:33] = (a[high,2]+a[low,2])/2
        q[33] = sides[0]
        scale = 2.0/sides.sum()
        q *= scale

        mirror = np.r_[np.arange(3), np.arange(12,21), np.arange(3,12)]
        pairs = []
        for i,j in zip(ii,jj):
            check()
            reflected = tuple(sorted((int(mirror[i]),int(mirror[j]))))
            if (int(i),int(j)) <= reflected:
                pairs.append((int(i),int(j)))
        pi = np.array([p[0] for p in pairs])
        pj = np.array([p[1] for p in pairs])
        DX = X[pi]-X[pj]
        DY = Y[pi]-Y[pj]
        RR = R[pi]+R[pj]
        # Only one representative of each reflected wall constraint.
        W = np.zeros(nvar)
        W[33] = 0.5
        wall = np.vstack((W-X[:12]-R[:12], W+X[:12]-R[:12],
                          -W-Y[:12]-R[:12]))
        offset = np.r_[np.zeros(24),np.ones(12)]
        objective = -np.sum(R,axis=0)

        def unpack(z):
            out = np.column_stack((X@z,Y@z,R@z))
            if swap:
                out[:,:2] = out[:,[1,0]]
            return out

        def values(z):
            check()
            dx,dy,rr = DX@z,DY@z,RR@z
            return np.r_[dx*dx+dy*dy-rr*rr, wall@z+offset]

        def jacobian(z):
            check()
            dx,dy,rr = DX@z,DY@z,RR@z
            return np.vstack((2*dx[:,None]*DX+2*dy[:,None]*DY-2*rr[:,None]*RR,wall))

        last = q.copy()
        def fun(z):
            nonlocal last
            check()
            last = z.copy()
            return float(objective@z)

        bounds = [(-1.0,1.0)]*12 + [(0.0,1.0)]*9 + [(1e-6,0.7)]*12 + [(0.15,1.85)]
        constraint = {'type':'ineq','fun':values,'jac':jacobian}
        def solve(z, iterations):
            check()
            result = minimize(fun,z,jac=lambda z:objective,method='SLSQP',
                              bounds=bounds,constraints=constraint,
                              options={'maxiter':iterations,'ftol':2e-13,'disp':False})
            check()
            consider(unpack(result.x))
            return result.x

        base = solve(q, min(55,int(params['maxiter'])))
        if np.min(values(base)) < -1e-7:
            base = q.copy()
        active = np.flatnonzero(values(base) < 2e-7)
        J = jacobian(base)[active]
        # A pseudoinverse maps coordinated contact openings to quotient motions.
        inverse = np.linalg.pinv(J,rcond=1e-9)
        attempts = int(params['attempts'])
        for trial in range(attempts):
            check()
            count = min(int(params['release_size']),len(active))
            # Prefer interacting contacts rather than unrelated isolated releases.
            first = int(rng.integers(len(active)))
            similarity = np.abs(J@J[first])
            weights = similarity + 0.08*max(1e-12,float(similarity.max()))
            weights[first] = 0
            others = rng.choice(len(active),count-1,replace=False,p=weights/weights.sum())
            chosen = np.r_[first,others]
            opening = np.zeros(len(active))
            opening[chosen] = rng.uniform(0.4,1.0,count)
            direction = inverse@opening
            size = float(np.max(np.abs(direction[:33])))
            if size < 1e-12:
                continue
            amplitude = float(params['step'])*(0.55+0.35*(trial%4))
            trial_q = base + direction*(amplitude/size)
            # Alternate feasible-side opening and simultaneous contact crossing.
            if trial % 3 == 2:
                trial_q = base - direction*(amplitude/size)
            trial_q[12:21] = np.clip(trial_q[12:21],0.0001,0.95)
            trial_q[21:33] = np.clip(trial_q[21:33],0.002,0.6)
            trial_q[33] = np.clip(trial_q[33],0.2,1.8)
            candidate = solve(trial_q,int(params['maxiter']))
            if objective@candidate < objective@base-1e-9 and np.min(values(candidate)) > -1e-9:
                base = candidate
                active = np.flatnonzero(values(base)<2e-7)
                J = jacobian(base)[active]
                inverse = np.linalg.pinv(J,rcond=1e-9)
    except StopSearch:
        pass
    return best
