import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"iterations": (160, 50, 400), "search_share": (0.78, 0.4, 0.9)}


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.69 * budget_s
    n = 21
    ii, jj = np.triu_indices(n, 1)
    best = None
    best_score = -np.inf
    original = None

    def score(a):
        if a is None or a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -np.inf
        d = a[ii, :2] - a[jj, :2]
        if np.any(np.sqrt(d[:, 0]**2 + d[:, 1]**2) < a[ii, 2] + a[jj, 2]):
            return -np.inf
        x, y, r = [[Fraction(float(v)) for v in a[:, j]] for j in range(3)]
        p = max(u+v for u,v in zip(x,r)) - min(u-v for u,v in zip(x,r))
        p += max(u+v for u,v in zip(y,r)) - min(u-v for u,v in zip(y,r))
        return float(np.sum(a[:, 2])) if p <= 2 else -np.inf

    if init is not None:
        try:
            original = np.asarray(init['circles'], float) if 'circles' in init else np.asarray(init['state'], float).reshape(3, 21).T
            s = score(original)
            if s > best_score:
                best, best_score = original.copy(), s
        except Exception:
            original = None

    def consider(a):
        nonlocal best, best_score
        a = np.array(a, dtype=float, copy=True)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return
        lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
        hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
        a[:, :2] -= (hi + lo)/2
        d = a[ii, :2] - a[jj, :2]
        ratio = np.min(np.sqrt(np.sum(d*d, axis=1))/(a[ii, 2]+a[jj, 2]))
        a[:, 2] *= min(1., float(ratio)) * (1.-3e-14)
        lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
        hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
        span = float(np.sum(hi-lo))
        if span <= 0:
            return
        a *= (2./span)*(1.-3e-14)
        s = score(a)
        if s > best_score:
            best, best_score = a, s

    class Stop(Exception):
        pass

    def optimize(a, end, symmetric):
        if time.monotonic() >= end:
            return
        if symmetric:
            # Axis circles 0:3; positive-side circles 3:12; reflected circles 12:21.
            X = np.zeros((21, 34))
            Y = np.zeros_like(X)
            R = np.zeros_like(X)
            for j in range(9):
                if time.monotonic() >= end:
                    return
                X[3+j, j], X[12+j, j] = 1., -1.
                Y[3+j, 12+j] = Y[12+j, 12+j] = 1.
                R[3+j, 24+j] = R[12+j, 24+j] = 1.
            for j in range(3):
                if time.monotonic() >= end:
                    return
                Y[j, 9+j] = 1.
                R[j, 21+j] = 1.
            reflection = np.r_[np.arange(3), np.arange(12,21), np.arange(3,12)]
            pairs = []
            for i,j in zip(ii,jj):
                if time.monotonic() >= end:
                    return
                mirrored = tuple(sorted((int(reflection[i]), int(reflection[j]))))
                if (int(i),int(j)) <= mirrored:
                    pairs.append((i,j))
            pi, pj = np.asarray(pairs).T
            reps = np.arange(12)
            W = np.zeros(34); W[-1] = 1.
            walls = np.vstack((.5*W-X[reps]-R[reps], -.5*W-Y[reps]-R[reps], -.5*W+Y[reps]-R[reps]))
            offset = np.r_[np.zeros(12), np.ones(24)]
            bounds = [(0.,1.)]*9 + [(-1.,1.)]*12 + [(1e-8,.5)]*12 + [(.02,1.98)]
        else:
            X = np.zeros((21,64)); X[:,:21] = np.eye(21)
            Y = np.zeros((21,64)); Y[:,21:42] = np.eye(21)
            R = np.zeros((21,64)); R[:,42:63] = np.eye(21)
            W = np.zeros(64); W[-1] = 1.
            pi,pj = ii,jj
            walls = np.vstack((.5*W-X-R, .5*W+X-R, -.5*W-Y-R, -.5*W+Y-R))
            offset = np.r_[np.zeros(42), np.ones(42)]
            bounds = [(-1.,1.)]*42 + [(1e-8,.5)]*21 + [(.02,1.98)]
        DX, DY, RR = X[pi]-X[pj], Y[pi]-Y[pj], R[pi]+R[pj]
        g = -np.sum(R,axis=0)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        a = a.copy(); a[:,:2] -= (hi+lo)/2
        if symmetric:
            q = np.r_[a[3:12,0],a[:3,1],a[3:12,1],a[:3,2],a[3:12,2],np.clip(hi[0]-lo[0],.02,1.98)]
        else:
            q = np.r_[a.T.ravel(),np.clip(hi[0]-lo[0],.02,1.98)]
        last = q.copy()
        counter = 0
        cache_q = None
        cache_d = None

        def check():
            if time.monotonic() >= end:
                raise Stop()

        def geometry(q):
            nonlocal cache_q, cache_d
            check()
            if cache_q is None or not np.array_equal(q,cache_q):
                dx,dy = DX@q,DY@q
                dist = np.maximum(np.sqrt(dx*dx+dy*dy),1e-18)
                cache_q = q.copy()
                cache_d = (dx,dy,dist)
            return cache_d

        def fun(q):
            check()
            return float(g@q)

        def con(q):
            dx,dy,dist = geometry(q)
            return np.r_[dist-RR@q,walls@q+offset]

        def jac(q):
            dx,dy,dist = geometry(q)
            return np.vstack(((dx/dist)[:,None]*DX+(dy/dist)[:,None]*DY-RR,walls))

        def unpack(q):
            return np.column_stack((X@q,Y@q,R@q))

        def callback(q):
            nonlocal last,counter
            last = q.copy()
            counter += 1
            check()
            if counter % 20 == 0:
                consider(unpack(q))

        try:
            result = minimize(fun,q,jac=lambda q:g,method='SLSQP',bounds=bounds,
                              constraints={'type':'ineq','fun':con,'jac':jac},callback=callback,
                              options={'maxiter':int(params['iterations']),'ftol':2e-12,'disp':False})
            last = result.x
        except (Stop,ValueError,FloatingPointError):
            pass
        consider(unpack(last))

    patterns = [(5,4,3,4,5),(4,3,4,3,4,3),(3,4,5,4,5),(4,5,4,5,3)]
    search_end = start + (deadline-start)*params['search_share']
    for t,pattern in enumerate(patterns):
        now = time.monotonic()
        if now >= search_end:
            break
        singles,pairs = [],[]
        for j,c in enumerate(pattern):
            if time.monotonic() >= search_end:
                break
            y = (j-(len(pattern)-1)/2)*np.sqrt(3.)
            if c%2:
                singles.append([0.,y,1.])
            for h in range(c//2):
                if time.monotonic() >= search_end:
                    break
                pairs.append([float(2*(h+1) if c%2 else 2*h+1),y,1.])
        if len(singles)!=3 or len(pairs)!=9:
            continue
        a = np.array(singles+pairs+[[-x,y,r] for x,y,r in pairs])
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        a *= 2./np.sum(hi-lo)
        consider(a)
        end = min(search_end,time.monotonic()+max(0.,search_end-time.monotonic())/(len(patterns)-t))
        optimize(a,end,True)
    if best is not None and time.monotonic()<deadline:
        optimize(best,deadline,False)
    if best is None:
        consider(np.array([[(i-1)*.2,(j-3)*.2,.099] for j in range(7) for i in range(3)]))
    if original is not None and score(original)>=best_score:
        return init
    return {'state':best.T.ravel().tolist()}
