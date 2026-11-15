import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {'iterations': (100, 40, 180), 'perturbation': (0.035, 0.005, 0.12), 'fraction': (0.69, 0.5, 0.72)}


def construct(params, seed, budget_s, init=None):
    deadline = time.monotonic() + params['fraction'] * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    best = init
    score_best = -float('inf')
    best_state = None

    class Stop(Exception):
        pass

    def check():
        if time.monotonic() >= deadline:
            raise Stop()

    def array(obj):
        if 'circles' in obj:
            return np.asarray(obj['circles'], dtype=float)
        return np.column_stack((obj['centers'], obj['radii']))

    def exact(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return -float('inf')
        d = a[ii, :2] - a[jj, :2]
        if np.any(np.sqrt(d[:, 0]**2 + d[:, 1]**2) < a[ii, 2] + a[jj, 2]):
            return -float('inf')
        x = list(map(Fraction, map(float, a[:, 0])))
        y = list(map(Fraction, map(float, a[:, 1])))
        r = list(map(Fraction, map(float, a[:, 2])))
        perimeter = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        perimeter += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        if perimeter > 2:
            return -float('inf')
        return float(np.sum(a[:, 2]))

    if init is not None:
        try:
            score_best = exact(array(init))
        except (ValueError, TypeError, KeyError):
            pass

    def retain(a, state=None):
        nonlocal best, score_best, best_state
        a = np.array(a, dtype=float, copy=True)
        if a.shape != (21,3) or not np.isfinite(a).all() or np.min(a[:,2]) <= 0:
            return
        a[:,:2] -= np.mean(a[:,:2],axis=0)
        d = a[ii,:2]-a[jj,:2]
        dist = np.sqrt(d[:,0]**2+d[:,1]**2)
        f = min(1.0, float(np.min(dist/(a[ii,2]+a[jj,2]))))
        if f <= 0:
            return
        a[:,2] *= f*(1-4e-14)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        a *= (2/np.sum(hi-lo))*(1-8e-14)
        s = exact(a)
        if s > score_best:
            score_best = s
            best = {'centers':a[:,:2].tolist(), 'radii':a[:,2].tolist()}
            if state is not None:
                best_state = (state[0], state[1].copy())

    fallback = np.array([[(i%7)*.2,(i//7)*.2,.099999999999] for i in range(21)])
    retain(fallback)

    def model(counts):
        check()
        ns = counts[0]
        m = ns + sum(counts[1:])
        npair = m-ns
        nvar = 2*m+npair+1
        X = np.zeros((21,nvar))
        Y = np.zeros_like(X)
        R = np.zeros_like(X)
        q = np.zeros(nvar)
        mirror = np.arange(21)
        reps = []
        p = 0
        g = 0
        for level, count in enumerate(counts):
            check()
            for col in range(count):
                check()
                q[g] = (col-(count-1)/2)*.2
                if level == 0 and count == 3:
                    q[g] *= 1.45
                q[m+npair+g] = .095
                reps.append(p)
                X[p,g] = 1
                R[p,m+npair+g] = 1
                if level:
                    k = m+g-ns
                    q[k] = level*.177
                    Y[p,k] = 1
                    X[p+1,g] = 1
                    R[p+1,m+npair+g] = 1
                    Y[p+1,k] = -1
                    mirror[p],mirror[p+1] = p+1,p
                    p += 2
                else:
                    p += 1
                g += 1
        a = np.column_stack((X@q,Y@q,R@q))
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        scale = 2/np.sum(hi-lo)
        q *= scale
        q[:m] -= lo[0]*scale
        q[-1] = (hi[0]-lo[0])*scale
        pairs = []
        for i,j in zip(ii,jj):
            check()
            other = tuple(sorted((int(mirror[i]),int(mirror[j]))))
            if (int(i),int(j)) <= other:
                pairs.append((i,j))
        pi,pj = np.array(pairs).T
        DX = X[pi]-X[pj]
        DY = Y[pi]-Y[pj]
        RS = R[pi]+R[pj]
        walls = []
        for p in reps:
            check()
            walls.append(X[p]-R[p])
            row = -X[p]-R[p]
            row[-1] = 1
            walls.append(row)
            row = -Y[p]-R[p]
            row[-1] = -.5
            walls.append(row)
        W = np.array(walls)
        offset = np.tile([0.,0.,1.],m)
        grad = -np.sum(R,axis=0)
        bounds = [(0.,2.)]*m + [(0.,1.)]*npair + [(1e-6, .7)]*m + [(.05,1.95)]
        return (q,X,Y,R,DX,DY,RS,W,offset,grad,bounds,m,npair)

    models = {}

    def optimize(counts, q=None, iterations=None):
        check()
        if counts not in models:
            models[counts] = model(counts)
        M = models[counts]
        base,X,Y,R,DX,DY,RS,W,offset,grad,bounds,m,npair = M
        if q is None:
            q = base.copy()
        latest = [q.copy()]
        cache = [None,None,None,None]
        def geometry(t):
            check()
            if cache[0] is None or not np.array_equal(cache[0], t):
                dx = DX@t
                dy = DY@t
                dist = np.sqrt(dx*dx+dy*dy)
                cache[:] = [t.copy(),dx,dy,dist]
            return cache[1:]
        def fun(t):
            check()
            return float(grad@t)
        def con(t):
            dx,dy,dist = geometry(t)
            return np.r_[dist-RS@t,W@t+offset]
        def jac(t):
            dx,dy,dist = geometry(t)
            den = np.maximum(dist,1e-15)
            return np.vstack(((dx/den)[:,None]*DX+(dy/den)[:,None]*DY-RS,W))
        def save(t):
            check()
            latest[0] = t.copy()
        try:
            res = minimize(fun,q,jac=lambda t:grad,method='SLSQP',bounds=bounds,
                           constraints={'type':'ineq','fun':con,'jac':jac},callback=save,
                           options={'maxiter':iterations or params['iterations'],'ftol':8e-12,'disp':False})
            latest[0] = res.x
        except Stop:
            t = latest[0]
            retain(np.column_stack((X@t,Y@t,R@t)),(counts,t))
            raise
        t = latest[0]
        retain(np.column_stack((X@t,Y@t,R@t)),(counts,t))
        return t

    try:
        for counts in ((3,5,4),(3,4,5),(5,4,4),(5,5,3)):
            check()
            optimize(counts)
        attempt = 0
        while time.monotonic() < deadline:
            check()
            if best_state is None:
                break
            counts,q = best_state
            q = q.copy()
            M = models[counts]
            m,npair = M[-2:]
            amplitude = params['perturbation']*(1+(attempt%3)*.6)
            selected = rng.choice(m,min(5,m),replace=False)
            q[selected] += rng.normal(0,amplitude,len(selected))
            q[m:m+npair] += rng.normal(0,amplitude*.65,npair)
            q[m:m+npair] = np.maximum(q[m:m+npair],.003)
            q[m+npair:m+npair+m] *= np.exp(rng.normal(0,.13,m))
            optimize(counts,q,70)
            attempt += 1
    except (Stop, ValueError, np.linalg.LinAlgError):
        pass
    return best
