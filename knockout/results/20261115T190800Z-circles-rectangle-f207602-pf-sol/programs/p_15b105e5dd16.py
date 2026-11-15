import time
import numpy as np
from fractions import Fraction
from scipy.optimize import minimize

PARAMS = {"iterations": (105, 45, 180), "jitter": (0.006, 0.0, 0.03), "starts": (5, 1, 12)}


def construct(params, seed, budget_s, init=None):
    deadline = time.monotonic() + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    best = init
    score_best = -float('inf')
    saved = None

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
        x, y, r = [[Fraction(float(t)) for t in a[:, k]] for k in range(3)]
        p = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        p += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        if p > 2:
            return -float('inf')
        d = a[ii,:2]-a[jj,:2]
        if np.any(np.sqrt(d[:,0]**2+d[:,1]**2) < a[ii,2]+a[jj,2]):
            return -float('inf')
        return float(np.sum(a[:,2]))

    if init is not None:
        try:
            score_best = exact(array(init))
        except (ValueError, KeyError, TypeError):
            pass

    def retain(a):
        nonlocal best, score_best, saved
        a = np.asarray(a).reshape(21,3).copy()
        if not np.isfinite(a).all() or np.min(a[:,2]) <= 0:
            return
        a[:,:2] -= np.mean(a[:,:2], axis=0)
        d = a[ii,:2]-a[jj,:2]
        ds = np.sqrt(d[:,0]**2+d[:,1]**2)
        f = min(1.0, float(np.min(ds/(a[ii,2]+a[jj,2]))))
        if f <= 0:
            return
        a[:,2] *= f*(1-5e-14)
        x, y, r = [[Fraction(float(t)) for t in a[:,k]] for k in range(3)]
        p = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        p += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        a *= float(Fraction(2)/p)*(1-5e-14)
        s = exact(a)
        if s > score_best:
            score_best = s
            best = {'centers': a[:,:2].tolist(), 'radii': a[:,2].tolist()}
            saved = a.copy()

    def layered(counts, jitter):
        pts, singles, pairs = [], [], []
        for level, count in enumerate(counts):
            check()
            for col in range(count):
                check()
                x = (col-(count-1)/2)*.2
                if level == 0 and count == 3:
                    x *= 1.45
                x += rng.normal(0,jitter)
                r = .095 + rng.normal(0,jitter*.12)
                if level == 0:
                    singles.append(len(pts))
                    pts.append([x,0,r])
                else:
                    y = level*.177 + rng.normal(0,jitter*.25)
                    i = len(pts)
                    pairs.append((i,i+1))
                    pts.extend([[x,y,r],[x,-y,r]])
        a = np.array(pts)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        s = 2/np.sum(hi-lo)
        a *= s
        a[:,0] -= lo[0]*s
        m = 2*len(singles)+3*len(pairs)+1
        B = np.zeros((64,m))
        k = 0
        for i in singles:
            check()
            B[3*i,k] = 1
            B[3*i+2,k+1] = 1
            k += 2
        for i,j in pairs:
            check()
            B[3*i,k] = B[3*j,k] = 1
            B[3*i+1,k+1] = 1
            B[3*j+1,k+1] = -1
            B[3*i+2,k+2] = B[3*j+2,k+2] = 1
            k += 3
        B[63,k] = 1
        v = np.r_[a.ravel(),(hi[0]-lo[0])*s]
        z = (B.T@v)/np.sum(B*B,axis=0)
        return z,B

    def optimize(z,B,iterations):
        check()
        X,Y,R = B[:63:3],B[1:63:3],B[2:63:3]
        W = B[63]
        D = X[ii]-X[jj]
        E = Y[ii]-Y[jj]
        S = R[ii]+R[jj]
        # Independent sign changes of dx and dy preserve the distance.
        keys = set()
        take = []
        for k in range(len(ii)):
            check()
            d,e = D[k].copy(),E[k].copy()
            nz = np.flatnonzero(d)
            if len(nz) and d[nz[0]] < 0:
                d = -d
            nz = np.flatnonzero(e)
            if len(nz) and e[nz[0]] < 0:
                e = -e
            key = (d.tobytes(),e.tobytes(),S[k].tobytes())
            if key not in keys:
                keys.add(key)
                take.append(k)
        D,E,S = D[take],E[take],S[take]
        A = np.vstack((X-R,W-X-R,Y-R-.5*W,-Y-R-.5*W,R))
        b = np.r_[np.zeros(42),np.ones(42),np.full(21,-1e-8)]
        AB = np.unique(np.column_stack((A,b)),axis=0)
        A,b = AB[:,:-1],AB[:,-1]
        g = -np.sum(R,axis=0)
        cache_z = None
        cache_c = None
        cache_j = None
        latest = [z.copy()]
        count = [0]

        def update(t):
            nonlocal cache_z,cache_c,cache_j
            check()
            if cache_z is not None and np.array_equal(t,cache_z):
                return
            dx,dy = D@t,E@t
            dist = np.sqrt(dx*dx+dy*dy)
            den = np.maximum(dist,1e-15)
            cache_z = t.copy()
            cache_c = np.r_[dist-S@t,A@t+b]
            cache_j = np.vstack(((dx/den)[:,None]*D+(dy/den)[:,None]*E-S,A))

        def fun(t):
            check()
            return float(g@t)

        def cons(t):
            update(t)
            return cache_c

        def jac(t):
            update(t)
            return cache_j

        def callback(t):
            check()
            latest[0] = t.copy()
            count[0] += 1
            if count[0] % 18 == 0:
                retain((B@t)[:63])

        try:
            res = minimize(fun,z,jac=lambda t:g,method='SLSQP',
                           constraints={'type':'ineq','fun':cons,'jac':jac},
                           callback=callback,
                           options={'maxiter':iterations,'ftol':1e-11,'disp':False})
            latest[0] = res.x
        except Stop:
            retain((B@latest[0])[:63])
            raise
        retain((B@latest[0])[:63])
        return latest[0]

    try:
        if score_best > 1.8:
            a = array(init).copy()
            lo = np.min(a[:,:2]-a[:,2,None],axis=0)
            hi = np.max(a[:,:2]+a[:,2,None],axis=0)
            a[:,0] -= lo[0]
            a[:,1] -= (lo[1]+hi[1])/2
            optimize(np.r_[a.ravel(),hi[0]-lo[0]],np.eye(64),60)
        families = [(3,5,4),(5,4,4),(5,5,3),(3,4,5),(3,5,4)]
        for attempt in range(params['starts']):
            check()
            z,B = layered(families[attempt%len(families)],params['jitter'])
            retain((B@z)[:63])
            optimize(z,B,params['iterations'])
        if saved is not None and time.monotonic() < deadline-.12:
            a = saved.copy()
            lo = np.min(a[:,:2]-a[:,2,None],axis=0)
            hi = np.max(a[:,:2]+a[:,2,None],axis=0)
            a[:,0] -= lo[0]
            a[:,1] -= (lo[1]+hi[1])/2
            optimize(np.r_[a.ravel(),hi[0]-lo[0]],np.eye(64),55)
    except (Stop, ValueError, np.linalg.LinAlgError):
        pass
    if best is None or score_best == -float('inf'):
        a = np.array([[(i%7)*.2,(i//7)*.2,.099999999999] for i in range(21)])
        return {'circles':a.tolist()}
    return best
