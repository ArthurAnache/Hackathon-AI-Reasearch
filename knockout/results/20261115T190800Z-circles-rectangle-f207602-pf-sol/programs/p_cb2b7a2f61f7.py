import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"iterations": (100, 40, 180), "perturbation": (0.018, 0.001, 0.06)}


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    best = init
    best_score = -float('inf')
    best_array = None
    best_model = None
    best_z = None

    class Stop(Exception):
        pass

    def check():
        if time.monotonic() >= deadline:
            raise Stop()

    def array(obj):
        if 'circles' in obj:
            return np.asarray(obj['circles'], dtype=float)
        return np.column_stack((obj['centers'], obj['radii']))

    def perimeter(a):
        x = [Fraction(float(t)) for t in a[:, 0]]
        y = [Fraction(float(t)) for t in a[:, 1]]
        r = [Fraction(float(t)) for t in a[:, 2]]
        return (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
                + max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)))

    def exact(a):
        if a.shape != (21,3) or not np.isfinite(a).all() or np.min(a[:,2]) <= 0:
            return -float('inf')
        if perimeter(a) > 2:
            return -float('inf')
        d = a[ii,:2]-a[jj,:2]
        if np.any(np.sqrt(d[:,0]**2+d[:,1]**2) < a[ii,2]+a[jj,2]):
            return -float('inf')
        return float(np.sum(a[:,2]))

    if init is not None:
        try:
            initial = array(init)
            best_score = exact(initial)
            if np.isfinite(best_score):
                best_array = initial.copy()
        except (ValueError, KeyError, TypeError):
            pass

    def retain(a, model=None, z=None):
        nonlocal best, best_score, best_array, best_model, best_z
        a = a.copy()
        if not np.isfinite(a).all() or np.min(a[:,2]) <= 0:
            return
        a[:,:2] -= np.mean(a[:,:2], axis=0)
        d = a[ii,:2]-a[jj,:2]
        f = min(1., float(np.min(np.sqrt(d[:,0]**2+d[:,1]**2)/(a[ii,2]+a[jj,2]))))
        if f <= 0:
            return
        a[:,2] *= f*(1-2e-14)
        p = perimeter(a)
        if p <= 0:
            return
        a *= float(Fraction(2)/p)*(1-2e-14)
        score = exact(a)
        if score > best_score:
            best_score = score
            best_array = a.copy()
            best = {'centers': a[:,:2].tolist(), 'radii': a[:,2].tolist()}
            if model is not None:
                best_model = model
                best_z = z.copy()

    def make_model(counts=None):
        # Each coordinate is a fixed linear map of the reduced variables.
        if counts is None:
            X = np.zeros((21,64)); Y = X.copy(); R = X.copy()
            for k in range(21):
                check()
                X[k,3*k] = 1; Y[k,3*k+1] = 1; R[k,3*k+2] = 1
            W = np.zeros(64); W[-1] = 1
            ids = np.arange(21)
            pi,pj = ii,jj
            walls = np.vstack((X-R,W-X-R,W*(-.5)+Y-R,W*(-.5)-Y-R))
            offsets = np.r_[np.zeros(42),np.ones(42)]
            lower = [(None,None)]*64
            for k in range(21):
                check()
                lower[3*k+2] = (1e-7,None)
            lower[-1] = (.05,1.95)
            return X,Y,R,W,pi,pj,walls,offsets,lower
        ns = counts[0]
        q = 2*ns+3*((21-ns)//2)+1
        X = np.zeros((21,q)); Y = X.copy(); R = X.copy()
        mate = np.arange(21); reps = []; positive = []; c = 0; k = 0
        bounds = [(None,None)]*q
        for level,n in enumerate(counts):
            check()
            for col in range(n):
                check()
                if level == 0:
                    X[k,c] = 1; R[k,c+1] = 1
                    bounds[c+1] = (1e-7,None)
                    reps.append(k); k += 1; c += 2
                else:
                    X[k:k+2,c] = 1
                    Y[k,c+1] = 1; Y[k+1,c+1] = -1
                    R[k:k+2,c+2] = 1
                    bounds[c+1] = (0,None); bounds[c+2] = (1e-7,None)
                    mate[k],mate[k+1] = k+1,k
                    reps.append(k); positive.append(k)
                    k += 2; c += 3
        W = np.zeros(q); W[-1] = 1; bounds[-1] = (.05,1.95)
        keep = []
        for t,(i,j) in enumerate(zip(ii,jj)):
            check()
            reflected = tuple(sorted((int(mate[i]),int(mate[j]))))
            if (int(i),int(j)) <= reflected:
                keep.append(t)
        pi,pj = ii[keep],jj[keep]
        reps = np.array(reps)
        walls = np.vstack((X[reps]-R[reps],W-X[reps]-R[reps],W*(-.5)-Y[reps]-R[reps]))
        offsets = np.r_[np.zeros(2*len(reps)),np.ones(len(reps))]
        return X,Y,R,W,pi,pj,walls,offsets,bounds

    def optimize(z, model, limit):
        X,Y,R,W,pi,pj,A,b,bounds = model
        DX = X[pi]-X[pj]; DY = Y[pi]-Y[pj]; SR = R[pi]+R[pj]
        grad = -np.sum(R,axis=0)
        latest = [z.copy()]; counter = [0]
        cache = [None,None,None]
        def calc(t):
            check()
            if cache[0] is None or not np.array_equal(t,cache[0]):
                dx = DX@t; dy = DY@t
                dist = np.sqrt(dx*dx+dy*dy)
                den = np.maximum(dist,1e-15)
                cache[0] = t.copy()
                cache[1] = np.r_[dist-SR@t,A@t+b]
                cache[2] = np.vstack(((dx/den)[:,None]*DX+(dy/den)[:,None]*DY-SR,A))
            return cache[1],cache[2]
        def save(t):
            retain(np.column_stack((X@t,Y@t,R@t)),model,t)
        def cb(t):
            check()
            latest[0] = t.copy(); counter[0] += 1
            if counter[0] % 10 == 0:
                save(t)
        try:
            res = minimize(lambda t: float(grad@t),z,jac=lambda t: grad,
                           method='SLSQP',bounds=bounds,
                           constraints={'type':'ineq','fun':lambda t: calc(t)[0],
                                        'jac':lambda t: calc(t)[1]},callback=cb,
                           options={'maxiter':limit,'ftol':8e-12,'disp':False})
            latest[0] = res.x
        except Stop:
            save(latest[0])
            raise
        save(latest[0])
        return latest[0]

    def seed_model(counts):
        model = make_model(counts)
        X,Y,R,W,*_ = model
        pts = []
        for level,n in enumerate(counts):
            check()
            for col in range(n):
                check()
                x = (col-(n-1)/2)*.20
                if level == 0 and n == 3:
                    x *= 1.45
                pts.append([x,level*.177,.095])
                if level:
                    pts.append([x,-level*.177,.095])
        a = np.array(pts)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        scale = 2/np.sum(hi-lo)
        a *= scale; a[:,0] -= lo[0]*scale
        z = (X.T@a[:,0]+Y.T@a[:,1]+R.T@a[:,2]+W*((hi[0]-lo[0])*scale))
        norm = np.sum(X*X+Y*Y+R*R,axis=0)+W*W
        return z/norm,model

    try:
        if best_array is not None and best_score > 2.2:
            m = make_model()
            a = best_array.copy()
            lo = np.min(a[:,:2]-a[:,2,None],axis=0)
            hi = np.max(a[:,:2]+a[:,2,None],axis=0)
            a[:,0] -= lo[0]; a[:,1] -= (lo[1]+hi[1])/2
            optimize(np.r_[a.ravel(),hi[0]-lo[0]],m,params['iterations'])
        for counts in ((3,5,4),(5,4,4),(5,5,3),(3,4,5)):
            check()
            if time.monotonic() > start+.52*budget_s:
                break
            z,m = seed_model(counts)
            retain(np.column_stack((m[0]@z,m[1]@z,m[2]@z)),m,z)
            optimize(z,m,params['iterations'])
        if best_array is not None:
            check()
            a = best_array.copy()
            lo = np.min(a[:,:2]-a[:,2,None],axis=0)
            hi = np.max(a[:,:2]+a[:,2,None],axis=0)
            a[:,0] -= lo[0]; a[:,1] -= (lo[1]+hi[1])/2
            optimize(np.r_[a.ravel(),hi[0]-lo[0]],make_model(),65)
        while time.monotonic() < deadline and best_model is not None:
            check()
            z = best_z.copy()
            z += rng.normal(0,params['perturbation'],len(z))
            for k,(lo,hi) in enumerate(best_model[-1]):
                check()
                if lo is not None: z[k] = max(z[k],lo)
                if hi is not None: z[k] = min(z[k],hi)
            optimize(z,best_model,65)
    except (Stop,ValueError,np.linalg.LinAlgError):
        pass
    if best_score == -float('inf'):
        return {'circles': [[(i%7)*.2,(i//7)*.2,.0999999999999] for i in range(21)]}
    return best
