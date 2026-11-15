import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'search_fraction': (0.70, 0.5, 0.75),
    'warm_seconds': (0.10, 0.02, 0.25),
    'stage_seconds': (0.22, 0.08, 0.35),
    'jitter': (0.008, 0.0, 0.04),
}

N = 21
II, JJ = np.triu_indices(N, 1)

class StopSearch(Exception):
    pass

def exact(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -float('inf')
    dx = a[II, 0] - a[JJ, 0]
    dy = a[II, 1] - a[JJ, 1]
    if np.any(np.sqrt(dx*dx + dy*dy) < a[II, 2] + a[JJ, 2]):
        return -float('inf')
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    w = max(u+v for u,v in zip(x,r)) - min(u-v for u,v in zip(x,r))
    h = max(u+v for u,v in zip(y,r)) - min(u-v for u,v in zip(y,r))
    if w+h > 2:
        return -float('inf')
    return float(np.sum(a[:, 2]))

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + float(params['search_fraction'])*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -float('inf')
    if init is not None:
        try:
            best_score = exact(np.asarray(init['circles'], dtype=float))
        except (ValueError, TypeError, KeyError):
            pass

    def check(t=None):
        if time.monotonic() >= (deadline if t is None else t):
            raise StopSearch()

    def retain(a):
        nonlocal best, best_score
        check()
        a = np.array(a, dtype=float, copy=True)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        dx = a[II, 0]-a[JJ, 0]
        dy = a[II, 1]-a[JJ, 1]
        f = min(1.0, float(np.min(np.sqrt(dx*dx+dy*dy)/(a[II, 2]+a[JJ, 2]))))
        if f <= 0:
            return -float('inf')
        a[:, 2] *= f*(1-4e-15)
        lo = np.min(a[:, :2]-a[:, 2,None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2,None], axis=0)
        a[:, :2] -= (lo+hi)*0.5
        a *= (2-2e-14)/float(np.sum(hi-lo))
        s = float(np.sum(a[:, 2]))
        if s > best_score:
            s = exact(a)
            if s > best_score:
                best_score = s
                best = {'circles': a.tolist()}
        return s

    def finish_model(X,Y,R,q,bounds):
        check()
        dim = len(q)
        width = np.zeros(dim); width[-1] = 0.5
        height = np.zeros(dim); height[-1] = -1.0
        B = np.concatenate((width-X-R, width+X-R, Y-R, height-Y-R))
        offset = np.r_[np.zeros(63), np.full(21,2.0)]
        dx = X[II]-X[JJ]; dy = Y[II]-Y[JJ]; sr = R[II]+R[JJ]
        # Exact structural signatures remove reflection-duplicate constraints.
        signatures = np.concatenate((sr, np.abs(dx), np.abs(dy)), axis=1)
        _, inds = np.unique(signatures, axis=0, return_index=True)
        return dict(X=X,Y=Y,R=R,q=q,bounds=bounds,DX=dx[inds],DY=dy[inds],SR=sr[inds],B=B,offset=offset,obj=-R.sum(axis=0),score=-float('inf'))

    def full_model(a):
        dim = 64
        X = np.zeros((21,dim)); Y = X.copy(); R = X.copy()
        X[np.arange(21),np.arange(21)] = 1
        Y[np.arange(21),21+np.arange(21)] = 1
        R[np.arange(21),42+np.arange(21)] = 1
        lo = np.min(a[:, :2]-a[:,2,None],axis=0)
        hi = np.max(a[:, :2]+a[:,2,None],axis=0)
        q = np.r_[a[:,0]-(lo[0]+hi[0])*0.5,a[:,1]-lo[1],a[:,2],hi[0]-lo[0]]
        bounds = [(None,None)]*42+[(1e-7,0.65)]*21+[(0.2,1.8)]
        return finish_model(X,Y,R,q,bounds)

    def family(rows, jitter):
        check()
        dim = sum(3*(n//2)+2*(n%2) for n in rows)+1
        X = np.zeros((21,dim)); Y = X.copy(); R = X.copy()
        q = np.zeros(dim)
        bounds = [(None,None)]*dim
        radius = 1/(max(rows)+1+(len(rows)-1)*np.sqrt(3)/2)
        c = 0; k = 0
        for row,n in enumerate(rows):
            check()
            yy = radius+row*np.sqrt(3)*radius
            if n%2:
                Y[c,k] = 1; R[c,k+1] = 1
                q[k:k+2] = [yy,radius]
                bounds[k+1] = (1e-7,0.65)
                c += 1; k += 2
            for j in range(n//2):
                check()
                X[c,k] = 1; X[c+1,k] = -1
                Y[c:c+2,k+1] = 1; R[c:c+2,k+2] = 1
                q[k:k+3] = [(2*j+1+n%2)*radius,yy,radius]
                q[k:k+2] += rng.normal(0,jitter,2)
                bounds[k+2] = (1e-7,0.65)
                c += 2; k += 3
        q[-1] = 2*max(rows)*radius
        bounds[-1] = (0.2,1.8)
        return finish_model(X,Y,R,q,bounds)

    def run(f, seconds, maxiter=150):
        local_end = min(deadline-0.008,time.monotonic()+seconds)
        DX,DY,SR,B = f['DX'],f['DY'],f['SR'],f['B']
        cache = [None,None,None]
        last = [f['q'].copy()]
        def constraints(q):
            check(local_end)
            if cache[0] is None or not np.array_equal(q,cache[0]):
                dx = DX@q; dy = DY@q
                d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-20)
                cache[0] = q.copy()
                cache[1] = np.r_[d-SR@q,B@q+f['offset']]
                cache[2] = np.concatenate(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-SR,B))
            return cache[1],cache[2]
        def objective(q):
            check(local_end)
            return float(f['obj']@q),f['obj']
        def callback(q):
            check(local_end)
            last[0] = q.copy()
            a = np.column_stack((f['X']@q,f['Y']@q,f['R']@q))
            s = retain(a)
            f['score'] = max(f['score'],s)
        try:
            res = minimize(objective,f['q'],jac=True,method='SLSQP',bounds=f['bounds'],
                constraints={'type':'ineq','fun':lambda q:constraints(q)[0],'jac':lambda q:constraints(q)[1]},
                callback=callback,options={'maxiter':maxiter,'ftol':3e-13,'disp':False})
            last[0] = res.x.copy()
        except StopSearch:
            pass
        f['q'] = last[0]
        if time.monotonic() < deadline-0.005:
            s = retain(np.column_stack((f['X']@f['q'],f['Y']@f['q'],f['R']@f['q'])))
            f['score'] = max(f['score'],s)

    try:
        if best is None:
            retain(np.array([[(i+.5)*.2,(j+.5)*.2,.099] for j in range(7) for i in range(3)]))
        if best is not None:
            warm = full_model(np.asarray(best['circles'],dtype=float))
            run(warm,float(params['warm_seconds']),45)
        archive = []
        layouts = [(5,4,3,4,5),(4,5,4,5,3),(7,7,7),(6,5,4,3,3)]
        for rows in layouts:
            check(deadline-0.12)
            f = family(rows,float(params['jitter']))
            archive.append(f)
            run(f,float(params['stage_seconds']),100)
        while time.monotonic() < deadline-0.04 and archive:
            check()
            archive.sort(key=lambda f:f['score'],reverse=True)
            f = archive[0]
            before = f['q'].copy()
            run(f,min(0.20,deadline-time.monotonic()-0.015),180)
            if np.max(np.abs(before-f['q'])) < 1e-8:
                archive.pop(0)
    except StopSearch:
        pass
    return best
