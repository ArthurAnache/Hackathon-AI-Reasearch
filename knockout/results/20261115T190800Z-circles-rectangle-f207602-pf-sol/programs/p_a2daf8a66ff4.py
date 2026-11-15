import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"search_fraction": (0.70, 0.5, 0.8), "stage_seconds": (0.19, 0.08, 0.4)}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + float(params['search_fraction']) * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    best = init
    best_score = -float('inf')

    def exact(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        d = a[ii, :2] - a[jj, :2]
        if np.any(np.sqrt(d[:, 0]**2 + d[:, 1]**2) < a[ii, 2] + a[jj, 2]):
            return -float('inf')
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        w = max(u+v for u,v in zip(x,r)) - min(u-v for u,v in zip(x,r))
        h = max(u+v for u,v in zip(y,r)) - min(u-v for u,v in zip(y,r))
        if w+h > 2:
            return -float('inf')
        return float(np.sum(a[:, 2]))

    if init is not None:
        try:
            best_score = exact(np.asarray(init['circles'], dtype=float))
        except Exception:
            pass

    def retain(a):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            raise StopSearch()
        a = np.asarray(a, dtype=float).copy()
        if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return
        d = a[ii, :2] - a[jj, :2]
        dist = np.sqrt(d[:, 0]**2 + d[:, 1]**2)
        factor = min(1.0, float(np.min(dist / (a[ii, 2] + a[jj, 2]))))
        if factor <= 0:
            return
        a[:, 2] *= factor * (1.0 - 2e-14)
        a[:, :2] -= np.mean(a[:, :2], axis=0)
        wh = np.sum(np.max(a[:, :2]+a[:, 2,None], axis=0) - np.min(a[:, :2]-a[:, 2,None], axis=0))
        a *= (2.0 / wh) * (1.0 - 2e-14)
        if float(np.sum(a[:, 2])) <= best_score:
            return
        score = exact(a)
        if score > best_score:
            best_score = score
            best = {'circles': a.tolist()}

    def family(rows):
        # Each row consists of independent mirrored pairs and an optional axis circle.
        nv = sum(3*(n//2)+2*(n%2) for n in rows)+1
        X = np.zeros((21,nv)); Y = X.copy(); R = X.copy()
        q = np.zeros(nv)
        k = 0; c = 0
        r = 1.0 / (max(rows) + 1 + (len(rows)-1)*np.sqrt(3)/2)
        for row,n in enumerate(rows):
            if time.monotonic() >= deadline:
                raise StopSearch()
            yy = r + row*np.sqrt(3)*r
            if n % 2:
                Y[c,k] = 1; R[c,k+1] = 1
                q[k:k+2] = [yy,r]
                k += 2; c += 1
            for j in range(n//2):
                if time.monotonic() >= deadline:
                    raise StopSearch()
                xx = (2*j+1+(n%2))*r
                X[c,k] = 1; X[c+1,k] = -1
                Y[c:c+2,k+1] = 1; R[c:c+2,k+2] = 1
                q[k:k+3] = [xx,yy,r]
                k += 3; c += 2
        q[-1] = 2*max(rows)*r
        dx = X[ii]-X[jj]; dy = Y[ii]-Y[jj]; sr = R[ii]+R[jj]
        # Eliminate duplicate pair inequalities induced by reflection.
        probe = np.random.default_rng(104).normal(size=nv)
        signatures = np.round(np.column_stack(((dx@probe)**2, (dy@probe)**2, sr@probe)), 11)
        _, unique = np.unique(signatures, axis=0, return_index=True)
        dx = dx[unique]; dy = dy[unique]; sr = sr[unique]
        width = np.zeros(nv); width[-1] = 0.5
        height = np.zeros(nv); height[-1] = -1
        wall = np.concatenate((width-X-R, width+X-R, Y-R, height-Y-R), axis=0)
        offset = np.concatenate((np.zeros(63), np.full(21,2.0)))
        objective = -np.sum(R, axis=0)
        bounds = [(None,None)]*nv
        for j in range(nv-1):
            if time.monotonic() >= deadline:
                raise StopSearch()
            if np.any(R[:,j]):
                bounds[j] = (1e-6,0.65)
        bounds[-1] = (0.2,1.8)
        return {'q':q,'X':X,'Y':Y,'R':R,'dx':dx,'dy':dy,'sr':sr,'wall':wall,'offset':offset,'obj':objective,'bounds':bounds,'score':0.0}

    def run(f, seconds, iterations=90):
        local_end = min(deadline, time.monotonic()+seconds)
        def check():
            if time.monotonic() >= local_end:
                raise StopSearch()
        def fun(q):
            check()
            return float(f['obj']@q)
        def con(q):
            check()
            x=f['dx']@q; y=f['dy']@q
            return np.concatenate((np.sqrt(x*x+y*y)-f['sr']@q, f['wall']@q+f['offset']))
        def jac(q):
            check()
            x=f['dx']@q; y=f['dy']@q
            d=np.maximum(np.sqrt(x*x+y*y),1e-15)
            return np.concatenate(((x/d)[:,None]*f['dx']+(y/d)[:,None]*f['dy']-f['sr'], f['wall']),axis=0)
        def callback(q):
            check()
            f['q'] = q.copy()
            a=np.column_stack((f['X']@q,f['Y']@q,f['R']@q))
            retain(a)
            v=con(q)
            if np.min(v)>-1e-6:
                f['score']=max(f['score'],float(-f['obj']@q))
        try:
            res=minimize(fun,f['q'],jac=lambda q:f['obj'],bounds=f['bounds'],constraints={'type':'ineq','fun':con,'jac':jac},method='SLSQP',callback=callback,options={'maxiter':iterations,'ftol':2e-12,'disp':False})
            f['q']=res.x.copy()
            retain(np.column_stack((f['X']@res.x,f['Y']@res.x,f['R']@res.x)))
        except StopSearch:
            pass

    families=[]
    layouts=[(4,5,4,5,3),(4,5,3,5,4),(7,7,7),(5,4,3,4,5),(5,6,5,5)]
    try:
        for rows in layouts:
            if time.monotonic() >= deadline:
                break
            f=family(rows)
            families.append(f)
            run(f,float(params['stage_seconds']))
        while time.monotonic() < deadline and families:
            families.sort(key=lambda f:f['score'],reverse=True)
            f=families[0]
            previous=f['q'].copy()
            run(f,min(0.24,deadline-time.monotonic()),120)
            if np.max(np.abs(f['q']-previous)) < 1e-8:
                if len(families)>1 and rng.random()<0.45:
                    f=families[1]
                q=f['q'].copy()
                q += rng.normal(0,0.002,len(q))
                for j,b in enumerate(f['bounds']):
                    if time.monotonic() >= deadline:
                        raise StopSearch()
                    if b[0] is not None:
                        q[j]=max(b[0]+1e-7,min(b[1]-1e-7,q[j]))
                f['q']=q
    except StopSearch:
        pass
    return best
