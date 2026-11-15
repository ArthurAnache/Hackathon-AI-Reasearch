import time
import numpy as np
from fractions import Fraction
from scipy.optimize import minimize

PARAMS = {"iterations": (160, 40, 300), "safety": (3e-13, 1e-14, 1e-9, "log")}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + 0.70 * budget_s
    safety = params['safety']
    ii, jj = np.triu_indices(21, 1)
    best = init
    best_score = -float('inf')
    best_array = None

    def exact(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return -float('inf')
        d = np.sqrt((a[ii, 0]-a[jj, 0])**2 + (a[ii, 1]-a[jj, 1])**2)
        if np.any(d < a[ii, 2]+a[jj, 2]):
            return -float('inf')
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        w = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        h = max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        if w+h > 2:
            return -float('inf')
        return float(np.sum(a[:, 2]))

    if init is not None:
        try:
            if 'circles' in init:
                a = np.asarray(init['circles'], dtype=float)
            else:
                a = np.asarray(init['state'], dtype=float).reshape(3,21).T.copy()
            best_score = exact(a)
            if np.isfinite(best_score):
                best_array = a.copy()
        except Exception:
            pass

    def consider(a):
        nonlocal best, best_score, best_array
        if time.monotonic() >= deadline:
            return
        a = np.array(a, dtype=float, copy=True)
        if a.shape != (21,3) or not np.isfinite(a).all() or np.min(a[:,2]) <= 0:
            return
        d = np.sqrt((a[ii,0]-a[jj,0])**2+(a[ii,1]-a[jj,1])**2)
        ratio = min(1.0, float(np.min(d/(a[ii,2]+a[jj,2]))))
        a[:,2] *= ratio*(1-safety)
        if np.min(a[:,2]) <= 0:
            return
        lo = np.min(a[:,:2]-a[:,2,None], axis=0)
        hi = np.max(a[:,:2]+a[:,2,None], axis=0)
        a[:,:2] -= (lo+hi)/2
        a *= (2.0/(hi-lo).sum())*(1-safety)
        if float(a[:,2].sum()) <= best_score:
            return
        s = exact(a)
        if s > best_score:
            best_score = s
            best_array = a.copy()
            best = {'circles': a.tolist()}

    def symmetric(rows):
        groups = []
        entries = []
        for row,k in enumerate(rows):
            if time.monotonic() >= deadline:
                raise StopSearch
            if k % 2:
                g = len(groups)
                groups.append((0., row*np.sqrt(3.), 1., False))
                entries.append((g,0.))
            for j in range(k//2):
                if time.monotonic() >= deadline:
                    raise StopSearch
                x = (2*j+1) if k%2 == 0 else (2*j+2)
                g = len(groups)
                groups.append((float(x), row*np.sqrt(3.), 1., True))
                entries.extend([(g,1.),(g,-1.)])
        ng = len(groups)
        nx = sum(int(g[3]) for g in groups)
        nv = nx+2*ng+1
        X = np.zeros((21,nv)); Y = X.copy(); R = X.copy()
        q = np.zeros(nv)
        xids = {}; p = 0
        for g, item in enumerate(groups):
            if time.monotonic() >= deadline:
                raise StopSearch
            if item[3]:
                xids[g] = p
                q[p] = item[0]
                p += 1
            q[nx+g] = item[1]
            q[nx+ng+g] = 1.
        for i,(g,sgn) in enumerate(entries):
            if time.monotonic() >= deadline:
                raise StopSearch
            if sgn:
                X[i,xids[g]] = sgn
            Y[i,nx+g] = 1.
            R[i,nx+ng+g] = 1.
        q[nx:nx+ng] -= (len(rows)-1)*np.sqrt(3.)/2
        arr = np.column_stack((X@q,Y@q,R@q))
        width = np.ptp(arr[:,0])+2
        height = np.ptp(arr[:,1])+2
        q[:-1] *= 2/(width+height)
        q[-1] = 2*width/(width+height)
        bounds = [(0.,1.)]*nx+[(-1.,1.)]*ng+[(1e-6,0.6)]*ng+[(0.1,1.9)]
        return X,Y,R,q,bounds

    def solve(X,Y,R,q,bounds,until):
        nv = len(q)
        dx = X[ii]-X[jj]
        dy = Y[ii]-Y[jj]
        rs = R[ii]+R[jj]
        keep = []
        seen = set()
        for k in range(len(ii)):
            if time.monotonic() >= until:
                raise StopSearch
            ax = dx[k].copy(); ay = dy[k].copy()
            nz = np.flatnonzero(ax)
            if len(nz) and ax[nz[0]] < 0:
                ax = -ax
            nz = np.flatnonzero(ay)
            if len(nz) and ay[nz[0]] < 0:
                ay = -ay
            key = (ax.tobytes(),ay.tobytes(),rs[k].tobytes())
            if key not in seen:
                seen.add(key); keep.append(k)
        dx=dx[keep]; dy=dy[keep]; rs=rs[keep]
        e = np.zeros(nv); e[-1]=0.5
        W = np.vstack((e-X-R,e+X-R,-e-Y-R,-e+Y-R))
        c = np.concatenate((np.zeros(42),np.ones(42)))
        ids = []; seen=set()
        for k in range(len(W)):
            if time.monotonic() >= until:
                raise StopSearch
            key=(W[k].tobytes(),float(c[k]))
            if key not in seen:
                seen.add(key); ids.append(k)
        W=W[ids]; c=c[ids]
        objective = -R.sum(axis=0)
        cache_q=None; cache_v=None; cache_j=None
        last_considered = [0.]
        def calc(z):
            nonlocal cache_q,cache_v,cache_j
            if time.monotonic() >= until:
                raise StopSearch
            if cache_q is None or not np.array_equal(z,cache_q):
                a=dx@z; b=dy@z
                d=np.sqrt(a*a+b*b)
                inv=1/np.maximum(d,1e-15)
                cache_v=np.concatenate((d-rs@z,W@z+c))
                cache_j=np.vstack(((a*inv)[:,None]*dx+(b*inv)[:,None]*dy-rs,W))
                cache_q=z.copy()
            return cache_v,cache_j
        def callback(z):
            now=time.monotonic()
            if now >= until:
                raise StopSearch
            if now-last_considered[0] > 0.014:
                consider(np.column_stack((X@z,Y@z,R@z)))
                last_considered[0]=now
        consider(np.column_stack((X@q,Y@q,R@q)))
        try:
            result=minimize(lambda z: float(objective@z),q,jac=lambda z:objective,
                bounds=bounds,constraints={'type':'ineq','fun':lambda z:calc(z)[0],'jac':lambda z:calc(z)[1]},
                method='SLSQP',callback=callback,
                options={'maxiter':params['iterations'],'ftol':2e-12,'disp':False})
            consider(np.column_stack((X@result.x,Y@result.x,R@result.x)))
        except StopSearch:
            pass

    families = [[3,4,3,4,3,4],[5,4,3,4,5],[7,7,7]]
    for k,rows in enumerate(families):
        if time.monotonic() >= started+budget_s*0.53:
            break
        try:
            X,Y,R,q,bounds=symmetric(rows)
            limit=min(deadline,started+budget_s*([0.27,0.49,0.56][k]))
            solve(X,Y,R,q,bounds,limit)
        except StopSearch:
            pass
    if best_array is not None and time.monotonic() < deadline-0.025:
        a=best_array.copy()
        X=np.zeros((21,64)); Y=X.copy(); R=X.copy()
        X[np.arange(21),np.arange(21)]=1.
        Y[np.arange(21),21+np.arange(21)]=1.
        R[np.arange(21),42+np.arange(21)]=1.
        width=np.max(a[:,0]+a[:,2])-np.min(a[:,0]-a[:,2])
        q=np.concatenate((a[:,0],a[:,1],a[:,2],[width]))
        bounds=[(-1.,1.)]*42+[(1e-7,0.6)]*21+[(0.05,1.95)]
        try:
            solve(X,Y,R,q,bounds,deadline)
        except StopSearch:
            pass
    if best is None:
        r=0.099999999999
        a=np.array([[(i%7-3)*0.2,(i//7-1)*0.2,r] for i in range(21)])
        best={'circles':a.tolist()}
    return best
