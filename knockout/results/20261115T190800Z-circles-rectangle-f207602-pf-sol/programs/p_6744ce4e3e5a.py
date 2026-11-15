import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"iterations": (130, 50, 250), "polish_fraction": (0.23, 0.1, 0.4)}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70 * budget_s
    ii, jj = np.triu_indices(21, 1)
    best = None
    best_score = -float('inf')

    def exact(candidate):
        try:
            a = np.asarray(candidate['circles'], dtype=float)
            if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
                return -float('inf')
            d = a[ii, :2] - a[jj, :2]
            if np.any(np.sqrt(d[:, 0]**2 + d[:, 1]**2) < a[ii, 2] + a[jj, 2]):
                return -float('inf')
            xs = [Fraction(float(v)) for v in a[:, 0]]
            ys = [Fraction(float(v)) for v in a[:, 1]]
            rs = [Fraction(float(v)) for v in a[:, 2]]
            w = max(x+r for x, r in zip(xs, rs))-min(x-r for x, r in zip(xs, rs))
            h = max(y+r for y, r in zip(ys, rs))-min(y-r for y, r in zip(ys, rs))
            if w+h > 2:
                return -float('inf')
            return float(np.sum(a[:, 2]))
        except Exception:
            return -float('inf')

    if init is not None:
        best_score = exact(init)
        if np.isfinite(best_score):
            best = init
    best_array = None
    numerical_best = best_score

    def repair(a):
        a = np.array(a, dtype=float, copy=True)
        if not np.isfinite(a).all():
            return None
        a[:, 2] = np.maximum(a[:, 2], 1e-10)
        d = a[ii, :2]-a[jj, :2]
        distances = np.sqrt(d[:, 0]**2+d[:, 1]**2)
        f = min(1.0, float(np.min(distances/(a[ii, 2]+a[jj, 2]))))
        if f <= 0:
            return None
        a[:, 2] *= f * (1-3e-13)
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        span = float(np.sum(hi-lo))
        if span <= 0:
            return None
        a[:, :2] -= (lo+hi)/2
        a *= (2-5e-13)/span
        return a

    def remember(a):
        nonlocal best_array, numerical_best
        a = repair(a)
        if a is not None:
            s = float(np.sum(a[:, 2]))
            if s > numerical_best:
                numerical_best = s
                best_array = a

    def model(counts=None):
        if counts is None:
            n = 64
            X = np.zeros((21,n)); Y = X.copy(); R = X.copy()
            X[np.arange(21),np.arange(21)] = 1
            Y[np.arange(21),21+np.arange(21)] = 1
            R[np.arange(21),42+np.arange(21)] = 1
            return X,Y,R
        groups = sum((c+1)//2 for c in counts)
        pairs = sum(c//2 for c in counts)
        n = 2*groups+pairs+1
        X = np.zeros((21,n)); Y = X.copy(); R = X.copy()
        row = 0; g = 0; p = 0
        for c in counts:
            if time.monotonic() >= end:
                raise StopSearch
            if c % 2:
                X[row,g]=1; R[row,groups+pairs+g]=1
                row += 1; g += 1
            for k in range(c//2):
                if time.monotonic() >= end:
                    raise StopSearch
                X[row:row+2,g]=1
                Y[row,groups+p]=1; Y[row+1,groups+p]=-1
                R[row:row+2,groups+pairs+g]=1
                row += 2; g += 1; p += 1
        return X,Y,R

    def run(X,Y,R,q,deadline):
        n = len(q)
        DX = X[ii]-X[jj]; DY = Y[ii]-Y[jj]; SR = R[ii]+R[jj]
        # Remove contacts duplicated by the imposed reflection.
        keep=[]; seen=set()
        for k in range(len(ii)):
            if time.monotonic() >= deadline:
                raise StopSearch
            x=DX[k]; y=DY[k]
            tx=tuple(x); ty=tuple(y)
            key=(min(tx,tuple(-x)),min(ty,tuple(-y)),tuple(SR[k]))
            if key not in seen:
                keep.append(k); seen.add(key)
        DX=DX[keep]; DY=DY[keep]; SR=SR[keep]
        wall = np.concatenate((-X-R,X-R,-Y-R,Y-R))
        wall[:42,-1] += .5
        wall[42:,-1] -= .5
        rhs=np.concatenate((np.zeros(42),np.ones(42)))
        _, inds=np.unique(np.column_stack((wall,rhs)),axis=0,return_index=True)
        wall=wall[inds]; rhs=rhs[inds]
        objective=-np.sum(R,axis=0)
        cache_q=None; cache_c=None; cache_j=None
        last=q.copy()
        def unpack(z):
            return np.column_stack((X@z,Y@z,R@z))
        def calc(z):
            nonlocal cache_q,cache_c,cache_j
            if time.monotonic() >= deadline:
                raise StopSearch
            if cache_q is None or not np.array_equal(cache_q,z):
                dx=DX@z; dy=DY@z
                dist=np.sqrt(dx*dx+dy*dy)
                safe=np.maximum(dist,1e-15)
                cache_c=np.concatenate((dist-SR@z,wall@z+rhs))
                cache_j=np.vstack((DX*(dx/safe)[:,None]+DY*(dy/safe)[:,None]-SR,wall))
                cache_q=z.copy()
            return cache_c
        def jac(z):
            calc(z)
            return cache_j
        def callback(z):
            nonlocal last
            if time.monotonic() >= deadline:
                raise StopSearch
            last=z.copy()
            remember(unpack(z))
        bounds=[(None,None)]*n
        for k in np.flatnonzero(np.sum(R,axis=0)):
            if time.monotonic() >= deadline:
                raise StopSearch
            bounds[k]=(1e-7,.6)
        bounds[-1]=(.15,1.85)
        remember(unpack(q))
        try:
            result=minimize(lambda z: float(objective@z),q,jac=lambda z:objective,
                constraints={'type':'ineq','fun':calc,'jac':jac},bounds=bounds,
                method='SLSQP',callback=callback,
                options={'maxiter':int(params['iterations']),'ftol':2e-12,'disp':False})
            last=result.x
            remember(unpack(last))
        except StopSearch:
            pass
        return last

    families=[[3,4,3,4,3,4],[5,4,3,4,5],[3,4,4,3,4,3]]
    quotient_end=start+(end-start)*(1-float(params['polish_fraction']))
    try:
        for index,counts in enumerate(families):
            now=time.monotonic()
            if now >= quotient_end:
                break
            X,Y,R=model(counts)
            groups=sum((c+1)//2 for c in counts)
            pairs=sum(c//2 for c in counts)
            q=np.zeros(X.shape[1]); g=0; p=0; xpos=0.0
            for col,c in enumerate(counts):
                if time.monotonic() >= quotient_end:
                    raise StopSearch
                if col:
                    xpos += np.sqrt(3.) if (c-counts[col-1])%2 else 2.0
                if c%2:
                    q[g]=xpos; q[groups+pairs+g]=1
                    g+=1
                for k in range(c//2):
                    if time.monotonic() >= quotient_end:
                        raise StopSearch
                    q[g]=xpos
                    q[groups+p]=2*k+(2 if c%2 else 1)
                    q[groups+pairs+g]=1
                    p+=1; g+=1
            a=np.column_stack((X@q,Y@q,R@q))
            lo=np.min(a[:,:2]-a[:,2,None],axis=0)
            hi=np.max(a[:,:2]+a[:,2,None],axis=0)
            q[:groups]-=(lo[0]+hi[0])/2
            scale=2/float(np.sum(hi-lo))
            q*=scale
            q[-1]=(hi[0]-lo[0])*scale
            remaining=len(families)-index
            deadline=min(quotient_end,now+(quotient_end-now)/remaining)
            run(X,Y,R,q,deadline)
    except StopSearch:
        pass
    if best_array is not None and time.monotonic() < end-.025:
        try:
            X,Y,R=model()
            a=best_array
            w=float(np.max(a[:,0]+a[:,2])-np.min(a[:,0]-a[:,2]))
            q=np.concatenate((a[:,0],a[:,1],a[:,2],[w]))
            run(X,Y,R,q,end-.015)
        except StopSearch:
            pass
    if best_array is not None:
        candidate={'circles':best_array.tolist()}
        score=exact(candidate)
        if score > best_score:
            best=candidate; best_score=score
    if best is None:
        a=np.array([[.2*(i%7-3),.2*(i//7-1),.09] for i in range(21)])
        a=repair(a)
        best={'circles':a.tolist()}
    return best
