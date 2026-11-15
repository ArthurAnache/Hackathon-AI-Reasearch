import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"iterations": (115, 40, 220), "fraction": (0.70, 0.4, 0.8), "perturb": (0.012, 0.0, 0.06)}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + float(params['fraction']) * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -float('inf')

    def canonical(obj):
        if obj is None:
            return None
        if 'circles' in obj:
            return obj
        return {'circles': [[float(p[0]), float(p[1]), float(r)] for p, r in zip(obj['centers'], obj['radii'])]}

    def exact_score(obj):
        try:
            c = np.asarray(canonical(obj)['circles'], dtype=float)
            if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
                return -float('inf')
            q = [[Fraction(float(v)) for v in row] for row in c]
            w = max(a+r for a,b,r in q)-min(a-r for a,b,r in q)
            h = max(b+r for a,b,r in q)-min(b-r for a,b,r in q)
            if w+h > 2:
                return -float('inf')
            for i in range(21):
                if time.monotonic() > start + .95*budget_s:
                    return -float('inf')
                d = c[i+1:, :2]-c[i, :2]
                if np.any(np.sqrt(d[:,0]**2+d[:,1]**2) < c[i+1:,2]+c[i,2]):
                    return -float('inf')
            return float(np.sum(c[:,2]))
        except Exception:
            return -float('inf')

    best_score = exact_score(init) if init is not None else -float('inf')

    def consider(c):
        nonlocal best, best_score
        if time.monotonic() > start + .88*budget_s:
            return
        c = np.array(c, dtype=float, copy=True)
        if c.shape != (21,3) or not np.isfinite(c).all() or np.any(c[:,2] <= 0):
            return
        ii,jj = np.triu_indices(21,1)
        d = c[ii,:2]-c[jj,:2]
        ratio = np.min(np.sqrt(np.sum(d*d,axis=1))/(c[ii,2]+c[jj,2]))
        c[:,2] *= min(1.,ratio)*(1.-3e-14)
        lo = np.min(c[:,:2]-c[:,2,None],axis=0)
        hi = np.max(c[:,:2]+c[:,2,None],axis=0)
        c[:,:2] -= (lo+hi)/2
        c *= (2./np.sum(hi-lo))*(1.-3e-14)
        obj = {'centers': c[:,:2].tolist(), 'radii': c[:,2].tolist()}
        s = exact_score(obj)
        if s > best_score:
            best_score, best = s, obj

    def solve(points):
        if time.monotonic() >= deadline:
            return
        singles = [p for p in points if abs(p[1]) < 1e-12]
        pairs = [p for p in points if p[1] > 1e-12]
        reps = np.array(singles+pairs, dtype=float)
        m = len(reps); a = len(singles); p = m-a
        if a+2*p != 21:
            return
        n = 3*m+1
        # Variables are representative x, y, radius, then rectangle width.
        lo = np.min(reps[:,0]-reps[:,2]); hi = np.max(reps[:,0]+reps[:,2])
        half = np.max(reps[:,1]+reps[:,2])
        scale = 2/(hi-lo+2*half)
        reps[:,0] = (reps[:,0]-lo)*scale
        reps[:,1:] *= scale
        z = np.r_[reps[:,0], reps[:,1], reps[:,2], (hi-lo)*scale]
        ai=[]; bi=[]; sg=[]
        for i in range(m):
            if time.monotonic() >= deadline:
                return
            for j in range(i+1,m):
                if time.monotonic() >= deadline:
                    return
                ai.append(i); bi.append(j); sg.append(1.)
                if i >= a:
                    ai.append(i); bi.append(j); sg.append(-1.)
        ai=np.array(ai,dtype=int); bi=np.array(bi,dtype=int); sg=np.array(sg)
        k=len(ai); rows=np.arange(k)
        linear=[]; const=[]
        for i in range(m):
            if time.monotonic() >= deadline:
                return
            v=np.zeros(n); v[i]=1; v[2*m+i]=-1
            linear.append(v); const.append(0.)
            v=np.zeros(n); v[-1]=1; v[i]=-1; v[2*m+i]=-1
            linear.append(v); const.append(0.)
            v=np.zeros(n); v[-1]=-.5; v[m+i]=-1; v[2*m+i]=-1
            linear.append(v); const.append(1.)
            if i>=a:
                v=np.zeros(n); v[m+i]=1; v[2*m+i]=-1
                linear.append(v); const.append(0.)
        L=np.array(linear); cc=np.array(const)
        obj=np.zeros(n); obj[2*m:3*m]=-2.; obj[2*m:2*m+a]=-1.
        cache_z=None; cache_v=None; cache_j=None
        last=[z.copy()]
        def compute(v):
            nonlocal cache_z,cache_v,cache_j
            if time.monotonic() >= deadline:
                raise StopSearch()
            if cache_z is not None and np.array_equal(v,cache_z):
                return
            dx=v[ai]-v[bi]
            dy=v[m+ai]-sg*v[m+bi]
            rr=v[2*m+ai]+v[2*m+bi]
            vals=dx*dx+dy*dy-rr*rr
            jac=np.zeros((k,n))
            jac[rows,ai]=2*dx; jac[rows,bi]=-2*dx
            jac[rows,m+ai]=2*dy; jac[rows,m+bi]=-2*sg*dy
            jac[rows,2*m+ai]=-2*rr; jac[rows,2*m+bi]=-2*rr
            cache_z=v.copy(); cache_v=np.r_[vals,L@v+cc]; cache_j=np.vstack((jac,L))
        def fun(v):
            compute(v)
            return cache_v
        def jac(v):
            compute(v)
            return cache_j
        def callback(v):
            if time.monotonic() >= deadline:
                raise StopSearch()
            last[0]=v.copy()
        bounds=[(0.,2.)]*m+[(0.,0.)]*a+[(0.,1.)]*p+[(1e-7,.7)]*m+[(.15,1.85)]
        try:
            res=minimize(lambda v: float(obj@v),z,jac=lambda v:obj,
                bounds=bounds,constraints={'type':'ineq','fun':fun,'jac':jac},
                method='SLSQP',callback=callback,
                options={'maxiter':int(params['iterations']),'ftol':2e-12,'disp':False})
            last[0]=res.x
        except StopSearch:
            pass
        v=last[0]
        out=np.column_stack((v[:m],v[m:2*m],v[2*m:3*m]))
        full=np.vstack((out,out[a:]*np.array([1.,-1.,1.])))
        consider(full)

    def row_seed(counts, noise=0.):
        points=[]; x=1.
        prev=None
        for count in counts:
            if time.monotonic() >= deadline:
                raise StopSearch()
            if prev is not None:
                x += np.sqrt(3.) if (prev-count)%2 else 2.
            for j in range(count):
                if time.monotonic() >= deadline:
                    raise StopSearch()
                y=2*j-(count-1)
                if y>=0:
                    points.append([x+noise*rng.normal(),float(y),1.+noise*rng.normal()])
            prev=count
        return points

    layouts=[[3,4,3,4,3,4],[5,4,3,4,5],[4,5,4,5,3],
             [4,5,4,4,4],[3,3,3,3,3,3,3],[5,6,5,5],
             [4,3,4,3,4,3],[3,5,5,5,3],[2,3,4,3,4,3,2]]
    try:
        for index, layout in enumerate(layouts):
            if time.monotonic() >= deadline:
                break
            pts=row_seed(layout,0. if index<3 else float(params['perturb']))
            solve(pts)
    except StopSearch:
        pass
    if best_score == -float('inf'):
        c=np.array([[.1+.2*(i%7),.1+.2*(i//7),.099] for i in range(21)])
        consider(c)
    return best
