import time
import math
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"family_time": (0.30, 0.15, 0.5), "iterations": (120, 50, 240)}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -float('inf')
    ia, ib = np.triu_indices(21, 1)

    def exact(c):
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return -float('inf')
        d = np.sqrt((c[ia, 0]-c[ib, 0])**2 + (c[ia, 1]-c[ib, 1])**2)
        if np.any(d < c[ia, 2]+c[ib, 2]):
            return -float('inf')
        xx = [Fraction(float(v)) for v in c[:, 0]]
        yy = [Fraction(float(v)) for v in c[:, 1]]
        rr = [Fraction(float(v)) for v in c[:, 2]]
        per = max(x+r for x,r in zip(xx,rr))-min(x-r for x,r in zip(xx,rr))
        per += max(y+r for y,r in zip(yy,rr))-min(y-r for y,r in zip(yy,rr))
        return float(np.sum(c[:, 2])) if per <= 2 else -float('inf')

    if init is not None:
        try:
            best_score = exact(np.array(init['circles'], dtype=float))
        except Exception:
            pass

    def retain(c):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            raise StopSearch()
        c = np.array(c, dtype=float, copy=True)
        if not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return
        c[:, :2] -= np.mean(c[:, :2], axis=0)
        d = np.sqrt(np.sum((c[ia, :2]-c[ib, :2])**2, axis=1))
        factor = min(1.0, float(np.min(d/(c[ia, 2]+c[ib, 2]))))
        c[:, 2] *= factor * (1.0-3e-14)
        p = np.ptp(c[:, 0]+0.0)  # bounding dimensions include individual radii below
        p = np.max(c[:, 0]+c[:, 2])-np.min(c[:, 0]-c[:, 2])
        p += np.max(c[:, 1]+c[:, 2])-np.min(c[:, 1]-c[:, 2])
        if p <= 0:
            return
        c *= (2.0/p)*(1.0-3e-14)
        if float(np.sum(c[:, 2])) > best_score:
            score = exact(c)
            if score > best_score:
                best_score = score
                best = {'circles': c.tolist()}

    saved = []

    def family(rows, allocation, warm=None):
        if time.monotonic() >= deadline:
            return
        local_end = min(deadline, time.monotonic()+allocation)
        groups = []
        coords = []
        for row, count in enumerate(rows):
            if time.monotonic() >= deadline:
                return
            if count % 2:
                groups.append(False)
                coords.append((0., row*math.sqrt(3)))
            for k in range(count//2):
                if time.monotonic() >= deadline:
                    return
                groups.append(True)
                coords.append((2*k+(1 if count % 2 == 0 else 2), row*math.sqrt(3)))
        ng = len(groups)
        pair_groups = [i for i,v in enumerate(groups) if v]
        nx = len(pair_groups)
        xindex = np.full(ng, -1, dtype=int)
        xindex[pair_groups] = np.arange(nx)
        dim = nx+2*ng+1
        X=[]; Y=[]; R=[]; mirror=[]
        for g, paired in enumerate(groups):
            if time.monotonic() >= deadline:
                return
            base = len(X)
            signs = [1., -1.] if paired else [0.]
            for s in signs:
                a=np.zeros(dim); b=np.zeros(dim); r=np.zeros(dim)
                if paired:
                    a[xindex[g]]=s
                b[nx+g]=1.; r[nx+ng+g]=1.
                X.append(a); Y.append(b); R.append(r)
            if paired:
                mirror.extend([base+1,base])
            else:
                mirror.append(base)
        X=np.array(X); Y=np.array(Y); R=np.array(R)
        pairs=[]
        for a,b in zip(ia,ib):
            if time.monotonic() >= deadline:
                return
            reflected=tuple(sorted((mirror[a],mirror[b])))
            if (int(a),int(b)) <= reflected:
                pairs.append((a,b))
        pa=np.array([p[0] for p in pairs]); pb=np.array([p[1] for p in pairs])
        DX=X[pa]-X[pb]; DY=Y[pa]-Y[pb]; SR=R[pa]+R[pb]
        walls=[]; offsets=[]
        for g in range(ng):
            if time.monotonic() >= deadline:
                return
            a=np.zeros(dim); a[-1]=.5; a[nx+ng+g]=-1
            if groups[g]: a[xindex[g]]=-1
            walls.append(a); offsets.append(0.)
            a=np.zeros(dim); a[nx+g]=1; a[nx+ng+g]=-1
            walls.append(a); offsets.append(0.)
            a=np.zeros(dim); a[-1]=-1; a[nx+g]=-1; a[nx+ng+g]=-1
            walls.append(a); offsets.append(2.)
        W=np.array(walls); off=np.array(offsets)
        objective=-np.sum(R,axis=0)
        z=np.zeros(dim)
        width=2*max(p[0]+1 for p in coords)
        height=(len(rows)-1)*math.sqrt(3)+2
        scale=2/(width+height)
        for g,(x,y) in enumerate(coords):
            if time.monotonic() >= deadline:
                return
            if groups[g]: z[xindex[g]]=x*scale
            z[nx+g]=(y+1)*scale
            z[nx+ng+g]=scale
        z[-1]=width*scale
        if warm is not None:
            z=warm.copy()
        bounds=[(0.,1.)]*nx+[(0.,2.)]*ng+[(1e-6,0.5)]*ng+[(.2,1.8)]
        cached_z=None; cached_c=None; cached_j=None
        def calculate(v):
            nonlocal cached_z,cached_c,cached_j
            if time.monotonic() >= local_end:
                raise StopSearch()
            if cached_z is None or not np.array_equal(v,cached_z):
                dx=DX@v; dy=DY@v
                distance=np.sqrt(dx*dx+dy*dy)
                safe=np.maximum(distance,1e-15)
                cached_c=np.concatenate((distance-SR@v,W@v+off))
                cached_j=np.vstack(((dx/safe)[:,None]*DX+(dy/safe)[:,None]*DY-SR,W))
                cached_z=v.copy()
            return cached_c,cached_j
        last=z.copy()
        def callback(v):
            nonlocal last
            if time.monotonic() >= local_end:
                raise StopSearch()
            last=v.copy()
            retain(np.column_stack((X@v,Y@v,R@v)))
        try:
            callback(z)
            result=minimize(lambda v: float(objective@v),z,jac=lambda v: objective,
                bounds=bounds, constraints={'type':'ineq','fun':lambda v: calculate(v)[0],
                'jac':lambda v: calculate(v)[1]},method='SLSQP',callback=callback,
                options={'maxiter':int(params['iterations']),'ftol':2e-12,'disp':False})
            last=result.x.copy()
            retain(np.column_stack((X@last,Y@last,R@last)))
        except StopSearch:
            pass
        if time.monotonic() < deadline:
            c=np.column_stack((X@last,Y@last,R@last))
            d=np.sqrt(np.sum((c[ia,:2]-c[ib,:2])**2,axis=1))
            repair=min(1.,float(np.min(d/np.maximum(c[ia,2]+c[ib,2],1e-15))))
            merit=float(np.sum(c[:,2]))*max(0.,repair)
            saved.append((merit,tuple(rows),last.copy()))

    families=[(4,5,4,5,3),(5,4,3,4,5),(4,5,3,5,4),(3,4,5,4,5)]
    try:
        for rows in families:
            if time.monotonic() >= deadline:
                break
            family(rows,float(params['family_time']))
        if saved and time.monotonic() < deadline:
            _,rows,z=max(saved,key=lambda p:p[0])
            family(rows,deadline-time.monotonic(),z)
    except StopSearch:
        pass
    if best_score == -float('inf'):
        c=[]
        for k in range(21):
            if time.monotonic() >= start+0.95*budget_s:
                break
            c.append([(k%5)*0.2,(k//5)*0.2,0.09])
        if len(c)==21:
            best={'circles':c}
    return best
