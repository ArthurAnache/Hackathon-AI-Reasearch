import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'removed': (5, 3, 8),
    'iterations': (90, 20, 200),
    'diversity': (0.025, 0.0, 0.12),
}
N = 21
II, JJ = np.triu_indices(N, 1)

class StopSearch(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[II, 0]-a[JJ, 0])**2 + (a[II, 1]-a[JJ, 1])**2)
    if np.any(d < a[II, 2]+a[JJ, 2]):
        return -np.inf
    x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
    s = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
    s += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
    return float(a[:, 2].sum()) if s <= 2 else -np.inf

def repaired(a):
    a = a.copy()
    if not np.isfinite(a).all():
        return None
    d = np.sqrt((a[II, 0]-a[JJ, 0])**2+(a[II, 1]-a[JJ, 1])**2)
    excess = np.max(a[II, 2]+a[JJ, 2]-d)
    a[:, 2] -= max(0.0, excess/2) + 3e-15
    if np.any(a[:, 2] <= 0):
        return None
    lo = np.min(a[:, :2]-a[:, 2,None], axis=0)
    hi = np.max(a[:, :2]+a[:, 2,None], axis=0)
    scale = (2.0-3e-14)/float(np.sum(hi-lo))
    a[:, :2] = (a[:, :2]-lo)*scale
    a[:, 2] *= scale
    a[:, 2] *= 1.0-2e-14
    return a

def cavity(disks, w, h, rng, diversity, deadline):
    if time.monotonic() >= deadline:
        raise StopSearch
    m = len(disks)
    triples = np.array(list(itertools.combinations(range(m+4), 3)), dtype=int)
    # Each triple yields two equations p = a + b*r, followed by one circle equation.
    rows = len(triples)
    mat = np.zeros((rows,2,2))
    rhs = np.zeros((rows,2))
    slope = np.zeros((rows,2))
    anchor = np.zeros((rows,3))
    used = np.zeros(rows, dtype=int)
    valid = np.ones(rows, dtype=bool)
    wallv = np.array([[1.,0.],[1.,0.],[0.,1.],[0.,1.]])
    wallrhs = np.array([0.,w,0.,h])
    wallr = np.array([1.,-1.,1.,-1.])
    # Sorted triples put obstacle circles before walls.
    has = triples[:,0] < m
    valid &= has
    anchor[has] = disks[triples[has,0]]
    for k in range(1,3):
        if time.monotonic() >= deadline:
            raise StopSearch
        ids = triples[:,k]
        circ = has & (ids < m)
        wall = has & ~circ
        ci = np.where(circ)[0]
        if len(ci):
            q = disks[ids[ci]]
            p = anchor[ci]
            mat[ci,k-1] = 2*(q[:,:2]-p[:,:2])
            rhs[ci,k-1] = np.sum(q[:,:2]**2-p[:,:2]**2,axis=1)-q[:,2]**2+p[:,2]**2
            slope[ci,k-1] = -2*(q[:,2]-p[:,2])
        wi = np.where(wall)[0]
        if len(wi):
            j = ids[wi]-m
            mat[wi,k-1] = wallv[j]
            rhs[wi,k-1] = wallrhs[j]
            slope[wi,k-1] = wallr[j]
    det = mat[:,0,0]*mat[:,1,1]-mat[:,0,1]*mat[:,1,0]
    valid &= np.abs(det)>1e-12
    ix = np.where(valid)[0]
    if not len(ix):
        return None
    t = mat[ix]
    inv = np.stack([t[:,1,1],-t[:,0,1],-t[:,1,0],t[:,0,0]],axis=1).reshape(-1,2,2)/det[ix,None,None]
    aa = np.einsum('nij,nj->ni',inv,rhs[ix])
    bb = np.einsum('nij,nj->ni',inv,slope[ix])
    cc = aa-anchor[ix,:2]
    ar = anchor[ix,2]
    qa = np.sum(bb*bb,axis=1)-1
    qb = 2*(np.sum(bb*cc,axis=1)-ar)
    qc = np.sum(cc*cc,axis=1)-ar*ar
    disc = qb*qb-4*qa*qc
    good = (disc>=0)&(np.abs(qa)>1e-13)
    aa,bb,qa,qb,disc = aa[good],bb[good],qa[good],qb[good],disc[good]
    roots = np.concatenate([(-qb+np.sqrt(disc))/(2*qa),(-qb-np.sqrt(disc))/(2*qa)])
    centers = np.concatenate([aa,aa])+np.concatenate([bb,bb])*roots[:,None]
    good = (roots>1e-7)&(centers[:,0]>=0)&(centers[:,0]<=w)&(centers[:,1]>=0)&(centers[:,1]<=h)
    centers = centers[good]
    # Include corners of the rectangle's inset geometry through wall-wall-circle triples above.
    if not len(centers):
        return None
    clearance = np.min(np.sqrt(np.sum((centers[:,None,:]-disks[None,:,:2])**2,axis=2))-disks[None,:,2],axis=1)
    clearance = np.minimum(clearance,np.minimum.reduce([centers[:,0],w-centers[:,0],centers[:,1],h-centers[:,1]]))
    best = float(clearance.max())
    eligible = np.where(clearance >= best*(1-diversity))[0]
    if best <= 1e-7:
        return None
    j = int(rng.choice(eligible))
    return np.r_[centers[j],clearance[j]*(1-1e-12)]

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.69*budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a = np.array([[(i%7+.5)/3.5,(i//7+.5)/1.5,0.065] for i in range(21)])
        a = repaired(a)
        init = {'circles':a.tolist()}
    raw = np.asarray(init['circles'], dtype=float)
    best = init
    best_score = exact_score(raw)
    lo = np.min(raw[:,:2]-raw[:,2,None],axis=0)
    hi = np.max(raw[:,:2]+raw[:,2,None],axis=0)
    factor = 2/np.sum(hi-lo)
    base = raw.copy()
    base[:,:2] = (base[:,:2]-lo)*factor
    base[:,2] *= factor
    w,h = (hi-lo)*factor
    objective = np.r_[np.zeros(42),-np.ones(21),0.]
    def check():
        if time.monotonic() >= deadline:
            raise StopSearch
    def consider(a):
        nonlocal best,best_score
        check()
        a = repaired(a)
        if a is not None:
            score = exact_score(a)
            if score > best_score:
                best_score = score
                best = {'circles':a.tolist()}
    def fun(z):
        check()
        return -z[42:63].sum(),objective
    def cons(z):
        check()
        x,y,r = z[:21],z[21:42],z[42:63]
        dx,dy = x[II]-x[JJ],y[II]-y[JJ]
        rr = r[II]+r[JJ]
        return np.r_[dx*dx+dy*dy-rr*rr,x-r,z[63]-x-r,y-r,2-z[63]-y-r]
    def jac(z):
        check()
        x,y,r = z[:21],z[21:42],z[42:63]
        j = np.zeros((294,64))
        ar = np.arange(210)
        j[ar,II]=2*(x[II]-x[JJ]);j[ar,JJ]=-j[ar,II]
        j[ar,21+II]=2*(y[II]-y[JJ]);j[ar,21+JJ]=-j[ar,21+II]
        j[ar,42+II]=-2*(r[II]+r[JJ]);j[ar,42+JJ]=j[ar,42+II]
        for k in range(4):
            check()
            rows=210+k*21+np.arange(21)
            j[rows,42+np.arange(21)]=-1
            j[rows,(0 if k<2 else 21)+np.arange(21)]=1 if k%2==0 else -1
            if k==1:j[rows,63]=1
            if k==3:j[rows,63]=-1
        return j
    last = [None]
    def callback(z):
        check()
        last[0]=z.copy()
        if np.max(-cons(z))<2e-7:
            consider(np.column_stack([z[:21],z[21:42],z[42:63]]))
    trial=0
    try:
        while time.monotonic()<deadline:
            k=int(params['removed'])
            anchor=int(rng.integers(21))
            distances=np.linalg.norm(base[:,:2]-base[anchor,:2],axis=1)
            if trial%3==0:
                removed=np.argsort(distances)[:k]
            elif trial%3==1:
                order=np.argsort(np.arctan2(base[:,1]-h/2,base[:,0]-w/2))
                removed=np.roll(order,-int(rng.integers(21)))[:k]
            else:
                removed=rng.choice(21,k,replace=False)
            disks=np.delete(base,removed,axis=0)
            success=True
            for q in range(k):
                check()
                new=cavity(disks,w,h,rng,float(params['diversity']) if q<k-2 else 0.,deadline)
                if new is None:
                    success=False
                    break
                disks=np.vstack([disks,new])
            if not success:
                trial+=1
                continue
            consider(disks)
            z=np.r_[disks[:,0],disks[:,1],disks[:,2],w]
            last[0]=None
            result=minimize(fun,z,jac=True,method='SLSQP',bounds=[(0,2)]*42+[(1e-7,1)]*21+[(0.15,1.85)],constraints={'type':'ineq','fun':cons,'jac':jac},callback=callback,options={'maxiter':int(params['iterations']),'ftol':2e-12,'disp':False})
            z=result.x
            consider(np.column_stack([z[:21],z[21:42],z[42:63]]))
            trial+=1
    except StopSearch:
        pass
    return best
