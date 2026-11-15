import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'removed': (3, 2, 5),
    'relax_iterations': (42, 10, 100),
    'finish_iterations': (110, 30, 250),
    'patch_bias': (0.65, 0.0, 1.0),
}

class Timeout(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    i, j = np.triu_indices(21, 1)
    d = np.sqrt((a[i, 0]-a[j, 0])**2 + (a[i, 1]-a[j, 1])**2)
    if np.any(d < a[i, 2]+a[j, 2]):
        return -np.inf
    x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
    perimeter = max(p+q for p,q in zip(x,r))-min(p-q for p,q in zip(x,r))
    perimeter += max(p+q for p,q in zip(y,r))-min(p-q for p,q in zip(y,r))
    if perimeter > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))

def repair(a):
    a = a.copy()
    if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return None
    n = len(a)
    i,j = np.triu_indices(n,1)
    d = np.sqrt(np.sum((a[i,:2]-a[j,:2])**2, axis=1))
    ratio = min(1.0, float(np.min(d/(a[i,2]+a[j,2]))))
    a[:,2] *= ratio*(1-3e-14)
    lo = np.min(a[:,:2]-a[:,2,None], axis=0)
    hi = np.max(a[:,:2]+a[:,2,None], axis=0)
    a[:,:2] -= lo
    a *= min(1.0, 2.0/float(np.sum(hi-lo)))*(1-3e-14)
    return a

def optimize(a, w, iterations, deadline):
    n = len(a)
    ii,jj = np.triu_indices(n,1)
    m = len(ii)
    z = np.r_[a[:,0], a[:,1], a[:,2], w]
    latest = [z.copy()]
    def check():
        if time.monotonic() >= deadline:
            raise Timeout()
    def objective(z):
        check()
        g = np.zeros(3*n+1)
        g[2*n:3*n] = -1
        return -np.sum(z[2*n:3*n]), g
    def con(z):
        check()
        x,y,r = z[:n],z[n:2*n],z[2*n:3*n]
        dx,dy = x[ii]-x[jj],y[ii]-y[jj]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        return np.r_[d-r[ii]-r[jj], x-r,z[-1]-x-r,y-r,2-z[-1]-y-r]
    def jac(z):
        check()
        x,y = z[:n],z[n:2*n]
        dx,dy = x[ii]-x[jj],y[ii]-y[jj]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        g = np.zeros((m+4*n,3*n+1))
        k = np.arange(m)
        g[k,ii]=dx/d; g[k,jj]=-dx/d
        g[k,n+ii]=dy/d; g[k,n+jj]=-dy/d
        g[k,2*n+ii]=-1; g[k,2*n+jj]=-1
        k = np.arange(n)
        g[m+k,k]=1; g[m+n+k,k]=-1
        g[m+2*n+k,n+k]=1; g[m+3*n+k,n+k]=-1
        for b in range(4):
            check()
            g[m+b*n+k,2*n+k]=-1
        g[m+n+k,-1]=1; g[m+3*n+k,-1]=-1
        return g
    def callback(z):
        latest[0] = z.copy()
        check()
    try:
        result = minimize(objective,z,jac=True,method='SLSQP',
            bounds=[(None,None)]*(2*n)+[(1e-7,1)]*n+[(0.15,1.85)],
            constraints={'type':'ineq','fun':con,'jac':jac},callback=callback,
            options={'maxiter':iterations,'ftol':2e-12,'disp':False})
        z = result.x
    except Timeout:
        z = latest[0]
    return np.column_stack((z[:n],z[n:2*n],z[2*n:3*n])),float(z[-1])

def insert_gap(a,w,deadline):
    n = len(a)
    h = 2-w
    if time.monotonic() >= deadline:
        raise Timeout()
    # Circle equations use an auxiliary t=x*x+y*y-r*r.
    rows = np.zeros((n+4,4))
    rhs = np.zeros(n+4)
    rows[:n,:3] = -2*a
    rows[:n,3] = 1
    rhs[:n] = a[:,2]**2-a[:,0]**2-a[:,1]**2
    rows[n:] = [[1,0,-1,0],[1,0,1,0],[0,1,-1,0],[0,1,1,0]]
    rhs[n:] = [0,w,0,h]
    triples = []
    for i in range(n):
        if time.monotonic() >= deadline:
            raise Timeout()
        triples.extend((i,j,k) for j,k in itertools.combinations(range(i+1,n+4),2))
    ids = np.asarray(triples,dtype=int)
    first,second,third = ids.T
    b = rows[second,:3]-rows[second,3,None]*rows[first,:3]
    c = rows[third,:3]-rows[third,3,None]*rows[first,:3]
    br = rhs[second]-rows[second,3]*rhs[first]
    cr = rhs[third]-rows[third,3]*rhs[first]
    bb = np.sum(b*b,axis=1); cc=np.sum(c*c,axis=1); bc=np.sum(b*c,axis=1)
    det=bb*cc-bc*bc
    good=det>1e-18
    b,c,br,cr,bb,cc,bc,det,first = [v[good] for v in (b,c,br,cr,bb,cc,bc,det,first)]
    u = ((br*cc-cr*bc)/det)[:,None]*b+((cr*bb-br*bc)/det)[:,None]*c
    v=np.cross(b,c)
    v/=np.sqrt(np.sum(v*v,axis=1))[:,None]
    delta=u-a[first]*np.array([1,1,-1])
    metric=np.array([1.,1.,-1.])
    qa=np.sum(v*v*metric,axis=1)
    qb=2*np.sum(delta*v*metric,axis=1)
    qc=np.sum(delta*delta*metric,axis=1)
    disc=qb*qb-4*qa*qc
    good=(disc>=0)&(np.abs(qa)>1e-12)
    u,v,qa,qb,disc=[t[good] for t in (u,v,qa,qb,disc)]
    roots=np.stack(((-qb+np.sqrt(disc))/(2*qa),(-qb-np.sqrt(disc))/(2*qa)),axis=1)
    points=(u[:,None,:]+roots[:,:,None]*v[:,None,:]).reshape(-1,3)
    points=points[np.isfinite(points).all(axis=1)&(points[:,2]>1e-8)]
    p=points[:,:2]
    if not len(p):
        raise Timeout()
    clearance=np.minimum.reduce([p[:,0],w-p[:,0],p[:,1],h-p[:,1]])
    for j in range(n):
        if time.monotonic() >= deadline:
            raise Timeout()
        clearance=np.minimum(clearance,np.sqrt(np.sum((p-a[j,:2])**2,axis=1))-a[j,2])
    k=int(np.argmax(clearance))
    return np.vstack((a,[p[k,0],p[k,1],max(1e-7,clearance[k]*(1-1e-11))]))

def construct(params, seed, budget_s, init=None):
    start=time.monotonic()
    deadline=start+0.69*budget_s
    rng=np.random.default_rng(seed)
    if init is None:
        a=np.array([[(i%5+.5)/5,(i//5+.5)/5,.045] for i in range(21)],float)
        init={'circles':a.tolist()}
    original=np.asarray(init['circles'],float)
    best=init
    best_score=exact_score(original)
    a=original.copy()
    lo=np.min(a[:,:2]-a[:,2,None],axis=0)
    hi=np.max(a[:,:2]+a[:,2,None],axis=0)
    scale=2/np.sum(hi-lo)
    a[:,:2]=(a[:,:2]-lo)*scale
    a[:,2]*=scale
    w=float((hi[0]-lo[0])*scale)
    count=int(params['removed'])
    order=np.argsort(a[:,2])
    anchor=int(order[int(rng.integers(0,12))])
    dist=np.sqrt(np.sum((a[:,:2]-a[anchor,:2])**2,axis=1))
    rank=dist-float(params['patch_bias'])*a[:,2]
    removed=np.argsort(rank)[:count]
    a=np.delete(a,removed,axis=0)
    try:
        a,w=optimize(a,w,int(params['relax_iterations']),min(deadline,start+0.24*budget_s))
        for unused in range(count):
            if time.monotonic() >= deadline:
                raise Timeout()
            a=insert_gap(a,w,deadline)
        a,w=optimize(a,w,int(params['finish_iterations']),deadline)
        if time.monotonic()<start+0.76*budget_s:
            candidate=repair(a)
            if candidate is not None:
                score=exact_score(candidate)
                if score>best_score:
                    best={'circles':candidate.tolist()}
    except Timeout:
        pass
    return best
