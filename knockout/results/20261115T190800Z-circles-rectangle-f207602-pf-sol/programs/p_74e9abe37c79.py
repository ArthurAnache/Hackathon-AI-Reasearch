import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"removed": (3, 2, 5), "iterations": (65, 20, 150), "choices": (4, 2, 10)}
I, J = np.triu_indices(21, 1)

class StopSearch(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    r = a[:, 2]
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < r[I]+r[J]):
        return -np.inf
    x, y, q = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
    if max(u+v for u,v in zip(x,q))-min(u-v for u,v in zip(x,q))+max(u+v for u,v in zip(y,q))-min(u-v for u,v in zip(y,q)) > 2:
        return -np.inf
    return float(np.sum(r))

def repair(a):
    a = a.copy()
    a[:, 2] = np.maximum(a[:, 2], 1e-12)
    r = a[:, 2]
    d = np.sqrt((a[I, 0]-a[J, 0])**2+(a[I, 1]-a[J, 1])**2)
    r *= min(1.0, float(np.min(d/(r[I]+r[J]))))*(1-3e-14)
    lo = np.min(a[:, :2]-r[:, None], axis=0)
    hi = np.max(a[:, :2]+r[:, None], axis=0)
    a[:, :2] -= lo
    a *= min(1.0, 2.0/float(np.sum(hi-lo)))*(1-3e-14)
    return a

def gaps(a, w, h, deadline):
    m = len(a)
    cc = np.array(list(itertools.combinations(range(m), 3)), dtype=int)
    pp = np.array(list(itertools.combinations(range(m), 2)), dtype=int)
    # Each triple supplies two equations p = A + B*r and one quadratic.
    bases=[]; mats=[]; constants=[]; slopes=[]
    if time.monotonic() > deadline:
        raise StopSearch
    c0=a[cc[:,0]]; c1=a[cc[:,1]]; c2=a[cc[:,2]]
    mats.append(2*np.stack((c1[:,:2]-c0[:,:2], c2[:,:2]-c0[:,:2]),axis=1))
    norm=lambda c: np.sum(c[:,:2]**2,axis=1)-c[:,2]**2
    constants.append(np.stack((norm(c1)-norm(c0),norm(c2)-norm(c0)),axis=1))
    slopes.append(-2*np.stack((c1[:,2]-c0[:,2],c2[:,2]-c0[:,2]),axis=1)); bases.append(c0)
    walls=[(0,0.,1.),(0,w,-1.),(1,0.,1.),(1,h,-1.)]
    for axis,k,s in walls:
        if time.monotonic()>deadline: raise StopSearch
        c0=a[pp[:,0]]; c1=a[pp[:,1]]
        mat=np.zeros((len(pp),2,2)); mat[:,0,:]=2*(c1[:,:2]-c0[:,:2]); mat[:,1,axis]=1
        mats.append(mat); constants.append(np.column_stack((norm(c1)-norm(c0),np.full(len(pp),k))))
        slopes.append(np.column_stack((-2*(c1[:,2]-c0[:,2]),np.full(len(pp),s)))); bases.append(c0)
    for wx in walls[:2]:
        if time.monotonic()>deadline: raise StopSearch
        for wy in walls[2:]:
            if time.monotonic()>deadline: raise StopSearch
            mats.append(np.tile(np.eye(2),(m,1,1)))
            constants.append(np.tile([wx[1],wy[1]],(m,1)))
            slopes.append(np.tile([wx[2],wy[2]],(m,1))); bases.append(a)
    mat=np.concatenate(mats); c=np.concatenate(constants); s=np.concatenate(slopes); base=np.concatenate(bases)
    det=mat[:,0,0]*mat[:,1,1]-mat[:,0,1]*mat[:,1,0]
    ok=np.abs(det)>1e-12
    mat=mat[ok]; c=c[ok]; s=s[ok]; base=base[ok]; det=det[ok]
    def solve(v):
        return np.column_stack(((v[:,0]*mat[:,1,1]-v[:,1]*mat[:,0,1])/det,(mat[:,0,0]*v[:,1]-mat[:,1,0]*v[:,0])/det))
    aa=solve(c); bb=solve(s); delta=aa-base[:,:2]
    qa=np.sum(bb*bb,axis=1)-1; qb=2*(np.sum(delta*bb,axis=1)-base[:,2]); qc=np.sum(delta*delta,axis=1)-base[:,2]**2
    disc=qb*qb-4*qa*qc
    good=(disc>=0)&(np.abs(qa)>1e-12)
    aa=aa[good]; bb=bb[good]; qa=qa[good]; qb=qb[good]; disc=disc[good]
    root=np.sqrt(disc)
    rr=np.concatenate(((-qb+root)/(2*qa),(-qb-root)/(2*qa)))
    pts=np.concatenate((aa,aa))+np.concatenate((bb,bb))*rr[:,None]
    ok=(rr>1e-8)&(pts[:,0]>0)&(pts[:,0]<w)&(pts[:,1]>0)&(pts[:,1]<h)
    pts=pts[ok]
    if len(pts)==0: return np.array([[w/2,h/2,1e-8]])
    clearance=np.sqrt(np.sum((pts[:,None,:]-a[None,:,:2])**2,axis=2))-a[None,:,2]
    rad=np.minimum(np.min(clearance,axis=1),np.minimum.reduce([pts[:,0],w-pts[:,0],pts[:,1],h-pts[:,1]]))
    order=np.argsort(-rad)
    selected=[]
    for j in order:
        if time.monotonic()>deadline: raise StopSearch
        if rad[j]<=0: break
        if all(np.linalg.norm(pts[j]-v[:2])>0.025 for v in selected):
            selected.append(np.r_[pts[j],rad[j]])
        if len(selected)>=12: break
    return np.array(selected) if selected else np.array([[w/2,h/2,1e-8]])

def construct(params, seed, budget_s, init=None):
    start=time.monotonic(); deadline=start+0.70*budget_s
    rng=np.random.default_rng(seed)
    if init is None:
        a=np.array([[(i%5+0.5)/5,(i//5+0.5)/5,0.04] for i in range(21)])
        best={"circles":a.tolist()}
    else:
        best=init; a=np.asarray(init['circles'],dtype=float).copy()
    bestscore=exact_score(a)
    lo=np.min(a[:,:2]-a[:,2,None],axis=0); hi=np.max(a[:,:2]+a[:,2,None],axis=0)
    a[:,:2]-=lo; scale=2/np.sum(hi-lo); a*=scale
    width=(hi[0]-lo[0])*scale
    original=a.copy()
    objective=np.r_[np.zeros(42),-np.ones(21),0.]
    def unpack(z): return z[:21],z[21:42],z[42:63],z[63]
    def fun(z):
        if time.monotonic()>deadline: raise StopSearch
        return float(objective@z),objective
    def con(z):
        if time.monotonic()>deadline: raise StopSearch
        x,y,r,w=unpack(z); dx=x[I]-x[J]; dy=y[I]-y[J]
        return np.r_[dx*dx+dy*dy-(r[I]+r[J])**2,x-r,w-x-r,y-r,2-w-y-r]
    def jac(z):
        if time.monotonic()>deadline: raise StopSearch
        x,y,r,w=unpack(z); n=len(I); out=np.zeros((n+84,64)); rows=np.arange(n)
        out[rows,I]=2*(x[I]-x[J]); out[rows,J]=-out[rows,I]
        out[rows,21+I]=2*(y[I]-y[J]); out[rows,21+J]=-out[rows,21+I]
        out[rows,42+I]=-2*(r[I]+r[J]); out[rows,42+J]=out[rows,42+I]
        k=np.arange(21)
        out[n+k,k]=1; out[n+21+k,k]=-1; out[n+42+k,21+k]=1; out[n+63+k,21+k]=-1
        for t in range(4):
            if time.monotonic()>deadline: raise StopSearch
            out[n+t*21+k,42+k]=-1
        out[n+21+k,63]=1; out[n+63+k,63]=-1
        return out
    def consider(z):
        nonlocal best,bestscore
        x,y,r,w=unpack(z); candidate=repair(np.column_stack((x,y,r)))
        score=exact_score(candidate)
        if score>bestscore:
            bestscore=score; best={'circles':candidate.tolist()}
    attempt=0
    while time.monotonic()<deadline:
        try:
            # Remove a correlated patch, then reinsert on different Voronoi branches.
            count=int(params['removed'])
            if attempt%2==0:
                pivot=int(rng.integers(21)); distance=np.sum((original[:,:2]-original[pivot,:2])**2,axis=1)
                remove=np.argsort(distance)[:count]
            else:
                remove=rng.choice(21,count,replace=False)
            work=np.delete(original,remove,axis=0)
            for k in range(count):
                if time.monotonic()>deadline: raise StopSearch
                candidates=gaps(work,width,2-width,deadline)
                choice=0 if k else min(len(candidates)-1,1+attempt%int(params['choices']))
                work=np.vstack((work,candidates[choice]))
            z=np.r_[work[:,0],work[:,1],work[:,2],width]
            latest=[z]
            def callback(v):
                if time.monotonic()>deadline: raise StopSearch
                latest[0]=v.copy()
            try:
                result=minimize(fun,z,jac=True,method='SLSQP',constraints={'type':'ineq','fun':con,'jac':jac},bounds=[(None,None)]*42+[(1e-7,0.8)]*21+[(0.1,1.9)],callback=callback,options={'maxiter':int(params['iterations']),'ftol':2e-12})
                consider(result.x)
            except StopSearch:
                consider(latest[0]); break
            attempt+=1
        except StopSearch:
            break
    return best
