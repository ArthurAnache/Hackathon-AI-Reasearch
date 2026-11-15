import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"iterations": (100, 30, 180), "safety": (2e-12, 1e-14, 1e-9)}

class StopSearch(Exception):
    pass

def exact_score(candidate):
    try:
        a = np.asarray(candidate['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -np.inf
        xs = [Fraction(float(v)) for v in a[:, 0]]
        ys = [Fraction(float(v)) for v in a[:, 1]]
        rs = [Fraction(float(v)) for v in a[:, 2]]
        if (max(x+r for x,r in zip(xs,rs))-min(x-r for x,r in zip(xs,rs)) +
            max(y+r for y,r in zip(ys,rs))-min(y-r for y,r in zip(ys,rs))) > 2:
            return -np.inf
        for i,j in itertools.combinations(range(21),2):
            d = np.sqrt((a[i,0]-a[j,0])**2 + (a[i,1]-a[j,1])**2)
            if d < a[i,2]+a[j,2]:
                return -np.inf
        return float(np.sum(a[:,2]))
    except (ValueError, TypeError, KeyError, OverflowError):
        return -np.inf

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70 * budget_s
    best = init
    bestscore = exact_score(init) if init is not None else -np.inf
    safety = float(params['safety'])
    # Three axis circles, nine reflected pairs, and rectangle width.
    A = np.zeros((21,3,34))
    for k in range(3):
        A[k,1,2*k] = 1
        A[k,2,2*k+1] = 1
    for k in range(9):
        q = 6+3*k
        A[3+k,0,q] = 1
        A[12+k,0,q] = -1
        A[3+k,1,q+1] = A[12+k,1,q+1] = 1
        A[3+k,2,q+2] = A[12+k,2,q+2] = 1
    ii,jj = np.triu_indices(21,1)
    # Keep just one constraint from each reflected orbit.
    mirror = np.array(list(range(3))+list(range(12,21))+list(range(3,12)))
    keep = []
    for k,(i,j) in enumerate(zip(ii,jj)):
        if time.monotonic() >= end:
            break
        reflected = tuple(sorted((int(mirror[i]),int(mirror[j]))))
        if (int(i),int(j)) <= reflected:
            keep.append(k)
    I,J = ii[keep],jj[keep]
    DX = A[I,0]-A[J,0]
    DY = A[I,1]-A[J,1]
    RR = A[I,2]+A[J,2]
    wallrows=[]
    wallconst=[]
    for k in range(21):
        if time.monotonic() >= end:
            break
        for sign in [-1,1]:
            row=sign*A[k,0]-A[k,2]
            row=row.copy(); row[33]+=0.5
            wallrows.append(row); wallconst.append(0.)
        wallrows.append(A[k,1]-A[k,2]); wallconst.append(0.)
        row=-A[k,1]-A[k,2]
        row=row.copy(); row[33]-=1
        wallrows.append(row); wallconst.append(2.)
    W=np.asarray(wallrows)
    C=np.asarray(wallconst)
    _, inds=np.unique(np.column_stack((W,C)),axis=0,return_index=True)
    W=W[inds]; C=C[inds]
    grad=-np.sum(A[:,2],axis=0)
    bounds=[(None,None)]*34
    for k in range(3):
        bounds[2*k+1]=(1e-6,None)
    for k in range(9):
        bounds[6+3*k]=(0.,None)
        bounds[8+3*k]=(1e-6,None)
    bounds[33]=(0.35,1.65)
    all_i,all_j=np.triu_indices(21,1)
    bestz=None

    def consider(z):
        nonlocal best,bestscore,bestz
        if time.monotonic() >= end:
            return
        a=np.einsum('ijk,k->ij',A,z)
        if not np.isfinite(a).all() or np.min(a[:,2])<=0:
            return
        delta=a[all_i,:2]-a[all_j,:2]
        distances=np.sqrt(np.sum(delta*delta,axis=1))
        ratio=float(np.min(distances/(a[all_i,2]+a[all_j,2])))
        a[:,2]*=min(1.,ratio)*(1.-safety)
        if np.min(a[:,2])<=0:
            return
        span=np.max(a[:,:2]+a[:,2,None],axis=0)-np.min(a[:,:2]-a[:,2,None],axis=0)
        a[:,:2]-=(np.max(a[:,:2],axis=0)+np.min(a[:,:2],axis=0))/2
        a*=2.*(1.-safety)/float(np.sum(span))
        score=float(np.sum(a[:,2]))
        if score>bestscore:
            candidate={'circles':a.tolist()}
            score=exact_score(candidate)
            if score>bestscore:
                bestscore=score; best=candidate; bestz=z.copy()

    def make_start(rows):
        r=2./(2.*max(rows)+2.+(len(rows)-1)*np.sqrt(3.))
        z=np.zeros(34); z[33]=2*max(rows)*r
        singles=[]; pairs=[]
        for row,n in enumerate(rows):
            if time.monotonic() >= end:
                raise StopSearch()
            y=r+np.sqrt(3.)*r*row
            if n%2:
                singles.append((y,r))
            for j in range(n//2):
                if time.monotonic() >= end:
                    raise StopSearch()
                x=(2*j+1+(n%2))*r
                pairs.append((x,y,r))
        z[:6]=np.asarray(singles).ravel()
        z[6:33]=np.asarray(pairs).ravel()
        return z

    def run(z, deadline, maxiter):
        last=[z.copy()]
        counter=[0]
        def check():
            if time.monotonic() >= min(end,deadline):
                raise StopSearch()
        def fun(x):
            check()
            return float(grad@x)
        def jac(x):
            check()
            return grad
        def con(x):
            check()
            dx=DX@x; dy=DY@x
            return np.concatenate((np.sqrt(dx*dx+dy*dy)-RR@x,W@x+C))
        def cj(x):
            check()
            dx=DX@x; dy=DY@x
            d=np.maximum(np.sqrt(dx*dx+dy*dy),1e-16)
            return np.vstack(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RR,W))
        def callback(x):
            check()
            last[0]=x.copy(); counter[0]+=1
            if counter[0]%4==0:
                consider(x)
        try:
            res=minimize(fun,z,jac=jac,bounds=bounds,
                         constraints={'type':'ineq','fun':con,'jac':cj},
                         method='SLSQP',callback=callback,
                         options={'maxiter':maxiter,'ftol':3e-12,'disp':False})
            last[0]=res.x
        except StopSearch:
            pass
        consider(last[0])

    families=[(4,5,3,5,4),(5,4,3,4,5),(5,4,5,4,3),(3,6,3,6,3),(3,4,7,4,3)]
    for index,rows in enumerate(families):
        now=time.monotonic()
        if now >= end-0.22*budget_s:
            break
        try:
            z=make_start(rows)
        except StopSearch:
            break
        consider(z)
        run(z,min(end-0.16*budget_s,now+0.115*budget_s),int(params['iterations']))
    if bestz is not None and time.monotonic()<end-0.015:
        run(bestz,end-0.008,180)
    if best is None:
        # A conservative valid fallback independent of optimization success.
        a=[]
        for k in range(21):
            if time.monotonic() >= start+0.90*budget_s:
                break
            a.append([(k%7)*0.14,(k//7)*0.14,0.06])
        if len(a)==21:
            best={'circles':a}
    return best
