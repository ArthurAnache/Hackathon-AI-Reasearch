import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'phase_seconds': (0.29, 0.1, 0.5),
    'iterations': (130, 40, 250),
    'margin': (8e-14, 1e-14, 1e-10, 'log')
}

class StopSearch(Exception):
    pass


def exact_score(data):
    try:
        c = np.asarray(data['circles'], dtype=float)
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return -float('inf')
        x = [Fraction(float(v)) for v in c[:, 0]]
        y = [Fraction(float(v)) for v in c[:, 1]]
        r = [Fraction(float(v)) for v in c[:, 2]]
        perimeter = max(a+b for a,b in zip(x,r))-min(a-b for a,b in zip(x,r))
        perimeter += max(a+b for a,b in zip(y,r))-min(a-b for a,b in zip(y,r))
        if perimeter > 2:
            return -float('inf')
        for i,j in itertools.combinations(range(21), 2):
            dx = c[i,0]-c[j,0]
            dy = c[i,1]-c[j,1]
            if np.sqrt(dx**2+dy**2) < c[i,2]+c[j,2]:
                return -float('inf')
        return float(np.sum(c[:,2]))
    except Exception:
        return -float('inf')


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    ii,jj = np.triu_indices(21,1)
    best_data = init
    best_exact = exact_score(init) if init is not None else -float('inf')
    best_est = best_exact
    best_c = None
    best_state = None
    margin = params['margin']

    def consider(c, state=None):
        nonlocal best_est, best_c, best_state
        if time.monotonic() >= deadline:
            raise StopSearch()
        if not np.isfinite(c).all() or np.min(c[:,2]) <= 0:
            return
        c = c.copy()
        d = np.sqrt(np.sum((c[ii,:2]-c[jj,:2])**2, axis=1))
        ratio = float(np.min(d/(c[ii,2]+c[jj,2])))
        c[:,2] *= min(1.0, ratio)*(1.0-margin)
        per = float(np.max(c[:,0]+c[:,2])-np.min(c[:,0]-c[:,2]) + np.max(c[:,1]+c[:,2])-np.min(c[:,1]-c[:,2]))
        if per <= 0:
            return
        c *= min(1.0, 2.0/per)*(1.0-margin)
        val = float(np.sum(c[:,2]))
        if val > best_est:
            best_est = val
            best_c = c
            best_state = state

    def model(rows):
        mid = len(rows)//2
        groups = []
        coords = []
        signs = []
        mirror = []
        m = max(rows)
        r = 1.0/(m+1+(len(rows)-1)*np.sqrt(3)/2)
        for level in range(mid+1):
            if time.monotonic() >= deadline:
                raise StopSearch()
            count = rows[mid+level]
            for j in range(count):
                x = (2*j-count+1)*r
                groups.append((x, level*np.sqrt(3)*r, r, level > 0))
        k = len(groups)
        paired = [g for g in range(k) if groups[g][3]]
        p = len(paired)
        nvar = 2*k+p+1
        yi = {g:k+t for t,g in enumerate(paired)}
        ridx = k+p+np.arange(k)
        X=[]; Y=[]; R=[]
        for g,(_,_,_,pair) in enumerate(groups):
            if time.monotonic() >= deadline:
                raise StopSearch()
            a=len(X)
            for sign in ([1,-1] if pair else [0]):
                xx=np.zeros(nvar); yy=np.zeros(nvar); rr=np.zeros(nvar)
                xx[g]=1; rr[ridx[g]]=1
                if pair:
                    yy[yi[g]]=sign
                X.append(xx); Y.append(yy); R.append(rr)
            mirror.extend([a+1,a] if pair else [a])
        X=np.asarray(X); Y=np.asarray(Y); R=np.asarray(R)
        keep=[]
        for a,b in zip(ii,jj):
            if time.monotonic() >= deadline:
                raise StopSearch()
            other=tuple(sorted((mirror[a],mirror[b])))
            if (int(a),int(b)) <= other:
                keep.append((a,b))
        a=np.array([q[0] for q in keep]); b=np.array([q[1] for q in keep])
        DX=X[a]-X[b]; DY=Y[a]-Y[b]; RS=R[a]+R[b]
        wall=np.zeros((3*k,nvar)); offset=np.zeros(3*k)
        for g in range(k):
            if time.monotonic() >= deadline:
                raise StopSearch()
            wall[3*g,g]=1; wall[3*g+1,g]=-1
            wall[3*g:3*g+2,-1]=0.5
            wall[3*g:3*g+3,ridx[g]]=-1
            wall[3*g+2,-1]=-0.5
            offset[3*g+2]=1
            if g in yi:
                wall[3*g+2,yi[g]]=-1
        weights=R.sum(axis=0)
        z=np.zeros(nvar)
        for g,(x,y,rad,pair) in enumerate(groups):
            if time.monotonic() >= deadline:
                raise StopSearch()
            z[g]=x; z[ridx[g]]=rad
            if pair:
                z[yi[g]]=y
        z[-1]=2*m*r
        bounds=[(-1.0,1.0)]*k+[(0.0,1.0)]*p+[(1e-6,0.65)]*k+[(0.15,1.85)]
        cache_z=None; cache_v=None; cache_j=None
        def both(v):
            nonlocal cache_z,cache_v,cache_j
            if cache_z is None or not np.array_equal(v,cache_z):
                dx=DX@v; dy=DY@v
                dist=np.maximum(np.sqrt(dx*dx+dy*dy),1e-16)
                cache_v=np.concatenate((dist-RS@v,wall@v+offset))
                cache_j=np.vstack(((dx/dist)[:,None]*DX+(dy/dist)[:,None]*DY-RS,wall))
                cache_z=v.copy()
            return cache_v,cache_j
        def unpack(v):
            return np.column_stack((X@v,Y@v,R@v))
        return z,bounds,weights,both,unpack

    def run(mod,z,seconds):
        phase_end=min(deadline,time.monotonic()+seconds)
        _,bounds,weights,both,unpack=mod
        last=z.copy()
        def callback(v):
            nonlocal last
            last=v.copy()
            consider(unpack(v),(mod,v.copy()))
            if time.monotonic() >= phase_end:
                raise StopSearch()
        def fun(v):
            if time.monotonic() >= phase_end:
                raise StopSearch()
            return -float(weights@v)
        try:
            result=minimize(fun,z,jac=lambda v:-weights,bounds=bounds,
                constraints={'type':'ineq','fun':lambda v:both(v)[0], 'jac':lambda v:both(v)[1]},
                method='SLSQP',callback=callback,
                options={'maxiter':int(params['iterations']),'ftol':3e-12,'disp':False})
            last=result.x
            consider(unpack(last),(mod,last.copy()))
        except StopSearch:
            pass
        return last

    families=[(4,5,3,5,4),(5,4,3,4,5),(4,4,5,4,4),(3,3,3,3,3,3,3)]
    try:
        for rows in families:
            if time.monotonic() >= deadline-0.10:
                break
            mod=model(rows)
            consider(mod[4](mod[0]),(mod,mod[0].copy()))
            run(mod,mod[0],float(params['phase_seconds']))
        if best_state is not None and time.monotonic() < deadline-0.025:
            mod,z=best_state
            run(mod,z,max(0.001,deadline-time.monotonic()-0.005))
    except StopSearch:
        pass

    if best_c is not None:
        candidate={'circles':best_c.tolist()}
        value=exact_score(candidate)
        if value > best_exact:
            best_data=candidate
            best_exact=value
    if best_exact == -float('inf'):
        # A simple certified fallback requires no numerical solver.
        c=[]
        for i in range(21):
            if time.monotonic()-start > budget_s*0.94:
                return {'circles': [[(j%7)*0.2,(j//7)*0.2,0.09] for j in range(21)]}
            c.append([(i%7)*0.2,(i//7)*0.2,0.09])
        best_data={'circles':c}
    return best_data
