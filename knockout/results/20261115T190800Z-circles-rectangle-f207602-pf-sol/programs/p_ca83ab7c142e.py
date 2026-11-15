import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'iterations': (140, 60, 240),
    'symmetry_fraction': (0.46, 0.30, 0.55),
    'safety': (3e-14, 1e-15, 1e-11, 'log'),
}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    best = init
    best_score = -float('inf')
    best_c = None
    safety = float(params['safety'])

    def exact(c):
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return -float('inf')
        x = [Fraction(float(v)) for v in c[:, 0]]
        y = [Fraction(float(v)) for v in c[:, 1]]
        r = [Fraction(float(v)) for v in c[:, 2]]
        w = max(a+b for a,b in zip(x,r))-min(a-b for a,b in zip(x,r))
        h = max(a+b for a,b in zip(y,r))-min(a-b for a,b in zip(y,r))
        if w+h > 2:
            return -float('inf')
        dx = c[ii,0]-c[jj,0]
        dy = c[ii,1]-c[jj,1]
        if np.any(np.sqrt(dx**2+dy**2) < c[ii,2]+c[jj,2]):
            return -float('inf')
        return float(np.sum(c[:,2]))

    if init is not None:
        try:
            if 'circles' in init:
                c0 = np.asarray(init['circles'], dtype=float)
            else:
                z = np.asarray(init['state'], dtype=float)
                c0 = np.column_stack((z[:21],z[21:42],z[42:63]))
            best_score = exact(c0)
            if np.isfinite(best_score):
                best_c = c0.copy()
        except Exception:
            pass

    def offer(c):
        nonlocal best, best_score, best_c
        if time.monotonic() >= end or not np.isfinite(c).all():
            return
        c = np.array(c, dtype=float, copy=True)
        if np.any(c[:,2] <= 0):
            return
        d = np.sqrt((c[ii,0]-c[jj,0])**2+(c[ii,1]-c[jj,1])**2)
        factor = min(1.0, float(np.min(d/(c[ii,2]+c[jj,2]))))
        if factor <= 0:
            return
        c[:,2] *= factor*(1-safety)
        lo = np.min(c[:,:2]-c[:,2,None],axis=0)
        hi = np.max(c[:,:2]+c[:,2,None],axis=0)
        c[:,:2] -= (lo+hi)/2
        c *= (2.0/float(np.sum(hi-lo)))*(1-safety)
        if float(np.sum(c[:,2])) <= best_score:
            return
        score = exact(c)
        if score > best_score:
            best_score = score
            best_c = c.copy()
            best = {'circles': c.tolist()}

    def symmetric(pattern, deadline):
        groups = []
        for row, count in enumerate(pattern):
            if time.monotonic() >= deadline:
                return
            if count % 2:
                groups.append((0.,float(row),False))
                for k in range(1,count//2+1):
                    groups.append((2.*k,float(row),True))
            else:
                for k in range(count//2):
                    groups.append((2.*k+1,float(row),True))
        g = len(groups)
        paired = np.array([k for k,a in enumerate(groups) if a[2]],dtype=int)
        p = len(paired)
        m = p+2*g+1
        X = np.zeros((g,m)); Y = np.zeros((g,m)); R = np.zeros((g,m))
        X[paired,np.arange(p)] = 1
        Y[np.arange(g),p+np.arange(g)] = 1
        R[np.arange(g),p+g+np.arange(g)] = 1
        a,b = np.triu_indices(g,1)
        DX = X[a]-X[b]; DY = Y[a]-Y[b]; RR = R[a]+R[b]
        W = np.zeros(m); W[-1] = 1
        L = np.vstack((0.5*W-X-R,-0.5*W-Y-R,-0.5*W+Y-R,X[paired]-R[paired]))
        offset = np.r_[np.zeros(g),np.ones(2*g),np.zeros(p)]
        obj = np.zeros(m)
        obj[p+g:p+2*g] = -1
        obj[p+g+paired] = -2
        radius = 2.0/(2*max(pattern)+2+(len(pattern)-1)*np.sqrt(3))
        z = np.zeros(m)
        z[:p] = np.array([groups[k][0] for k in paired])*radius
        z[p:p+g] = (np.array([v[1] for v in groups])-(len(pattern)-1)/2)*np.sqrt(3)*radius
        z[p+g:p+2*g] = radius
        z[-1] = 2*max(pattern)*radius
        def unpack(v):
            x=X@v; y=Y@v; r=R@v
            out=[]
            for k in range(g):
                if time.monotonic() >= end:
                    raise StopSearch()
                out.append((x[k],y[k],r[k]))
                if groups[k][2]:
                    out.append((-x[k],y[k],r[k]))
            return np.array(out)
        def check():
            if time.monotonic() >= deadline:
                raise StopSearch()
        def fun(v):
            check()
            return float(obj@v)
        def con(v):
            check()
            dx=DX@v; dy=DY@v; rr=RR@v
            return np.r_[dx*dx+dy*dy-rr*rr,L@v+offset]
        def jac(v):
            check()
            dx=DX@v; dy=DY@v; rr=RR@v
            return np.vstack((2*(dx[:,None]*DX+dy[:,None]*DY-rr[:,None]*RR),L))
        counter=[0]
        def callback(v):
            check()
            counter[0]+=1
            if counter[0]%3==0:
                offer(unpack(v))
        bounds=[(1e-7,1.)]*p+[(-1.,1.)]*g+[(1e-7,0.7)]*g+[(0.15,1.85)]
        try:
            offer(unpack(z))
            res=minimize(fun,z,jac=lambda v:obj,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':con,'jac':jac},callback=callback,
                options={'maxiter':int(params['iterations']),'ftol':2e-12,'disp':False})
            offer(unpack(res.x))
        except StopSearch:
            pass

    sym_end = min(end, start+float(params['symmetry_fraction'])*budget_s)
    patterns = ([3,4,3,4,3,4],[4,5,4,5,3])
    for k, pattern in enumerate(patterns):
        now=time.monotonic()
        if now >= sym_end:
            break
        deadline = now+(sym_end-now)/(len(patterns)-k)
        symmetric(pattern,deadline)

    if best_c is not None and time.monotonic() < end-0.035:
        c=best_c.copy()
        lo=np.min(c[:,:2]-c[:,2,None],axis=0)
        hi=np.max(c[:,:2]+c[:,2,None],axis=0)
        c[:,:2]-=(lo+hi)/2
        z=np.r_[c[:,0],c[:,1],c[:,2],(hi[0]-lo[0])]
        m=64
        obj=np.r_[np.zeros(42),-np.ones(21),0.]
        L=np.zeros((84,m))
        for k in range(21):
            if time.monotonic() >= end:
                break
            L[k,k]=-1;L[k,42+k]=-1;L[k,63]=.5
            L[21+k,k]=1;L[21+k,42+k]=-1;L[21+k,63]=.5
            L[42+k,21+k]=-1;L[42+k,42+k]=-1;L[42+k,63]=-.5
            L[63+k,21+k]=1;L[63+k,42+k]=-1;L[63+k,63]=-.5
        offset=np.r_[np.zeros(42),np.ones(42)]
        rows=np.arange(len(ii))
        def checkfull():
            if time.monotonic() >= end-0.012:
                raise StopSearch()
        def f(v):
            checkfull()
            return float(obj@v)
        def cf(v):
            checkfull()
            dx=v[ii]-v[jj];dy=v[21+ii]-v[21+jj];rr=v[42+ii]+v[42+jj]
            return np.r_[dx*dx+dy*dy-rr*rr,L@v+offset]
        def jf(v):
            checkfull()
            dx=v[ii]-v[jj];dy=v[21+ii]-v[21+jj];rr=v[42+ii]+v[42+jj]
            J=np.zeros((len(ii),64))
            J[rows,ii]=2*dx;J[rows,jj]=-2*dx
            J[rows,21+ii]=2*dy;J[rows,21+jj]=-2*dy
            J[rows,42+ii]=-2*rr;J[rows,42+jj]=-2*rr
            return np.vstack((J,L))
        def cb(v):
            checkfull()
            offer(np.column_stack((v[:21],v[21:42],v[42:63])))
        try:
            res=minimize(f,z,jac=lambda v:obj,method='SLSQP',
                bounds=[(-1.,1.)]*42+[(1e-8,.7)]*21+[(.1,1.9)],
                constraints={'type':'ineq','fun':cf,'jac':jf},callback=cb,
                options={'maxiter':160,'ftol':1e-13,'disp':False})
            offer(np.column_stack((res.x[:21],res.x[21:42],res.x[42:63])))
        except StopSearch:
            pass
    if best is None:
        r=1.0/22.0
        best={'circles':[[(k-10)*2*r,0.,r*(1-1e-12)] for k in range(21)]}
    return best
