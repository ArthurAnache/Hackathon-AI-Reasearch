import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"tilt": (0.65, 0.1, 1.5), "iterations": (100, 30, 220)}

class StopSearch(Exception):
    pass


def exact_score(candidate):
    try:
        a = np.asarray(candidate['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        perimeter = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        perimeter += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        if perimeter > 2:
            return -float('inf')
        for i in range(21):
            for j in range(i):
                dx,dy = a[i,:2]-a[j,:2]
                if np.sqrt(dx*dx+dy*dy) < a[i,2]+a[j,2]:
                    return -float('inf')
        return float(np.sum(a[:,2]))
    except (TypeError, ValueError, KeyError, OverflowError):
        return -float('inf')


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    best = init
    score = exact_score(init) if init is not None else -float('inf')
    ii,jj = np.triu_indices(21,1)

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def retain(a):
        nonlocal best, score
        check()
        a = np.asarray(a, dtype=float).copy()
        if not np.isfinite(a).all() or np.min(a[:,2]) <= 0:
            return
        delta = a[ii,:2]-a[jj,:2]
        dist = np.sqrt(np.sum(delta*delta,axis=1))
        shrink = min(1., float(np.min(dist/(a[ii,2]+a[jj,2]))))
        if shrink <= 0:
            return
        a[:,2] *= shrink*(1.-3e-15)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        a[:,:2] -= (lo+hi)/2
        a *= (2./float(np.sum(hi-lo)))*(1.-3e-15)
        if float(np.sum(a[:,2])) <= score:
            return
        candidate = {'circles':a.tolist()}
        value = exact_score(candidate)
        if value > score:
            best,score = candidate,value

    try:
        if init is None:
            rows = []
            for col,count in enumerate((5,4,5,4,3)):
                check()
                for k in range(count):
                    check()
                    rows.append([np.sqrt(3.)*col,2*k-count+1,1.])
            a = np.array(rows)
            a *= 2./np.sum(np.ptp(a[:,:2],axis=0)+2)
            retain(a)
        else:
            a = np.array(init['circles'],dtype=float)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        a[:,:2] -= (lo+hi)/2
        # Detect the reflection direction rather than assuming input ordering.
        errors = []
        for axis in (0,1):
            check()
            reflected = a.copy()
            reflected[:,axis] *= -1
            errors.append(float(np.sum(np.min(np.sum((a[:,None,:]-reflected[None,:,:])**2,axis=2),axis=1))))
        if errors[0] < errors[1]:
            a[:,[0,1]] = a[:,[1,0]]
        used = set()
        groups = []
        for i in np.argsort(-a[:,1]):
            check()
            i = int(i)
            if i in used:
                continue
            if abs(a[i,1]) < 1e-6:
                groups.append((i,))
                used.add(i)
            else:
                target = a[i].copy()
                target[1] *= -1
                distances = np.sum((a-target)**2,axis=1)
                distances[i] = np.inf
                for j in used:
                    check()
                    distances[j] = np.inf
                j = int(np.argmin(distances))
                if not np.isfinite(distances[j]):
                    return best
                groups.append((i,j))
                used.update((i,j))
        m = len(groups)
        pairs = [g for g,t in enumerate(groups) if len(t)==2]
        p = len(pairs)
        nv = 2*m+p+1
        yi = {g:m+k for k,g in enumerate(pairs)}
        ri = m+p
        X = np.zeros((21,nv)); Y = X.copy(); R = X.copy()
        z = np.zeros(nv)
        labels = []; signs = []
        row = 0
        for g,t in enumerate(groups):
            check()
            z[g] = np.mean(a[list(t),0])
            z[ri+g] = np.mean(a[list(t),2])
            if len(t)==2:
                z[yi[g]] = np.mean(np.abs(a[list(t),1]))
            for k in range(len(t)):
                check()
                X[row,g] = 1
                R[row,ri+g] = 1
                s = (1 if k==0 else -1) if len(t)==2 else 0
                if s:
                    Y[row,yi[g]] = s
                labels.append(g); signs.append(s)
                row += 1
        width = float(np.max(a[:,0]+a[:,2])-np.min(a[:,0]-a[:,2]))
        z[:m] += width/2
        z[-1] = width
        keep = []; signatures = set()
        for i,j in zip(ii,jj):
            check()
            key = (min(labels[i],labels[j]),max(labels[i],labels[j]),signs[i]*signs[j])
            if key not in signatures:
                signatures.add(key); keep.append((i,j))
        pi,pj = np.asarray(keep).T
        DX = X[pi]-X[pj]; DY = Y[pi]-Y[pj]; RS = R[pi]+R[pj]
        W = np.zeros((3*m,nv)); b = np.zeros(3*m)
        for g in range(m):
            check()
            W[g,g] = 1; W[g,ri+g] = -1
            W[m+g,-1] = 1; W[m+g,g] = -1; W[m+g,ri+g] = -1
            W[2*m+g,-1] = -.5; W[2*m+g,ri+g] = -1
            if g in yi:
                W[2*m+g,yi[g]] = -1
            b[2*m+g] = 1
        weights = np.array([len(t) for t in groups],dtype=float)
        basegrad = np.zeros(nv); basegrad[ri:ri+m] = -weights
        def unpack(v):
            return np.column_stack((X@v,Y@v,R@v))
        def fun(v):
            check()
            dx,dy = DX@v,DY@v
            return np.r_[np.sqrt(dx*dx+dy*dy)-RS@v,W@v+b]
        def jac(v):
            check()
            dx,dy = DX@v,DY@v
            d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-30)
            return np.vstack(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RS,W))
        def callback(v):
            check()
            retain(unpack(v))
        bounds = [(0.,2.)]*m+[(0.,1.)]*p+[(1e-7,.6)]*m+[(.1,1.9)]
        def solve(v,gradient,iterations):
            def objective(t):
                check()
                return float(gradient@t)
            def derivative(t):
                check()
                return gradient
            result = minimize(objective,v,jac=derivative,bounds=bounds,method='SLSQP',
                constraints={'type':'ineq','fun':fun,'jac':jac},callback=callback,
                options={'maxiter':iterations,'ftol':1e-13,'disp':False})
            retain(unpack(result.x))
            return result.x
        z = solve(z,basegrad,int(params['iterations']))
        anchor = z.copy()
        # Solve the isolated contact equations directly to remove SQP termination error.
        active = np.flatnonzero(fun(z)<1e-7)
        for iteration in range(5):
            check()
            f = fun(z)[active]
            J = jac(z)[active]
            step = np.linalg.lstsq(J,-f,rcond=1e-12)[0]
            if np.max(np.abs(step)) > 1e-4:
                break
            z += step
            retain(unpack(z))
        if np.min(fun(z)) >= -1e-9:
            anchor = z.copy()
        attempt = 0
        while time.monotonic() < deadline:
            check()
            tilt = rng.normal(size=m)
            tilt -= np.dot(weights,tilt)/np.sum(weights)
            tilted = basegrad.copy()
            tilted[ri:ri+m] *= np.exp(float(params['tilt'])*tilt)
            current = solve(anchor.copy(),tilted, min(65,int(params['iterations'])))
            current = solve(current,basegrad,int(params['iterations']))
            if -basegrad@current > -basegrad@anchor+1e-9 and np.min(fun(current)) > -1e-8:
                anchor = current.copy()
            attempt += 1
    except (StopSearch, ValueError, FloatingPointError, np.linalg.LinAlgError):
        pass
    return best
