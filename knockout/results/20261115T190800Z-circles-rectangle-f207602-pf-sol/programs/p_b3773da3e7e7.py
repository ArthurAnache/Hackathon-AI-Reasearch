import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"search_fraction": (0.69, 0.5, 0.75), "iterations": (95, 30, 180), "jitter": (0.009, 0.0, 0.04)}

class TimeLimit(Exception):
    pass


def exact_score(data):
    try:
        c = np.asarray(data['circles'], dtype=float)
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return -float('inf')
        x = [Fraction(float(v)) for v in c[:, 0]]
        y = [Fraction(float(v)) for v in c[:, 1]]
        r = [Fraction(float(v)) for v in c[:, 2]]
        w = max(a+b for a,b in zip(x,r))-min(a-b for a,b in zip(x,r))
        h = max(a+b for a,b in zip(y,r))-min(a-b for a,b in zip(y,r))
        if w+h > 2:
            return -float('inf')
        for i in range(21):
            for j in range(i):
                dx = c[i,0]-c[j,0]
                dy = c[i,1]-c[j,1]
                if np.sqrt(dx*dx+dy*dy) < c[i,2]+c[j,2]:
                    return -float('inf')
        return float(np.sum(c[:,2]))
    except Exception:
        return -float('inf')


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + params['search_fraction'] * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = exact_score(init) if init is not None else -float('inf')
    ii, jj = np.triu_indices(21, 1)
    best_model = None
    best_z = None

    def retain(c, model=None, z=None):
        nonlocal best, best_score, best_model, best_z
        if time.monotonic() >= end:
            return
        c = np.array(c, dtype=float, copy=True)
        if c.shape != (21,3) or not np.isfinite(c).all() or np.min(c[:,2]) <= 0:
            return
        d = np.sqrt(np.sum((c[ii,:2]-c[jj,:2])**2, axis=1))
        ratio = min(1.0, float(np.min(d/(c[ii,2]+c[jj,2]))))
        if ratio <= 0:
            return
        c[:,2] *= ratio * (1.0-4e-13)
        lo = np.min(c[:,:2]-c[:,2,None], axis=0)
        hi = np.max(c[:,:2]+c[:,2,None], axis=0)
        perimeter = float(np.sum(hi-lo))
        if perimeter <= 0:
            return
        c[:,:2] -= (hi+lo)*0.5
        c *= (2.0/perimeter)*(1.0-4e-13)
        value = float(np.sum(c[:,2]))
        if value <= best_score:
            return
        candidate = {'circles': c.tolist()}
        value = exact_score(candidate)
        if value > best_score:
            best_score = value
            best = candidate
            best_model = model
            best_z = None if z is None else z.copy()

    # A guaranteed, compact fallback independent of the supplied start.
    fallback = []
    for row, count in enumerate([4,5,4,5,3]):
        for col in range(count):
            fallback.append([2.0*(col-(count-1)/2), row*np.sqrt(3.0), 1.0])
    retain(fallback)

    def make_model(rows):
        orbit = []
        for row, count in enumerate(rows):
            if count % 2:
                orbit.append((row, 0.0))
            for k in range(count//2):
                orbit.append((row, float(k+1) if count%2 else float(k)+0.5))
        n = len(orbit)
        positive = [i for i,(_,x) in enumerate(orbit) if x > 0]
        nx = len(positive)
        nv = nx+2*n+1
        xm = {o:k for k,o in enumerate(positive)}
        expansion = []
        mirror = []
        for o, (_, x) in enumerate(orbit):
            first = len(expansion)
            expansion.append((o,1))
            if x > 0:
                expansion.append((o,-1))
                mirror.extend([first+1,first])
            else:
                mirror.append(first)
        X = np.zeros((21,nv)); Y = X.copy(); R = X.copy()
        for i,(o,s) in enumerate(expansion):
            if o in xm:
                X[i,xm[o]] = s
            Y[i,nx+o] = 1
            R[i,nx+n+o] = 1
        pairs = []
        for i in range(21):
            for j in range(i):
                a,b = sorted((mirror[i],mirror[j]), reverse=True)
                if (i,j) <= (a,b):
                    pairs.append((i,j))
        p = np.array([a for a,b in pairs]); q = np.array([b for a,b in pairs])
        DX = X[p]-X[q]; DY = Y[p]-Y[q]; DR = R[p]+R[q]
        wall = np.zeros((3*n,nv)); const = np.zeros(3*n)
        for o in range(n):
            if o in xm:
                wall[3*o,xm[o]] = -1
            wall[3*o,-1] = 0.5
            wall[3*o,nx+n+o] = -1
            for k,s in ((1,1),(2,-1)):
                wall[3*o+k,-1] = -0.5
                wall[3*o+k,nx+o] = s
                wall[3*o+k,nx+n+o] = -1
                const[3*o+k] = 1
        obj = -np.sum(R,axis=0)
        bounds = [(0.00001,0.95)]*nx + [(-0.99,0.99)]*n + [(0.00005,0.45)]*n + [(0.25,1.75)]
        z = np.zeros(nv)
        maxcount = max(rows)
        unit = 2.0/(2*maxcount + 2 + np.sqrt(3)*(len(rows)-1))
        for o,(row,x) in enumerate(orbit):
            if o in xm:
                z[xm[o]] = 2*x*unit
            z[nx+o] = (row-(len(rows)-1)/2)*np.sqrt(3)*unit
            z[nx+n+o] = unit*0.96
        z[-1] = 2*maxcount*unit
        z[:nx] *= 1+rng.uniform(-params['jitter'],params['jitter'],nx)
        z[nx:nx+n] += rng.uniform(-params['jitter'],params['jitter'],n)*unit
        return (X,Y,R,DX,DY,DR,wall,const,obj,bounds), z

    def run(model, initial, deadline, maxiter):
        X,Y,R,DX,DY,DR,wall,const,obj,bounds = model
        last = initial.copy()
        def check():
            if time.monotonic() >= min(end,deadline):
                raise TimeLimit()
        def unpack(z):
            return np.column_stack((X@z,Y@z,R@z))
        def fun(z):
            check()
            return float(obj@z)
        def con(z):
            check()
            dx=DX@z; dy=DY@z; rr=DR@z
            return np.concatenate((dx*dx+dy*dy-rr*rr,wall@z+const))
        def jac(z):
            check()
            dx=DX@z; dy=DY@z; rr=DR@z
            return np.vstack((2*(dx[:,None]*DX+dy[:,None]*DY-rr[:,None]*DR),wall))
        def callback(z):
            nonlocal last
            check()
            last=z.copy()
            retain(unpack(z),model,z)
        try:
            result=minimize(fun,initial,jac=lambda z:obj,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':con,'jac':jac},callback=callback,
                options={'maxiter':maxiter,'ftol':2e-12,'disp':False})
            last=result.x
        except TimeLimit:
            pass
        if time.monotonic()<end:
            retain(unpack(last),model,last)
        return last

    families = [[5,4,3,4,5],[5,4,5,4,3],[4,5,3,5,4],[4,5,4,5,3]]
    search_end = start + 0.78*(end-start)
    for k, rows in enumerate(families):
        now=time.monotonic()
        if now >= search_end:
            break
        model,z=make_model(rows)
        deadline=now+(search_end-now)/(len(families)-k)
        run(model,z,deadline,params['iterations'])
    if best_model is not None and best_z is not None and time.monotonic()<end:
        model=best_model
        z=best_z.copy()
        run(model,z,end,params['iterations']*2)
    return best
