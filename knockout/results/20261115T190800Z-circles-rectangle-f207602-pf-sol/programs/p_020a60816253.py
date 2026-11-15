import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'search_fraction': (0.69, 0.5, 0.75),
    'max_iterations': (85, 30, 150),
    'jitter': (0.012, 0.0, 0.05),
}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + params['search_fraction'] * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -float('inf')
    ii, jj = np.triu_indices(21, 1)

    def exact_score(data):
        try:
            a = np.asarray(data['circles'], dtype=float)
            if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
                return -float('inf')
            d = a[ii, :2] - a[jj, :2]
            if np.any(np.sqrt(np.sum(d*d, axis=1)) < a[ii, 2] + a[jj, 2]):
                return -float('inf')
            xs = [Fraction(float(v)) for v in a[:, 0]]
            ys = [Fraction(float(v)) for v in a[:, 1]]
            rs = [Fraction(float(v)) for v in a[:, 2]]
            perimeter = (max(x+r for x,r in zip(xs,rs)) - min(x-r for x,r in zip(xs,rs))
                         + max(y+r for y,r in zip(ys,rs)) - min(y-r for y,r in zip(ys,rs)))
            if perimeter > 2:
                return -float('inf')
            return float(np.sum(a[:, 2]))
        except (ValueError, KeyError, TypeError, OverflowError):
            return -float('inf')

    if init is not None:
        best_score = exact_score(init)

    def consider(a):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        a = np.array(a, dtype=float, copy=True)
        if a.shape != (21,3) or not np.isfinite(a).all() or np.min(a[:,2]) <= 0:
            return
        a[:, :2] -= np.mean(a[:, :2], axis=0)
        d = a[ii,:2] - a[jj,:2]
        distances = np.sqrt(np.sum(d*d, axis=1))
        ratio = min(1.0, float(np.min(distances/(a[ii,2]+a[jj,2]))))
        if ratio <= 0:
            return
        a[:,2] *= ratio * (1.0-8e-14)
        wh = np.max(a[:,:2]+a[:,2,None], axis=0)-np.min(a[:,:2]-a[:,2,None], axis=0)
        a *= (2.0/float(np.sum(wh))) * (1.0-8e-14)
        if float(np.sum(a[:,2])) <= best_score:
            return
        candidate = {'circles': a.tolist()}
        score = exact_score(candidate)
        if score > best_score:
            best, best_score = candidate, score

    fallback = np.array([[0.2*x, 0.2*y, 0.1] for y in range(3) for x in range(7)])
    consider(fallback)
    records = []

    def model(rows):
        entities = []
        for row, count in enumerate(rows):
            if count % 2:
                entities.append((row, 0.0, True))
            for k in range(count//2):
                entities.append((row, float(2*k + (2 if count%2 else 1)), False))
        nv = 1 + sum(2 if single else 3 for _,_,single in entities)
        X = np.zeros((21,nv)); Y = X.copy(); R = X.copy()
        z = np.zeros(nv)
        z[0] = 1.05
        mirror = []
        reps = []
        bounds = [(0.25, 1.75)]
        col = 1; c = 0
        rad = 0.105
        for row, offset, single in entities:
            reps.append(c)
            if single:
                Y[c,col] = 1; R[c,col+1] = 1
                z[col:col+2] = [rad + row*np.sqrt(3)*rad, rad]
                bounds.extend([(0.0,2.0),(1e-6,0.5)])
                mirror.append(c)
                col += 2; c += 1
            else:
                X[c,col] = 1; X[c+1,col] = -1
                Y[c:c+2,col+1] = 1; R[c:c+2,col+2] = 1
                z[col:col+3] = [offset*rad, rad + row*np.sqrt(3)*rad, rad]
                bounds.extend([(0.0,1.0),(0.0,2.0),(1e-6,0.5)])
                mirror.extend([c+1,c])
                col += 3; c += 2
        selected = []
        for i,j in zip(ii,jj):
            if time.monotonic() >= deadline:
                raise StopSearch
            reflected = tuple(sorted((mirror[i],mirror[j])))
            if (int(i),int(j)) <= reflected:
                selected.append((i,j))
        p = np.array([v[0] for v in selected]); q = np.array([v[1] for v in selected])
        DX = X[p]-X[q]; DY = Y[p]-Y[q]; SR = R[p]+R[q]
        A = []
        constants = []
        for i in reps:
            if time.monotonic() >= deadline:
                raise StopSearch
            t = -X[i]-R[i]; t = t.copy(); t[0] += .5
            A.append(t); constants.append(0.)
            A.append(Y[i]-R[i]); constants.append(0.)
            t = -Y[i]-R[i]; t = t.copy(); t[0] -= 1.
            A.append(t); constants.append(2.)
        A = np.asarray(A); constants = np.asarray(constants)
        objective = -np.sum(R,axis=0)
        return (X,Y,R,DX,DY,SR,A,constants,objective,bounds), z

    def run(m, z, seconds, iterations):
        X,Y,R,DX,DY,SR,A,b,obj,bounds = m
        local_end = min(deadline,time.monotonic()+seconds)
        latest = z.copy()
        counter = 0
        def check():
            if time.monotonic() >= local_end:
                raise StopSearch
        def fun(v):
            check()
            return float(obj@v)
        def jac(v):
            check()
            return obj
        def con(v):
            check()
            dx=DX@v; dy=DY@v
            return np.concatenate((np.sqrt(dx*dx+dy*dy)-SR@v, A@v+b))
        def cjac(v):
            check()
            dx=DX@v; dy=DY@v
            d=np.maximum(np.sqrt(dx*dx+dy*dy),1e-15)
            return np.vstack(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-SR,A))
        def callback(v):
            nonlocal latest,counter
            check()
            latest=v.copy(); counter+=1
            if counter%4 == 0:
                consider(np.column_stack((X@v,Y@v,R@v)))
        try:
            result=minimize(fun,z,jac=jac,method='SLSQP',bounds=bounds,
                            constraints={'type':'ineq','fun':con,'jac':cjac},
                            callback=callback,
                            options={'maxiter':iterations,'ftol':2e-12,'disp':False})
            latest=result.x.copy()
        except StopSearch:
            pass
        a=np.column_stack((X@latest,Y@latest,R@latest))
        consider(a)
        return latest, float(np.sum(a[:,2]))

    families = [(3,4,5,4,5),(5,4,3,4,5),(4,5,3,5,4),(4,4,5,4,4),(5,3,5,3,5)]
    try:
        for index, rows in enumerate(families):
            if time.monotonic() >= deadline-0.07:
                break
            m,z=model(rows)
            remaining=deadline-time.monotonic()
            allocation=min(0.25*budget_s, remaining/(min(3,len(families)-index)+0.25))
            z,value=run(m,z,allocation,int(params['max_iterations']))
            records.append((value,m,z))
        records.sort(key=lambda v:v[0],reverse=True)
        attempt=0
        while records and time.monotonic() < deadline-0.035:
            _,m,z=records[attempt%min(2,len(records))]
            trial=z.copy()
            if attempt:
                noise=rng.normal(0,params['jitter'],len(z))
                noise[0]*=0.5
                trial+=noise
                for k,(lo,hi) in enumerate(m[-1]):
                    if time.monotonic() >= deadline:
                        raise StopSearch
                    trial[k]=min(hi,max(lo,trial[k]))
            out,value=run(m,trial,min(0.18*budget_s,deadline-time.monotonic()),int(params['max_iterations']))
            if value>records[0][0]:
                records[0]=(value,m,out)
            attempt+=1
    except StopSearch:
        pass
    return best
