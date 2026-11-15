import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"search_fraction": (0.70, 0.5, 0.75), "stage_fraction": (0.25, 0.15, 0.4)}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + params['search_fraction'] * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    best = init
    best_score = -float('inf')

    def exact(c):
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return -float('inf')
        dx = c[ii, 0] - c[jj, 0]
        dy = c[ii, 1] - c[jj, 1]
        if np.any(np.sqrt(dx * dx + dy * dy) < c[ii, 2] + c[jj, 2]):
            return -float('inf')
        xs = [Fraction(float(v)) for v in c[:, 0]]
        ys = [Fraction(float(v)) for v in c[:, 1]]
        rs = [Fraction(float(v)) for v in c[:, 2]]
        w = max(x+r for x,r in zip(xs,rs))-min(x-r for x,r in zip(xs,rs))
        h = max(y+r for y,r in zip(ys,rs))-min(y-r for y,r in zip(ys,rs))
        if w+h > 2:
            return -float('inf')
        return float(np.sum(c[:, 2]))

    if init is not None:
        try:
            best_score = exact(np.asarray(init['circles'], dtype=float))
        except Exception:
            pass

    def retain(c):
        nonlocal best, best_score
        c = np.asarray(c, dtype=float).copy()
        if not np.isfinite(c).all() or np.min(c[:,2]) <= 0:
            return -float('inf')
        c[:, :2] -= np.mean(c[:, :2], axis=0)
        dx = c[ii,0]-c[jj,0]
        dy = c[ii,1]-c[jj,1]
        ratios = np.sqrt(dx*dx+dy*dy)/(c[ii,2]+c[jj,2])
        shrink = min(1.0, float(np.min(ratios)))
        if shrink <= 0:
            return -float('inf')
        c[:,2] *= shrink * (1-3e-13)
        wh = np.ptp(c[:,0:2]+0, axis=0)
        width = np.max(c[:,0]+c[:,2])-np.min(c[:,0]-c[:,2])
        height = np.max(c[:,1]+c[:,2])-np.min(c[:,1]-c[:,2])
        c *= (2.0/(width+height))*(1-4e-14)
        score = exact(c)
        if score > best_score:
            best_score = score
            best = {'circles': c.tolist()}
        return score

    fallback = np.array([[i*.22, j*.22, .1] for i in range(7) for j in range(3)], dtype=float)
    retain(fallback)

    # Three axis circles and nine mirrored pairs: 34 independent variables.
    nv = 34
    X = np.zeros((21,nv)); Y = X.copy(); R = X.copy()
    for k in range(3):
        X[k,2*k] = 1
        R[k,2*k+1] = 1
    for k in range(9):
        a = 6+3*k
        for q, sign in ((3+k,1),(12+k,-1)):
            X[q,a] = 1
            Y[q,a+1] = sign
            R[q,a+2] = 1
    mirror = np.array(list(range(3))+list(range(12,21))+list(range(3,12)))
    pairs = []
    for i,j in zip(ii,jj):
        if time.monotonic() >= deadline:
            return best
        reflected = tuple(sorted((int(mirror[i]),int(mirror[j]))))
        if (int(i),int(j)) <= reflected:
            pairs.append((i,j))
    pi = np.array([p[0] for p in pairs]); pj = np.array([p[1] for p in pairs])
    DX = X[pi]-X[pj]; DY = Y[pi]-Y[pj]; RR = R[pi]+R[pj]
    reps = np.arange(12)
    wall = np.vstack((X[reps]-R[reps], -X[reps]-R[reps], -Y[reps]-R[reps]))
    wall[12:24,33] += 1
    wall[24:36,33] -= .5
    wall_offset = np.r_[np.zeros(24),np.ones(12)]
    objective = -np.sum(R,axis=0)
    bounds = []
    for k in range(nv):
        if time.monotonic() >= deadline:
            return best
        if k == 33:
            bounds.append((.35,1.65))
        elif np.any(R[:,k]):
            bounds.append((1e-6,.4))
        else:
            bounds.append((0,2))

    cache_z = None
    cache_g = None
    cache_j = None
    def update(z):
        nonlocal cache_z,cache_g,cache_j
        if cache_z is not None and np.array_equal(z,cache_z):
            return
        dx = DX@z; dy = DY@z
        dist = np.sqrt(dx*dx+dy*dy+1e-28)
        cache_g = np.r_[dist-RR@z,wall@z+wall_offset]
        cache_j = np.vstack(((dx/dist)[:,None]*DX+(dy/dist)[:,None]*DY-RR,wall))
        cache_z = z.copy()
    def funcon(z):
        update(z)
        return cache_g
    def jaccon(z):
        update(z)
        return cache_j
    def unpack(z):
        return np.column_stack((X@z,Y@z,R@z))

    trajectories = []
    def run(z, until, maxiter=150):
        latest = z.copy()
        steps = 0
        local_best = (-float('inf'),z.copy())
        def save(q):
            nonlocal local_best
            score = retain(unpack(q))
            if score > local_best[0]:
                local_best = (score,q.copy())
        def callback(q):
            nonlocal latest,steps
            latest = q.copy(); steps += 1
            if steps % 9 == 0:
                save(q)
            if time.monotonic() >= until:
                raise StopSearch()
        try:
            result = minimize(lambda q: float(objective@q),z,jac=lambda q: objective,
                              constraints={'type':'ineq','fun':funcon,'jac':jaccon},
                              bounds=bounds,method='SLSQP',callback=callback,
                              options={'maxiter':maxiter,'ftol':2e-12,'disp':False})
            latest = result.x
        except StopSearch:
            pass
        except (ValueError,FloatingPointError):
            pass
        save(latest)
        trajectories.append(local_best)
        return local_best

    patterns = [(4,5,4,5,3),(5,4,3,4,5),(4,3,5,4,5)]
    for number, counts in enumerate(patterns):
        now = time.monotonic()
        if now >= deadline-.08:
            break
        singles = []; doubles = []
        for row,count in enumerate(counts):
            if time.monotonic() >= deadline:
                return best
            xx = row*np.sqrt(3.0)
            if count % 2:
                singles.append((xx,0,1))
                distances = range(2,count,2)
            else:
                distances = range(1,count,2)
            for yy in distances:
                if time.monotonic() >= deadline:
                    return best
                doubles.append((xx,float(yy),1))
        z = np.zeros(nv)
        raw = np.array(singles+doubles)
        lo = np.min(raw[:,0]-raw[:,2])
        width = np.max(raw[:,0]+raw[:,2])-lo
        height = 2*np.max(raw[:,1]+raw[:,2])
        scale = 2/(width+height)
        for k,(xx,yy,rr) in enumerate(singles):
            z[2*k:2*k+2] = [(xx-lo)*scale,rr*scale*.99]
        for k,(xx,yy,rr) in enumerate(doubles):
            z[6+3*k:9+3*k] = [(xx-lo)*scale,yy*scale,rr*scale*.99]
        z[33] = width*scale
        allotment = params['stage_fraction']*budget_s
        run(z,min(deadline-.04,time.monotonic()+allotment),110)

    if trajectories and time.monotonic() < deadline-.025:
        trajectories.sort(key=lambda a:a[0],reverse=True)
        z = trajectories[0][1]
        run(z,deadline-.01,250)
    return best
