import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"iterations": (110, 40, 200), "safety": (3e-13, 1e-14, 1e-10, "log")}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    bestscore = -float('inf')
    saved = None
    savedscore = -float('inf')
    safety = params['safety']

    def exact(c):
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return -float('inf')
        fs = [[Fraction(float(v)) for v in row] for row in c]
        w = max(x+r for x,y,r in fs)-min(x-r for x,y,r in fs)
        h = max(y+r for x,y,r in fs)-min(y-r for x,y,r in fs)
        if w+h > 2:
            return -float('inf')
        for i in range(21):
            if time.monotonic() > start + 0.93*budget_s:
                return -float('inf')
            for j in range(i):
                d = np.sqrt((c[i,0]-c[j,0])**2+(c[i,1]-c[j,1])**2)
                if d < c[i,2]+c[j,2]:
                    return -float('inf')
        return float(np.sum(c[:,2]))

    if init is not None:
        try:
            bestscore = exact(np.asarray(init['circles'], dtype=float))
        except (ValueError, KeyError, TypeError):
            pass

    ii, jj = np.triu_indices(21, 1)
    def retain(c):
        nonlocal saved, savedscore
        if not np.isfinite(c).all() or np.min(c[:,2]) <= 0:
            return
        c = c.copy()
        d = np.hypot(c[ii,0]-c[jj,0], c[ii,1]-c[jj,1])
        rat = min(1., float(np.min(d/(c[ii,2]+c[jj,2]))))
        c[:,2] *= rat*(1-safety)
        extent = np.ptp(np.concatenate((c[:,:2]-c[:,2,None], c[:,:2]+c[:,2,None]), axis=0), axis=0).sum()
        if extent <= 0:
            return
        c *= (2./extent)*(1-safety)
        score = float(c[:,2].sum())
        if score > savedscore:
            savedscore = score
            saved = c

    def family(rows):
        singles = []
        pairs = []
        for k, n in enumerate(rows):
            if time.monotonic() >= deadline:
                raise StopSearch
            if n % 2:
                singles.append((0., k*np.sqrt(3.), 0.96))
            for j in range(n//2):
                x = 2*j + (2 if n % 2 else 1)
                pairs.append((float(x), k*np.sqrt(3.), 0.96))
        a = len(singles)
        p = len(pairs)
        m = a+p
        base = np.array(singles+pairs, dtype=float)
        lo = np.min(base[:,1]-base[:,2])
        base[:,1] -= lo
        W = 2*float(np.max(base[:,0]+base[:,2]))
        H = float(np.max(base[:,1]+base[:,2]))
        scale = 2/(W+H)
        base *= scale
        z = np.r_[W*scale, base[:,1], base[:,2], base[a:,0]]
        return a,p,z

    def optimize(a,p,z,until,maxiter):
        m = a+p
        nv = 1+2*m+p
        groups = np.r_[np.arange(m), np.arange(a,m)]
        signs = np.r_[np.zeros(a), np.ones(p), -np.ones(p)]
        X = np.zeros((21,nv)); Y = np.zeros_like(X); R = np.zeros_like(X)
        for k in range(21):
            if time.monotonic() >= deadline:
                raise StopSearch
            g = groups[k]
            Y[k,1+g] = 1
            R[k,1+m+g] = 1
            if g >= a:
                X[k,1+2*m+g-a] = signs[k]
        mirror = np.r_[np.arange(a), np.arange(m,21), np.arange(a,m)]
        pairs = []
        for i,j in zip(ii,jj):
            if time.monotonic() >= deadline:
                raise StopSearch
            mi,mj = sorted((int(mirror[i]),int(mirror[j])))
            if (int(i),int(j)) <= (mi,mj):
                pairs.append((i,j))
        pi,pj = np.array(pairs).T
        DX = X[pi]-X[pj]; DY = Y[pi]-Y[pj]; RR = R[pi]+R[pj]
        wall = np.vstack((-X[:m]-R[:m], Y[:m]-R[:m], -Y[:m]-R[:m]))
        wall[:m,0] += 0.5
        wall[2*m:,0] -= 1
        offset = np.r_[np.zeros(2*m), np.full(m,2.)]
        objective = -np.sum(R, axis=0)
        cachez = None
        cachef = None
        cachej = None
        def guard():
            if time.monotonic() >= until:
                raise StopSearch
        def calculate(v):
            nonlocal cachez,cachef,cachej
            guard()
            if cachez is None or not np.array_equal(cachez,v):
                dx = DX@v; dy = DY@v
                d = np.maximum(np.hypot(dx,dy),1e-18)
                cachef = np.r_[d-RR@v, wall@v+offset]
                cachej = np.vstack(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RR, wall))
                cachez = v.copy()
            return cachef
        def jac(v):
            calculate(v)
            return cachej
        def unpack(v):
            return np.column_stack((X@v,Y@v,R@v))
        latest = z.copy()
        def callback(v):
            nonlocal latest
            guard()
            latest = v.copy()
            retain(unpack(v))
        bounds = [(0.15,1.85)] + [(-0.2,2.2)]*m + [(0.00001,0.7)]*m + [(0.,1.)]*p
        retain(unpack(z))
        try:
            out = minimize(lambda v: (guard(),float(objective@v))[1], z,
                jac=lambda v: objective, bounds=bounds,
                constraints={'type':'ineq','fun':calculate,'jac':jac},
                callback=callback, method='SLSQP',
                options={'maxiter':maxiter,'ftol':2e-12,'disp':False})
            latest = out.x
            retain(unpack(latest))
        except StopSearch:
            pass
        return latest

    trajectories = []
    families = [[5,4,5,4,3], [4,5,4,5,3], [3,4,3,4,3,4], [7,7,7]]
    try:
        for index, rows in enumerate(families):
            now = time.monotonic()
            if now >= deadline-0.07:
                break
            a,p,z = family(rows)
            until = min(deadline-0.035, now + max(0.10, (deadline-now)*0.32))
            z = optimize(a,p,z,until,int(params['iterations']))
            trajectories.append((a,p,z))
        while time.monotonic() < deadline-0.035 and trajectories:
            a,p,z = trajectories[0]
            if saved is not None:
                # Recover quotient variables from the best compatible symmetric packing.
                m = a+p
                if np.max(np.abs(saved[:a,0])) < 1e-7:
                    c = saved
                    width = 2*np.max(np.abs(c[:,0])+c[:,2])
                    height = np.max(c[:,1]+c[:,2])-np.min(c[:,1]-c[:,2])
                    if width+height <= 2.0000001:
                        z = np.r_[width,c[:m,1]-np.min(c[:,1]-c[:,2]),c[:m,2],np.abs(c[a:m,0])]
            z = optimize(a,p,z,deadline,int(params['iterations']))
            trajectories[0] = (a,p,z)
            break
    except StopSearch:
        pass

    if saved is not None:
        # The extra shrink also makes the exact rational bounding-box check robust.
        for attempt in range(3):
            if time.monotonic() > start+0.88*budget_s:
                break
            score = exact(saved)
            if score > bestscore:
                bestscore = score
                best = {'circles':saved.tolist()}
                break
            saved[:,2] *= 1-1e-11
            saved[:,:2] *= 1-1e-12
    if best is None or bestscore == -float('inf'):
        c = np.array([[(i%7)*0.1,(i//7)*0.1,0.049] for i in range(21)],float)
        return {'circles':c.tolist()}
    return best
