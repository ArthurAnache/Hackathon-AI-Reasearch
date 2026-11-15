import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    "surgery_iterations": (24, 8, 60),
    "relax_iterations": (48, 10, 120),
    "locality": (0.22, 0.05, 0.8),
}
N = 21
II, JJ = np.triu_indices(N, 1)

class StopSearch(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[II, 0]-a[JJ, 0])**2 + (a[II, 1]-a[JJ, 1])**2)
    if np.any(d < a[II, 2]+a[JJ, 2]):
        return -np.inf
    x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
    per = (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
           +max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)))
    if per > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a = np.array([[(i % 7 + .5)*.2, (i // 7 + .5)*.2, .095] for i in range(21)])
        init = {"circles": a.tolist()}
    original = np.asarray(init["circles"], float)
    best = init
    best_score = exact_score(original)

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def consider(a):
        nonlocal best, best_score
        if time.monotonic() >= deadline + .07*budget_s:
            return
        a = np.array(a, dtype=float, copy=True)
        if not np.isfinite(a).all() or np.min(a[:,2]) <= 0:
            return
        # Repair float-level overlap, then normalize the actual bounding box.
        for _ in range(3):
            if time.monotonic() >= deadline + .07*budget_s:
                return
            d = np.sqrt((a[II,0]-a[JJ,0])**2 + (a[II,1]-a[JJ,1])**2)
            ratio = min(1., float(np.min(d/(a[II,2]+a[JJ,2]))))
            a[:,2] *= ratio*(1.-3e-15)
            lo = np.min(a[:,:2]-a[:,2,None], axis=0)
            hi = np.max(a[:,:2]+a[:,2,None], axis=0)
            scale = (2.-2e-14)/float(np.sum(hi-lo))
            a[:,:2] = (a[:,:2]-lo)*scale
            a[:,2] *= scale
        score = exact_score(a)
        if score > best_score:
            best_score = score
            best = {"circles": a.tolist()}

    a = original.copy()
    lo = np.min(a[:,:2]-a[:,2,None], axis=0)
    hi = np.max(a[:,:2]+a[:,2,None], axis=0)
    a[:,:2] -= lo
    scale = 2./np.sum(hi-lo)
    a *= scale
    sides = (hi-lo)*scale

    def reflection_groups(v, height):
        singles = list(np.argsort(abs(v[:,1]-height/2))[:3])
        remaining = set(range(N))-set(singles)
        pairs = []
        error = float(np.sum((v[singles,1]-height/2)**2))
        while remaining:
            check()
            i = min(remaining)
            remaining.remove(i)
            target = np.array([v[i,0], height-v[i,1], v[i,2]])
            j = min(remaining, key=lambda k: float(np.sum((v[k]-target)**2)))
            remaining.remove(j)
            if v[i,1] > v[j,1]:
                i,j = j,i
            pairs.append((i,j))
            error += float(np.sum((v[j]-np.array([v[i,0],height-v[i,1],v[i,2]]))**2))
        return error, singles, pairs

    try:
        g0 = reflection_groups(a, sides[1])
        swapped = a[:,[1,0,2]]
        g1 = reflection_groups(swapped, sides[0])
        if g1[0] < g0[0]:
            a = swapped.copy()
            sides = sides[::-1]
            _, singles, pairs = g1
        else:
            _, singles, pairs = g0
        groups = [(i,) for i in singles]+pairs
        # q contains twelve shared x's, nine lower y's, twelve radii, and width.
        D = 34
        X = np.zeros((N,D)); Y = np.zeros((N,D)); R = np.zeros((N,D))
        yo = np.zeros(N)
        q = np.zeros(D)
        q[-1] = sides[0]
        gid = np.zeros(N, dtype=int)
        for k, group in enumerate(groups):
            check()
            q[k] = np.mean(a[list(group),0])
            q[21+k] = np.mean(a[list(group),2])
            for i in group:
                gid[i] = k
                X[i,k] = 1
                R[i,21+k] = 1
            if len(group) == 1:
                Y[group[0],-1] = -.5
                yo[group[0]] = 1
            else:
                i,j = group
                q[12+k-3] = (a[i,1]+2-q[-1]-a[j,1])/2
                Y[i,12+k-3] = 1
                Y[j,12+k-3] = -1
                Y[j,-1] = -1
                yo[j] = 2
        DX = X[II]-X[JJ]
        DY = Y[II]-Y[JJ]
        DR = R[II]+R[JJ]
        dyo = yo[II]-yo[JJ]
        W = np.zeros(D); W[-1] = 1
        B = np.vstack((X-R, W-X-R, Y-R, -W-Y-R))
        bo = np.r_[np.zeros(N),np.zeros(N),yo,2-yo]
        objective = -np.sum(R,axis=0)
        last = q.copy()
        cache_q = None
        cache_c = None
        cache_j = None

        def calculations(v):
            nonlocal cache_q, cache_c, cache_j, last
            check()
            if cache_q is not None and np.array_equal(v,cache_q):
                return cache_c,cache_j
            dx = DX@v
            dy = DY@v+dyo
            dist = np.maximum(np.sqrt(dx*dx+dy*dy),1e-18)
            c = np.r_[dist-DR@v, B@v+bo]
            j = np.vstack(((dx[:,None]*DX+dy[:,None]*DY)/dist[:,None]-DR,B))
            cache_q = v.copy(); cache_c = c; cache_j = j
            last = v.copy()
            return c,j

        def unpack(v):
            return np.column_stack((X@v,Y@v+yo,R@v))

        bounds = [(None,None)]*D
        for k in range(21,33):
            bounds[k] = (1e-7,None)
        bounds[-1] = (.15,1.85)

        def optimize(v, iterations, forced=()):
            forced = np.asarray(forced,dtype=int)
            keep = np.ones(294,dtype=bool)
            keep[forced] = False
            cons = [{"type":"ineq", "fun":lambda z: calculations(z)[0][keep],
                     "jac":lambda z: calculations(z)[1][keep]}]
            if len(forced):
                cons.append({"type":"eq", "fun":lambda z: calculations(z)[0][forced],
                             "jac":lambda z: calculations(z)[1][forced]})
            result = minimize(lambda z: float(objective@z),v,jac=lambda z: objective,
                              bounds=bounds,constraints=cons,method="SLSQP",
                              options={"maxiter":iterations,"ftol":2e-13,"disp":False})
            return result.x

        q = optimize(q,40)
        consider(unpack(q))
        # Resolve the nearly square contact system without penalty bias.
        polished = q.copy()
        active = np.flatnonzero(calculations(q)[0] < 2e-7)
        for _ in range(5):
            check()
            c,j = calculations(polished)
            step = np.linalg.lstsq(j[active],-c[active],rcond=1e-11)[0]
            if np.linalg.norm(step) > .005:
                break
            polished += step
        consider(unpack(polished))
        base = polished if np.min(calculations(polished)[0]) > -1e-8 else q
        gaps,_ = calculations(base)
        # Collapse reflected duplicates before choosing coordinated new contacts.
        edges = {}
        for e in range(len(II)):
            check()
            if 1e-5 < gaps[e] < .24:
                key = tuple(sorted((int(gid[II[e]]),int(gid[JJ[e]]))))
                if key not in edges or gaps[e] < gaps[edges[key]]:
                    edges[key] = e
        available = sorted(edges.values(),key=lambda e:gaps[e])
        attempt = 0
        while len(available) >= 3:
            check()
            anchor = available[attempt % min(12,len(available))]
            pts = unpack(base)[:,:2]
            mids = (pts[II]+pts[JJ])/2
            center = mids[anchor]
            ranked = sorted(available,key=lambda e: gaps[e]+float(params['locality'])*np.linalg.norm(mids[e]-center)+rng.uniform(0,.025))
            chosen = [anchor]+[e for e in ranked if e != anchor][:2]
            trial = optimize(base.copy(),int(params['surgery_iterations']),chosen)
            consider(unpack(trial))
            check()
            trial = optimize(trial,int(params['relax_iterations']))
            consider(unpack(trial))
            attempt += 1
    except StopSearch:
        if 'last' in locals() and 'unpack' in locals():
            consider(unpack(last))
    return best
