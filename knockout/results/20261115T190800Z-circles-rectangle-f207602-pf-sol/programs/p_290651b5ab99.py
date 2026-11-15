import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {'polish_iterations': (100, 30, 180), 'release': (0.018, 0.002, 0.06), 'contacts': (4, 3, 7), 'excursion_iterations': (23, 10, 50)}

class StopSearch(Exception):
    pass

def canonical(obj):
    if obj is None:
        return None
    if 'circles' in obj:
        return obj
    a = []
    for x, y, r, m in obj['orbits']:
        a.append([x, y, r])
        if m == 2:
            a.append([x, -y, r])
    return {'circles': a}

def exact_score(obj):
    try:
        a = np.asarray(canonical(obj)['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        x, y, r = [[Fraction(float(t)) for t in a[:, k]] for k in range(3)]
        p = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        p += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        if p > 2:
            return -float('inf')
        i,j = np.triu_indices(21,1)
        dx = a[i,0]-a[j,0]
        dy = a[i,1]-a[j,1]
        if np.any(np.sqrt(dx*dx+dy*dy) < a[i,2]+a[j,2]):
            return -float('inf')
        return float(np.sum(a[:,2]))
    except (ValueError, TypeError, KeyError, OverflowError):
        return -float('inf')

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + .69*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = exact_score(init)
    fallback = {'circles': [[(i%7-3)*.19999999999999, (i//7-1)*.19999999999999, .09999999999999] for i in range(21)]}
    fscore = exact_score(fallback)
    if fscore > best_score:
        best, best_score = fallback, fscore

    def tick():
        if time.monotonic() >= deadline:
            raise StopSearch()

    try:
        group = np.array([0,1,2]+[g for g in range(3,12) for _ in range(2)])
        sign = np.array([0.,0.,0.]+[s for _ in range(9) for s in (1.,-1.)])
        X = np.zeros((21,34)); Y = np.zeros_like(X); R = np.zeros_like(X)
        for k in range(21):
            tick()
            g = group[k]
            X[k,g] = 1
            R[k,21+g] = 1
            if g >= 3:
                Y[k,9+g] = sign[k]
        pairs = []
        seen = set()
        for i in range(21):
            tick()
            for j in range(i):
                tick()
                key = (min(group[i],group[j]), max(group[i],group[j]), int(sign[i]*sign[j]))
                if key not in seen:
                    seen.add(key)
                    pairs.append((i,j))
        pi,pj = np.array(pairs).T
        DX,DY,RS = X[pi]-X[pj],Y[pi]-Y[pj],R[pi]+R[pj]
        walls = np.zeros((36,34)); offset = np.zeros(36)
        for g in range(12):
            tick()
            walls[g,g] = 1
            walls[g,21+g] = -1
            walls[g,33] = .5
            walls[12+g,g] = -1
            walls[12+g,21+g] = -1
            walls[12+g,33] = .5
            walls[24+g,21+g] = -1
            walls[24+g,33] = -.5
            offset[24+g] = 1
            if g >= 3:
                walls[24+g,9+g] = -1
        weights = np.array([1.]*3+[2.]*9)
        objective = np.zeros(34); objective[21:33] = -weights
        bounds = [(-1.,1.)]*12+[(0.,1.)]*9+[(1e-8,.6)]*12+[(.08,1.92)]
        release = np.zeros(len(pi))
        ai,aj = np.triu_indices(21,1)
        cache_v = None
        cache_g = None
        cache_j = None
        champion = None
        champion_value = -float('inf')

        def calc(v):
            nonlocal cache_v, cache_g, cache_j
            tick()
            if cache_v is not None and np.array_equal(v,cache_v):
                return cache_g,cache_j
            dx,dy = DX@v,DY@v
            dd = np.maximum(np.sqrt(dx*dx+dy*dy),1e-30)
            cache_v = v.copy()
            cache_g = np.r_[dd-RS@v,walls@v+offset]
            cache_j = np.vstack(((dx/dd)[:,None]*DX+(dy/dd)[:,None]*DY-RS,walls))
            return cache_g,cache_j

        def constraints(v):
            g = calc(v)[0].copy()
            g[:len(pi)] -= release
            return g

        def retain(v):
            nonlocal best,best_score,champion,champion_value
            tick()
            if not np.isfinite(v).all() or np.any(v[21:33] <= 0):
                return
            a = np.column_stack((X@v,Y@v,R@v))
            dx = a[ai,0]-a[aj,0]; dy = a[ai,1]-a[aj,1]
            factor = min(1.,float(np.min(np.sqrt(dx*dx+dy*dy)/(a[ai,2]+a[aj,2]))))
            if factor <= 0:
                return
            rr = v[21:33]*factor*(1-8e-15)
            xx = v[:12]
            yy = np.r_[np.zeros(3),v[12:21]]
            extent = np.max(xx+rr)-np.min(xx-rr)+2*np.max(yy+rr)
            scale = min(1.,2./extent)*(1-8e-15)
            val = float(weights@rr)*scale
            if val > champion_value:
                champion_value = val
                champion = v.copy()
            if val <= best_score:
                return
            obj = {'orbits': [[float(xx[g]*scale),float(yy[g]*scale),float(rr[g]*scale),1 if g<3 else 2] for g in range(12)]}
            value = exact_score(obj)
            if value > best_score:
                best,best_score = obj,value

        def solve(v, n, seconds=None):
            tick()
            local_end = deadline if seconds is None else min(deadline,time.monotonic()+seconds)
            last = [v.copy()]
            count = [0]
            def fun(z):
                tick()
                return float(objective@z)
            def cb(z):
                tick()
                last[0] = z.copy()
                count[0] += 1
                if count[0]%4 == 0:
                    retain(z)
                if time.monotonic() >= local_end:
                    raise StopSearch()
            try:
                res = minimize(fun,v,jac=lambda z: objective,method='SLSQP',bounds=bounds,
                    constraints={'type':'ineq','fun':constraints,'jac':lambda z:calc(z)[1]},
                    callback=cb,options={'maxiter':n,'ftol':3e-14,'disp':False})
                last[0] = res.x
            except StopSearch:
                if time.monotonic() >= deadline:
                    raise
            retain(last[0])
            return last[0]

        def contact_polish(v):
            q = v.copy()
            g,J = calc(q)
            active = np.flatnonzero(g < 2e-7)
            if len(active) < 30:
                return q
            for _ in range(7):
                tick()
                g,J = calc(q)
                step,_,rank,_ = np.linalg.lstsq(J[active],-g[active],rcond=1e-11)
                if rank < 33 or np.max(np.abs(step)) > .002:
                    break
                q += step
                retain(q)
                if np.max(np.abs(step)) < 2e-15:
                    break
            return q

        def lattice(counts):
            data = [[2.*(j-1),0.,1.] for j in range(3)]
            for k,count in enumerate(counts):
                tick()
                for j in range(count):
                    tick()
                    data.append([2.*(j-(count-1)/2), (k+1)*np.sqrt(3.),1.])
            a = np.asarray(data)
            w = np.ptp(a[:,0])+2
            h = 2*np.max(a[:,1]+a[:,2])
            scale = 2/(w+h)
            a *= scale
            v = np.zeros(34)
            v[:12] = a[:,0]; v[12:21] = a[3:,1]; v[21:33] = a[:,2]
            v[33] = w*scale
            return v

        anchor = None
        if init is not None:
            a = np.asarray(canonical(init)['circles'],float).copy()
            lo = np.min(a[:,:2]-a[:,2,None],axis=0)
            hi = np.max(a[:,:2]+a[:,2,None],axis=0)
            a[:,:2] -= (lo+hi)/2
            projections = []
            for swap in (False,True):
                tick()
                b = a.copy()
                if swap:
                    b[:,[0,1]] = b[:,[1,0]]
                order = np.argsort(b[:,1])
                lower,singles,upper = order[:9],order[9:12],order[12:]
                ref = b[lower].copy(); ref[:,1] *= -1
                cost = np.sum((b[upper,None,:]-ref[None,:,:])**2,axis=2)
                u,j = linear_sum_assignment(cost)
                orbits = np.vstack((np.column_stack((b[singles,0],np.zeros(3),b[singles,2])),(b[upper[u]]+ref[j])/2))
                v = np.zeros(34)
                v[:12] = orbits[:,0]; v[12:21] = np.maximum(orbits[3:,1],0)
                v[21:33] = orbits[:,2]
                v[33] = np.max(orbits[:,0]+orbits[:,2])-np.min(orbits[:,0]-orbits[:,2])
                err = float(np.sum(cost[u,j])+np.sum(b[singles,1]**2))
                projections.append((err,v))
            anchor = min(projections,key=lambda t:t[0])[1]
        if anchor is None:
            anchor = lattice((4,5))
        anchor = solve(anchor,int(params['polish_iterations']),.26*budget_s)
        contact_polish(anchor)
        if champion is not None:
            anchor = champion.copy()
        attempt = 0
        while time.monotonic() < deadline-.025:
            tick()
            release[:] = 0
            if best_score < 2.36 and attempt < 3:
                v = solve(lattice(((4,5),(3,3,3),(5,4))[attempt]),75,.20*budget_s)
            else:
                gaps,J = calc(anchor)
                near = np.flatnonzero(gaps[:len(pi)] < .002)
                if len(near) < 3:
                    break
                k = min(int(params['contacts'])+attempt%3,len(near))
                selected = rng.choice(near,k,replace=False)
                release[selected] = float(params['release'])*rng.uniform(.6,1.4,k)
                v = solve(anchor.copy(),int(params['excursion_iterations']),.10*budget_s)
                release[:] = 0
                v = solve(v,65,.17*budget_s)
            release[:] = 0
            contact_polish(v)
            if champion is not None:
                anchor = champion.copy()
            attempt += 1
    except (StopSearch, ValueError, FloatingPointError, np.linalg.LinAlgError):
        pass
    return best
