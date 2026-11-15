import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {
    'first_share': (0.38, 0.2, 0.65),
    'release': (0.009, 0.002, 0.04),
    'contacts': (3, 3, 5),
    'iterations': (100, 40, 180)
}


def canonical(obj):
    if obj is None:
        return None
    if 'circles' in obj:
        return obj
    rows = []
    for x, y, r, m in obj['orbits']:
        rows.append([x, y, r])
        if m == 2:
            rows.append([x, -y, r])
    return {'circles': rows}


def score(obj):
    try:
        a = np.asarray(canonical(obj)['circles'], dtype=np.float64)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        w = max(u+t for u,t in zip(x,r))-min(u-t for u,t in zip(x,r))
        h = max(u+t for u,t in zip(y,r))-min(u-t for u,t in zip(y,r))
        if w+h > 2:
            return -float('inf')
        for i in range(21):
            for j in range(i):
                dx = a[i,0]-a[j,0]
                dy = a[i,1]-a[j,1]
                if np.sqrt(dx*dx+dy*dy) < a[i,2]+a[j,2]:
                    return -float('inf')
        return float(np.sum(a[:,2]))
    except (TypeError, ValueError, KeyError, OverflowError):
        return -float('inf')


class StopSearch(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + 0.69*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = score(init)
    best_z = None

    def tick():
        if time.monotonic() >= deadline:
            raise StopSearch()

    # A certified fallback also covers malformed starts.
    fallback = np.array([[2*(i%7), 2*(i//7), 1.] for i in range(21)], float)/10.000000000001
    fallback_obj = {'circles': fallback.tolist()}
    fallback_score = score(fallback_obj)
    if fallback_score > best_score:
        best, best_score = fallback_obj, fallback_score
    try:
        a = np.asarray(canonical(init)['circles'], float).copy()
        if a.shape != (21,3) or not np.isfinite(a).all():
            a = fallback.copy()
    except (TypeError, ValueError, KeyError):
        a = fallback.copy()

    try:
        tick()
        lo = np.min(a[:,:2]-a[:,2,None], axis=0)
        hi = np.max(a[:,:2]+a[:,2,None], axis=0)
        a[:,:2] -= (lo+hi)/2
        projections = []
        for swap in (False, True):
            tick()
            b = a.copy()
            if swap:
                b[:,[0,1]] = b[:,[1,0]]
            order = np.argsort(b[:,1])
            lower, singles, upper = order[:9], order[9:12], order[12:]
            reflected = b[lower].copy()
            reflected[:,1] *= -1
            cost = np.sum((b[upper,None,:]-reflected[None,:,:])**2, axis=2)
            u,v = linear_sum_assignment(cost)
            g = np.zeros((12,3))
            g[:3,0] = b[singles,0]
            g[:3,2] = b[singles,2]
            error = float(np.sum(b[singles,1]**2))
            for k,(p,q) in enumerate(zip(u,v)):
                tick()
                g[3+k] = (b[upper[p]]+reflected[q])/2
                error += float(cost[p,q])
            projections.append((error,g))
        projections.sort(key=lambda q:q[0])

        groups = np.array([0,1,2]+[g for g in range(3,12) for _ in (0,1)])
        signs = np.array([0.,0.,0.]+[s for _ in range(9) for s in (1.,-1.)])
        X = np.zeros((21,34)); Y = X.copy(); R = X.copy()
        for k,g in enumerate(groups):
            tick()
            X[k,g] = 1.
            R[k,21+g] = 1.
            if g >= 3:
                Y[k,12+g-3] = signs[k]
        pairs = []
        seen = set()
        for i in range(21):
            tick()
            for j in range(i):
                tick()
                g,h = sorted((int(groups[i]),int(groups[j])))
                key = (g,h,int(signs[i]*signs[j]))
                if key not in seen:
                    seen.add(key)
                    pairs.append((i,j))
        pi,pj = np.array(pairs).T
        DX = X[pi]-X[pj]
        DY = Y[pi]-Y[pj]
        RS = R[pi]+R[pj]
        walls = np.zeros((36,34))
        offsets = np.zeros(36)
        for g in range(12):
            tick()
            walls[g,g] = 1.
            walls[g,21+g] = -1.
            walls[g,33] = .5
            walls[12+g,g] = -1.
            walls[12+g,21+g] = -1.
            walls[12+g,33] = .5
            walls[24+g,21+g] = -1.
            walls[24+g,33] = -.5
            offsets[24+g] = 1.
            if g >= 3:
                walls[24+g,12+g-3] = -1.
        weights = np.array([1.]*3+[2.]*9)
        grad = np.zeros(34)
        grad[21:33] = -weights
        bounds = [(-1.,1.)]*12+[(0.,1.)]*9+[(1e-7,.6)]*12+[(.15,1.85)]
        ii,jj = np.triu_indices(21,1)
        release = np.zeros(len(pi))

        def encode(g):
            v = np.zeros(34)
            v[:12] = g[:,0]
            v[12:21] = np.abs(g[3:,1])
            v[21:33] = np.maximum(g[:,2],1e-6)
            w = np.max(g[:,0]+v[21:33])-np.min(g[:,0]-v[21:33])
            h = 2*np.max(np.abs(g[:,1])+v[21:33])
            factor = 2/max(w+h,1e-6)
            v[:33] *= factor
            v[33] = np.clip(w*factor,.15,1.85)
            return v

        def retain(v):
            nonlocal best,best_score,best_z
            tick()
            if not np.isfinite(v).all():
                return
            r = v[21:33].copy()
            if np.any(r <= 0):
                return
            c = np.column_stack((X@v,Y@v,R@v))
            dx = c[ii,0]-c[jj,0]
            dy = c[ii,1]-c[jj,1]
            ratio = np.min(np.sqrt(dx*dx+dy*dy)/(c[ii,2]+c[jj,2]))
            r *= max(0.,min(1.,float(ratio)))*(1.-8e-15)
            if np.any(r <= 0):
                return
            xx = v[:12].copy()
            yy = np.r_[np.zeros(3),np.abs(v[12:21])]
            xmin = np.min(xx-r)
            xmax = np.max(xx+r)
            xx -= (xmin+xmax)/2
            width = xmax-xmin
            height = 2*np.max(yy+r)
            scale = 2/(width+height)*(1.-8e-15)
            value_hint = float(weights@r)*scale
            if value_hint <= best_score:
                return
            obj = {'orbits': [[float(xx[g]*scale),float(yy[g]*scale),float(r[g]*scale),1 if g<3 else 2] for g in range(12)]}
            value = score(obj)
            if value > best_score:
                best,best_score = obj,value
                best_z = np.zeros(34)
                best_z[:12] = xx*scale
                best_z[12:21] = yy[3:]*scale
                best_z[21:33] = r*scale
                best_z[33] = width*scale+1e-14

        def constraints(v):
            tick()
            dx,dy = DX@v,DY@v
            return np.r_[np.sqrt(dx*dx+dy*dy)-RS@v-release,walls@v+offsets]

        def derivatives(v):
            tick()
            dx,dy = DX@v,DY@v
            d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-25)
            return np.vstack(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RS,walls))

        def objective(v):
            tick()
            return float(grad@v)

        def jac(v):
            tick()
            return grad

        def solve(v, seconds, iterations=100, tolerance=2e-13):
            end = min(deadline,time.monotonic()+max(0.,seconds))
            last = v.copy()
            count = 0
            def callback(q):
                nonlocal last,count
                tick()
                last = q.copy()
                count += 1
                if count%3 == 0:
                    retain(q)
                if time.monotonic() >= end:
                    raise TimeoutError()
            try:
                result = minimize(objective,v,jac=jac,bounds=bounds,method='SLSQP',
                    constraints={'type':'ineq','fun':constraints,'jac':derivatives},
                    callback=callback,options={'maxiter':iterations,'ftol':tolerance,'disp':False})
                last = result.x
            except TimeoutError:
                pass
            retain(last)
            return last

        available = max(0.,deadline-time.monotonic())
        initial = encode(projections[0][1])
        retain(initial)
        z = solve(initial, available*float(params['first_share']),int(params['iterations']))
        anchor = best_z.copy() if best_z is not None else z.copy()

        # A compact five-row family: three axis singles and nine pairs.
        g = np.zeros((12,3))
        g[:3,0] = [-2.,0.,2.]
        g[:3,2] = 1.
        g[3:8,0] = [-4.,-2.,0.,2.,4.]
        g[3:8,1] = np.sqrt(3.)
        g[3:8,2] = 1.
        g[8:,0] = [-3.,-1.,1.,3.]
        g[8:,1] = 2*np.sqrt(3.)
        g[8:,2] = 1.
        remaining = deadline-time.monotonic()
        if remaining > .12:
            solve(encode(g),remaining*.38,int(params['iterations']),1e-11)
        if best_z is not None:
            anchor = best_z.copy()
        remaining = deadline-time.monotonic()
        if remaining > .08:
            anchor = solve(anchor,remaining*.30,60,8e-15)
        attempt = 0
        while time.monotonic() < deadline:
            tick()
            if best_z is not None:
                anchor = best_z.copy()
            release[:] = 0.
            dx,dy = DX@anchor,DY@anchor
            gaps = np.sqrt(dx*dx+dy*dy)-RS@anchor
            near = np.flatnonzero(gaps < .002)
            if len(near) < 3:
                break
            remaining = deadline-time.monotonic()
            if remaining < .045:
                break
            k = min(int(params['contacts']),len(near))
            selected = rng.choice(near,k,replace=False)
            release[selected] = float(params['release'])*rng.uniform(.7,1.3,k)*(1+.3*(attempt%3))
            displaced = solve(anchor,min(.10,remaining*.25),20,1e-10)
            release[:] = 0.
            remaining = deadline-time.monotonic()
            solve(displaced,min(.22,remaining*.85),70,2e-14)
            attempt += 1
    except (StopSearch, ValueError, FloatingPointError):
        pass
    return best
