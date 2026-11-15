import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {'screen': (65, 30, 120), 'polish': (95, 40, 180), 'release': (0.012, 0.002, 0.04), 'slice': (0.16, 0.08, 0.3)}


def canonical(obj):
    if 'circles' in obj:
        return obj
    out = []
    for x, y, r, m in obj['orbits']:
        out.append([x, y, r])
        if m == 2:
            out.append([x, -y, r])
    return {'circles': out}


def score(obj):
    try:
        a = np.asarray(canonical(obj)['circles'], dtype=np.float64)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
        perimeter = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        perimeter += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        if perimeter > 2:
            return -float('inf')
        for i in range(21):
            for j in range(i):
                dx = a[i,0]-a[j,0]
                dy = a[i,1]-a[j,1]
                if np.sqrt(dx*dx+dy*dy) < a[i,2]+a[j,2]:
                    return -float('inf')
        return float(np.sum(a[:,2]))
    except (ValueError, TypeError, KeyError, OverflowError):
        return -float('inf')


class Finished(Exception):
    pass


class SliceDone(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + 0.69*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = score(init) if init is not None else -float('inf')
    fallback = {'circles': [[(i%7-3)*0.1999999999999, (i//7-1)*0.1999999999999, 0.0999999999999] for i in range(21)]}
    fs = score(fallback)
    if fs > best_score:
        best, best_score = fallback, fs

    def tick():
        if time.monotonic() >= deadline:
            raise Finished()

    groups = np.array([0,1,2]+[g for g in range(3,12) for _ in range(2)])
    signs = np.array([0.,0.,0.]+[s for _ in range(9) for s in (1.,-1.)])
    X = np.zeros((21,34)); Y = np.zeros_like(X); R = np.zeros_like(X)
    for k in range(21):
        if time.monotonic() >= deadline:
            return best
        g = groups[k]
        X[k,g] = 1
        R[k,21+g] = 1
        if g >= 3:
            Y[k,9+g] = signs[k]
    pairs = []
    seen = set()
    for i in range(21):
        if time.monotonic() >= deadline:
            return best
        for j in range(i):
            if time.monotonic() >= deadline:
                return best
            key = (min(groups[i],groups[j]), max(groups[i],groups[j]), int(signs[i]*signs[j]))
            if key not in seen:
                seen.add(key)
                pairs.append((i,j))
    pi,pj = np.array(pairs).T
    DX = X[pi]-X[pj]; DY = Y[pi]-Y[pj]; RS = R[pi]+R[pj]
    walls = np.zeros((36,34)); offsets = np.zeros(36)
    for g in range(12):
        if time.monotonic() >= deadline:
            return best
        walls[g,g] = 1; walls[g,21+g] = -1; walls[g,33] = .5
        walls[12+g,g] = -1; walls[12+g,21+g] = -1; walls[12+g,33] = .5
        walls[24+g,21+g] = -1; walls[24+g,33] = -.5; offsets[24+g] = 1
        if g >= 3:
            walls[24+g,9+g] = -1
    weights = np.array([1.]*3+[2.]*9)
    grad = np.zeros(34); grad[21:33] = -weights
    bounds = [(-1.,1.)]*12+[(0.,1.)]*9+[(1e-7,.6)]*12+[(.08,1.92)]
    release = np.zeros(len(pi))
    ii,jj = np.triu_indices(21,1)
    champion = None
    champion_score = -float('inf')
    slice_end = deadline

    def check():
        tick()
        if time.monotonic() >= slice_end:
            raise SliceDone()

    def retain(v):
        nonlocal best, best_score, champion, champion_score
        tick()
        if not np.isfinite(v).all() or np.any(v[21:33] <= 0):
            return
        a = np.column_stack((X@v,Y@v,R@v))
        d = a[ii,:2]-a[jj,:2]
        ratio = float(np.min(np.sqrt(np.sum(d*d,axis=1))/(a[ii,2]+a[jj,2])))
        if ratio <= 0:
            return
        r = v[21:33]*min(1.,ratio)*(1.-2e-14)
        xx = v[:12]
        yy = np.r_[np.zeros(3),v[12:21]]
        perimeter = np.max(xx+r)-np.min(xx-r)+2*np.max(yy+r)
        scale = min(1.,2./perimeter)*(1.-2e-14)
        value = float(weights@r)*scale
        if value > champion_score:
            champion_score = value
            champion = v.copy()
        if value <= best_score:
            return
        obj = {'orbits': [[float(xx[g]*scale),float(yy[g]*scale),float(r[g]*scale),1 if g<3 else 2] for g in range(12)]}
        verified = score(obj)
        if verified > best_score:
            best, best_score = obj, verified

    def fun(v):
        check()
        return float(grad@v)

    def jac(v):
        check()
        return grad

    def con(v):
        check()
        dx,dy = DX@v,DY@v
        return np.r_[np.sqrt(dx*dx+dy*dy)-RS@v-release,walls@v+offsets]

    def cjac(v):
        check()
        dx,dy = DX@v,DY@v
        d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-25)
        return np.vstack(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RS,walls))

    def solve(v, iterations, seconds):
        nonlocal slice_end
        tick()
        slice_end = min(deadline,time.monotonic()+seconds)
        last = v.copy()
        count = 0
        def callback(z):
            nonlocal last,count
            tick()
            last = z.copy()
            count += 1
            if count % 3 == 0:
                retain(z)
            check()
        try:
            result = minimize(fun,v,jac=jac,method='SLSQP',bounds=bounds,
                              constraints={'type':'ineq','fun':con,'jac':cjac},callback=callback,
                              options={'maxiter':iterations,'ftol':2e-13,'disp':False})
            last = result.x
        except SliceDone:
            pass
        retain(last)
        return last

    def encode(g):
        z = np.zeros(34)
        z[:12] = g[:,0]
        z[12:21] = np.maximum(g[3:,1],0.)
        z[21:33] = np.maximum(g[:,2],1e-6)
        z[33] = np.clip(np.max(g[:,0]+g[:,2])-np.min(g[:,0]-g[:,2]),.08,1.92)
        return z

    def lattice(rows):
        data = [[2.*(j-1),0.,1.] for j in range(3)]
        for k,n in enumerate(rows):
            tick()
            for j in range(n):
                tick()
                data.append([2.*(j-(n-1)/2),(k+1)*np.sqrt(3.),1.])
        g = np.asarray(data)
        w = np.ptp(g[:,0])+2
        h = 2*np.max(g[:,1]+g[:,2])
        return encode(g*(2./(w+h)))

    try:
        starts = []
        if init is not None:
            a = np.asarray(canonical(init)['circles'],float).copy()
            if a.shape == (21,3) and np.isfinite(a).all():
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
                    reflected = b[lower].copy(); reflected[:,1] *= -1
                    cost = np.sum((b[upper,None,:]-reflected[None,:,:])**2,axis=2)
                    u,v = linear_sum_assignment(cost)
                    g = np.vstack((np.column_stack((b[singles,0],np.zeros(3),b[singles,2])),(b[upper[u]]+reflected[v])/2))
                    error = float(np.sum(cost[u,v])+np.sum(b[singles,1]**2))
                    projections.append((error,encode(g)))
                starts.append(min(projections,key=lambda t:t[0])[1])
        starts.extend([lattice((4,5)),lattice((5,4)),lattice((4,3,2)),lattice((3,3,3))])
        screen_end = started + .46*budget_s
        for z in starts:
            tick()
            if time.monotonic() >= screen_end and champion is not None:
                break
            release[:] = 0
            solve(z,int(params['screen']),float(params['slice'])*budget_s)
        if champion is not None:
            release[:] = 0
            anchor = solve(champion.copy(),int(params['polish']),.13*budget_s)
        else:
            anchor = starts[0]
        attempt = 0
        while time.monotonic() < deadline:
            tick()
            release[:] = 0
            dx,dy = DX@anchor,DY@anchor
            gaps = np.sqrt(dx*dx+dy*dy)-RS@anchor
            near = np.flatnonzero(gaps < .003)
            if len(near) < 3:
                break
            selected = rng.choice(near,min(4,len(near)),replace=False)
            release[selected] = float(params['release'])*rng.uniform(.7,1.3,len(selected))
            moved = solve(anchor.copy(),22,.055*budget_s)
            release[:] = 0
            solve(moved,int(params['polish']),.11*budget_s)
            if champion is not None:
                anchor = champion.copy()
            attempt += 1
    except (Finished,ValueError,FloatingPointError):
        pass
    return best
