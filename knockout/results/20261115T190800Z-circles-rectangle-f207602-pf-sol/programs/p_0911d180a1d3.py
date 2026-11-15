import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {'warm_iterations': (105, 50, 160), 'screen_iterations': (75, 30, 130), 'polish_iterations': (120, 50, 180), 'release': (0.009, 0.002, 0.035)}


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
        x, y, r = [[Fraction(float(t)) for t in a[:, k]] for k in range(3)]
        wh = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        wh += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        if wh > 2:
            return -float('inf')
        i,j = np.triu_indices(21,1)
        dx,dy = a[i,0]-a[j,0],a[i,1]-a[j,1]
        if np.any(np.sqrt(dx*dx+dy*dy) < a[i,2]+a[j,2]):
            return -float('inf')
        return float(np.sum(a[:,2]))
    except (ValueError, TypeError, KeyError, OverflowError):
        return -float('inf')


class StopSearch(Exception):
    pass


class StopSlice(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    begin = time.monotonic()
    deadline = begin + .69*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = score(init) if init is not None else -float('inf')
    fallback = {'circles': [[(i%7-3)*.1999999999999, (i//7-1)*.1999999999999, .0999999999999] for i in range(21)]}
    fs = score(fallback)
    if fs > best_score:
        best,best_score = fallback,fs

    def tick():
        if time.monotonic() >= deadline:
            raise StopSearch()

    group = np.array([0,1,2]+[g for g in range(3,12) for _ in range(2)])
    signs = np.array([0.,0.,0.]+[s for _ in range(9) for s in (1.,-1.)])
    X = np.zeros((21,34)); Y = np.zeros_like(X); R = np.zeros_like(X)
    try:
        for k,g in enumerate(group):
            tick()
            X[k,g] = 1
            R[k,21+g] = 1
            if g >= 3:
                Y[k,9+g] = signs[k]
        pairs = []; seen = set()
        for i in range(21):
            tick()
            for j in range(i):
                tick()
                key = (min(group[i],group[j]),max(group[i],group[j]),int(signs[i]*signs[j]))
                if key not in seen:
                    seen.add(key); pairs.append((i,j))
        pi,pj = np.asarray(pairs).T
        DX,DY,RS = X[pi]-X[pj],Y[pi]-Y[pj],R[pi]+R[pj]
        walls = np.zeros((36,34)); offsets = np.zeros(36)
        for g in range(12):
            tick()
            walls[g,g] = 1; walls[g,21+g] = -1; walls[g,33] = .5
            walls[12+g,g] = -1; walls[12+g,21+g] = -1; walls[12+g,33] = .5
            walls[24+g,21+g] = -1; walls[24+g,33] = -.5; offsets[24+g] = 1
            if g >= 3:
                walls[24+g,9+g] = -1
    except StopSearch:
        return best

    weights = np.array([1.]*3+[2.]*9)
    grad = np.zeros(34); grad[21:33] = -weights
    bounds = [(-1.,1.)]*12+[(0.,1.)]*9+[(1e-7,.6)]*12+[(.08,1.92)]
    release = np.zeros(len(pi))
    ii,jj = np.triu_indices(21,1)
    champion = None
    champion_score = -float('inf')
    cache_v = None; cache_c = None; cache_j = None
    slice_end = deadline

    def retain(v):
        nonlocal best,best_score,champion,champion_score
        tick()
        if not np.isfinite(v).all() or np.any(v[21:33] <= 0):
            return
        a = np.column_stack((X@v,Y@v,R@v))
        dx,dy = a[ii,0]-a[jj,0],a[ii,1]-a[jj,1]
        ratio = min(1.,float(np.min(np.sqrt(dx*dx+dy*dy)/(a[ii,2]+a[jj,2]))))
        if ratio <= 0:
            return
        rr = v[21:33]*ratio*(1-2e-14)
        xx = v[:12]; yy = np.r_[np.zeros(3),v[12:21]]
        wh = np.max(xx+rr)-np.min(xx-rr)+2*np.max(np.abs(yy)+rr)
        scale = min(1.,2./wh)*(1-2e-14)
        estimate = float(weights@rr)*scale
        if estimate <= min(best_score,champion_score):
            return
        obj = {'orbits': [[float(xx[g]*scale),float(yy[g]*scale),float(rr[g]*scale),1 if g<3 else 2] for g in range(12)]}
        exact = score(obj)
        if exact > champion_score:
            champion_score,champion = exact,v.copy()
        if exact > best_score:
            best_score,best = exact,obj

    def check():
        tick()
        if time.monotonic() >= slice_end:
            raise StopSlice()

    def update(v):
        nonlocal cache_v,cache_c,cache_j
        check()
        if cache_v is not None and np.array_equal(v,cache_v):
            return
        dx,dy = DX@v,DY@v
        d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-25)
        cache_c = np.r_[d-RS@v,walls@v+offsets]
        cache_j = np.vstack(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RS,walls))
        cache_v = v.copy()

    def fun(v):
        check()
        return float(grad@v)

    def jac(v):
        check()
        return grad

    def con(v):
        update(v)
        z = cache_c.copy(); z[:len(pi)] -= release
        return z

    def cjac(v):
        update(v)
        return cache_j

    def solve(v,n,seconds):
        nonlocal slice_end
        tick()
        slice_end = min(deadline,time.monotonic()+max(.001,seconds))
        last = v.copy(); count = 0
        def callback(z):
            nonlocal last,count
            tick()
            last = z.copy(); count += 1
            if count%6 == 0:
                retain(z)
            check()
        try:
            res = minimize(fun,v,jac=jac,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':con,'jac':cjac},callback=callback,
                options={'maxiter':n,'ftol':1e-13,'disp':False})
            last = res.x
        except StopSlice:
            pass
        slice_end = deadline
        retain(last)
        return last

    def encode(g):
        z = np.zeros(34)
        z[:12] = g[:,0]; z[12:21] = np.maximum(g[3:,1],0.)
        z[21:33] = np.maximum(g[:,2],1e-7)
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
        wh = np.ptp(g[:,0])+2+2*np.max(g[:,1]+g[:,2])
        return encode(g*(2./wh))

    try:
        projections = []
        if init is not None:
            try:
                a = np.asarray(canonical(init)['circles'],float).copy()
                if a.shape == (21,3) and np.isfinite(a).all() and np.all(a[:,2]>0):
                    lo = np.min(a[:,:2]-a[:,2,None],axis=0)
                    hi = np.max(a[:,:2]+a[:,2,None],axis=0)
                    a[:,:2] -= (lo+hi)/2
                    for swap in (False,True):
                        tick()
                        b = a.copy()
                        if swap:
                            b[:,[0,1]] = b[:,[1,0]]
                        order = np.argsort(b[:,1])
                        lower,singles,upper = order[:9],order[9:12],order[12:]
                        ref = b[lower].copy(); ref[:,1] *= -1
                        cost = np.sum((b[upper,None,:]-ref[None,:,:])**2,axis=2)
                        u,v = linear_sum_assignment(cost)
                        g = np.vstack((np.column_stack((b[singles,0],np.zeros(3),b[singles,2])),(b[upper[u]]+ref[v])/2))
                        error = float(np.sum(cost[u,v])+np.sum(b[singles,1]**2))
                        projections.append((error,encode(g)))
            except (ValueError,TypeError,KeyError):
                pass
        projections.sort(key=lambda p:p[0])
        if projections:
            solve(projections[0][1],int(params['warm_iterations']),.245*budget_s)
        if champion_score < 2.3658323:
            solve(lattice((4,3,2)),int(params['screen_iterations']),.175*budget_s)
        if champion_score < 2.3658323 and time.monotonic() < begin+.43*budget_s:
            if len(projections)>1 and projections[1][0] < .02:
                z = projections[1][1]
            else:
                z = lattice((5,4))
            solve(z,int(params['screen_iterations']),.105*budget_s)
        if champion is not None:
            solve(champion.copy(),int(params['polish_iterations']),min(.17*budget_s,deadline-time.monotonic()-.035*budget_s))
        elif time.monotonic() < deadline:
            solve(lattice((4,5)),int(params['polish_iterations']),deadline-time.monotonic()-.005)
        while time.monotonic() < deadline-.035*budget_s:
            tick()
            if champion is None:
                break
            if champion_score < 2.3657:
                solve(champion.copy(),int(params['polish_iterations']),.07*budget_s)
                if time.monotonic() >= deadline-.06*budget_s:
                    break
            anchor = champion.copy()
            dx,dy = DX@anchor,DY@anchor
            gaps = np.sqrt(dx*dx+dy*dy)-RS@anchor
            near = np.flatnonzero(gaps < .002)
            if len(near)<3:
                break
            chosen = rng.choice(near,min(4,len(near)),replace=False)
            release[chosen] = float(params['release'])*rng.uniform(.7,1.3,len(chosen))
            moved = solve(anchor,20,.035*budget_s)
            release[:] = 0
            solve(moved,int(params['polish_iterations']),.085*budget_s)
    except (StopSearch,ValueError,FloatingPointError):
        pass
    return best
