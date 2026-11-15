import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {
    'screen': (42, 20, 90),
    'polish': (110, 40, 180),
    'release': (0.012, 0.002, 0.04),
    'contacts': (4, 3, 5)
}


def canonical(obj):
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
        x, y, r = [[Fraction(float(t)) for t in a[:, k]] for k in range(3)]
        width = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        height = max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        if width+height > 2:
            return -float('inf')
        ii, jj = np.triu_indices(21, 1)
        dx = a[ii,0]-a[jj,0]
        dy = a[ii,1]-a[jj,1]
        if np.any(np.sqrt(dx*dx+dy*dy) < a[ii,2]+a[jj,2]):
            return -float('inf')
        return float(np.sum(a[:,2]))
    except (ValueError, TypeError, KeyError, OverflowError):
        return -float('inf')


class EndSearch(Exception):
    pass


class EndSlice(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    begin = time.monotonic()
    deadline = begin + .69*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = score(init) if init is not None else -float('inf')
    fallback = {'circles': [[(i%7-3)*.19999999999998, (i//7-1)*.19999999999998, .09999999999998] for i in range(21)]}
    val = score(fallback)
    if val > best_score:
        best, best_score = fallback, val

    def tick():
        if time.monotonic() >= deadline:
            raise EndSearch()

    group = np.array([0,1,2]+[g for g in range(3,12) for _ in range(2)])
    signs = np.array([0.,0.,0.]+[s for _ in range(9) for s in (1.,-1.)])
    X = np.zeros((21,34))
    Y = np.zeros_like(X)
    R = np.zeros_like(X)
    for k,g in enumerate(group):
        if time.monotonic() >= deadline:
            return best
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
            key = (min(group[i],group[j]),max(group[i],group[j]),int(signs[i]*signs[j]))
            if key not in seen:
                seen.add(key)
                pairs.append((i,j))
    pi,pj = np.asarray(pairs).T
    DX,DY,RS = X[pi]-X[pj],Y[pi]-Y[pj],R[pi]+R[pj]
    walls = np.zeros((36,34))
    offsets = np.zeros(36)
    for g in range(12):
        if time.monotonic() >= deadline:
            return best
        walls[g,g] = 1
        walls[g,21+g] = -1
        walls[g,33] = .5
        walls[12+g,g] = -1
        walls[12+g,21+g] = -1
        walls[12+g,33] = .5
        walls[24+g,21+g] = -1
        walls[24+g,33] = -.5
        offsets[24+g] = 1
        if g >= 3:
            walls[24+g,9+g] = -1
    weights = np.array([1.]*3+[2.]*9)
    grad = np.zeros(34)
    grad[21:33] = -weights
    bounds = [(-1.,1.)]*12+[(0.,1.)]*9+[(1e-7,.6)]*12+[(.08,1.92)]
    release = np.zeros(len(pi))
    ii,jj = np.triu_indices(21,1)
    champion = None
    champion_value = -float('inf')
    cache_v = None
    cache_c = None
    cache_j = None
    slice_end = deadline

    def retain(v):
        nonlocal best,best_score,champion,champion_value
        tick()
        if not np.isfinite(v).all() or np.any(v[21:33] <= 0):
            return
        xx = v[:12]
        yy = np.r_[np.zeros(3),v[12:21]]
        a = np.column_stack((X@v,Y@v,R@v))
        dx = a[ii,0]-a[jj,0]
        dy = a[ii,1]-a[jj,1]
        ratio = min(1.,float(np.min(np.sqrt(dx*dx+dy*dy)/(a[ii,2]+a[jj,2]))))
        if ratio <= 0:
            return
        rr = v[21:33]*ratio*(1-2e-14)
        perimeter = np.max(xx+rr)-np.min(xx-rr)+2*np.max(np.abs(yy)+rr)
        scale = min(1.,2./perimeter)*(1-2e-14)
        estimate = float(weights@rr)*scale
        if estimate <= min(best_score,champion_value):
            return
        obj = {'orbits': [[float(xx[g]*scale),float(yy[g]*scale),float(rr[g]*scale),1 if g<3 else 2] for g in range(12)]}
        exact = score(obj)
        if exact > champion_value:
            champion_value,champion = exact,v.copy()
        if exact > best_score:
            best_score,best = exact,obj

    def check():
        tick()
        if time.monotonic() >= slice_end:
            raise EndSlice()

    def update(v):
        nonlocal cache_v,cache_c,cache_j
        check()
        if cache_v is not None and np.array_equal(v,cache_v):
            return
        dx,dy = DX@v,DY@v
        dd = np.maximum(np.sqrt(dx*dx+dy*dy),1e-25)
        cache_c = np.r_[dd-RS@v,walls@v+offsets]
        cache_j = np.vstack(((dx/dd)[:,None]*DX+(dy/dd)[:,None]*DY-RS,walls))
        cache_v = v.copy()

    def fun(v):
        check()
        return float(grad@v)

    def jac(v):
        check()
        return grad

    def con(v):
        update(v)
        z = cache_c.copy()
        z[:len(pi)] -= release
        return z

    def cjac(v):
        update(v)
        return cache_j

    def solve(v,n,seconds):
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
            if count%4 == 0:
                retain(z)
            check()
        try:
            result = minimize(fun,v,jac=jac,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':con,'jac':cjac},callback=callback,
                options={'maxiter':n,'ftol':3e-13,'disp':False})
            last = result.x
        except EndSlice:
            pass
        retain(last)
        slice_end = deadline
        return last

    def encode(g):
        v = np.zeros(34)
        v[:12] = g[:,0]
        v[12:21] = np.maximum(g[3:,1],0.)
        v[21:33] = np.maximum(g[:,2],1e-7)
        v[33] = np.clip(np.max(g[:,0]+g[:,2])-np.min(g[:,0]-g[:,2]),.08,1.92)
        return v

    def lattice(rows):
        data = [[2.*(j-1),0.,1.] for j in range(3)]
        for k,count in enumerate(rows):
            tick()
            for j in range(count):
                tick()
                data.append([2.*(j-(count-1)/2),(k+1)*np.sqrt(3.),1.])
        a = np.asarray(data)
        w = np.ptp(a[:,0])+2
        h = 2*np.max(a[:,1]+a[:,2])
        return encode(a*(2/(w+h)))

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
                        ref = b[lower].copy()
                        ref[:,1] *= -1
                        cost = np.sum((b[upper,None,:]-ref[None,:,:])**2,axis=2)
                        u,v = linear_sum_assignment(cost)
                        g = np.vstack((np.column_stack((b[singles,0],np.zeros(3),b[singles,2])),(b[upper[u]]+ref[v])/2))
                        error = float(np.sum(cost[u,v])+np.sum(b[singles,1]**2))
                        projections.append((error,encode(g)))
            except (ValueError,TypeError,KeyError):
                pass
        projections.sort(key=lambda t:t[0])
        starts = []
        if projections:
            starts.append(projections[0][1])
        starts.extend([lattice((4,3,2)),lattice((5,4)),lattice((3,3,3))])
        if len(projections)>1 and projections[1][0] < .015:
            starts.insert(2,projections[1][1])
        finalists = []
        for v in starts:
            tick()
            if time.monotonic()-begin > .43*budget_s and finalists:
                break
            z = solve(v,int(params['screen']),.125*budget_s)
            dx,dy = DX@z,DY@z
            feasible = np.min(np.r_[np.sqrt(dx*dx+dy*dy)-RS@z,walls@z+offsets])
            finalists.append((float(-grad@z) if feasible > -1e-6 else -1.,z))
            if champion_value > 2.3658323:
                break
        if champion is not None:
            anchor = solve(champion.copy(),int(params['polish']),.19*budget_s)
        elif finalists:
            anchor = solve(max(finalists,key=lambda t:t[0])[1],int(params['polish']),.19*budget_s)
        else:
            return best
        attempt = 0
        while time.monotonic() < deadline:
            tick()
            release[:] = 0
            if champion is not None:
                anchor = champion.copy()
            dx,dy = DX@anchor,DY@anchor
            gaps = np.sqrt(dx*dx+dy*dy)-RS@anchor
            near = np.flatnonzero(gaps < .003)
            if len(near)<3:
                break
            chosen = rng.choice(near,min(int(params['contacts']),len(near)),replace=False)
            release[chosen] = float(params['release'])*rng.uniform(.7,1.3,len(chosen))
            moved = solve(anchor,22,.055*budget_s)
            release[:] = 0
            solve(moved,int(params['polish']),.13*budget_s)
            attempt += 1
    except (EndSearch,ValueError,FloatingPointError):
        pass
    return best
