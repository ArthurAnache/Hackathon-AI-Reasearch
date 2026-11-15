import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {'iterations': (85, 35, 150), 'release': (0.012, 0.002, 0.05), 'portfolio': (4, 2, 6), 'screen_iterations': (23, 12, 40)}

class StopSearch(Exception):
    pass

class EndSlice(Exception):
    pass

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
        a = np.asarray(canonical(obj)['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
        if max(u+t for u,t in zip(x,r))-min(u-t for u,t in zip(x,r))+max(u+t for u,t in zip(y,r))-min(u-t for u,t in zip(y,r)) > 2:
            return -float('inf')
        for i in range(21):
            for j in range(i):
                dx = a[i,0]-a[j,0]
                dy = a[i,1]-a[j,1]
                if np.sqrt(dx*dx+dy*dy) < a[i,2]+a[j,2]:
                    return -float('inf')
        return float(np.sum(a[:,2]))
    except Exception:
        return -float('inf')

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + 0.69*budget_s
    rng = np.random.default_rng(seed)
    best = init
    bestscore = score(init) if init is not None else -float('inf')
    def tick():
        if time.monotonic() >= deadline:
            raise StopSearch()
    groups = []
    signs = []
    for g in range(12):
        tick()
        groups.append(g)
        signs.append(0 if g < 3 else 1)
        if g >= 3:
            groups.append(g)
            signs.append(-1)
    groups = np.array(groups)
    signs = np.array(signs)
    X = np.zeros((21,34))
    Y = X.copy()
    R = X.copy()
    X[np.arange(21),groups] = 1
    R[np.arange(21),21+groups] = 1
    mask = groups >= 3
    Y[np.flatnonzero(mask),groups[mask]+9] = signs[mask]
    pairs = []
    seen = set()
    for i in range(21):
        tick()
        for j in range(i):
            tick()
            key = (min(groups[i],groups[j]),max(groups[i],groups[j]),signs[i]*signs[j])
            if key not in seen:
                seen.add(key)
                pairs.append((i,j))
    pi,pj = np.array(pairs).T
    DX = X[pi]-X[pj]
    DY = Y[pi]-Y[pj]
    RS = R[pi]+R[pj]
    W = np.zeros((36,34))
    off = np.zeros(36)
    for g in range(12):
        tick()
        W[g,g] = 1
        W[g,21+g] = -1
        W[g,33] = .5
        W[12+g,g] = -1
        W[12+g,21+g] = -1
        W[12+g,33] = .5
        W[24+g,21+g] = -1
        W[24+g,33] = -.5
        off[24+g] = 1
        if g >= 3:
            W[24+g,g+9] = -1
    weights = np.r_[np.ones(3),np.full(9,2.)]
    gradient = np.zeros(34)
    gradient[21:33] = -weights
    release = np.zeros(len(pi))
    bounds = [(-1.,1.)]*12+[(0.,1.)]*9+[(1e-7,.5)]*12+[(.12,1.88)]
    ii,jj = np.triu_indices(21,1)
    bestz = None

    def repaired(z):
        tick()
        if not np.isfinite(z).all() or np.any(z[21:33] <= 0):
            return None, -float('inf'), None
        c = np.column_stack((X@z,Y@z,R@z))
        d = np.hypot(c[ii,0]-c[jj,0],c[ii,1]-c[jj,1])
        shrink = min(1.,float(np.min(d/(c[ii,2]+c[jj,2]))))*(1.-2e-13)
        if shrink <= 0:
            return None, -float('inf'), None
        rr = z[21:33]*shrink
        xx = z[:12]
        yy = np.r_[np.zeros(3),z[12:21]]
        width = np.max(xx+rr)-np.min(xx-rr)
        perimeter = width+2*np.max(yy+rr)
        scale = 2./perimeter*(1.-2e-13)
        predicted = float(weights@rr)*scale
        zz = z.copy()
        zz[:21] *= scale
        zz[21:33] = rr*scale
        zz[33] = width*scale
        return zz,predicted,(xx,yy,rr,scale)

    def retain(z):
        nonlocal best,bestscore,bestz
        zz,predicted,parts = repaired(z)
        if predicted <= bestscore or parts is None:
            return predicted
        xx,yy,rr,scale = parts
        obj = {'orbits': [[float(xx[g]*scale),float(yy[g]*scale),float(rr[g]*scale),1 if g<3 else 2] for g in range(12)]}
        val = score(obj)
        if val > bestscore:
            best = obj
            bestscore = val
            bestz = zz
        return predicted

    local_deadline = deadline
    def check_solve():
        tick()
        if time.monotonic() >= local_deadline:
            raise EndSlice()
    def fun(z):
        check_solve()
        return float(gradient@z)
    def jac(z):
        check_solve()
        return gradient
    def con(z):
        check_solve()
        dx = DX@z
        dy = DY@z
        return np.r_[np.hypot(dx,dy)-RS@z-release,W@z+off]
    def cj(z):
        check_solve()
        dx = DX@z
        dy = DY@z
        d = np.maximum(np.hypot(dx,dy),1e-25)
        return np.vstack(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RS,W))

    def solve(z,it,seconds=None):
        nonlocal local_deadline
        tick()
        local_deadline = deadline if seconds is None else min(deadline,time.monotonic()+seconds)
        last = z.copy()
        count = 0
        def callback(v):
            nonlocal last,count
            tick()
            last = v.copy()
            count += 1
            if count%5 == 0:
                retain(v)
            check_solve()
        try:
            res = minimize(fun,z,jac=jac,method='SLSQP',bounds=bounds,constraints={'type':'ineq','fun':con,'jac':cj},callback=callback,options={'maxiter':it,'ftol':2e-12,'disp':False})
            last = res.x
        except EndSlice:
            pass
        retain(last)
        return last

    def lattice(rows,variant=0):
        x = [-2.,0.,2.]
        y = []
        for level,n in enumerate(rows,1):
            tick()
            xs = 2*np.arange(n)-(n-1)
            if variant:
                xs = xs+(.25 if level%2 else -.25)
            x.extend(xs.tolist())
            y.extend([level*np.sqrt(3.)]*n)
        x = np.array(x)
        y = np.array(y)
        width = float(np.max(x+1)-np.min(x-1))
        height = 2*(max(y)+1)
        s = 2/(width+height)
        z = np.zeros(34)
        z[:12] = x*s
        z[12:21] = y*s
        z[21:33] = s
        z[33] = width*s
        return z

    try:
        z = lattice((4,5))
        retain(z)
        starts = []
        if init is not None and score(init) > 2.25:
            a = np.asarray(canonical(init)['circles'],float)
            for swap in (False,True):
                tick()
                b = a.copy()
                if swap:
                    b[:,[0,1]] = b[:,[1,0]]
                lo = np.min(b[:,:2]-b[:,2,None],axis=0)
                hi = np.max(b[:,:2]+b[:,2,None],axis=0)
                b[:,:2] -= (lo+hi)/2
                order = np.argsort(np.abs(b[:,1]))
                singles = order[:3]
                upper = np.flatnonzero(b[:,1]>1e-6)
                if len(upper)==9 and np.max(np.abs(b[singles,1]))<1e-5:
                    z = np.zeros(34)
                    inds = np.r_[singles,upper]
                    z[:12] = b[inds,0]
                    z[12:21] = b[upper,1]
                    z[21:33] = b[inds,2]
                    z[33] = hi[0]-lo[0]
                    starts.append(z)
        for rows,var in [((4,5),0),((3,3,3),0),((4,3,2),0),((5,4),0),((3,3,3),1),((4,5),1)][:int(params['portfolio'])]:
            tick()
            starts.append(lattice(rows,var))
        # Screen all families within a shared fraction of the search budget.
        screened = []
        screen_end = started+0.36*budget_s
        for k,z in enumerate(starts):
            tick()
            allowance = max(.012,(screen_end-time.monotonic())/(len(starts)-k))
            result = solve(z,int(params['screen_iterations']),allowance)
            feasible,merit,_ = repaired(result)
            screened.append((merit,k,result))
        screened.sort(key=lambda item:(-item[0],item[1]))
        # Preserve raw trajectories: global radius repair is for scoring only.
        for rank,(_,_,z) in enumerate(screened[:2]):
            tick()
            remaining = deadline-time.monotonic()
            allowance = remaining*(.62 if rank == 0 else .75)
            solve(z,int(params['iterations']),allowance)
        if bestz is not None:
            solve(bestz.copy(),int(params['iterations']),max(.01,(deadline-time.monotonic())*.55))
        while time.monotonic() < deadline:
            tick()
            if bestz is None:
                break
            anchor = bestz.copy()
            gaps = np.hypot(DX@anchor,DY@anchor)-RS@anchor
            near = np.flatnonzero(gaps<.002)
            if len(near)<3:
                break
            chosen = rng.choice(near,min(4,len(near)),replace=False)
            release[chosen] = float(params['release'])*rng.uniform(.6,1.4,len(chosen))
            displaced = solve(anchor,22)
            release[:] = 0
            solve(displaced,int(params['iterations']))
    except (StopSearch,ValueError,FloatingPointError):
        pass
    return best
