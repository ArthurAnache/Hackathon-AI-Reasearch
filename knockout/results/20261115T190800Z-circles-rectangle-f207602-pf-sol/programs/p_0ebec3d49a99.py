import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {
    'release_gap': (0.0012, 0.00005, 0.02, 'log'),
    'release_count': (3, 3, 6),
    'probes': (3, 1, 8),
}

class Timeout(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    bestscore = -float('inf')
    ii, jj = np.triu_indices(21, 1)

    def exact_score(obj):
        try:
            c = np.asarray(obj['circles'], dtype=float)
            if c.shape != (21, 3) or not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
                return -float('inf')
            d = c[ii, :2] - c[jj, :2]
            if np.any(np.sqrt(d[:, 0]**2 + d[:, 1]**2) < c[ii, 2] + c[jj, 2]):
                return -float('inf')
            xs = [Fraction(float(v)) for v in c[:, 0]]
            ys = [Fraction(float(v)) for v in c[:, 1]]
            rs = [Fraction(float(v)) for v in c[:, 2]]
            w = max(x+r for x,r in zip(xs,rs))-min(x-r for x,r in zip(xs,rs))
            h = max(y+r for y,r in zip(ys,rs))-min(y-r for y,r in zip(ys,rs))
            if w+h > 2:
                return -float('inf')
            return float(np.sum(c[:, 2]))
        except Exception:
            return -float('inf')

    if init is not None:
        if 'circles' not in init:
            cc = []
            for x,y,r,m in init['orbits']:
                if time.monotonic() >= deadline:
                    return init
                cc.append([x,y,r])
                if m == 2:
                    cc.append([x,-y,r])
            init = {'circles': cc}
            best = init
        bestscore = exact_score(init)
        raw = np.array(init['circles'], dtype=float)
    else:
        cc = []
        for row,n in enumerate([4,4,5,4,4]):
            if time.monotonic() >= deadline:
                return best
            for k in range(n):
                if time.monotonic() >= deadline:
                    return best
                cc.append([(k-(n-1)/2)*0.20, (row-2)*0.18, 0.09])
        raw = np.array(cc)

    def save(c):
        nonlocal best, bestscore
        if time.monotonic() >= deadline:
            return
        c = np.array(c, copy=True)
        if not np.isfinite(c).all() or np.min(c[:,2]) <= 0:
            return
        c[:,:2] -= np.mean(c[:,:2], axis=0)
        d = c[ii,:2]-c[jj,:2]
        distances = np.sqrt(d[:,0]**2+d[:,1]**2)
        ratio = float(np.min(distances/(c[ii,2]+c[jj,2])))
        if ratio < 1.0:
            c[:,2] *= max(0.0, ratio)*(1-3e-15)
        w = np.max(c[:,0]+c[:,2])-np.min(c[:,0]-c[:,2])
        h = np.max(c[:,1]+c[:,2])-np.min(c[:,1]-c[:,2])
        c *= min(1.0, 2.0/(w+h))*(1-4e-15)
        obj = {'circles': c.tolist()}
        score = exact_score(obj)
        if score > bestscore:
            bestscore, best = score, obj

    # Identify the reflection axis in either orientation.
    projections = []
    for flip in (False, True):
        if time.monotonic() >= deadline:
            return best
        c = raw.copy()
        if flip:
            c[:,[0,1]] = c[:,[1,0]]
        c[:,1] -= (np.max(c[:,1]+c[:,2])+np.min(c[:,1]-c[:,2]))/2
        singles = np.argsort(np.abs(c[:,1]))[:3]
        rest = np.array([i for i in range(21) if i not in singles])
        order = rest[np.argsort(c[rest,1])]
        neg,pos = order[:9],order[9:]
        cost = ((c[pos,None,0]-c[neg,0])**2 +
                (c[pos,None,1]+c[neg,1])**2 +
                (c[pos,None,2]-c[neg,2])**2)
        a,b = linear_sum_assignment(cost)
        err = np.sum(cost[a,b])+np.sum(c[singles,1]**2)
        projections.append((err,c,singles,pos[a],neg[b]))
    _, c, singles, pos, neg = min(projections, key=lambda q:q[0])
    # Variables: 12 x coordinates, nine positive y coordinates, 12 radii, width.
    nv = 34
    M = np.zeros((21,3,nv))
    z = np.zeros(nv)
    c[:,0] -= np.min(c[:,0]-c[:,2])
    for k in range(12):
        if time.monotonic() >= deadline:
            return best
        if k < 3:
            ids = [int(singles[k])]
            z[k] = c[ids[0],0]
            z[21+k] = c[ids[0],2]
        else:
            ids = [int(pos[k-3]),int(neg[k-3])]
            z[k] = np.mean(c[ids,0])
            z[12+k-3] = (c[ids[0],1]-c[ids[1],1])/2
            z[21+k] = np.mean(c[ids,2])
        for j,index in enumerate(ids):
            if time.monotonic() >= deadline:
                return best
            M[index,0,k] = 1
            M[index,2,21+k] = 1
            if k >= 3:
                M[index,1,12+k-3] = 1 if j == 0 else -1
    z[33] = np.max(c[:,0]+c[:,2])
    # Rectangle is [0,w] x [-(2-w)/2,(2-w)/2].
    X,Y,R = M[:,0,:],M[:,1,:],M[:,2,:]
    ew = np.zeros(nv); ew[33] = 1
    A = np.vstack((X-R, ew-X-R, Y-R-ew/2, -Y-R-ew/2))
    bnd = np.r_[np.zeros(42),np.ones(42)]
    DX,DY,DR = X[ii]-X[jj],Y[ii]-Y[jj],R[ii]+R[jj]
    objective = -np.sum(R,axis=0)
    local_end = deadline
    cached_z = None
    cached_g = None
    cached_j = None

    def check():
        if time.monotonic() >= min(deadline,local_end):
            raise Timeout()

    def calc(v):
        nonlocal cached_z,cached_g,cached_j
        check()
        if cached_z is None or not np.array_equal(v,cached_z):
            dx,dy,rr = DX@v,DY@v,DR@v
            dd = np.maximum(np.sqrt(dx*dx+dy*dy),1e-30)
            cached_g = np.r_[A@v+bnd,dd-rr]
            cached_j = np.vstack((A,(dx/dd)[:,None]*DX+(dy/dd)[:,None]*DY-DR))
            cached_z = v.copy()
        return cached_g,cached_j

    # Remove exact reflection duplicates, retaining all distinct constraints.
    g,jac = calc(z)
    keys = np.round(np.c_[g,jac],11)
    _,keep = np.unique(keys,axis=0,return_index=True)
    keep = np.sort(keep)
    offsets = np.zeros(len(keep))
    last = z.copy()
    polished = z.copy()

    def fun(v):
        check()
        return float(objective@v)
    def grad(v):
        check()
        return objective
    def con(v):
        return calc(v)[0][keep]-offsets
    def jaccon(v):
        return calc(v)[1][keep]
    def callback(v):
        nonlocal last
        check()
        last = v.copy()
        if np.min(calc(v)[0]) > -1e-7:
            save(np.einsum('ijk,k->ij',M,v))

    bounds = [(None,None)]*nv
    for k in range(12,21):
        if time.monotonic() >= deadline:
            return best
        bounds[k] = (0,1)
    for k in range(21,33):
        if time.monotonic() >= deadline:
            return best
        bounds[k] = (1e-6,1)
    bounds[33] = (0.1,1.9)

    def run(v, seconds, iterations=100):
        nonlocal local_end,last
        local_end = min(deadline,time.monotonic()+seconds)
        last = v.copy()
        try:
            res = minimize(fun,v,jac=grad,bounds=bounds,
                constraints={'type':'ineq','fun':con,'jac':jaccon},
                method='SLSQP',callback=callback,
                options={'maxiter':iterations,'ftol':2e-14,'disp':False})
            last = res.x.copy()
        except (Timeout,ValueError,np.linalg.LinAlgError):
            pass
        return last.copy()

    z = run(z, max(0.12,0.30*budget_s),90)
    local_end = deadline
    if time.monotonic() < deadline:
        save(np.einsum('ijk,k->ij',M,z))
    polished = z.copy()
    # Tight contact equations can be solved more accurately than SLSQP's merit function.
    try:
        g,jac = calc(polished)
        active = keep[g[keep] < 2e-7]
        if len(active) >= nv:
            for iteration in range(5):
                check()
                g,jac = calc(z)
                step = np.linalg.lstsq(jac[active],-g[active],rcond=1e-12)[0]
                if np.linalg.norm(step) > 0.01:
                    break
                z += step
                save(np.einsum('ijk,k->ij',M,z))
            if np.min(calc(z)[0]) > -1e-8:
                polished = z.copy()
    except (Timeout,np.linalg.LinAlgError):
        return best

    for probe in range(int(params['probes'])):
        if time.monotonic() + 0.12 >= deadline:
            break
        local_end = deadline
        try:
            g,jac = calc(polished)
            active = np.flatnonzero(g[keep] < 2e-6)
            multipliers = np.linalg.lstsq(jac[keep[active]].T,-objective,rcond=1e-10)[0]
            pair_active = active[keep[active] >= 84]
            if len(pair_active) < 3:
                break
            strength = {int(a):abs(float(m)) for a,m in zip(active,multipliers)}
            ranked = sorted(pair_active,key=lambda a:strength[int(a)])
            pool = np.array(ranked[:max(6,len(ranked)//2)])
            selected = rng.choice(pool,size=min(int(params['release_count']),len(pool)),replace=False)
            offsets[:] = 0
            offsets[selected] = float(params['release_gap'])*(1.6**probe)
            perturbed = run(polished,0.12,30)
            offsets[:] = 0
            result = run(perturbed,min(0.24,max(0.01,deadline-time.monotonic())),65)
            local_end = deadline
            if time.monotonic() < deadline:
                save(np.einsum('ijk,k->ij',M,result))
        except (Timeout,ValueError,np.linalg.LinAlgError):
            break
    return best
