import time
import numpy as np
from fractions import Fraction
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {"first_slice": (0.55, 0.2, 0.8), "jitter": (0.012, 0.001, 0.05)}


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -float('inf')
    ii, jj = np.triu_indices(21, 1)

    def exact(c):
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return -float('inf')
        d = np.sqrt((c[ii, 0]-c[jj, 0])**2 + (c[ii, 1]-c[jj, 1])**2)
        if np.any(d < c[ii, 2]+c[jj, 2]):
            return -float('inf')
        xs = [Fraction(float(v)) for v in c[:, 0]]
        ys = [Fraction(float(v)) for v in c[:, 1]]
        rs = [Fraction(float(v)) for v in c[:, 2]]
        w = max(x+r for x,r in zip(xs,rs))-min(x-r for x,r in zip(xs,rs))
        h = max(y+r for y,r in zip(ys,rs))-min(y-r for y,r in zip(ys,rs))
        if w+h > 2:
            return -float('inf')
        return float(np.sum(c[:, 2]))

    raw = None
    if init is not None:
        try:
            if 'circles' in init:
                raw = np.asarray(init['circles'], dtype=float)
            else:
                q = []
                for x,y,r,m in init['orbits']:
                    if time.monotonic() >= deadline:
                        return init
                    q.append([x,y,r])
                    if m == 2:
                        q.append([x,-y,r])
                raw = np.array(q)
            best_score = exact(raw)
        except Exception:
            raw = None

    def consider(c):
        nonlocal best, best_score
        if time.monotonic() >= deadline or not np.isfinite(c).all():
            return
        c = c.copy()
        c[:, 2] = np.maximum(c[:, 2], 1e-12)
        dist = np.sqrt((c[ii,0]-c[jj,0])**2+(c[ii,1]-c[jj,1])**2)
        shrink = min(1.0, float(np.min(dist/(c[ii,2]+c[jj,2]))))
        if shrink <= 0:
            return
        c[:,2] *= shrink * (1-3e-14)
        c[:,:2] -= np.mean(c[:,:2], axis=0)
        wh = np.sum(np.max(c[:,:2]+c[:,2,None], axis=0)-np.min(c[:,:2]-c[:,2,None], axis=0))
        c *= (2.0/wh)*(1-3e-14)
        score = exact(c)
        if score > best_score:
            best_score = score
            best = {'circles': c.tolist()}

    # x[0:12], positive y[12:21], radius[21:33], width[33].
    a,b = np.triu_indices(12,1)
    mult = np.r_[np.ones(3), np.full(9,2.)]
    grad = np.r_[np.zeros(21), -mult, 0.]

    def unpack(z):
        return z[:12], np.r_[np.zeros(3),z[12:21]], z[21:33], z[33]

    def expand(z):
        x,y,r,w = unpack(z)
        c = np.column_stack((x,y,r))
        return np.vstack((c, np.column_stack((x[3:],-y[3:],r[3:]))))

    def projection(c, transpose=False):
        c = c.copy()
        if transpose:
            c[:,[0,1]] = c[:,[1,0]]
        order = np.argsort(c[:,1])
        lo,mid,hi = order[:9],order[9:12],order[12:]
        cy = np.mean(c[mid,1])
        cost = (c[lo,0,None]-c[hi,0])**2 + (c[lo,1,None]+c[hi,1]-2*cy)**2
        cost += .4*(c[lo,2,None]-c[hi,2])**2
        p,q = linear_sum_assignment(cost)
        low,high = c[lo[p]],c[hi[q]]
        x = np.r_[c[mid,0],(low[:,0]+high[:,0])/2]
        y = (high[:,1]-low[:,1])/2
        r = np.r_[c[mid,2],(low[:,2]+high[:,2])/2]
        left = np.min(x-r)
        x -= left
        width = np.max(x+r)
        height = 2*np.max(np.r_[r[:3], y+r[3:]])
        scale = 2/(width+height)
        z = np.r_[x*scale,y*scale,r*scale,width*scale]
        return float(np.sum(cost[p,q])), z

    def lattice(reverse=False):
        d = 2/(6+2*np.sqrt(3))
        x0 = np.array([1.,2.,3.])
        row1 = np.arange(4)+.5 if reverse else np.arange(5)
        row2 = np.arange(5) if reverse else np.arange(4)+.5
        x = np.r_[x0,row1,row2]+.5
        y = np.r_[np.full(len(row1),np.sqrt(3)/2),np.full(len(row2),np.sqrt(3))]
        return np.r_[x*d,y*d,np.full(12,.5*d),5*d]

    class Stop(Exception):
        pass

    best_z = None
    best_z_score = -float('inf')
    bounds = [(0.,2.)]*12+[(0.,1.)]*9+[(1e-6,.7)]*12+[(.1,1.9)]

    def solve(z, seconds, iterations=180):
        nonlocal best_z, best_z_score
        until = min(deadline-0.012, time.monotonic()+seconds)
        if time.monotonic() >= until:
            return z
        cache_x = None
        cache_c = None
        cache_j = None
        last = z.copy()
        def calc(v):
            nonlocal cache_x,cache_c,cache_j
            if time.monotonic() >= until:
                raise Stop()
            if cache_x is not None and np.array_equal(cache_x,v):
                return cache_c,cache_j
            x,y,r,w = unpack(v)
            dx=x[a]-x[b]; dy=y[a]-y[b]; sr=r[a]+r[b]
            cp=dx*dx+dy*dy-sr*sr
            c=np.r_[cp,x-r,w-x-r,1-w/2-y-r,y[3:]-r[3:]]
            j=np.zeros((111,34))
            rows=np.arange(66)
            j[rows,a]=2*dx; j[rows,b]=-2*dx
            mask=a>=3
            j[rows[mask],12+a[mask]-3]=2*dy[mask]
            mask=b>=3
            j[rows[mask],12+b[mask]-3]=-2*dy[mask]
            j[rows,21+a]=-2*sr; j[rows,21+b]=-2*sr
            t=np.arange(12)
            j[66+t,t]=1; j[66+t,21+t]=-1
            j[78+t,t]=-1; j[78+t,21+t]=-1; j[78+t,33]=1
            j[90+t,21+t]=-1; j[90+t,33]=-.5
            j[93+np.arange(9),12+np.arange(9)]=-1
            j[102+np.arange(9),12+np.arange(9)]=1
            j[102+np.arange(9),24+np.arange(9)]=-1
            cache_x=v.copy(); cache_c=c; cache_j=j
            return c,j
        def callback(v):
            nonlocal last
            last=v.copy()
            if time.monotonic() >= until:
                raise Stop()
        try:
            res=minimize(lambda v: float(grad@v),z,jac=lambda v:grad,
                         constraints={'type':'ineq','fun':lambda v:calc(v)[0], 'jac':lambda v:calc(v)[1]},
                         bounds=bounds,method='SLSQP',callback=callback,
                         options={'maxiter':iterations,'ftol':2e-12,'disp':False})
            last=res.x
        except (Stop, ValueError, FloatingPointError):
            pass
        if time.monotonic() < deadline:
            c=expand(last)
            before=best_score
            consider(c)
            if best_score > best_z_score and best_score > before:
                best_z_score=best_score
                best_z=last.copy()
        return last

    seeds=[]
    if raw is not None and raw.shape == (21,3) and np.isfinite(raw).all():
        seeds=[projection(raw,False),projection(raw,True)]
        seeds.sort(key=lambda t:t[0])
    else:
        seeds=[(0.,lattice(False))]
    z=solve(seeds[0][1], float(params['first_slice']))
    if time.monotonic() < deadline-.12:
        # A second reflection direction can recover perturbed pair assignments.
        if len(seeds)>1 and seeds[1][0] < 4*seeds[0][0]+.001:
            solve(seeds[1][1], .27)
        else:
            solve(lattice(False), .25)
    if time.monotonic() < deadline-.15:
        solve(lattice(True), .23)
    if best_z is not None and time.monotonic() < deadline-.08:
        solve(best_z, max(.02,deadline-time.monotonic()-.035),240)
    while time.monotonic() < deadline-.12:
        base=best_z.copy() if best_z is not None else z.copy()
        base[:21] += rng.normal(0,float(params['jitter']),21)
        base[:12]=np.clip(base[:12],.001,1.999)
        base[12:21]=np.clip(base[12:21],.001,.999)
        solve(base,min(.18,deadline-time.monotonic()-.025),70)
    if best is None:
        c=np.array([[.2*(k%5),.2*(k//5),.099999999999] for k in range(21)])
        return {'circles':c.tolist()}
    return best
