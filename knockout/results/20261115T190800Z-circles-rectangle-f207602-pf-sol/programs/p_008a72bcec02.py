import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {"probe_iterations": (65, 25, 120), "release": (0.012, 0.002, 0.04), "contacts": (4, 3, 6)}


def canonical(obj):
    if 'circles' in obj:
        return obj
    a = []
    for x,y,r,m in obj['orbits']:
        a.append([x,y,r])
        if m == 2:
            a.append([x,-y,r])
    return {'circles': a}


def score(obj):
    try:
        a = np.asarray(canonical(obj)['circles'], dtype=float)
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return -float('inf')
        x = list(map(lambda v: Fraction(float(v)), a[:,0]))
        y = list(map(lambda v: Fraction(float(v)), a[:,1]))
        r = list(map(lambda v: Fraction(float(v)), a[:,2]))
        if max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))+max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)) > 2:
            return -float('inf')
        i,j = np.triu_indices(21,1)
        d = np.sqrt((a[i,0]-a[j,0])**2+(a[i,1]-a[j,1])**2)
        if np.any(d < a[i,2]+a[j,2]):
            return -float('inf')
        return float(np.sum(a[:,2]))
    except Exception:
        return -float('inf')


class StopSearch(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    deadline = time.monotonic()+0.70*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_value = score(init) if init is not None else -float('inf')
    def tick():
        if time.monotonic() >= deadline:
            raise StopSearch()

    # Twelve orbits: three on the axis, nine above it.
    groups = np.r_[np.arange(3),np.repeat(np.arange(3,12),2)]
    signs = np.r_[np.zeros(3),np.tile([1.,-1.],9)]
    X = np.zeros((21,34)); Y = X.copy(); R = X.copy()
    X[np.arange(21),groups] = 1.
    R[np.arange(21),21+groups] = 1.
    mask = groups >= 3
    Y[np.flatnonzero(mask),groups[mask]+9] = signs[mask]
    ai,aj = np.triu_indices(21,1)
    keys = np.column_stack((np.minimum(groups[ai],groups[aj]),np.maximum(groups[ai],groups[aj]),signs[ai]*signs[aj]))
    _,keep = np.unique(keys,axis=0,return_index=True)
    pi,pj = ai[keep],aj[keep]
    DX,DY,RS = X[pi]-X[pj],Y[pi]-Y[pj],R[pi]+R[pj]
    W = np.zeros((36,34)); offset = np.zeros(36)
    q = np.arange(12)
    W[q,q] = 1; W[q,21+q] = -1; W[q,33] = .5
    W[12+q,q] = -1; W[12+q,21+q] = -1; W[12+q,33] = .5
    W[24+q,21+q] = -1; W[24+q,33] = -.5
    W[27+np.arange(9),12+np.arange(9)] = -1
    offset[24:] = 1
    weights = np.r_[np.ones(3),np.full(9,2.)]
    objective = np.zeros(34); objective[21:33] = -weights
    bounds = [(-1.,1.)]*12+[(0.,1.)]*9+[(1e-7,.5)]*12+[(.12,1.88)]
    release = np.zeros(len(pi))
    best_z = None
    last = None

    def retain(v):
        nonlocal best,best_value,best_z
        if v is None or not np.isfinite(v).all():
            return
        a = np.column_stack((X@v,Y@v,R@v))
        if np.any(a[:,2] <= 0):
            return
        d = np.sqrt((a[ai,0]-a[aj,0])**2+(a[ai,1]-a[aj,1])**2)
        factor = min(1.,float(np.min(d/(a[ai,2]+a[aj,2]))))*(1.-2e-14)
        if factor <= 0:
            return
        a[:,2] *= factor
        span = np.max(a[:,:2]+a[:,2,None],axis=0)-np.min(a[:,:2]-a[:,2,None],axis=0)
        scale = 2./float(np.sum(span))*(1.-2e-14)
        a *= scale
        if float(np.sum(a[:,2])) <= best_value:
            return
        obj = {'circles':a.tolist()}
        val = score(obj)
        if val > best_value:
            best,best_value = obj,val
            best_z = v.copy()
            best_z[:33] *= scale
            best_z[21:33] *= factor
            best_z[33] = float(np.max(a[:,0]+a[:,2])-np.min(a[:,0]-a[:,2]))
            # Recenter, preserving the horizontal reflection axis.
            center = .5*(np.max(a[:,0]+a[:,2])+np.min(a[:,0]-a[:,2]))
            best_z[:12] -= center

    def fun(v):
        tick()
        return float(objective@v)
    def jac(v):
        tick()
        return objective
    def con(v):
        tick()
        dx,dy = DX@v,DY@v
        return np.r_[np.sqrt(dx*dx+dy*dy)-RS@v-release,W@v+offset]
    def cjac(v):
        tick()
        dx,dy = DX@v,DY@v
        d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-24)
        return np.vstack(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RS,W))
    calls = 0
    def callback(v):
        nonlocal last,calls
        last = v.copy()
        calls += 1
        tick()
        if calls % 12 == 0:
            retain(v)
    def solve(v,n):
        nonlocal last
        tick()
        last = v.copy()
        out = minimize(fun,v,jac=jac,bounds=bounds,method='SLSQP',constraints={'type':'ineq','fun':con,'jac':cjac},callback=callback,options={'maxiter':int(n),'ftol':2e-12,'disp':False})
        retain(out.x)
        return out.x

    def lattice(which):
        v = np.zeros(34)
        if which == 0:
            xx = np.r_[[-2.,0.,2.],[-3.,-1.,1.,3.],[-4.,-2.,0.,2.,4.]]
            yy = np.r_[np.full(4,np.sqrt(3.)),np.full(5,2*np.sqrt(3.))]
        else:
            xx = np.r_[[-2.,0.,2.],[-1.,1.,3.],[-2.,0.,2.],[-1.,1.,3.]]
            yy = np.repeat(np.arange(1,4)*np.sqrt(3.),3)
        rr = np.ones(12)
        width = np.max(xx+rr)-np.min(xx-rr)
        height = 2*np.max(np.r_[np.zeros(3),yy]+rr)
        s = 2./(width+height)*(1.-1e-13)
        xx -= .5*(np.max(xx+rr)+np.min(xx-rr))
        v[:12] = xx*s; v[12:21] = yy*s; v[21:33] = s; v[33] = width*s
        return v

    try:
        starts = [lattice(0),lattice(1)]
        if best_value > 2.2:
            a = np.asarray(canonical(init)['circles'],float).copy()
            low = np.min(a[:,:2]-a[:,2,None],axis=0)
            high = np.max(a[:,:2]+a[:,2,None],axis=0)
            a[:,:2] -= (low+high)/2
            projections = []
            for swap in (False,True):
                tick()
                b = a[:,[1,0,2]].copy() if swap else a.copy()
                order = np.argsort(b[:,1]); lower,axis,upper = order[:9],order[9:12],order[12:]
                reflected = b[lower].copy(); reflected[:,1] *= -1
                costs = np.sum((b[upper,None,:]-reflected[None,:,:])**2,axis=2)
                u,v = linear_sum_assignment(costs)
                g = np.vstack((b[axis],(b[upper[u]]+reflected[v])/2))
                error = np.sum(costs[u,v])+np.sum(b[axis,1]**2)
                z = np.zeros(34); z[:12] = g[:,0]; z[12:21] = np.abs(g[3:,1]); z[21:33] = g[:,2]; z[33] = high[1 if swap else 0]-low[1 if swap else 0]
                projections.append((error,z))
            starts.insert(0,min(projections,key=lambda p:p[0])[1])
        for z in starts:
            tick()
            retain(z)
            solve(z,params['probe_iterations'])
        attempt = 0
        while time.monotonic() < deadline:
            tick()
            anchor = best_z.copy() if best_z is not None else starts[0].copy()
            release[:] = 0
            if attempt % 3 != 0:
                dx,dy = DX@anchor,DY@anchor
                gap = np.sqrt(dx*dx+dy*dy)-RS@anchor
                near = np.flatnonzero(gap < .004)
                if len(near) >= 3:
                    chosen = rng.choice(near,min(len(near),int(params['contacts'])),replace=False)
                    release[chosen] = params['release']*rng.uniform(.7,1.3,len(chosen))
                    anchor = solve(anchor,20)
                    release[:] = 0
            solve(anchor,80)
            attempt += 1
    except (StopSearch,ValueError,FloatingPointError):
        pass
    retain(last)
    return best
