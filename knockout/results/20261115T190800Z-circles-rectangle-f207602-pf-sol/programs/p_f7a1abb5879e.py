import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {'probe_iters': (48, 20, 100), 'polish_iters': (95, 40, 160), 'jitter': (0.024, 0.003, 0.08), 'release': (0.009, 0.002, 0.03)}

class StopSearch(Exception):
    pass

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
        a = np.asarray(canonical(obj)['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
        w = max(u+t for u,t in zip(x,r))-min(u-t for u,t in zip(x,r))
        h = max(u+t for u,t in zip(y,r))-min(u-t for u,t in zip(y,r))
        if w+h > 2:
            return -float('inf')
        i,j = np.triu_indices(21,1)
        d = np.sqrt((a[i,0]-a[j,0])**2+(a[i,1]-a[j,1])**2)
        if np.any(d < a[i,2]+a[j,2]):
            return -float('inf')
        return float(np.sum(a[:,2]))
    except Exception:
        return -float('inf')

def construct(params, seed, budget_s, init=None):
    deadline = time.monotonic()+0.70*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_value = score(init)
    fallback = {'circles': [[2*(k%7)/10.00000000001, 2*(k//7)/10.00000000001, 1/10.00000000001] for k in range(21)]}
    fv = score(fallback)
    if fv > best_value:
        best, best_value = fallback, fv
    def tick():
        if time.monotonic() >= deadline:
            raise StopSearch()

    groups = np.r_[np.arange(3), np.repeat(np.arange(3,12),2)]
    signs = np.r_[np.zeros(3), np.tile([1.,-1.],9)]
    X = np.zeros((21,34)); Y = np.zeros_like(X); R = np.zeros_like(X)
    X[np.arange(21),groups] = 1
    R[np.arange(21),21+groups] = 1
    mask = groups >= 3
    Y[np.flatnonzero(mask),groups[mask]+9] = signs[mask]
    ai,aj = np.triu_indices(21,1)
    keys = np.column_stack((np.minimum(groups[ai],groups[aj]),np.maximum(groups[ai],groups[aj]),signs[ai]*signs[aj]))
    _, keep = np.unique(keys,axis=0,return_index=True)
    pi,pj = ai[keep],aj[keep]
    DX,DY,RS = X[pi]-X[pj],Y[pi]-Y[pj],R[pi]+R[pj]
    W = np.zeros((36,34)); off = np.zeros(36)
    q = np.arange(12)
    W[q,q] = 1; W[q,21+q] = -1; W[q,33] = .5
    W[12+q,q] = -1; W[12+q,21+q] = -1; W[12+q,33] = .5
    W[24+q,21+q] = -1; W[24+q,33] = -.5
    W[27+np.arange(9),12+np.arange(9)] = -1
    off[24:] = 1
    weights = np.r_[np.ones(3),np.full(9,2.)]
    grad = np.zeros(34); grad[21:33] = -weights
    bounds = [(-1.,1.)]*12+[(0.,1.)]*9+[(1e-8,.5)]*12+[(.12,1.88)]
    release = np.zeros(len(pi))
    best_z = None
    last = None

    def retain(z):
        nonlocal best,best_value,best_z
        if z is None or not np.isfinite(z).all() or np.any(z[21:33] <= 0):
            return
        a = np.column_stack((X@z,Y@z,R@z))
        d = np.sqrt((a[ai,0]-a[aj,0])**2+(a[ai,1]-a[aj,1])**2)
        factor = min(1.,float(np.min(d/(a[ai,2]+a[aj,2]))))*(1-3e-14)
        if factor <= 0:
            return
        a[:,2] *= factor
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        scale = 2/float(np.sum(hi-lo))*(1-3e-14)
        a *= scale
        if float(np.sum(a[:,2])) <= best_value:
            return
        obj = {'circles': a.tolist()}
        value = score(obj)
        if value > best_value:
            best,best_value = obj,value
            best_z = z.copy()
            best_z[:33] *= scale
            best_z[21:33] *= factor
            center = .5*(np.max(a[:,0]+a[:,2])+np.min(a[:,0]-a[:,2]))
            best_z[:12] -= center
            best_z[33] = float(np.max(a[:,0]+a[:,2])-np.min(a[:,0]-a[:,2]))

    def fun(z):
        tick()
        return float(grad@z)
    def jac(z):
        tick()
        return grad
    def con(z):
        tick()
        dx,dy = DX@z,DY@z
        return np.r_[np.sqrt(dx*dx+dy*dy)-RS@z-release,W@z+off]
    def cj(z):
        tick()
        dx,dy = DX@z,DY@z
        d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-24)
        return np.vstack(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RS,W))
    def solve(z,n):
        nonlocal last
        tick()
        last = z.copy()
        counter = 0
        def cb(v):
            nonlocal last,counter
            last = v.copy()
            tick()
            counter += 1
            if counter % 6 == 0:
                retain(v)
        res = minimize(fun,z,jac=jac,bounds=bounds,method='SLSQP',constraints={'type':'ineq','fun':con,'jac':cj},callback=cb,options={'maxiter':int(n),'ftol':1e-12,'disp':False})
        retain(res.x)
        return res.x
    def pack(g):
        z = np.zeros(34)
        z[:12] = g[:,0]
        z[12:21] = np.abs(g[3:,1])
        z[21:33] = np.maximum(g[:,2],1e-5)
        lo = np.min(g[:,0]-z[21:33]); hi = np.max(g[:,0]+z[21:33])
        z[:12] -= (lo+hi)/2
        h = 2*np.max(np.abs(g[:,1])+z[21:33])
        s = 1.99/(hi-lo+h)
        z[:33] *= s
        z[33] = (hi-lo)*s
        return z
    def lattice(which):
        if which == 0:
            xx = np.r_[[-2.,0.,2.],[-3.,-1.,1.,3.],[-4.,-2.,0.,2.,4.]]
            yy = np.r_[np.zeros(3),np.full(4,np.sqrt(3.)),np.full(5,2*np.sqrt(3.))]
        else:
            xx = np.r_[[-2.,0.,2.],[-1.,1.,3.],[-2.,0.,2.],[-1.,1.,3.]]
            yy = np.r_[np.zeros(3),np.repeat(np.arange(1,4)*np.sqrt(3.),3)]
        return pack(np.column_stack((xx,yy,np.ones(12))))

    try:
        seeds = []
        projected = []
        try:
            a = np.asarray(canonical(init)['circles'],float).copy()
            if a.shape == (21,3) and np.isfinite(a).all() and np.all(a[:,2]>0):
                lo = np.min(a[:,:2]-a[:,2,None],axis=0)
                hi = np.max(a[:,:2]+a[:,2,None],axis=0)
                a[:,:2] -= (lo+hi)/2
                for swap in (False,True):
                    tick()
                    b = a[:,[1,0,2]].copy() if swap else a.copy()
                    order = np.argsort(b[:,1])
                    lower,axis,upper = order[:9],order[9:12],order[12:]
                    reflected = b[lower].copy(); reflected[:,1] *= -1
                    costs = np.sum((b[upper,None,:]-reflected[None,:,:])**2,axis=2)
                    u,v = linear_sum_assignment(costs)
                    singles = b[axis].copy(); singles[:,1] = 0
                    g = np.vstack((singles,(b[upper[u]]+reflected[v])/2))
                    error = float(np.sum(costs[u,v])+np.sum(b[axis,1]**2))
                    projected.append((error,pack(g)))
                projected.sort(key=lambda t:t[0])
        except (KeyError,TypeError,ValueError):
            pass
        if projected and best_value > 2.25:
            seeds.append(projected[0][1])
        seeds.append(lattice(0))
        if projected and best_value <= 2.25:
            seeds.append(projected[0][1])
        seeds.append(lattice(1))
        results = []
        for z in seeds:
            tick()
            retain(z)
            z = solve(z,params['probe_iters'])
            violation = max(0.,-float(np.min(con(z))))
            results.append((float(weights@z[21:33])-30*violation,z.copy()))
        results.sort(key=lambda t:t[0],reverse=True)
        anchor = best_z.copy() if best_z is not None else results[0][1].copy()
        anchor = solve(anchor,params['polish_iters'])
        attempt = 0
        while time.monotonic() < deadline:
            tick()
            release[:] = 0
            z = best_z.copy() if best_z is not None else anchor.copy()
            if attempt == 0 and len(results)>1:
                z = results[1][1].copy()
            elif attempt % 3 == 2:
                dx,dy = DX@z,DY@z
                gap = np.sqrt(dx*dx+dy*dy)-RS@z
                near = np.flatnonzero(gap < .003)
                if len(near) >= 4:
                    chosen = rng.choice(near,4,replace=False)
                    release[chosen] = params['release']*rng.uniform(.7,1.3,4)
                    z = solve(z,18)
                    release[:] = 0
            else:
                z[:21] += rng.normal(0,params['jitter'],21)
                z[:12] = np.clip(z[:12],-.98,.98)
                z[12:21] = np.clip(z[12:21],.0001,.98)
            solve(z,params['polish_iters'])
            attempt += 1
    except (StopSearch,ValueError,FloatingPointError):
        pass
    retain(last)
    return best
