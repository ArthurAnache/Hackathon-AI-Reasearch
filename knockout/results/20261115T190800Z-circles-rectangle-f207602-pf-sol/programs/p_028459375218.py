import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment, root
from scipy.linalg import qr

PARAMS = {'release': (0.009, 0.002, 0.04), 'contacts': (4, 3, 7), 'probe_iters': (24, 10, 60), 'polish_iters': (65, 20, 120), 'jitter': (0.012, 0.001, 0.08)}

class Timeout(Exception):
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
        a = np.asarray(canonical(obj)['circles'], dtype=np.float64)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        xs = [Fraction(float(v)) for v in a[:, 0]]
        ys = [Fraction(float(v)) for v in a[:, 1]]
        rs = [Fraction(float(v)) for v in a[:, 2]]
        w = max(x+r for x,r in zip(xs,rs))-min(x-r for x,r in zip(xs,rs))
        h = max(y+r for y,r in zip(ys,rs))-min(y-r for y,r in zip(ys,rs))
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
    deadline = time.monotonic()+0.69*budget_s
    rng = np.random.default_rng(seed)
    best = init
    bestscore = score(init) if init is not None else -float('inf')
    def tick():
        if time.monotonic() >= deadline:
            raise Timeout()
    fallback = np.array([[2*(i%7),2*(i//7),1.] for i in range(21)],float)/10.000000000001
    fb = {'circles': fallback.tolist()}
    fs = score(fb)
    if fs > bestscore:
        best,bestscore = fb,fs
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
    W = np.zeros((36,34)); off = np.zeros(36)
    q = np.arange(12)
    W[q,q] = 1.; W[q,21+q] = -1.; W[q,33] = .5
    W[12+q,q] = -1.; W[12+q,21+q] = -1.; W[12+q,33] = .5
    W[24+q,21+q] = -1.; W[24+q,33] = -.5
    W[27+np.arange(9),12+np.arange(9)] = -1.
    off[24:] = 1.
    grad = np.zeros(34); grad[21:33] = -np.r_[np.ones(3),np.full(9,2.)]
    bounds = [(-1.,1.)]*12+[(0.,1.)]*9+[(1e-6,.5)]*12+[(.15,1.85)]
    release = np.zeros(len(pi))
    last = None
    anchor = None
    def values(z):
        tick()
        dx,dy = DX@z,DY@z
        return np.r_[np.sqrt(dx*dx+dy*dy)-RS@z,W@z+off]
    def derivatives(z):
        tick()
        dx,dy = DX@z,DY@z
        d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-25)
        return np.vstack(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RS,W))
    def con(z):
        v = values(z)
        v[:len(pi)] -= release
        return v
    def retain(z):
        nonlocal best,bestscore
        if z is None or not np.isfinite(z).all() or np.min(z[21:33]) <= 0:
            return
        c = np.column_stack((X@z,Y@z,R@z))
        d = np.sqrt((c[ai,0]-c[aj,0])**2+(c[ai,1]-c[aj,1])**2)
        f = min(1.,float(np.min(d/(c[ai,2]+c[aj,2]))))*(1.-2e-15)
        if f <= 0:
            return
        c[:,2] *= f
        lo = np.min(c[:,:2]-c[:,2,None],axis=0)
        hi = np.max(c[:,:2]+c[:,2,None],axis=0)
        c[:,:2] -= (lo+hi)/2
        c *= 2./float(np.sum(hi-lo))*(1.-2e-15)
        if float(np.sum(c[:,2])) <= bestscore:
            return
        obj = {'circles':c.tolist()}
        v = score(obj)
        if v > bestscore:
            best,bestscore = obj,v
    def fun(z):
        tick()
        return float(grad@z)
    def jac(z):
        tick()
        return grad
    def cb(z):
        nonlocal last
        last = z.copy()
        tick()
    def solve(z,n):
        nonlocal last
        tick()
        last = z.copy()
        res = minimize(fun,z,jac=jac,bounds=bounds,method='SLSQP',constraints={'type':'ineq','fun':con,'jac':derivatives},callback=cb,options={'maxiter':int(n),'ftol':8e-15,'disp':False})
        retain(res.x)
        return res.x
    def contact_polish(z):
        nonlocal last
        tick()
        v = values(z)
        active = np.flatnonzero(np.abs(v)<2e-7)
        if len(active)<34:
            return z
        J = derivatives(z)[active]
        _,rr,piv = qr(J.T,pivoting=True,mode='economic')
        if abs(rr[33,33]) < 1e-9:
            return z
        chosen = active[piv[:34]]
        def f(u):
            nonlocal last
            last = u.copy()
            return values(u)[chosen]
        def j(u):
            return derivatives(u)[chosen]
        out = root(f,z,jac=j,method='hybr',options={'xtol':1e-11,'maxfev':30})
        retain(out.x)
        if np.min(values(out.x)) > -1e-10:
            return out.x
        return z
    try:
        a = np.asarray(canonical(best)['circles'],float).copy()
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        a[:,:2] -= (lo+hi)/2
        projections = []
        for swap in (False,True):
            tick()
            b = a[:,[1,0,2]].copy() if swap else a.copy()
            order = np.argsort(b[:,1]); lower,axis,upper = order[:9],order[9:12],order[12:]
            reflected = b[lower].copy(); reflected[:,1] *= -1
            costs = np.sum((b[upper,None,:]-reflected[None,:,:])**2,axis=2)
            u,v = linear_sum_assignment(costs)
            g = np.vstack((b[axis],(b[upper[u]]+reflected[v])/2))
            z = np.zeros(34)
            z[:12] = g[:,0]; z[12:21] = np.abs(g[3:,1]); z[21:33] = g[:,2]
            z[33] = hi[int(swap)]-lo[int(swap)]
            error = float(np.sum(costs[u,v])+np.sum(b[axis,1]**2))
            projections.append((error,z))
        projections.sort(key=lambda item:item[0])
        anchor = projections[0][1]
        retain(anchor)
        anchor = contact_polish(anchor)
        anchor = solve(anchor,params['polish_iters'])
        anchor = contact_polish(anchor)
        attempt = 0
        while time.monotonic() < deadline:
            tick()
            z = anchor.copy()
            release[:] = 0.
            if attempt % 3 != 2:
                gap = values(z)[:len(pi)]
                near = np.flatnonzero(gap < .003)
                if len(near) >= 3:
                    chosen = rng.choice(near,min(len(near),int(params['contacts'])),replace=False)
                    release[chosen] = float(params['release'])*rng.uniform(.6,1.4,len(chosen))
                    z = solve(z,params['probe_iters'])
                    release[:] = 0.
            else:
                z[:21] += rng.normal(0.,params['jitter'],21)
                z[12:21] = np.maximum(z[12:21],.001)
            z = solve(z,params['polish_iters'])
            if np.min(values(z)) > -1e-8:
                z = contact_polish(z)
                if grad@z < grad@anchor-1e-13:
                    anchor = z.copy()
            attempt += 1
    except (Timeout,ValueError,FloatingPointError,np.linalg.LinAlgError):
        pass
    retain(last)
    return best
