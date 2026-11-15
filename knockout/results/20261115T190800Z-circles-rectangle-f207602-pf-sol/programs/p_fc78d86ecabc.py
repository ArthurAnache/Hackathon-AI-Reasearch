import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {'probe_time': (0.23, 0.1, 0.4), 'ftol': (8e-13, 1e-14, 1e-9, 'log'), 'safety': (3e-14, 1e-14, 1e-10, 'log')}

class StopSearch(Exception):
    pass

def canonical(obj):
    if 'circles' in obj:
        return obj
    rows = []
    for x,y,r,m in obj['orbits']:
        rows.append([x,y,r])
        if m == 2:
            rows.append([x,-y,r])
    return {'circles': rows}

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + .69*budget_s
    rng = np.random.default_rng(seed)
    ii,jj = np.triu_indices(21,1)
    best = init
    best_score = -float('inf')
    safety = params['safety']
    def tick(end=None):
        if time.monotonic() >= (deadline if end is None else min(end,deadline)):
            raise StopSearch()
    def exact(c):
        if c.shape != (21,3) or not np.isfinite(c).all() or np.any(c[:,2] <= 0):
            return -float('inf')
        d = c[ii,:2]-c[jj,:2]
        if np.any(np.sqrt(d[:,0]**2+d[:,1]**2) < c[ii,2]+c[jj,2]):
            return -float('inf')
        xs = [Fraction(float(v)) for v in c[:,0]]
        ys = [Fraction(float(v)) for v in c[:,1]]
        rs = [Fraction(float(v)) for v in c[:,2]]
        w = max(x+r for x,r in zip(xs,rs))-min(x-r for x,r in zip(xs,rs))
        h = max(y+r for y,r in zip(ys,rs))-min(y-r for y,r in zip(ys,rs))
        if w+h > 2:
            return -float('inf')
        return float(np.sum(c[:,2]))
    def retain(c):
        nonlocal best,best_score
        c = np.array(c,dtype=float,copy=True)
        if c.shape != (21,3) or not np.isfinite(c).all() or np.min(c[:,2]) <= 0:
            return
        d = c[ii,:2]-c[jj,:2]
        ratio = np.min(np.sqrt(d[:,0]**2+d[:,1]**2)/(c[ii,2]+c[jj,2]))
        c[:,2] *= min(1.,float(ratio))*(1-safety)
        lo = np.min(c[:,:2]-c[:,2,None],axis=0)
        hi = np.max(c[:,:2]+c[:,2,None],axis=0)
        perimeter = float(np.sum(hi-lo))
        if perimeter <= 0:
            return
        c[:,:2] -= (lo+hi)/2
        c *= (2/perimeter)*(1-safety)
        if float(np.sum(c[:,2])) > best_score:
            val = exact(c)
            if val > best_score:
                best_score = val
                best = {'circles':c.tolist()}
    fallback = np.array([[2*(i%7),2*(i//7),1.] for i in range(21)],float)/10.000000001
    initial = fallback
    try:
        a = np.array(canonical(init)['circles'],float)
        if a.shape == (21,3) and np.isfinite(a).all() and np.min(a[:,2]) > 0:
            initial = a
            best_score = exact(a)
    except (TypeError,ValueError,KeyError):
        pass
    retain(fallback)
    retain(initial)

    # The same analytic solver supports both the orbit quotient and all circles.
    def model(symmetric):
        tick()
        if symmetric:
            ids = np.array([0,1,2]+[g for g in range(3,12) for _ in range(2)])
            signs = np.array([0.,0.,0.]+[s for _ in range(9) for s in (1.,-1.)])
            nv = 34
            X = np.zeros((21,nv)); Y = X.copy(); R = X.copy()
            for k,g in enumerate(ids):
                tick()
                X[k,g] = 1
                R[k,21+g] = 1
                if g >= 3:
                    Y[k,9+g] = signs[k]
            seen = set(); pairs = []
            for i,j in zip(ii,jj):
                tick()
                key = (min(ids[i],ids[j]),max(ids[i],ids[j]),int(signs[i]*signs[j]))
                if key not in seen:
                    seen.add(key); pairs.append((i,j))
            p,q = np.array(pairs).T
            walls = []; offsets = []
            for g in range(12):
                tick()
                for sg in (1.,-1.):
                    row = np.zeros(nv); row[g] = sg; row[21+g] = -1; row[-1] = .5
                    walls.append(row); offsets.append(0.)
                row = np.zeros(nv); row[21+g] = -1; row[-1] = -.5
                if g >= 3:
                    row[9+g] = -1
                walls.append(row); offsets.append(1.)
            bounds = [(-1.,1.)]*12+[(0.,1.)]*9+[(1e-7,.5)]*12+[(.1,1.9)]
        else:
            nv = 64
            X = np.zeros((21,nv)); Y = X.copy(); R = X.copy()
            ar = np.arange(21)
            X[ar,ar] = 1; Y[ar,21+ar] = 1; R[ar,42+ar] = 1
            p,q = ii,jj
            walls = np.vstack((X-R+.5*np.eye(1,nv,nv-1),-X-R+.5*np.eye(1,nv,nv-1),Y-R-.5*np.eye(1,nv,nv-1),-Y-R-.5*np.eye(1,nv,nv-1)))
            offsets = np.r_[np.zeros(42),np.ones(42)]
            bounds = [(-1.,1.)]*42+[(1e-7,.5)]*21+[(.1,1.9)]
        W = np.asarray(walls); off = np.asarray(offsets)
        DX = X[p]-X[q]; DY = Y[p]-Y[q]; RS = R[p]+R[q]
        grad = -np.sum(R,axis=0)
        return X,Y,R,DX,DY,RS,W,off,grad,bounds
    try:
        sym = model(True)
        full = model(False)
    except StopSearch:
        return best

    def expand(z,m):
        return np.column_stack((m[0]@z,m[1]@z,m[2]@z))
    def pack_full(c):
        c = np.array(c,copy=True)
        lo = np.min(c[:,:2]-c[:,2,None],axis=0); hi = np.max(c[:,:2]+c[:,2,None],axis=0)
        c[:,:2] -= (lo+hi)/2
        scale = 2/np.sum(hi-lo)
        c *= scale
        return np.r_[c[:,0],c[:,1],c[:,2],(hi[0]-lo[0])*scale]
    def pack_orbits(g):
        lo = np.min(g[:,0]-g[:,2]); hi = np.max(g[:,0]+g[:,2])
        w = hi-lo; h = 2*np.max(np.abs(g[:,1])+g[:,2])
        g = g.copy(); g[:,0] -= (lo+hi)/2
        g *= 2/(w+h)
        return np.r_[g[:,0],np.abs(g[3:,1]),g[:,2],2*w/(w+h)]
    def solve(z,m,seconds,iters=100):
        end = min(deadline,time.monotonic()+seconds)
        X,Y,R,DX,DY,RS,W,off,grad,bounds = m
        last = [z.copy()]; cache = [None,None,None]; count = [0]
        def calc(v):
            tick(end)
            if cache[0] is not None and np.array_equal(v,cache[0]):
                return cache[1],cache[2]
            dx = DX@v; dy = DY@v
            d = np.hypot(dx,dy); den = np.maximum(d,1e-24)
            f = np.r_[d-RS@v,W@v+off]
            j = np.vstack(((dx/den)[:,None]*DX+(dy/den)[:,None]*DY-RS,W))
            cache[:] = [v.copy(),f,j]
            return f,j
        def cb(v):
            tick(end)
            last[0] = v.copy(); count[0] += 1
            if count[0]%7 == 0:
                retain(expand(v,m))
        try:
            res = minimize(lambda v:float(grad@v),z,jac=lambda v:grad,method='SLSQP',bounds=bounds,constraints={'type':'ineq','fun':lambda v:calc(v)[0],'jac':lambda v:calc(v)[1]},callback=cb,options={'maxiter':iters,'ftol':params['ftol'],'disp':False})
            last[0] = res.x
        except StopSearch:
            pass
        retain(expand(last[0],m))
        return last[0]

    try:
        a = initial.copy()
        lo = np.min(a[:,:2]-a[:,2,None],axis=0); hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        a[:,:2] -= (lo+hi)/2
        projections = []
        for swap in (False,True):
            tick()
            b = a.copy()
            if swap:
                b[:,[0,1]] = b[:,[1,0]]
            order = np.argsort(b[:,1]); lower,axis,upper = order[:9],order[9:12],order[12:]
            reflected = b[lower].copy(); reflected[:,1] *= -1
            cost = np.sum((b[upper,None,:]-reflected[None,:,:])**2,axis=2)
            u,v = linear_sum_assignment(cost)
            g = np.vstack((np.column_stack((b[axis,0],np.zeros(3),b[axis,2])),(b[upper[u]]+reflected[v])/2))
            projections.append((float(np.sum(cost[u,v])+np.sum(b[axis,1]**2)),pack_orbits(g)))
        projections.sort(key=lambda p:p[0])
        # First preserve the incumbent's basin, then race a known compact family.
        if projections[0][0] < 1e-8:
            solve(projections[0][1],sym,params['probe_time'])
        else:
            solve(pack_full(initial),full,params['probe_time'])
        tick()
        rows = [[x,0.,1.] for x in (-2.,0.,2.)]
        for level,count in enumerate((4,5),1):
            tick()
            rows.extend([[2*k-(count-1),level*np.sqrt(3.),1.] for k in range(count)])
        solve(pack_orbits(np.asarray(rows,float)),sym,.34)
        tick()
        if projections[0][0] >= 1e-8:
            solve(projections[0][1],sym,.20)
        else:
            solve(projections[1][1],sym,.18)
        tick()
        # Unrestricted final polishing avoids locking in an incorrect reflection.
        c = np.asarray(canonical(best)['circles'],float)
        solve(pack_full(c),full,min(.30,max(.01,deadline-time.monotonic())))
        while time.monotonic() < deadline-.045:
            tick()
            c = np.asarray(canonical(best)['circles'],float)
            z = pack_full(c)
            z[:42] += rng.normal(0.,.012,42)
            solve(z,full,min(.18,deadline-time.monotonic()))
    except (StopSearch,ValueError,FloatingPointError,np.linalg.LinAlgError):
        pass
    return best
