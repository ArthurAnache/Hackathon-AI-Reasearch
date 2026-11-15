import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {'round_iters': (48, 20, 100), 'polish_iters': (110, 40, 200), 'jitter': (0.035, 0.005, 0.12), 'contact_threshold': (2e-6, 1e-8, 1e-4, 'log')}

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

def score(obj):
    try:
        a = np.asarray(canonical(obj)['circles'], dtype=float)
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return -float('inf')
        x = [Fraction(float(v)) for v in a[:,0]]
        y = [Fraction(float(v)) for v in a[:,1]]
        r = [Fraction(float(v)) for v in a[:,2]]
        w = max(u+t for u,t in zip(x,r))-min(u-t for u,t in zip(x,r))
        h = max(u+t for u,t in zip(y,r))-min(u-t for u,t in zip(y,r))
        if w+h > 2:
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
    deadline = time.monotonic() + 0.69*budget_s
    rng = np.random.default_rng(seed)
    best = init
    bestscore = score(init) if init is not None else -float('inf')
    def tick():
        if time.monotonic() >= deadline:
            raise StopSearch()
    fallback = np.array([[2*(i%7),2*(i//7),1.] for i in range(21)],float)/10.000000000001
    fb = {'circles':fallback.tolist()}
    fs = score(fb)
    if fs > bestscore:
        best,bestscore = fb,fs
    a = fallback.copy()
    try:
        aa = np.asarray(canonical(init)['circles'],float)
        if aa.shape == (21,3) and np.isfinite(aa).all():
            a = aa.copy()
    except Exception:
        pass
    ids = np.array([0,1,2]+[g for g in range(3,12) for _ in range(2)])
    signs = np.array([0.,0.,0.]+[s for _ in range(9) for s in (1.,-1.)])
    X = np.zeros((21,34)); Y = np.zeros_like(X); R = np.zeros_like(X)
    for k,g in enumerate(ids):
        if time.monotonic() >= deadline:
            return best
        X[k,g] = 1.; R[k,21+g] = 1.
        if g >= 3:
            Y[k,9+g] = signs[k]
    seen = set(); pairs = []
    for i in range(21):
        if time.monotonic() >= deadline:
            return best
        for j in range(i):
            if time.monotonic() >= deadline:
                return best
            key = (min(ids[i],ids[j]),max(ids[i],ids[j]),int(signs[i]*signs[j]))
            if key not in seen:
                seen.add(key); pairs.append((i,j))
    p,q = np.array(pairs).T
    DX = X[p]-X[q]; DY = Y[p]-Y[q]; RS = R[p]+R[q]
    walls = np.zeros((36,34)); off = np.zeros(36)
    for g in range(12):
        if time.monotonic() >= deadline:
            return best
        walls[g,g] = 1.; walls[g,21+g] = -1.; walls[g,33] = .5
        walls[12+g,g] = -1.; walls[12+g,21+g] = -1.; walls[12+g,33] = .5
        walls[24+g,21+g] = -1.; walls[24+g,33] = -.5; off[24+g] = 1.
        if g >= 3:
            walls[24+g,9+g] = -1.
    weights = np.array([1.]*3+[2.]*9)
    grad = np.zeros(34); grad[21:33] = -weights
    bounds = [(-.999,.999)]*12+[(0.,.999)]*9+[(1e-7,.499999)]*12+[(.12,1.88)]
    ii,jj = np.triu_indices(21,1)
    def retain(z):
        nonlocal best,bestscore
        tick()
        if not np.isfinite(z).all() or np.min(z[21:33]) <= 0:
            return
        c = np.column_stack((X@z,Y@z,R@z))
        dist = np.hypot(c[ii,0]-c[jj,0],c[ii,1]-c[jj,1])
        factor = min(1.,float(np.min(dist/(c[ii,2]+c[jj,2]))))*(1.-2e-14)
        c[:,2] *= factor
        lo = np.min(c[:,:2]-c[:,2,None],axis=0)
        hi = np.max(c[:,:2]+c[:,2,None],axis=0)
        scale = min(1.,2./float(np.sum(hi-lo)))*(1.-2e-14)
        c *= scale
        if float(np.sum(c[:,2])) <= bestscore:
            return
        obj = {'circles':c.tolist()}
        val = score(obj)
        if val > bestscore:
            best,bestscore = obj,val
    def fun(z):
        tick(); return float(grad@z)
    def jac(z):
        tick(); return grad
    def con(z):
        tick()
        dx,dy = DX@z,DY@z
        return np.r_[np.hypot(dx,dy)-RS@z,walls@z+off]
    def cj(z):
        tick()
        dx,dy = DX@z,DY@z
        d = np.maximum(np.hypot(dx,dy),1e-24)
        return np.vstack(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RS,walls))
    def contact_polish(z):
        tick()
        residual = con(z)
        if np.min(residual) < -1e-5:
            return z
        active = np.flatnonzero(np.abs(residual) < params['contact_threshold'])
        if len(active) < 34:
            return z
        original = z.copy()
        for _ in range(6):
            tick()
            f = con(z)[active]
            J = cj(z)[active]
            delta, _, rank, _ = np.linalg.lstsq(J, -f, rcond=1e-11)
            if rank < 34 or np.max(np.abs(delta)) > .002:
                return original
            trial = z + delta
            if np.min(trial[21:33]) <= 0 or not np.isfinite(trial).all():
                return original
            z = trial
            retain(z)
            if np.max(np.abs(f)) < 3e-16:
                break
        if np.min(con(z)) >= -1e-10 and grad@z <= grad@original + 1e-11:
            return z
        return original
    def solve(z,n):
        tick()
        counter = [0]
        def cb(v):
            tick(); counter[0] += 1
            if counter[0] % 4 == 0:
                retain(v)
        res = minimize(fun,z,jac=jac,bounds=bounds,method='SLSQP',constraints={'type':'ineq','fun':con,'jac':cj},callback=cb,options={'maxiter':n,'ftol':2e-13,'disp':False})
        retain(res.x)
        return contact_polish(res.x)
    def pack(g):
        tick()
        z = np.zeros(34)
        z[:12] = g[:,0]; z[12:21] = np.abs(g[3:,1]); z[21:33] = np.maximum(g[:,2],.002)
        w = np.max(g[:,0]+z[21:33])-np.min(g[:,0]-z[21:33])
        h = 2*np.max(np.abs(g[:,1])+z[21:33])
        scale = 1.96/max(w+h,1e-9)
        z[:33] *= scale; z[33] = np.clip(w*scale,.12,1.88)
        return z
    try:
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        a[:,:2] -= (lo+hi)/2
        projected = []
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
            if bestscore < 1.:
                g[:,2] = .045
            projected.append((float(np.sum(cost[u,v])+np.sum(b[axis,1]**2)),pack(g)))
        projected.sort(key=lambda t:t[0])
        seeds = [projected[0][1]]
        if bestscore < 2.3:
            for counts in ((4,5),(3,3,3)):
                tick()
                rows = [[x,0.,1.] for x in (-2.,0.,2.)]
                for level,count in enumerate(counts,1):
                    tick()
                    rows.extend([[2*k-(count-1),level*np.sqrt(3.),1.] for k in range(count)])
                seeds.append(pack(np.asarray(rows,float)))
            seeds.append(projected[1][1])
        results = []
        for z in seeds:
            tick()
            z = solve(z,int(params['round_iters']))
            merit = float(weights@z[21:33])-20*max(0.,-float(np.min(con(z))))
            results.append((merit,z))
        results.sort(key=lambda t:t[0],reverse=True)
        anchor = solve(results[0][1],int(params['polish_iters']))
        attempt = 0
        while time.monotonic() < deadline:
            tick()
            if attempt < len(results)-1:
                z = results[attempt+1][1].copy()
            else:
                z = anchor.copy()
                z[:21] += rng.normal(0.,float(params['jitter']),21)
                z[12:21] = np.maximum(z[12:21],.001)
                z[:12] = np.clip(z[:12],-.95,.95)
            z = solve(z,int(params['polish_iters']))
            if np.min(con(z)) > -1e-7 and weights@z[21:33] > weights@anchor[21:33]:
                anchor = z.copy()
            attempt += 1
    except (StopSearch,ValueError,FloatingPointError,np.linalg.LinAlgError):
        pass
    return best
