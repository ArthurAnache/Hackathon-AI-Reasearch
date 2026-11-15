import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {"first_fraction": (0.45, 0.25, 0.55), "iterations": (180, 60, 300), "safety": (2e-14, 1e-15, 1e-11, "log")}

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


def evaluate(obj):
    try:
        a = np.asarray(canonical(obj)['circles'], dtype=np.float64)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        perimeter = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        perimeter += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        if perimeter > 2:
            return -float('inf')
        i,j = np.triu_indices(21, 1)
        d = np.sqrt((a[i,0]-a[j,0])**2+(a[i,1]-a[j,1])**2)
        if np.any(d < a[i,2]+a[j,2]):
            return -float('inf')
        return float(np.sum(a[:,2]))
    except (TypeError, ValueError, KeyError, OverflowError):
        return -float('inf')


def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + .70*budget_s
    phase_deadline = deadline
    best = init
    best_value = evaluate(init) if init is not None else -float('inf')
    safety = float(params['safety'])
    ii,jj = np.triu_indices(21,1)

    def tick():
        if time.monotonic() >= phase_deadline:
            raise StopSearch()

    fallback = np.array([[.2*(i%7-3), .2*(i//7-1), .1*(1-1e-13)] for i in range(21)])
    obj = {'circles': fallback.tolist()}
    value = evaluate(obj)
    if value > best_value:
        best, best_value = obj, value
    try:
        a = np.asarray(canonical(init)['circles'], float).copy()
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            a = fallback.copy()
    except (TypeError, ValueError, KeyError):
        a = fallback.copy()

    def center(b):
        b = b.copy()
        lo = np.min(b[:,:2]-b[:,2,None],axis=0)
        hi = np.max(b[:,:2]+b[:,2,None],axis=0)
        b[:,:2] -= (lo+hi)/2
        p = float(np.sum(hi-lo))
        if p > 2:
            b *= 2/p
        return b

    a = center(a)

    def retain(b):
        nonlocal best,best_value
        tick()
        if not np.isfinite(b).all() or np.any(b[:,2] <= 0):
            return
        b = b.copy()
        dx = b[ii,0]-b[jj,0]
        dy = b[ii,1]-b[jj,1]
        ratio = float(np.min(np.sqrt(dx*dx+dy*dy)/(b[ii,2]+b[jj,2])))
        b[:,2] *= min(1., ratio)*(1-safety)
        lo = np.min(b[:,:2]-b[:,2,None],axis=0)
        hi = np.max(b[:,:2]+b[:,2,None],axis=0)
        scale = min(1.,2/float(np.sum(hi-lo)))*(1-safety)
        b *= scale
        if float(np.sum(b[:,2])) <= best_value:
            return
        candidate = {'circles': b.tolist()}
        value = evaluate(candidate)
        if value > best_value:
            best,best_value = candidate,value

    def solve(b, groups=None, iterations=180):
        tick()
        if groups is None:
            n = 64
            X = np.zeros((21,n)); Y = X.copy(); R = X.copy()
            X[np.arange(21),np.arange(21)] = 1
            Y[np.arange(21),21+np.arange(21)] = 1
            R[np.arange(21),42+np.arange(21)] = 1
            z = np.r_[b[:,0],b[:,1],b[:,2],1.]
            bounds = [(-1.,1.)]*42+[(1e-7,.6)]*21+[(.15,1.85)]
        else:
            g = groups
            n = 34
            X = np.zeros((21,n)); Y = X.copy(); R = X.copy()
            k = 0
            for t in range(12):
                tick()
                X[k,t] = 1; R[k,21+t] = 1
                if t >= 3:
                    Y[k,12+t-3] = 1
                k += 1
                if t >= 3:
                    X[k,t] = 1; R[k,21+t] = 1; Y[k,12+t-3] = -1
                    k += 1
            z = np.r_[g[:,0],g[3:,1],g[:,2],1.]
            bounds = [(-1.,1.)]*12+[(0.,1.)]*9+[(1e-7,.6)]*12+[(.15,1.85)]
        z[-1] = np.clip(np.ptp(b[:,0])+2*np.mean(b[:,2]), .2, 1.8)
        DX = X[ii]-X[jj]; DY = Y[ii]-Y[jj]; RS = R[ii]+R[jj]
        walls = np.vstack((X-R,-X-R,Y-R,-Y-R))
        walls[:42,-1] += .5
        walls[42:,-1] -= .5
        offsets = np.r_[np.zeros(42),np.ones(42)]
        grad = -np.sum(R,axis=0)
        last = z.copy()
        count = 0

        def unpack(v):
            return np.column_stack((X@v,Y@v,R@v))

        def fun(v):
            tick()
            return float(grad@v)

        def jac(v):
            tick()
            return grad

        def con(v):
            tick()
            dx,dy = DX@v,DY@v
            return np.r_[np.sqrt(dx*dx+dy*dy)-RS@v,walls@v+offsets]

        def cj(v):
            tick()
            dx,dy = DX@v,DY@v
            d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-25)
            return np.vstack(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RS,walls))

        def callback(v):
            nonlocal last,count
            tick()
            last = v.copy()
            count += 1
            if count % 2 == 0:
                retain(unpack(v))

        try:
            result = minimize(fun,z,jac=jac,bounds=bounds,method='SLSQP',constraints={'type':'ineq','fun':con,'jac':cj},callback=callback,options={'maxiter':iterations,'ftol':2e-13,'disp':False})
            last = result.x
            retain(unpack(last))
        except StopSearch:
            pass
        return unpack(last)

    phase_deadline = min(deadline,started+float(params['first_fraction'])*budget_s)
    try:
        solve(a,iterations=int(params['iterations']))
    except (StopSearch, ValueError, FloatingPointError):
        pass

    phase_deadline = deadline
    try:
        tick()
        repaired = center(np.asarray(canonical(best)['circles'],float))
        projections = []
        for swap in (False,True):
            tick()
            b = repaired.copy()
            if swap:
                b[:,[0,1]] = b[:,[1,0]]
            order = np.argsort(b[:,1])
            lower,mid,upper = order[:9],order[9:12],order[12:]
            reflected = b[lower].copy(); reflected[:,1] *= -1
            cost = np.sum((b[upper,None,:]-reflected[None,:,:])**2,axis=2)
            p,q = linear_sum_assignment(cost)
            g = [[b[k,0],0.,b[k,2]] for k in mid]
            error = float(np.sum(b[mid,1]**2)+np.sum(cost[p,q]))
            for u,v in zip(p,q):
                tick()
                t = (b[upper[u]]+reflected[v])/2
                t[1] = abs(t[1])
                g.append(t.tolist())
            projections.append((error,np.asarray(g)))
        error,g = min(projections,key=lambda x:x[0])
        if error < .025:
            expanded = []
            for k,(x,y,r) in enumerate(g):
                tick()
                expanded.append([x,y,r])
                if k >= 3:
                    expanded.append([x,-y,r])
            solve(np.asarray(expanded),groups=g,iterations=int(params['iterations']))
        else:
            solve(repaired,iterations=int(params['iterations']))
        if time.monotonic() < deadline-.04:
            solve(center(np.asarray(canonical(best)['circles'],float)),iterations=int(params['iterations']))
    except (StopSearch, ValueError, FloatingPointError):
        pass
    return best
