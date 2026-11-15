import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {"orbit_fraction": (0.28, 0.15, 0.4), "free_fraction": (0.17, 0.08, 0.25), "safety": (1e-14, 2e-15, 1e-11, "log"), "iterations": (140, 60, 240)}

class Stop(Exception):
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


def evaluate(obj):
    try:
        a = np.asarray(canonical(obj)['circles'], dtype=np.float64)
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return -float('inf')
        x,y,r = [[Fraction(float(v)) for v in a[:,k]] for k in range(3)]
        perimeter = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        perimeter += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        if perimeter > 2:
            return -float('inf')
        i,j = np.triu_indices(21,1)
        d = np.sqrt((a[i,0]-a[j,0])**2+(a[i,1]-a[j,1])**2)
        if np.any(d < a[i,2]+a[j,2]):
            return -float('inf')
        return float(np.sum(a[:,2]))
    except (TypeError, ValueError, KeyError, OverflowError):
        return -float('inf')


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + .69*budget_s
    phase = deadline
    best = init
    best_value = evaluate(init)
    safety = float(params['safety'])
    ii,jj = np.triu_indices(21,1)

    def tick():
        if time.monotonic() >= phase:
            raise Stop()

    fallback = np.array([[.2*(i%7-3),.2*(i//7-1),.1*(1-1e-12)] for i in range(21)])
    obj = {'circles': fallback.tolist()}
    value = evaluate(obj)
    if value > best_value:
        best,best_value = obj,value
    try:
        original = np.asarray(canonical(init)['circles'],float).copy()
        if original.shape != (21,3) or not np.isfinite(original).all() or np.any(original[:,2] <= 0):
            original = fallback.copy()
    except (TypeError,ValueError,KeyError):
        original = fallback.copy()

    def center(a):
        a = a.copy()
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        a[:,:2] -= (lo+hi)/2
        a *= min(1.,2/float(np.sum(hi-lo)))
        return a

    def retain(a):
        nonlocal best,best_value
        tick()
        if not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        a = a.copy()
        d = np.sqrt((a[ii,0]-a[jj,0])**2+(a[ii,1]-a[jj,1])**2)
        a[:,2] *= min(1.,float(np.min(d/(a[ii,2]+a[jj,2]))))*(1-safety)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        a[:,:2] -= (lo+hi)/2
        a *= min(1.,2/float(np.sum(hi-lo)))*(1-safety)
        if float(np.sum(a[:,2])) <= best_value:
            return
        obj = {'circles':a.tolist()}
        value = evaluate(obj)
        if value > best_value:
            best,best_value = obj,value

    def project(a):
        a = center(a)
        choices = []
        for swap in (False,True):
            tick()
            b = a.copy()
            if swap:
                b[:,[0,1]] = b[:,[1,0]]
            order = np.argsort(b[:,1])
            lower,mid,upper = order[:9],order[9:12],order[12:]
            reflected = b[lower].copy()
            reflected[:,1] *= -1
            costs = np.sum((b[upper,None,:]-reflected[None,:,:])**2,axis=2)
            p,q = linear_sum_assignment(costs)
            groups = [[b[k,0],0.,b[k,2]] for k in mid]
            for u,v in zip(p,q):
                tick()
                t = (b[upper[u]]+reflected[v])/2
                t[1] = abs(t[1])
                groups.append(t.tolist())
            choices.append((float(np.sum(costs[p,q])+np.sum(b[mid,1]**2)),np.asarray(groups)))
        return min(choices,key=lambda t:t[0])[1]

    def solve(data,symmetric=True):
        tick()
        if symmetric:
            n = 34
            X = np.zeros((21,n)); Y = X.copy(); R = X.copy()
            orbit = []; signs = []
            for g in range(12):
                tick()
                orbit.append(g); signs.append(0 if g<3 else 1)
                if g>=3:
                    orbit.append(g); signs.append(-1)
            for k,g in enumerate(orbit):
                tick()
                X[k,g] = 1; R[k,21+g] = 1
                if g>=3:
                    Y[k,12+g-3] = signs[k]
            chosen = []; seen = set()
            for i,j in zip(ii,jj):
                tick()
                key = (min(orbit[i],orbit[j]),max(orbit[i],orbit[j]),signs[i]*signs[j])
                if key not in seen:
                    seen.add(key); chosen.append((i,j))
            pi,pj = np.asarray(chosen).T
            z = np.r_[data[:,0],data[3:,1],data[:,2],1.]
            bounds = [(-1.,1.)]*12+[(0.,1.)]*9+[(1e-7,.6)]*12+[(.1,1.9)]
        else:
            n = 64
            X = np.zeros((21,n)); Y = X.copy(); R = X.copy()
            ar = np.arange(21)
            X[ar,ar] = 1; Y[ar,21+ar] = 1; R[ar,42+ar] = 1
            pi,pj = ii,jj
            z = np.r_[data[:,0],data[:,1],data[:,2],1.]
            bounds = [(-1.,1.)]*42+[(1e-7,.6)]*21+[(.1,1.9)]
        a = np.column_stack((X@z,Y@z,R@z))
        z[-1] = np.clip(np.max(a[:,0]+a[:,2])-np.min(a[:,0]-a[:,2]),.1,1.9)
        DX = X[pi]-X[pj]; DY = Y[pi]-Y[pj]; RS = R[pi]+R[pj]
        walls = np.vstack((X-R,-X-R,Y-R,-Y-R))
        walls[:42,-1] += .5
        walls[42:,-1] -= .5
        offsets = np.r_[np.zeros(42),np.ones(42)]
        grad = -np.sum(R,axis=0)
        cache_x = None
        cache_c = None
        cache_j = None
        last = z.copy()
        count = 0

        def calculate(v):
            nonlocal cache_x,cache_c,cache_j
            tick()
            if cache_x is None or not np.array_equal(v,cache_x):
                dx = DX@v; dy = DY@v
                d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-25)
                cache_c = np.r_[d-RS@v,walls@v+offsets]
                cache_j = np.vstack(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RS,walls))
                cache_x = v.copy()

        def fun(v):
            tick()
            return float(grad@v)

        def jac(v):
            tick()
            return grad

        def con(v):
            calculate(v)
            return cache_c

        def cj(v):
            calculate(v)
            return cache_j

        def unpack(v):
            return np.column_stack((X@v,Y@v,R@v))

        def callback(v):
            nonlocal last,count
            tick()
            last = v.copy(); count += 1
            if count%3 == 0:
                retain(unpack(v))

        try:
            result = minimize(fun,z,jac=jac,bounds=bounds,method='SLSQP',constraints={'type':'ineq','fun':con,'jac':cj},callback=callback,options={'maxiter':int(params['iterations']),'ftol':3e-14,'disp':False})
            last = result.x
            retain(unpack(last))
        except Stop:
            pass
        return unpack(last)

    phase = min(deadline,start+float(params['orbit_fraction'])*budget_s)
    try:
        solve(project(original))
    except (Stop,ValueError,FloatingPointError):
        pass

    phase = min(deadline,start+(float(params['orbit_fraction'])+float(params['free_fraction']))*budget_s)
    try:
        solve(center(np.asarray(canonical(best)['circles'],float)),False)
    except (Stop,ValueError,FloatingPointError):
        pass

    phase = min(deadline,start+.61*budget_s)
    try:
        tick()
        g = []
        for x in (-.2,0.,.2):
            tick()
            g.append([x,0.,.1])
        for row,count in ((1,4),(2,5)):
            tick()
            for k in range(count):
                tick()
                g.append([.2*(k-(count-1)/2),row*np.sqrt(.03),.1])
        solve(np.asarray(g))
    except (Stop,ValueError,FloatingPointError):
        pass

    phase = deadline
    try:
        tick()
        solve(project(np.asarray(canonical(best)['circles'],float)))
    except (Stop,ValueError,FloatingPointError):
        pass
    return best
