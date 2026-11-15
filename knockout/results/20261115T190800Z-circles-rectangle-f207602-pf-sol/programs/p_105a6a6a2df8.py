import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {'iterations': (100, 50, 160), 'jitter': (0.018, 0.002, 0.07), 'polish_reserve': (0.20, 0.08, 0.35)}


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    best = init
    best_score = -float('inf')
    best_array = None
    saved = None

    class Stop(Exception):
        pass

    def check():
        if time.monotonic() >= deadline:
            raise Stop()

    def asarray(obj):
        if 'circles' in obj:
            return np.asarray(obj['circles'], dtype=float)
        return np.column_stack((obj['centers'], obj['radii']))

    def perimeter(a):
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        return (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
                +max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)))

    def exact(a):
        if a.shape != (21,3) or not np.isfinite(a).all() or np.min(a[:,2]) <= 0:
            return -float('inf')
        if perimeter(a) > 2:
            return -float('inf')
        d = a[ii,:2]-a[jj,:2]
        if np.any(np.sqrt(d[:,0]**2+d[:,1]**2) < a[ii,2]+a[jj,2]):
            return -float('inf')
        return float(np.sum(a[:,2]))

    if init is not None:
        try:
            a = asarray(init)
            best_score = exact(a)
            if np.isfinite(best_score):
                best_array = a.copy()
        except (ValueError, TypeError, KeyError):
            pass

    def retain(a, model=None, z=None):
        nonlocal best, best_score, best_array, saved
        a = a.copy()
        if a.shape != (21,3) or not np.isfinite(a).all() or np.min(a[:,2]) <= 0:
            return
        a[:,:2] -= np.mean(a[:,:2],axis=0)
        d = a[ii,:2]-a[jj,:2]
        factor = min(1.,float(np.min(np.sqrt(d[:,0]**2+d[:,1]**2)/(a[ii,2]+a[jj,2]))))
        if factor <= 0:
            return
        a[:,2] *= factor*(1-3e-14)
        pf = np.sum(np.max(a[:,:2]+a[:,2,None],axis=0)-np.min(a[:,:2]-a[:,2,None],axis=0))
        if 2*np.sum(a[:,2])/pf <= best_score:
            return
        p = perimeter(a)
        a *= float(Fraction(2)/p)*(1-3e-14)
        score = exact(a)
        if score > best_score:
            best_score = score
            best_array = a.copy()
            best = {'centers':a[:,:2].tolist(),'radii':a[:,2].tolist()}
            if model is not None:
                saved = (model,z.copy())

    def model_for(counts=None, a=None):
        check()
        if counts is None:
            n = 21
            size = 64
            X = np.zeros((n,size)); Y = X.copy(); R = X.copy()
            ids = np.arange(n)
            X[ids,3*ids] = 1
            Y[ids,3*ids+1] = 1
            R[ids,3*ids+2] = 1
            pi,pj = ii,jj
            DX = X[pi]-X[pj]
            DY = Y[pi]-Y[pj]
            RS = R[pi]+R[pj]
            lo = np.min(a[:,:2]-a[:,2,None],axis=0)
            hi = np.max(a[:,:2]+a[:,2,None],axis=0)
            a = a.copy()
            a[:,0] -= lo[0]
            a[:,1] -= (hi[1]+lo[1])/2
            z = np.r_[a.ravel(),hi[0]-lo[0]]
            weights = np.ones(n)
            bounds = [(None,None)]*size
            for k in range(n):
                check()
                bounds[3*k+2] = (1e-8,1.)
            reflected = False
        else:
            s = counts[0]
            m = sum(counts[1:])
            n = s+m
            ir = n+m
            size = ir+n+1
            ids = np.arange(n)
            X = np.zeros((n,size)); Y = X.copy(); R = X.copy()
            X[ids,ids] = 1
            Y[np.arange(s,n),np.arange(n,ir)] = 1
            R[ids,ir+ids] = 1
            p,q = np.triu_indices(n,1)
            u,v = np.triu_indices(m,1)
            pi = np.r_[p,u+s,np.arange(s,n)]
            pj = np.r_[q,v+s,np.arange(s,n)]
            signs = np.r_[np.ones(len(p)),-np.ones(len(u)+m)]
            DX = X[pi]-X[pj]
            DY = Y[pi]-signs[:,None]*Y[pj]
            RS = R[pi]+R[pj]
            z = np.zeros(size)
            row = 0
            for level,count in enumerate(counts):
                check()
                for col in range(count):
                    check()
                    z[row] = (col-(count-1)/2)*.20
                    if level == 0 and s == 3:
                        z[row] *= 1.45
                    if level:
                        z[n+row-s] = level*.177
                    z[ir+row] = .095
                    row += 1
            x,y,r = X@z,Y@z,R@z
            lo = min(x-r); hi = max(x+r); h = max(y+r)
            scale = 2/(hi-lo+2*h)
            z *= scale
            z[:n] -= lo*scale
            z[-1] = (hi-lo)*scale
            weights = np.r_[np.ones(s),np.full(m,2.)]
            bounds = [(None,None)]*size
            for k in range(n,ir):
                check()
                bounds[k] = (0.,1.)
            for k in range(ir,ir+n):
                check()
                bounds[k] = (1e-8,1.)
            reflected = True
        bounds[-1] = (.05,1.95)
        W = np.zeros(size); W[-1] = 1
        if reflected:
            L = np.vstack((X-R,W-X-R,-.5*W-Y-R))
            b = np.r_[np.zeros(2*n),np.ones(n)]
        else:
            L = np.vstack((X-R,W-X-R,-.5*W+Y-R,-.5*W-Y-R))
            b = np.r_[np.zeros(2*n),np.ones(2*n)]
        grad = -(weights@R)
        def expand(t):
            arr = np.column_stack((X@t,Y@t,R@t))
            if reflected:
                other = arr[s:].copy()
                other[:,1] *= -1
                arr = np.vstack((arr,other))
            return arr
        return {'DX':DX,'DY':DY,'RS':RS,'L':L,'b':b,'grad':grad,
                'bounds':bounds,'expand':expand,'movable':np.flatnonzero(np.any(X!=0,axis=0)|np.any(Y!=0,axis=0))},z

    def optimize(model,z,iterations,local_deadline=None):
        check()
        DX,DY,RS,L,b = (model[k] for k in ('DX','DY','RS','L','b'))
        grad = model['grad']
        cache = [None,None,None]
        latest = [z.copy()]
        calls = [0]
        def update(t):
            check()
            if local_deadline is not None and time.monotonic() >= local_deadline:
                raise Stop()
            if cache[0] is None or not np.array_equal(t,cache[0]):
                dx,dy = DX@t,DY@t
                dist = np.sqrt(dx*dx+dy*dy)
                den = np.maximum(dist,1e-15)
                cache[0] = t.copy()
                cache[1] = np.r_[dist-RS@t,L@t+b]
                cache[2] = np.vstack(((dx/den)[:,None]*DX+(dy/den)[:,None]*DY-RS,L))
        def fun(t):
            update(t)
            return cache[1]
        def jac(t):
            update(t)
            return cache[2]
        def callback(t):
            check()
            latest[0] = t.copy()
            calls[0] += 1
            if calls[0]%20 == 0:
                retain(model['expand'](t),model,t)
        try:
            res = minimize(lambda t:float(grad@t),z,jac=lambda t:grad,
                           method='SLSQP',bounds=model['bounds'],callback=callback,
                           constraints={'type':'ineq','fun':fun,'jac':jac},
                           options={'maxiter':int(iterations),'ftol':2e-12,'disp':False})
            latest[0] = res.x
        except Stop:
            retain(model['expand'](latest[0]),model,latest[0])
            raise
        retain(model['expand'](latest[0]),model,latest[0])
        return latest[0]

    try:
        reserve = min(params['polish_reserve'],.16*budget_s)
        search_end = deadline-reserve
        if best_score > 2.30:
            model,z = model_for(a=best_array)
            optimize(model,z,55)
        else:
            for counts in ((3,5,4),(5,4,4),(3,4,5),(5,5,3),(3,3,4,2)):
                check()
                if time.monotonic() >= search_end:
                    break
                model,z = model_for(counts)
                retain(model['expand'](z),model,z)
                try:
                    optimize(model,z,params['iterations'],search_end)
                except Stop:
                    break
        attempt = 0
        while saved is not None and time.monotonic() < search_end-.10:
            check()
            model,z = saved
            z = z.copy()
            movable = model['movable']
            choose = rng.choice(movable,min(7,len(movable)),replace=False)
            z[choose] += rng.normal(0,params['jitter']*(1+attempt%3),len(choose))
            try:
                optimize(model,z,65,search_end)
            except Stop:
                break
            attempt += 1
        if best_array is not None and time.monotonic() < deadline-.045:
            model,z = model_for(a=best_array)
            optimize(model,z,55)
    except (Stop,ValueError,np.linalg.LinAlgError):
        pass
    if best is None or not np.isfinite(best_score):
        return {'circles':[[(i%7)*.2,(i//7)*.2,.09999999999998] for i in range(21)]}
    return best
