import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {
    'projection_fraction': (0.20, 0.10, 0.30),
    'lattice_fraction': (0.23, 0.12, 0.32),
    'alternative_fraction': (0.10, 0.04, 0.18),
    'jitter': (0.009, 0.001, 0.04)
}

class StopSearch(Exception):
    pass

class SliceEnd(Exception):
    pass

def canonical(obj):
    if obj is None or 'circles' in obj:
        return obj
    rows = []
    for x, y, r, m in obj['orbits']:
        rows.append([x, y, r])
        if m == 2:
            rows.append([x, -y, r])
    return {'circles': rows}

def exact_score(obj):
    try:
        a = np.asarray(canonical(obj)['circles'], dtype=np.float64)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        xs, ys, rs = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
        w = max(x+r for x,r in zip(xs,rs))-min(x-r for x,r in zip(xs,rs))
        h = max(y+r for y,r in zip(ys,rs))-min(y-r for y,r in zip(ys,rs))
        if w+h > 2:
            return -float('inf')
        i,j = np.triu_indices(21,1)
        d = np.sqrt((a[i,0]-a[j,0])**2+(a[i,1]-a[j,1])**2)
        if np.any(d < a[i,2]+a[j,2]):
            return -float('inf')
        return float(np.sum(a[:,2]))
    except (TypeError, ValueError, KeyError, OverflowError):
        return -float('inf')

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + .70*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = exact_score(init)
    ai, aj = np.triu_indices(21,1)
    champion = None
    champion_score = -float('inf')

    def tick():
        if time.monotonic() >= deadline:
            raise StopSearch()

    fallback = np.array([[(k%7-3)*.2, (k//7-1)*.2, .09999999999999] for k in range(21)])
    fb = {'circles': fallback.tolist()}
    fs = exact_score(fb)
    if fs > best_score:
        best, best_score = fb, fs

    try:
        pi, pj, signs = [], [], []
        for i in range(12):
            tick()
            for j in range(i+1,12):
                tick()
                pi.append(i); pj.append(j); signs.append(1.)
                if i >= 3:
                    pi.append(i); pj.append(j); signs.append(-1.)
        pi = np.array(pi); pj = np.array(pj); signs = np.array(signs)
        m = len(pi)
        rows = np.arange(m)
        im = pi >= 3; jm = pj >= 3
        L = np.zeros((45,34)); offset = np.zeros(45)
        k = np.arange(12)
        L[k,k] = 1.; L[k,21+k] = -1.
        L[12+k,k] = -1.; L[12+k,21+k] = -1.; L[12+k,33] = 1.
        L[24+k,21+k] = -1.; L[24+k,33] = -.5; offset[24+k] = 1.
        k = np.arange(9)
        L[27+k,12+k] = -1.
        L[36+k,12+k] = 1.; L[36+k,24+k] = -1.
        objective = np.zeros(34)
        objective[21:24] = -1.; objective[24:33] = -2.
        bounds = [(0.,2.)]*12+[(0.,1.)]*9+[(1e-7,.6)]*12+[(.12,1.88)]
        lower = np.array([b[0] for b in bounds]); upper = np.array([b[1] for b in bounds])
        cached_v = None; cached_g = None; cached_J = None

        def calc(v):
            nonlocal cached_v, cached_g, cached_J
            tick()
            if cached_v is not None and np.array_equal(v,cached_v):
                return cached_g,cached_J
            y = np.r_[np.zeros(3),v[12:21]]
            dx = v[pi]-v[pj]; dy = y[pi]-signs*y[pj]
            d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-25)
            J = np.zeros((m,34))
            J[rows,pi] = dx/d; J[rows,pj] = -dx/d
            J[rows[im],9+pi[im]] = dy[im]/d[im]
            J[rows[jm],9+pj[jm]] = -signs[jm]*dy[jm]/d[jm]
            J[rows,21+pi] = -1.; J[rows,21+pj] = -1.
            cached_v = v.copy()
            cached_g = np.r_[d-v[21+pi]-v[21+pj],L@v+offset]
            cached_J = np.vstack((J,L))
            return cached_g,cached_J

        def expand(v):
            a = np.zeros((21,3))
            a[:3,0] = v[:3]; a[:3,2] = v[21:24]
            a[3::2,0] = v[3:12]; a[4::2,0] = v[3:12]
            a[3::2,1] = v[12:21]; a[4::2,1] = -v[12:21]
            a[3::2,2] = v[24:33]; a[4::2,2] = v[24:33]
            return a

        def retain(v):
            nonlocal best,best_score,champion,champion_score
            tick()
            a = expand(v)
            if not np.isfinite(a).all() or np.any(a[:,2] <= 0):
                return
            lo = np.min(a[:,:2]-a[:,2,None],axis=0)
            hi = np.max(a[:,:2]+a[:,2,None],axis=0)
            a[:,:2] -= (lo+hi)/2
            d = np.sqrt((a[ai,0]-a[aj,0])**2+(a[ai,1]-a[aj,1])**2)
            shrink = min(1.,float(np.min(d/(a[ai,2]+a[aj,2]))))
            if shrink <= 0:
                return
            a[:,2] *= shrink*(1.-2e-14)
            extent = np.max(a[:,:2]+a[:,2,None],axis=0)-np.min(a[:,:2]-a[:,2,None],axis=0)
            scale = 2./float(np.sum(extent))*(1.-2e-14)
            a *= scale
            estimate = float(np.sum(a[:,2]))
            if estimate > champion_score:
                champion_score = estimate
                champion = v.copy()
            if estimate > best_score:
                obj = {'circles':a.tolist()}
                val = exact_score(obj)
                if val > best_score:
                    best,best_score = obj,val

        def polish(v):
            tick()
            g,J = calc(v)
            active = np.flatnonzero(np.abs(g)<3e-6)
            if len(active)<34 or np.min(g)<-1e-5:
                return
            w = v.copy()
            for _ in range(4):
                tick()
                g,J = calc(w)
                delta,_,rank,_ = np.linalg.lstsq(J[active],-g[active],rcond=1e-11)
                if rank<34 or np.max(np.abs(delta))>.003:
                    break
                w += delta
                retain(w)
                if np.max(np.abs(g[active]))<3e-16:
                    break

        def solve(v, seconds, iterations=100):
            tick()
            local_end = min(deadline-.003,time.monotonic()+max(.002,seconds))
            last = np.clip(v,lower,upper)
            count = 0
            def fun(w):
                tick()
                if time.monotonic()>=local_end:
                    raise SliceEnd()
                return float(objective@w)
            def jac(w):
                tick()
                return objective
            def callback(w):
                nonlocal last,count
                tick()
                last = w.copy(); count += 1
                if count%5 == 0:
                    retain(w)
                if time.monotonic()>=local_end:
                    raise SliceEnd()
            try:
                result = minimize(fun,last,jac=jac,method='SLSQP',bounds=bounds,
                    constraints={'type':'ineq','fun':lambda w:calc(w)[0], 'jac':lambda w:calc(w)[1]},
                    callback=callback,options={'maxiter':iterations,'ftol':3e-14,'disp':False})
                last = result.x
            except SliceEnd:
                pass
            retain(last)
            if time.monotonic()<deadline-.008:
                polish(last)
            return last

        def vector(g):
            tick()
            v = np.zeros(34)
            left = np.min(g[:,0]-g[:,2])
            v[:12] = g[:,0]-left
            v[12:21] = np.abs(g[3:,1])
            v[21:33] = np.maximum(g[:,2],1e-6)
            v[33] = np.max(g[:,0]+g[:,2])-left
            return v

        def lattice(counts):
            g = [[2.*(j-1),0.,1.] for j in range(3)]
            for level,count in enumerate(counts,1):
                tick()
                for j in range(count):
                    tick()
                    g.append([2.*j-(count-1),level*np.sqrt(3.),1.])
            g = np.array(g)
            g *= 2./(np.ptp(g[:,0])+2.+2.*np.max(g[:,1]+g[:,2]))
            return vector(g)

        projections = []
        try:
            a = np.asarray(canonical(init)['circles'],float).copy()
            if a.shape == (21,3) and np.isfinite(a).all() and np.all(a[:,2]>0):
                a[:,:2] -= (np.max(a[:,:2]+a[:,2,None],axis=0)+np.min(a[:,:2]-a[:,2,None],axis=0))/2
                for swap in (False,True):
                    tick()
                    b = a[:,[1,0,2]].copy() if swap else a.copy()
                    order = np.argsort(b[:,1]); low,axis,high = order[:9],order[9:12],order[12:]
                    reflected = b[low].copy(); reflected[:,1] *= -1.
                    cost = np.sum((b[high,None,:]-reflected[None,:,:])**2,axis=2)
                    u,j = linear_sum_assignment(cost)
                    g = np.vstack((np.column_stack((b[axis,0],np.zeros(3),b[axis,2])),(b[high[u]]+reflected[j])/2))
                    error = float(np.sum(cost[u,j])+np.sum(b[axis,1]**2))
                    projections.append((error,vector(g)))
        except (TypeError,ValueError,KeyError):
            pass
        projections.sort(key=lambda z:z[0])
        if projections:
            solve(projections[0][1],params['projection_fraction']*budget_s,120)
        solve(lattice((4,5)),params['lattice_fraction']*budget_s,150)
        if champion_score < 2.3657:
            if len(projections)>1 and projections[1][0]<.006:
                alternative = projections[1][1]
            else:
                g = [[x,0.,1.] for x in (-np.sqrt(3.),0.,np.sqrt(3.))]
                for x,ys in ((-2*np.sqrt(3.),(1.,3.)),(-np.sqrt(3.),(2.,4.)),(0.,(2.,)),(np.sqrt(3.),(2.,4.)),(2*np.sqrt(3.),(1.,3.))):
                    tick()
                    for y in ys:
                        tick()
                        g.append([x,y,1.])
                g = np.array(g)* (2./(4*np.sqrt(3.)+12.))
                alternative = vector(g)
            solve(alternative,params['alternative_fraction']*budget_s,100)
        attempt = 0
        while time.monotonic()<deadline-.012:
            tick()
            if champion is None:
                break
            v = champion.copy()
            if attempt:
                v[:21] += rng.normal(0.,params['jitter']*(1.+.5*(attempt%3)),21)
                v = np.clip(v,lower,upper)
            remaining = deadline-time.monotonic()
            solve(v,min(.15*budget_s,remaining-.006),110)
            attempt += 1
    except (StopSearch,ValueError,FloatingPointError,np.linalg.LinAlgError):
        pass
    return best
