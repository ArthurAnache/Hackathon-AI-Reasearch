import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment
from scipy.linalg import null_space, lstsq

PARAMS = {
    'release_scale': (0.055, 0.008, 0.16),
    'active_threshold': (2e-6, 1e-8, 1e-3, 'log'),
    'iterations': (65, 20, 160),
    'safety': (3e-15, 1e-15, 1e-11, 'log')
}
I, J = np.triu_indices(21, 1)

class EndSearch(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2] + a[J, 2]):
        return -np.inf
    x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
    p = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
    p += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
    return float(np.sum(a[:, 2])) if p <= 2 else -np.inf

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    best = init
    score = -np.inf
    if init is not None:
        a = np.asarray(init['circles'], dtype=float) if 'circles' in init else np.asarray(init['state'], dtype=float).reshape(3,21).T
        a = a.copy()
        score = exact_score(a)
    else:
        a = np.array([[0.2*(k%3), 0.2*(k//3), 0.099999999999] for k in range(21)])
    eps = params['safety']
    def check():
        if time.monotonic() >= deadline:
            raise EndSearch()
    def offer(c):
        nonlocal best, score
        check()
        c = np.asarray(c, dtype=float).copy()
        if not np.isfinite(c).all() or np.min(c[:,2]) <= 0:
            return
        d = np.sqrt((c[I,0]-c[J,0])**2+(c[I,1]-c[J,1])**2)
        q = min(1., float(np.min(d/(c[I,2]+c[J,2]))))
        if q <= 0:
            return
        c[:,2] *= q*(1-eps)
        lo = np.min(c[:,:2]-c[:,2,None], axis=0)
        hi = np.max(c[:,:2]+c[:,2,None], axis=0)
        s = (2-12*eps)/np.sum(hi-lo)
        c[:,:2] = (c[:,:2]-(hi+lo)/2)*s
        c[:,2] *= s
        if float(np.sum(c[:,2])) > score:
            sc = exact_score(c)
            if sc > score:
                score = sc
                best = {'circles': c.tolist()}
    try:
        offer(a)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        a[:,:2] -= (hi+lo)/2
        # Infer the reflection axis rather than relying on the input ordering.
        choices = []
        for swap in (False, True):
            check()
            b = a[:,[1,0,2]].copy() if swap else a.copy()
            singles = np.argsort(np.abs(b[:,1]))[:3]
            remain = np.array([i for i in range(21) if i not in set(singles)])
            order = remain[np.argsort(b[remain,1])]
            lower, upper = order[:9], order[9:]
            cost = (b[upper,None,0]-b[lower,0])**2 + (b[upper,None,1]+b[lower,1])**2 + (b[upper,None,2]-b[lower,2])**2
            rows, cols = linear_sum_assignment(cost)
            lower = lower[cols]
            err = float(np.sum(cost[rows,cols])+np.sum(b[singles,1]**2))
            choices.append((err,b,singles,upper,lower))
        _, b, singles, upper, lower = min(choices,key=lambda t:t[0])
        ids = np.r_[np.arange(12),np.arange(3,12)]
        xi = ids
        yi = np.r_[np.zeros(3,dtype=int),np.arange(12,21),np.arange(12,21)]
        signs = np.r_[np.zeros(3),np.ones(9),-np.ones(9)]
        ri = 21+ids
        z = np.zeros(34)
        z[:3] = b[singles,0]
        z[3:12] = (b[upper,0]+b[lower,0])/2
        z[12:21] = (b[upper,1]-b[lower,1])/2
        z[21:24] = b[singles,2]
        z[24:33] = (b[upper,2]+b[lower,2])/2
        z[-1] = np.max(b[:,0]+b[:,2])-np.min(b[:,0]-b[:,2])
        pairs = [(i,j) for i in range(12) for j in range(i+1,12)]
        pairs += [(i,12+j-3) for i in range(3,12) for j in range(i,12)]
        pi,pj = np.array(pairs).T
        m = len(pi)
        rr = np.arange(m)
        wall = np.arange(12)
        k = np.arange(12)
        B = np.zeros((48,34))
        B[k,xi[wall]] = 1
        B[12+k,xi[wall]] = -1
        B[:24,-1] = .5
        B[k,ri[wall]] = -1
        B[12+k,ri[wall]] = -1
        B[24+k,yi[wall]] = signs[wall]
        B[36+k,yi[wall]] = -signs[wall]
        B[24+k,ri[wall]] -= 1
        B[36+k,ri[wall]] -= 1
        B[24:,-1] = -.5
        offset = np.r_[np.zeros(24),np.ones(24)]
        obj = np.zeros(34)
        obj[21:24] = -1
        obj[24:33] = -2
        bounds = [(None,None)]*34
        for q in range(21,33):
            check()
            bounds[q] = (1e-5,.6)
        bounds[-1] = (.2,1.8)
        cache = [None,None,None]
        def unpack(v):
            return np.column_stack((v[xi],signs*v[yi],v[ri]))
        def calc(v):
            check()
            if cache[0] is not None and np.array_equal(cache[0],v):
                return cache[1],cache[2]
            x,y,r = v[xi],signs*v[yi],v[ri]
            dx,dy,rad = x[pi]-x[pj], y[pi]-y[pj],r[pi]+r[pj]
            d = np.sqrt(dx*dx+dy*dy+1e-30)
            c = np.r_[d-rad,B@v+offset]
            jac = np.zeros((m+48,34))
            tx,ty = dx/d,dy/d
            np.add.at(jac,(rr,xi[pi]),tx)
            np.add.at(jac,(rr,xi[pj]),-tx)
            np.add.at(jac,(rr,yi[pi]),ty*signs[pi])
            np.add.at(jac,(rr,yi[pj]),-ty*signs[pj])
            np.add.at(jac,(rr,ri[pi]),-1.)
            np.add.at(jac,(rr,ri[pj]),-1.)
            jac[m:] = B
            cache[:] = [v.copy(),c,jac]
            return c,jac
        def polish(v):
            v = v.copy()
            c,jac = calc(v)
            active = np.where(c < 3e-7)[0]
            if len(active) < 32:
                return v
            for _ in range(5):
                check()
                c,jac = calc(v)
                delta = lstsq(jac[active],-c[active],cond=1e-11,check_finite=False,lapack_driver='gelsy')[0]
                if np.linalg.norm(delta) > .003:
                    break
                v += delta
                if np.max(np.abs(delta)) < 2e-15:
                    break
            offer(unpack(v))
            return v
        def run(v, allowance):
            local_end = min(deadline,time.monotonic()+allowance)
            last = [v.copy()]
            def fun(t):
                check()
                if time.monotonic() >= local_end:
                    raise EndSearch()
                return float(obj@t),obj
            def cb(t):
                check()
                last[0] = t.copy()
            try:
                result = minimize(fun,v,jac=True,method='SLSQP',bounds=bounds,
                    constraints={'type':'ineq','fun':lambda t:calc(t)[0],'jac':lambda t:calc(t)[1]},
                    callback=cb, options={'maxiter':params['iterations'],'ftol':2e-14,'disp':False})
                v = result.x
            except EndSearch:
                v = last[0]
            check()
            offer(unpack(v))
            return polish(v)
        z = run(z,min(.30,.22*budget_s))
        anchor = z.copy()
        anchor_value = float(obj@anchor)
        attempt = 0
        while time.monotonic() < deadline-.06:
            check()
            c,jac = calc(anchor)
            active = np.where(c < params['active_threshold'])[0]
            trial = anchor.copy()
            if len(active) >= 31:
                drop = rng.choice(len(active),size=min(3,len(active)),replace=False)
                retained = np.delete(active,drop)
                basis = null_space(jac[retained],rcond=1e-9)
                if basis.shape[1]:
                    direction = basis@rng.normal(size=basis.shape[1])
                    direction /= max(np.linalg.norm(direction),1e-20)
                    trial += direction*params['release_scale']*(.65+1.1*rng.random())
            else:
                trial[:21] += rng.normal(0,.012,21)
            trial[21:33] = np.maximum(trial[21:33],.002)
            trial[-1] = np.clip(trial[-1],.21,1.79)
            trial = run(trial,min(.26,max(.07,deadline-time.monotonic()-.015)))
            feasible = np.min(calc(trial)[0]) > -2e-8
            if feasible and float(obj@trial) < anchor_value-1e-12:
                anchor,anchor_value = trial.copy(),float(obj@trial)
            attempt += 1
    except EndSearch:
        pass
    return best
