import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {'symmetric_share': (0.70, 0.4, 0.85), 'ftol': (2e-12, 1e-14, 1e-8, 'log')}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + 0.70 * budget_s
    best = init
    best_score = -float('inf')
    best_array = None
    ii, jj = np.triu_indices(21, 1)

    def exact(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        d = a[ii, :2] - a[jj, :2]
        if np.any(np.sqrt(d[:, 0]**2 + d[:, 1]**2) < a[ii, 2] + a[jj, 2]):
            return -float('inf')
        xs = [Fraction(float(v)) for v in a[:, 0]]
        ys = [Fraction(float(v)) for v in a[:, 1]]
        rs = [Fraction(float(v)) for v in a[:, 2]]
        w = max(x+r for x,r in zip(xs,rs)) - min(x-r for x,r in zip(xs,rs))
        h = max(y+r for y,r in zip(ys,rs)) - min(y-r for y,r in zip(ys,rs))
        return float(np.sum(a[:, 2])) if w+h <= 2 else -float('inf')

    def retain(a, repair=True):
        nonlocal best, best_score, best_array
        a = np.array(a, dtype=float, copy=True)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.min(a[:,2]) <= 0:
            return
        if repair:
            a[:, :2] -= np.mean(a[:, :2], axis=0)
            delta = a[ii,:2] - a[jj,:2]
            dist = np.sqrt(np.sum(delta*delta, axis=1))
            f = min(1.0, float(np.min(dist/(a[ii,2]+a[jj,2]))))
            a[:,2] *= f*(1-4e-14)
            ext = np.max(a[:,:2]+a[:,2,None],axis=0)-np.min(a[:,:2]-a[:,2,None],axis=0)
            if np.sum(ext) <= 0:
                return
            a *= (2/float(np.sum(ext)))*(1-4e-14)
        s = exact(a)
        if s > best_score:
            best_score = s
            best_array = a.copy()
            best = {'circles': a.tolist()}

    a0 = None
    if init is not None:
        if 'circles' in init:
            a0 = np.asarray(init['circles'], dtype=float)
        elif 'orbits' in init:
            rows = []
            for x,y,r,m in init['orbits']:
                if time.monotonic() >= deadline:
                    break
                rows.append([x,y,r])
                if m == 2:
                    rows.append([x,-y,r])
            a0 = np.asarray(rows, dtype=float)
        if a0 is not None and a0.shape == (21,3):
            s = exact(a0)
            if s > best_score:
                best_score = s
                best_array = a0.copy()
            retain(a0)
    if a0 is None or a0.shape != (21,3):
        rows = []
        for y,xx in [(0,[-3,-1,1]),(1,[-4,-2,0,2,4]),(-1,[-4,-2,0,2,4]),(2,[-3,-1,1,3]),(-2,[-3,-1,1,3])]:
            if time.monotonic() >= deadline:
                break
            rows.extend([[x*.09,y*np.sqrt(3)*.09,.089] for x in xx])
        a0 = np.asarray(rows)
        retain(a0)

    def projection(a, transpose):
        a = a.copy()
        if transpose:
            a[:,[0,1]] = a[:,[1,0]]
        a[:,:2] -= np.mean(a[:,:2],axis=0)
        order = np.argsort(a[:,1])
        lo, ax, hi = order[:9], order[9:12], order[12:]
        cost = (a[lo,0,None]-a[hi,0][None,:])**2 + (a[lo,1,None]+a[hi,1][None,:])**2
        u,v = linear_sum_assignment(cost)
        lp,hp = lo[u],hi[v]
        x = np.r_[a[ax,0],(a[lp,0]+a[hp,0])/2]
        y = (a[hp,1]-a[lp,1])/2
        r = np.r_[a[ax,2],(a[lp,2]+a[hp,2])/2]
        err = float(np.sum(cost[u,v])+np.sum(a[ax,1]**2))
        return err,x,y,r

    def optimize(a, symmetric, until, proj=None):
        if time.monotonic() >= until:
            return
        if symmetric:
            n = 34
            X = np.zeros((21,n)); Y = X.copy(); R = X.copy()
            idx = 0
            for k in range(12):
                if time.monotonic() >= until:
                    return
                signs = [0] if k < 3 else [1,-1]
                for sign in signs:
                    X[idx,k] = 1
                    R[idx,21+k] = 1
                    if k >= 3:
                        Y[idx,12+k-3] = sign
                    idx += 1
            _,x,y,r = proj
            z = np.r_[x,y,np.maximum(r,1e-6),1.0]
        else:
            n = 64
            X = np.zeros((21,n)); Y = X.copy(); R = X.copy()
            X[:,:21] = np.eye(21); Y[:,21:42] = np.eye(21); R[:,42:63] = np.eye(21)
            aa = a.copy()
            ctr = (np.max(aa[:,:2]+aa[:,2,None],axis=0)+np.min(aa[:,:2]-aa[:,2,None],axis=0))/2
            aa[:,:2] -= ctr
            z = np.r_[aa[:,0],aa[:,1],np.maximum(aa[:,2],1e-6),1.0]
        expanded = np.column_stack((X@z,Y@z,R@z))
        width = np.ptp(expanded[:,0])+2*np.mean(expanded[:,2])
        z[-1] = np.clip(width,0.15,1.85)
        W = np.zeros(n); W[-1] = .5
        L = np.vstack((X-R+W,-X-R+W,Y*-1-R-W,Y-R-W))
        constant = np.r_[np.zeros(42),np.ones(42)]
        DX = X[ii]-X[jj]; DY = Y[ii]-Y[jj]; SR = R[ii]+R[jj]
        objective = -np.sum(R,axis=0)
        cache_z = None; cache_g = None; cache_j = None
        def calc(v):
            nonlocal cache_z,cache_g,cache_j
            if time.monotonic() >= until:
                raise StopSearch()
            if cache_z is None or not np.array_equal(v,cache_z):
                dx,dy,rs = DX@v,DY@v,SR@v
                cache_g = np.r_[L@v+constant,dx*dx+dy*dy-rs*rs]
                cache_j = np.vstack((L,2*(dx[:,None]*DX+dy[:,None]*DY-rs[:,None]*SR)))
                cache_z = v.copy()
            return cache_g
        def jac(v):
            calc(v)
            return cache_j
        def callback(v):
            if time.monotonic() >= until:
                raise StopSearch()
            retain(np.column_stack((X@v,Y@v,R@v)))
        bounds = [(None,None)]*n
        for k in (range(21,33) if symmetric else range(42,63)):
            if time.monotonic() >= until:
                return
            bounds[k] = (1e-7,.8)
        bounds[-1] = (.05,1.95)
        try:
            res = minimize(lambda v: float(objective@v),z,jac=lambda v:objective,
                method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':calc,'jac':jac},callback=callback,
                options={'maxiter':180,'ftol':params['ftol'],'disp':False})
            callback(res.x)
        except StopSearch:
            pass

    if time.monotonic() < deadline:
        p0 = projection(a0,False)
        p1 = projection(a0,True)
        p = p0 if p0[0] <= p1[0] else p1
        split = started + params['symmetric_share']*(deadline-started)
        optimize(a0,True,split,p)
    if time.monotonic() < deadline and best_array is not None:
        optimize(best_array,False,deadline)
    return best
