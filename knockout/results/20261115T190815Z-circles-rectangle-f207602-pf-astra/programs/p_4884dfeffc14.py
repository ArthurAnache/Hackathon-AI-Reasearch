import time
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize

PARAMS = {
    'trust': (0.20, 0.05, 0.40),
    'lp_steps': (7, 2, 20),
    'fraction': (0.70, 0.50, 0.72),
    'perturbation': (0.009, 0.001, 0.04)
}
N = 21
I, J = np.triu_indices(N, 1)
K = np.arange(N)
M = len(I)

class Deadline(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    r = a[:, 2]
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < r[I]+r[J]):
        return -np.inf
    x = list(map(Fraction, map(float, a[:, 0])))
    y = list(map(Fraction, map(float, a[:, 1])))
    r = list(map(Fraction, map(float, r)))
    w = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
    h = max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
    return float(np.sum(a[:, 2])) if w+h <= 2 else -np.inf

def construct(params, seed, budget_s, init=None):
    begin = time.monotonic()
    end = begin + float(params['fraction'])*budget_s
    stop = end
    rng = np.random.default_rng(seed)
    best_score = -np.inf
    best_obj = None
    best_a = None
    raw = None

    def guard():
        if time.monotonic() >= stop:
            raise Deadline()

    def keep(a, original=None):
        nonlocal best_score, best_obj, best_a
        if float(np.sum(a[:, 2])) <= best_score:
            return
        sc = exact_score(a)
        if sc > best_score:
            best_score = sc
            best_a = a.copy()
            best_obj = original if original is not None else {'coordinate_rows': a.T.tolist()}

    if init is not None:
        try:
            raw = np.asarray(init['circles'] if 'circles' in init else np.asarray(init['coordinate_rows']).T, dtype=float)
            if raw.shape == (N,3) and np.isfinite(raw).all():
                keep(raw, init)
            else:
                raw = None
        except (ValueError, TypeError, KeyError, OverflowError):
            raw = None
    k = np.arange(N)
    fallback = np.column_stack(((k % 5+0.5)*0.2, (k//5+0.5)*0.2, np.full(N,0.1*(1-1e-12))))
    keep(fallback)
    if raw is None:
        raw = fallback.copy()

    B = np.zeros((4*N, 3*N+1))
    B[K,K] = 1
    B[N+K,K] = -1
    B[2*N+K,N+K] = 1
    B[3*N+K,N+K] = -1
    B[K,2*N+K] = 1
    B[N+K,2*N+K] = 1
    B[2*N+K,2*N+K] = 1
    B[3*N+K,2*N+K] = 1
    B[:2*N,-1] = -0.5
    B[2*N:,-1] = 0.5
    bb = np.r_[np.zeros(2*N), np.ones(2*N)]
    objective = np.r_[np.zeros(2*N), -np.ones(N), 0.0]

    def pack(a):
        a = a.copy()
        a[:,2] = np.maximum(np.abs(a[:,2]), 1e-7)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        sides = hi-lo
        scale = 2/max(float(sides.sum()),1e-12)
        a[:,:2] = (a[:,:2]-(lo+hi)/2)*scale
        a[:,2] *= scale
        return np.r_[a[:,0],a[:,1],a[:,2],sides[0]*scale]

    def consider(v):
        guard()
        a = np.column_stack((v[:N],v[N:2*N],np.maximum(v[2*N:3*N],1e-10)))
        d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
        f = min(1.0,float(np.min(d/(a[I,2]+a[J,2]))))
        if not np.isfinite(f) or f <= 0:
            return
        a[:,2] *= f*(1-2e-12)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        a[:,:2] -= (lo+hi)/2
        a *= (2/float((hi-lo).sum()))*(1-2e-12)
        keep(a)

    def lattice(counts):
        a = []
        groups = []
        for row,count in enumerate(counts):
            guard()
            base = len(a)
            for j in range(count):
                guard()
                a.append([j-(count-1)/2, (row-(len(counts)-1)/2)*np.sqrt(3)/2, 0.48])
            for j in range((count+1)//2):
                guard()
                groups.append((base+j,base+count-1-j))
        pairs = sum(i != j for i,j in groups)
        g = len(groups)
        T = np.zeros((64,pairs+2*g+1))
        perm = np.arange(N)
        p = 0
        for h,(i,j) in enumerate(groups):
            guard()
            perm[i],perm[j] = j,i
            if i != j:
                T[i,p],T[j,p] = -1,1
                p += 1
            T[N+i,pairs+h] = 1
            T[N+j,pairs+h] = 1
            T[2*N+i,pairs+g+h] = 1
            T[2*N+j,pairs+g+h] = 1
        T[-1,-1] = 1
        return np.asarray(a), T, pairs+g, perm

    def run(a, T=None, nr=None, perm=None, steps=7):
        guard()
        if T is None:
            T = np.eye(64)
            nr = 42
        v = pack(a)
        q = (T.T@v)/np.sum(T*T,axis=0)
        dim = len(q)
        if perm is None:
            ii,jj = I,J
        else:
            reflected_i = np.minimum(perm[I],perm[J])
            reflected_j = np.maximum(perm[I],perm[J])
            mask = (I*N+J <= reflected_i*N+reflected_j)
            ii,jj = I[mask],J[mask]
        m = len(ii)
        p = np.arange(m)
        BT = B@T
        _, unique = np.unique(np.column_stack((BT,bb)),axis=0,return_index=True)
        BT,bt = BT[unique],bb[unique]
        c = objective@T
        bounds = [(None,None)]*nr + [(1e-7,0.5)]*(dim-nr-1)+[(0.05,1.95)]
        base = np.zeros((m,64))
        base[p,2*N+ii] = 1
        base[p,2*N+jj] = 1
        lp_end = min(stop, time.monotonic()+0.075*budget_s)
        previous = -float(c@q)
        stagnant = 0
        consider(T@q)
        for iteration in range(steps):
            guard()
            if time.monotonic() >= lp_end:
                break
            v = T@q
            dx,dy = v[ii]-v[jj],v[N+ii]-v[N+jj]
            d = np.maximum(np.hypot(dx,dy),1e-15)
            base[p,ii],base[p,jj] = -dx/d,dx/d
            base[p,N+ii],base[p,N+jj] = -dy/d,dy/d
            local_bounds = list(bounds)
            for h in range(nr):
                guard()
                local_bounds[h] = (q[h]-params['trust'],q[h]+params['trust'])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res = linprog(c,A_ub=np.vstack((base@T,BT)),b_ub=np.r_[np.zeros(m),bt],bounds=local_bounds,method='highs',options={'threads':1,'time_limit':max(0.001,lp_end-time.monotonic()),'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
            guard()
            if not res.success:
                break
            q = res.x
            consider(T@q)
            value = -float(c@q)
            stagnant = stagnant+1 if value-previous < 2e-6 else 0
            previous = value
            if stagnant >= 2:
                break

        def con(q):
            guard()
            v = T@q
            dx,dy = v[ii]-v[jj],v[N+ii]-v[N+jj]
            return np.r_[np.hypot(dx,dy)-v[2*N+ii]-v[2*N+jj],bt-BT@q]

        def jac(q):
            guard()
            v = T@q
            dx,dy = v[ii]-v[jj],v[N+ii]-v[N+jj]
            d = np.maximum(np.hypot(dx,dy),1e-15)
            G = -base.copy()
            G[p,ii],G[p,jj] = dx/d,-dx/d
            G[p,N+ii],G[p,N+jj] = dy/d,-dy/d
            return np.vstack((G@T,-BT))

        def fun(q):
            guard()
            return float(c@q)

        result = minimize(fun,q,jac=lambda q:c,method='SLSQP',bounds=bounds,constraints={'type':'ineq','fun':con,'jac':jac},callback=lambda q:consider(T@q),options={'maxiter':140,'ftol':2e-12,'disp':False})
        consider(T@result.x)

    fractions = [0.30,0.49,0.67,0.80,1.0]
    for stage in range(5):
        if time.monotonic() >= end:
            break
        stop = begin + (end-begin)*fractions[stage]
        if time.monotonic() >= stop-0.012:
            continue
        try:
            if stage == 0:
                a,T,nr,perm = lattice([5,4,3,4,5])
                run(a,T,nr,perm,int(params['lp_steps']))
            elif stage == 1:
                run(raw,steps=int(params['lp_steps']))
            elif stage == 2:
                a,T,nr,perm = lattice([4,3,4,3,4,3])
                run(a,T,nr,perm,int(params['lp_steps']))
            elif stage == 3:
                a = best_a.copy()
                a[:,:2] += rng.normal(0,float(params['perturbation']),size=(N,2))
                run(a,steps=3)
            else:
                run(best_a.copy(),steps=0)
        except (Deadline,ValueError,FloatingPointError):
            pass
    return best_obj
