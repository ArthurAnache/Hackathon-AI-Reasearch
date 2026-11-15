import time
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize

PARAMS = {
    'screen_iterations': (22, 8, 45),
    'screen_fraction': (0.28, 0.15, 0.40),
    'symmetric_fraction': (0.60, 0.45, 0.65),
    'finalists': (3, 1, 4),
    'polish_iterations': (95, 30, 160),
    'radius_floor': (0.00003, 0.000001, 0.001, 'log'),
    'jitter': (0.012, 0.0, 0.08)
}
N = 21
I, J = np.triu_indices(N, 1)
K = np.arange(N)
D = 65

class Deadline(Exception):
    pass

class StageDone(Exception):
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
    q = list(map(Fraction, map(float, r)))
    w = max(u+v for u,v in zip(x,q))-min(u-v for u,v in zip(x,q))
    h = max(u+v for u,v in zip(y,q))-min(u-v for u,v in zip(y,q))
    return float(np.sum(r)) if w+h <= 2 else -np.inf

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70*budget_s
    screen_end = start + params['screen_fraction']*budget_s
    sym_end = start + params['symmetric_fraction']*budget_s
    rng = np.random.default_rng(seed)
    best, score, best_full = None, -np.inf, None
    raw = None

    def guard():
        if time.monotonic() >= end:
            raise Deadline()

    if init is not None:
        try:
            raw = np.asarray(init['circles'] if 'circles' in init else init.get('packing_rows', init.get('coordinate_rows')), dtype=float)
            if 'circles' not in init:
                raw = raw.T
            s = exact_score(raw)
            if s > score:
                best, score = init, s
        except (ValueError, TypeError, KeyError, OverflowError):
            raw = None

    fallback = np.column_stack(((np.arange(N)%5-2)*0.2, (np.arange(N)//5-2)*0.2, np.full(N, 0.1*(1-1e-12))))
    fs = exact_score(fallback)
    if fs > score:
        best, score = {'packing_rows': fallback.T.tolist()}, fs

    B = np.zeros((4*N, D))
    B[K,K] = 1
    B[K,2*N+K] = 1
    B[K,-2] = -0.5
    B[N+K,K] = -1
    B[N+K,2*N+K] = 1
    B[N+K,-2] = -0.5
    B[2*N+K,N+K] = 1
    B[2*N+K,2*N+K] = 1
    B[2*N+K,-1] = -0.5
    B[3*N+K,N+K] = -1
    B[3*N+K,2*N+K] = 1
    B[3*N+K,-1] = -0.5
    floor = float(params['radius_floor'])

    def normalized(a):
        a = a.copy()
        lo = np.min(a[:,:2]-a[:,2,None], axis=0)
        hi = np.max(a[:,:2]+a[:,2,None], axis=0)
        total = float(a[:,2].sum())
        a[:,:2] -= (lo+hi)/2
        a /= total
        return np.r_[a[:,0], a[:,1], a[:,2], (hi-lo)/total]

    def consider(v):
        nonlocal best, score, best_full
        guard()
        a = np.column_stack((v[:N],v[N:2*N],np.maximum(v[2*N:3*N],1e-15)))
        d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
        shrink = min(1.0, float(np.min(d/(a[I,2]+a[J,2]))))
        if not np.isfinite(shrink) or shrink <= 0:
            return
        a[:,2] *= shrink*(1-2e-12)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        size = float(np.sum(hi-lo))
        if size <= 0:
            return
        a[:,:2] -= (lo+hi)/2
        a *= (2/size)*(1-2e-12)
        if float(a[:,2].sum()) > score:
            s = exact_score(a)
            if s > score:
                best, score = {'packing_rows': a.T.tolist()}, s
                best_full = normalized(a)

    def model(T, bounds):
        gx = T[I]-T[J]
        gy = T[N+I]-T[N+J]
        gr = T[2*N+I]+T[2*N+J]
        walls = B@T
        eq = T[2*N:3*N].sum(axis=0)[None,:]
        obj = T[-2]+T[-1]
        return T,bounds,gx,gy,gr,walls,eq,obj

    def symmetric(singles, inner, outer):
        guard()
        p = inner+outer
        m = singles+p
        dim = m+p+m+2
        T = np.zeros((D,dim))
        ids = np.r_[np.arange(singles), np.arange(singles,m), np.arange(singles,m)]
        T[K,ids] = 1
        T[2*N+K,m+p+ids] = 1
        T[N+singles+np.arange(p),m+np.arange(p)] = 1
        T[N+singles+p+np.arange(p),m+np.arange(p)] = -1
        T[-2,-2] = T[-1,-1] = 1
        x = np.r_[2*np.arange(singles)-(singles-1), 2*np.arange(inner)-(inner-1), 2*np.arange(outer)-(outer-1)].astype(float)
        x += rng.normal(0,params['jitter'],m)
        y = np.r_[np.full(inner,np.sqrt(3.0)),np.full(outer,2*np.sqrt(3.0))]
        r = np.ones(m)
        w = 2*np.max(np.abs(x)+r)
        h = 2*np.max(y+1)
        q = np.r_[x,y,r,w,h]/N
        bounds = [(-2.,2.)]*m+[(0.,2.)]*p+[(floor,.5)]*m+[(.01,2.)]*2
        return model(T,bounds), q

    def lp(mod,q,count,until):
        T,bounds,gx,gy,gr,walls,eq,obj = mod
        stale = 0
        for it in range(count):
            guard()
            now = time.monotonic()
            if now >= until-0.004:
                break
            dx,dy = gx@q,gy@q
            d = np.maximum(np.hypot(dx,dy),1e-15)
            A = np.vstack((gr-(dx/d)[:,None]*gx-(dy/d)[:,None]*gy,walls))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res = linprog(obj,A_ub=A,b_ub=np.zeros(len(A)),A_eq=eq,b_eq=[1.],bounds=bounds,method='highs',options={'threads':1,'time_limit':max(.001,until-now),'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
            guard()
            if not res.success:
                break
            change = abs(float(obj@(q-res.x)))
            q = res.x
            consider(T@q)
            stale = stale+1 if change < 1e-10 else 0
            if stale >= 2:
                break
        return q

    def polish(mod,q,until):
        T,bounds,gx,gy,gr,walls,eq,obj = mod
        latest = [q.copy()]
        def check():
            guard()
            if time.monotonic() >= until:
                raise StageDone()
        def fun(z):
            check()
            return float(obj@z)
        def con(z):
            check()
            return np.r_[np.hypot(gx@z,gy@z)-gr@z,-walls@z]
        def jac(z):
            check()
            dx,dy = gx@z,gy@z
            d = np.maximum(np.hypot(dx,dy),1e-15)
            return np.vstack(((dx/d)[:,None]*gx+(dy/d)[:,None]*gy-gr,-walls))
        def callback(z):
            check()
            latest[0] = z.copy()
            consider(T@z)
        try:
            res = minimize(fun,q,jac=lambda z:obj,method='SLSQP',bounds=bounds,constraints=[{'type':'ineq','fun':con,'jac':jac},{'type':'eq','fun':lambda z:eq@z-1,'jac':lambda z:eq}],callback=callback,options={'maxiter':params['polish_iterations'],'ftol':2e-13,'disp':False})
            latest[0] = res.x
            consider(T@res.x)
        except StageDone:
            pass
        return latest[0]

    try:
        guard()
        if raw is not None and exact_score(raw) > 2.3:
            best_full = normalized(raw)
        layouts = [(3,5,4),(3,4,5),(5,4,4),(3,6,3)]
        survivors = []
        for index,layout in enumerate(layouts):
            guard()
            now = time.monotonic()
            if now >= screen_end-.008:
                break
            mod,q = symmetric(*layout)
            until = now+(screen_end-now)/(len(layouts)-index)
            q = lp(mod,q,params['screen_iterations'],until)
            survivors.append((float(mod[-1]@q),mod,q))
        survivors.sort(key=lambda v:v[0])
        count = min(params['finalists'],len(survivors))
        for index in range(count):
            guard()
            now = time.monotonic()
            if now >= sym_end-.008:
                break
            _,mod,q = survivors[index]
            until = now+(sym_end-now)/(count-index)
            q = polish(mod,q,until)
            consider(mod[0]@q)
        guard()
        if best_full is not None and time.monotonic() < end-.015:
            bounds = [(-2.,2.)]*(2*N)+[(floor,.5)]*N+[(.01,2.)]*2
            mod = model(np.eye(D),bounds)
            polish(mod,best_full,end-.006)
    except Deadline:
        pass
    return best
