import time
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize

PARAMS = {
    'trust': (0.18, 0.04, 0.4),
    'screen_steps': (18, 8, 35),
    'screen_fraction': (0.31, 0.15, 0.4),
    'lp_fraction': (0.47, 0.35, 0.56),
    'stable_rounds': (2, 1, 6),
    'polish_steps': (100, 30, 180)
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
K = np.arange(N)

class Deadline(Exception):
    pass


def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return -np.inf
    x = list(map(Fraction, map(float, a[:, 0])))
    y = list(map(Fraction, map(float, a[:, 1])))
    r = list(map(Fraction, map(float, a[:, 2])))
    perimeter = (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
                 +max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)))
    return float(np.sum(a[:, 2])) if perimeter <= 2 else -np.inf


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70*budget_s
    lp_end = start + params['lp_fraction']*budget_s
    best = None
    best_score = -np.inf
    raw = None

    def guard(stop=None):
        if time.monotonic() >= (end if stop is None else min(end, stop)):
            raise Deadline()

    def encode(a):
        return {'coordinate_rows': a.T.tolist()}

    if init is not None:
        try:
            raw = np.asarray(init['circles'] if 'circles' in init else np.asarray(init['coordinate_rows']).T, dtype=float)
            s = exact_score(raw)
            if s > best_score:
                best, best_score = init, s
        except (ValueError, KeyError, TypeError, OverflowError):
            raw = None
    ids = np.arange(N)
    fallback = np.column_stack(((ids%5+0.5)*0.2, (ids//5+0.5)*0.2, np.full(N, 0.1*(1-1e-12))))
    fs = exact_score(fallback)
    if fs > best_score:
        best, best_score = encode(fallback), fs

    B = np.zeros((84,64))
    B[K,K] = -1
    B[K,42+K] = 1
    B[21+K,K] = 1
    B[21+K,42+K] = 1
    B[21+K,-1] = -1
    B[42+K,21+K] = -1
    B[42+K,42+K] = 1
    B[42+K,-1] = 0.5
    B[63+K,21+K] = 1
    B[63+K,42+K] = 1
    B[63+K,-1] = 0.5
    bb = np.r_[np.zeros(42),np.ones(42)]
    rhs = np.r_[np.zeros(M),bb]

    def consider(v):
        nonlocal best, best_score
        guard()
        a = np.column_stack((v[:21],v[21:42],np.maximum(v[42:63],1e-12)))
        d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
        shrink = min(1.0,float(np.min(d/(a[I,2]+a[J,2]))))
        if not np.isfinite(shrink) or shrink <= 0:
            return
        a[:,2] *= shrink*(1-2e-12)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        span = float(np.sum(hi-lo))
        if span <= 0:
            return
        a[:,:2] -= (lo+hi)*0.5
        a *= (2.0/span)*(1-2e-12)
        if float(np.sum(a[:,2])) > best_score:
            s = exact_score(a)
            if s > best_score:
                best, best_score = encode(a), s

    def model(T,v,pos,rad):
        return {'T':T,'v':v,'dx':T[I]-T[J],'dy':T[21+I]-T[21+J],
                'rr':T[42+I]+T[42+J],'B':B@T,'obj':-T[42:63].sum(axis=0),
                'pos':pos,'rad':rad,'value':-np.inf,'stale':0,'stable':0,
                'contacts':None,'steps':0,'done':False}

    def unrestricted(a):
        a = a.copy()
        a[:,2] = np.maximum(np.abs(a[:,2]),1e-7)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        scale = 2/max(float(np.sum(hi-lo)),1e-12)
        a[:,0] = (a[:,0]-lo[0])*scale
        a[:,1] = (a[:,1]-(lo[1]+hi[1])*0.5)*scale
        a[:,2] *= scale
        v = np.r_[a[:,0],a[:,1],a[:,2],(hi[0]-lo[0])*scale]
        return model(np.eye(64),v,np.arange(42),np.arange(42,63))

    def symmetric(counts):
        T = np.zeros((64,34))
        T[K,np.r_[np.arange(12),np.arange(3,12)]] = 1
        T[42+K,np.r_[np.arange(21,33),np.arange(24,33)]] = 1
        T[21+np.arange(3,12),np.arange(12,21)] = 1
        T[21+np.arange(12,21),np.arange(12,21)] = -1
        T[-1,-1] = 1
        v = np.zeros(34)
        v[:3] = [0.2,0.5,0.8]
        offset = 3
        for row,count in enumerate(counts):
            guard()
            v[offset:offset+count] = (np.arange(count)+0.5)/count
            v[12+offset-3:12+offset-3+count] = 0.20*(row+1)
            offset += count
        v[21:33] = 0.04
        v[-1] = 1
        return model(T,v,np.arange(21),np.arange(21,33))

    def triangular():
        pieces = []
        for row,count in enumerate((4,3,4,3,4,3)):
            guard()
            x = np.arange(count,dtype=float)+(0.5 if count == 3 else 0)
            pieces.append(np.column_stack((x,np.full(count,row*np.sqrt(3)/2),np.full(count,0.49))))
        return unrestricted(np.vstack(pieces))

    def advance(m,steps,stop):
        for it in range(steps):
            guard()
            if time.monotonic() >= stop-0.004 or m['done']:
                break
            v = m['v']
            dx,dy = m['dx']@v,m['dy']@v
            dist = np.maximum(np.hypot(dx,dy),1e-15)
            A = np.vstack((m['rr']-(dx/dist)[:,None]*m['dx']-(dy/dist)[:,None]*m['dy'],m['B']))
            bounds = np.empty((len(v),2))
            bounds[:,0],bounds[:,1] = -2,2
            p = m['pos']
            trust = params['trust']
            bounds[p,0] = v[p]-trust
            bounds[p,1] = v[p]+trust
            bounds[m['rad'],0] = 1e-7
            bounds[m['rad'],1] = 0.5
            bounds[-1] = [0.05,1.95]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res = linprog(m['obj'],A_ub=A,b_ub=rhs,bounds=bounds,method='highs',
                    options={'threads':1,'time_limit':max(0.001,stop-time.monotonic()),
                             'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
            guard()
            if not res.success or res.x is None:
                break
            gain = -float(res.fun)+float(m['obj']@v)
            m['v'] = res.x
            m['value'] = -float(res.fun)
            m['steps'] += 1
            consider(m['T']@res.x)
            m['stale'] = m['stale']+1 if gain < 2e-10 else 0
            contacts = (rhs-A@res.x < 1e-7).tobytes()
            m['stable'] = m['stable']+1 if contacts == m['contacts'] else 0
            m['contacts'] = contacts
            limited = np.any(np.minimum(res.x[p]-bounds[p,0],bounds[p,1]-res.x[p]) < 1e-7)
            if m['stale'] >= 3 or (m['steps'] >= 9 and not limited and
                    m['stable'] >= params['stable_rounds'] and 0 <= gain < 0.001):
                m['done'] = True

    def polish(m,stop):
        def fun(v):
            guard(stop)
            return float(m['obj']@v)
        def con(v):
            guard(stop)
            dx,dy = m['dx']@v,m['dy']@v
            return np.r_[np.hypot(dx,dy)-m['rr']@v,bb-m['B']@v]
        def jac(v):
            guard(stop)
            dx,dy = m['dx']@v,m['dy']@v
            d = np.maximum(np.hypot(dx,dy),1e-15)
            return np.vstack(((dx/d)[:,None]*m['dx']+(dy/d)[:,None]*m['dy']-m['rr'],-m['B']))
        def callback(v):
            guard(stop)
            m['v'] = v.copy()
            consider(m['T']@v)
        bounds = [(None,None)]*len(m['v'])
        for index in m['rad']:
            guard(stop)
            bounds[index] = (1e-7,0.5)
        bounds[-1] = (0.05,1.95)
        try:
            res = minimize(fun,m['v'],jac=lambda v:m['obj'],method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':con,'jac':jac},callback=callback,
                options={'ftol':1e-12,'maxiter':params['polish_steps'],'disp':False})
            m['v'] = res.x
            consider(m['T']@res.x)
        except Deadline:
            pass

    try:
        valid_raw = raw is not None and raw.shape == (21,3) and np.isfinite(raw).all()
        models = [unrestricted(raw if valid_raw else fallback),symmetric((5,4)),triangular(),symmetric((4,5))]
        screen_end = start+params['screen_fraction']*budget_s
        for j,m in enumerate(models):
            guard()
            now = time.monotonic()
            if now >= screen_end:
                break
            stop = now+(screen_end-now)/(len(models)-j)
            advance(m,params['screen_steps'],stop)
        models.sort(key=lambda m:m['value'],reverse=True)
        for j,m in enumerate(models[:2]):
            guard()
            now = time.monotonic()
            if now >= lp_end:
                break
            stop = now+0.65*(lp_end-now) if j == 0 else lp_end
            advance(m,65,stop)
        models.sort(key=lambda m:m['value'],reverse=True)
        winner = models[0]
        if time.monotonic() < end-0.03:
            polish(winner,end-0.06)
        if len(winner['v']) < 64 and time.monotonic() < end-0.035:
            full = winner['T']@winner['v']
            released = model(np.eye(64),full,np.arange(42),np.arange(42,63))
            polish(released,end)
    except Deadline:
        pass
    return best
