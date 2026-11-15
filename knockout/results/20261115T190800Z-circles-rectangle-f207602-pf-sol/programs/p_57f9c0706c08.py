import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {
    'release_size': (0.0015, 0.00001, 0.03, 'log'),
    'contacts': (3, 3, 6),
    'polish_iterations': (65, 20, 120),
}


def evaluate(obj):
    try:
        c = np.asarray(obj['circles'], dtype=np.float64)
        if c.shape != (21, 3) or not np.isfinite(c).all() or not (c[:, 2] > 0).all():
            return -float('inf')
        x = [Fraction(float(t)) for t in c[:, 0]]
        y = [Fraction(float(t)) for t in c[:, 1]]
        r = [Fraction(float(t)) for t in c[:, 2]]
        p = max(a+b for a,b in zip(x,r))-min(a-b for a,b in zip(x,r))
        p += max(a+b for a,b in zip(y,r))-min(a-b for a,b in zip(y,r))
        if p > 2:
            return -float('inf')
        for i in range(21):
            d = c[i+1:, :2]-c[i, :2]
            if np.any(np.sqrt(d[:,0]**2+d[:,1]**2) < c[i+1:,2]+c[i,2]):
                return -float('inf')
        return float(np.sum(c[:,2]))
    except Exception:
        return -float('inf')


class StopSearch(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    if init is not None and 'orbits' in init:
        rows = []
        for x,y,r,m in init['orbits']:
            if time.monotonic() >= end:
                return init
            rows.append([x,y,r])
            if m == 2:
                rows.append([x,-y,r])
        original = {'circles': rows}
    else:
        original = init
    if original is None:
        original = {'circles': [[(i%7-3)*0.2, (i//7-1)*0.2, 0.09999999999999] for i in range(21)]}
    best = init if init is not None else original
    best_score = evaluate(original)
    c0 = np.array(original['circles'], dtype=float)

    def check():
        if time.monotonic() >= end:
            raise StopSearch()

    # Find the reflection orientation and average matched reflected circles.
    projections = []
    for swap in (False, True):
        if time.monotonic() >= end:
            return best
        c = c0.copy()
        if swap:
            c[:, [0,1]] = c[:, [1,0]]
        lo = np.min(c[:,:2]-c[:,2,None], axis=0)
        hi = np.max(c[:,:2]+c[:,2,None], axis=0)
        c[:,:2] -= (lo+hi)/2
        axis = np.argsort(np.abs(c[:,1]))[:3]
        rest = np.array([i for i in range(21) if i not in set(axis)])
        order = rest[np.argsort(c[rest,1])]
        neg, pos = order[:9], order[9:]
        target = c[neg].copy()
        target[:,1] *= -1
        cost = np.sum((c[pos,None,:]-target[None,:,:])**2, axis=2)
        aa, bb = linear_sum_assignment(cost)
        paired = (c[pos[aa]] + target[bb])/2
        orbs = np.vstack((c[axis], paired))
        error = float(np.sum(cost[aa,bb]) + np.sum(c[axis,1]**2))
        orbs[:3,1] = 0
        orbs[3:,1] = np.abs(orbs[3:,1])
        v = np.zeros(34)
        v[:12] = orbs[:,0]
        v[12:21] = orbs[3:,1]
        v[21:33] = orbs[:,2]
        v[33] = hi[0]-lo[0]
        projections.append((error,v))
    v0 = min(projections, key=lambda z:z[0])[1]

    # Linear maps for the twelve reflection orbits.
    X = np.zeros((12,34)); Y = np.zeros((12,34)); R = np.zeros((12,34))
    X[np.arange(12),np.arange(12)] = 1
    Y[np.arange(3,12),np.arange(12,21)] = 1
    R[np.arange(12),np.arange(21,33)] = 1
    dxs=[]; dys=[]; rss=[]
    for a in range(12):
        check()
        for b in range(a,12):
            check()
            if a == b:
                if a >= 3:
                    dxs.append(X[a]-X[b]); dys.append(2*Y[a]); rss.append(2*R[a])
            else:
                dxs.append(X[a]-X[b]); dys.append(Y[a]-Y[b]); rss.append(R[a]+R[b])
                if a >= 3 and b >= 3:
                    dxs.append(X[a]-X[b]); dys.append(Y[a]+Y[b]); rss.append(R[a]+R[b])
    DX=np.array(dxs); DY=np.array(dys); RS=np.array(rss)
    walls=[]; const=[]
    for a in range(12):
        check()
        z=-X[a]-R[a]; z[33]+=0.5; walls.append(z); const.append(0.)
        z=X[a]-R[a]; z[33]+=0.5; walls.append(z); const.append(0.)
        z=-Y[a]-R[a]; z[33]-=0.5; walls.append(z); const.append(1.)
    W=np.array(walls); wb=np.array(const)
    weights=np.zeros(34); weights[21:24]=1; weights[24:33]=2
    bounds=[(None,None)]*12+[(0.,1.)]*9+[(1e-6,0.5)]*12+[(0.2,1.8)]

    def expand(v):
        rows=[]
        for k in range(12):
            check()
            x=float(v[k]); y=0. if k<3 else float(v[k+9]); r=float(v[21+k])
            rows.append([x,y,r])
            if k>=3:
                rows.append([x,-y,r])
        return np.array(rows)

    def consider(v):
        nonlocal best,best_score
        check()
        if not np.isfinite(v).all():
            return
        c=expand(v)
        if np.min(c[:,2])<=0:
            return
        # Remove all floating-point overlap before perimeter normalization.
        ratio=1.0
        for i in range(20):
            check()
            d=c[i+1:,:2]-c[i,:2]
            dist=np.sqrt(d[:,0]**2+d[:,1]**2)
            ratio=min(ratio,float(np.min(dist/(c[i,2]+c[i+1:,2]))))
        c[:,2]*=max(0.,ratio)*(1.-2e-15)
        if np.min(c[:,2])<=0:
            return
        xs=[Fraction(float(t)) for t in c[:,0]]
        ys=[Fraction(float(t)) for t in c[:,1]]
        rs=[Fraction(float(t)) for t in c[:,2]]
        p=max(x+r for x,r in zip(xs,rs))-min(x-r for x,r in zip(xs,rs))
        p+=max(y+r for y,r in zip(ys,rs))-min(y-r for y,r in zip(ys,rs))
        scale=float(Fraction(2)/p)*(1.-3e-15)
        c*=scale
        if float(np.sum(c[:,2])) <= best_score:
            return
        obj={'circles':c.tolist()}
        score=evaluate(obj)
        if score>best_score:
            best_score=score; best=obj

    offset=np.zeros(len(DX))
    cache_v=None; cache_f=None; cache_j=None
    def compute(v):
        nonlocal cache_v,cache_f,cache_j
        check()
        if cache_v is not None and np.array_equal(cache_v,v):
            return
        dx=DX@v; dy=DY@v
        d=np.sqrt(dx*dx+dy*dy)
        safe=np.maximum(d,1e-30)
        cache_f=np.concatenate((W@v+wb,d-RS@v-offset))
        cache_j=np.vstack((W,(dx/safe)[:,None]*DX+(dy/safe)[:,None]*DY-RS))
        cache_v=v.copy()
    def fun(v):
        compute(v); return cache_f
    def jac(v):
        compute(v); return cache_j
    def objective(v):
        check(); return -float(weights@v)
    calls=0
    def callback(v):
        nonlocal calls
        check(); calls+=1
        if calls%7==0:
            consider(v)
    def solve(v,n):
        nonlocal cache_v
        cache_v=None
        res=minimize(objective,v,jac=lambda z:-weights,method='SLSQP',bounds=bounds,
                     constraints={'type':'ineq','fun':fun,'jac':jac},callback=callback,
                     options={'maxiter':n,'ftol':2e-14,'disp':False})
        consider(res.x)
        return res.x
    try:
        base=solve(v0,params['polish_iterations'])
        consider(base)
        anchor=base.copy()
        attempt=0
        while time.monotonic()<end:
            check()
            offset[:]=0
            base=anchor.copy()
            gaps=np.sqrt((DX@base)**2+(DY@base)**2)-RS@base
            active=np.flatnonzero(gaps<0.0001)
            if len(active)<3:
                active=np.argsort(gaps)[:20]
            count=min(int(params['contacts']),len(active))
            chosen=rng.choice(active,size=count,replace=False)
            amplitude=params['release_size']*(0.35,1.,3.,8.)[attempt%4]
            offset[chosen]=amplitude
            trial=solve(base,14)
            offset[:]=0
            trial=solve(trial,params['polish_iterations'])
            if weights@trial > weights@anchor+1e-9 and np.min(fun(trial))>-1e-8:
                anchor=trial.copy()
            attempt+=1
    except StopSearch:
        pass
    return best
