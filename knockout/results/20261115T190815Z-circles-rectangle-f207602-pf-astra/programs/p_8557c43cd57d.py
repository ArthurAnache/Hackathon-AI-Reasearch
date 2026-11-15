import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

PARAMS = {
    'temperature': (0.11, 0.01, 0.4, 'log'),
    'stages': (4, 2, 7),
    'anneal': (0.23, 0.1, 0.6),
    'iterations': (100, 30, 250),
    'equalization': (0.35, 0.0, 1.0),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class Stop(Exception):
    pass


def score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return -np.inf
    x = list(map(lambda v: Fraction(float(v)), a[:, 0]))
    y = list(map(lambda v: Fraction(float(v)), a[:, 1]))
    r = list(map(lambda v: Fraction(float(v)), a[:, 2]))
    perimeter = (max(map(sum, zip(x, r))) - min(a-b for a,b in zip(x,r))
                 + max(map(sum, zip(y, r))) - min(a-b for a,b in zip(y,r)))
    return float(np.sum(a[:, 2])) if perimeter <= 2 else -np.inf


def repair(a):
    a = np.array(a, dtype=float, copy=True)
    if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return None
    d = np.sqrt((a[I,0]-a[J,0])**2 + (a[I,1]-a[J,1])**2)
    factor = min(1.0, float(np.min(d/(a[I,2]+a[J,2]))))
    a[:,2] *= factor * (1-4e-14)
    lo = np.min(a[:,:2]-a[:,2,None], axis=0)
    hi = np.max(a[:,:2]+a[:,2,None], axis=0)
    a[:,:2] -= lo
    a *= (2.0/np.sum(hi-lo))*(1-4e-14)
    return a


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf
    if init is not None:
        base = np.array(init['circles'], dtype=float)
        best_score = score(base)
    else:
        base = np.array([[(i%7+0.5)/7, (i//7+0.5)/3, 0.03] for i in range(21)])
        best = {'circles': base.tolist()}
        best_score = score(base)

    def consider(a):
        nonlocal best, best_score
        if a is None or time.monotonic() >= end:
            return
        value = score(a)
        if value > best_score:
            best_score = value
            best = {'circles': a.tolist()}

    lo = np.min(base[:,:2]-base[:,2,None], axis=0)
    hi = np.max(base[:,:2]+base[:,2,None], axis=0)
    scale = 2/np.sum(hi-lo)
    p = (base[:,:2]-lo)*scale
    r = base[:,2]*scale
    w = (hi[0]-lo[0])*scale
    z0 = np.r_[p[:,0]/w, p[:,1]/(2-w), np.log(r), w]
    bounds = [(1e-5,1-1e-5)]*42 + [(-10.0,1.0)]*21 + [(0.25,1.75)]
    last = z0.copy()
    cutoff = end

    def smooth(z, tau):
        nonlocal last
        if time.monotonic() >= cutoff:
            raise Stop()
        last = z.copy()
        x,y,l,w = z[:21],z[21:42],z[42:63],z[63]
        h = 2-w
        dx = w*(x[I]-x[J])
        dy = h*(y[I]-y[J])
        d2 = np.maximum(dx*dx+dy*dy, 1e-25)
        pair_l = np.logaddexp(l[I],l[J])
        logs = np.r_[pair_l-0.5*np.log(d2), l-np.log(w*x),
                     l-np.log(w*(1-x)), l-np.log(h*y), l-np.log(h*(1-y))]
        normalizer = logsumexp(logs/tau)
        weights = np.exp(logs/tau-normalizer)
        v = weights[:M]
        left,right,bottom,top = weights[M:].reshape(4,21)
        gx = np.bincount(I, -v*dx*w/d2, minlength=21)+np.bincount(J,v*dx*w/d2,minlength=21)
        gy = np.bincount(I, -v*dy*h/d2, minlength=21)+np.bincount(J,v*dy*h/d2,minlength=21)
        gx += -left/x+right/(1-x)
        gy += -bottom/y+top/(1-y)
        q = np.exp(l-logsumexp(l))
        gl = np.bincount(I,v*np.exp(l[I]-pair_l),minlength=21)+np.bincount(J,v*np.exp(l[J]-pair_l),minlength=21)
        gl += left+right+bottom+top-q
        gw = np.sum(-v*(dx*(x[I]-x[J])-dy*(y[I]-y[J]))/d2)
        gw += -(left.sum()+right.sum())/w+(bottom.sum()+top.sum())/h
        return tau*normalizer-logsumexp(l), np.r_[gx,gy,gl,gw]

    def decode(z):
        w = z[63]
        p = np.c_[w*z[:21],(2-w)*z[21:42]]
        r = np.exp(z[42:63])
        d = np.sqrt(np.sum((p[I]-p[J])**2,axis=1))
        congestion = max(float(np.max((r[I]+r[J])/np.maximum(d,1e-30))),
                         float(np.max(r/p[:,0])), float(np.max(r/(w-p[:,0]))),
                         float(np.max(r/p[:,1])),float(np.max(r/(2-w-p[:,1]))))
        return repair(np.c_[p,r/congestion])

    # The softened maximum changes the contact load before returning to the
    # hard worst-contact problem. Two starts use different radius proportions.
    finalists = []
    for attempt in range(2):
        if time.monotonic() >= start+0.43*budget_s:
            break
        z = z0.copy()
        if attempt:
            mix = float(params['equalization'])
            z[42:63] = (1-mix)*z[42:63]+mix*np.mean(z[42:63])
            z[:42] = np.clip(z[:42]+rng.normal(0,0.025,42),0.01,0.99)
        cutoff = min(start+(0.225+attempt*0.205)*budget_s, end)
        for stage in range(int(params['stages'])):
            if time.monotonic() >= cutoff:
                break
            tau = float(params['temperature'])*float(params['anneal'])**stage
            last = z.copy()
            try:
                z = minimize(smooth,z,args=(tau,),jac=True,method='L-BFGS-B',bounds=bounds,
                             options={'maxiter':int(params['iterations']),'ftol':2e-12,'gtol':1e-7,'maxls':25}).x
            except Stop:
                z = last
                break
        if time.monotonic() < end:
            candidate = decode(z)
            consider(candidate)
            if candidate is not None:
                finalists.append(candidate)

    # Switch back to the original radius-sum objective, with analytic Jacobian.
    row = np.arange(M)
    k = np.arange(21)
    def constraints(z):
        if time.monotonic() >= cutoff:
            raise Stop()
        x,y,r,w = z[:21],z[21:42],z[42:63],z[63]
        dx,dy = x[I]-x[J],y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        return np.r_[d-r[I]-r[J],x-r,w-x-r,y-r,2-w-y-r]

    def jacobian(z):
        if time.monotonic() >= cutoff:
            raise Stop()
        dx,dy = z[I]-z[J],z[21+I]-z[21+J]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        a = np.zeros((M+84,64))
        a[row,I] = dx/d
        a[row,J] = -dx/d
        a[row,21+I] = dy/d
        a[row,21+J] = -dy/d
        a[row,42+I] = -1
        a[row,42+J] = -1
        a[M+k,k] = 1
        a[M+21+k,k] = -1
        a[M+21+k,63] = 1
        a[M+42+k,21+k] = 1
        a[M+63+k,21+k] = -1
        a[M+63+k,63] = -1
        a[M+k,42+k] = -1
        a[M+21+k,42+k] = -1
        a[M+42+k,42+k] = -1
        a[M+63+k,42+k] = -1
        return a

    gradient = np.r_[np.zeros(42),-np.ones(21),0.0]
    def objective(z):
        if time.monotonic() >= cutoff:
            raise Stop()
        return -np.sum(z[42:63]),gradient

    saved = None
    def callback(z):
        nonlocal saved
        saved = z.copy()
        if time.monotonic() >= cutoff:
            raise Stop()

    finalists.sort(key=lambda a: -float(a[:,2].sum()))
    for index,a in enumerate(finalists):
        now = time.monotonic()
        if now >= end-0.025:
            break
        cutoff = min(end-0.015,now+(end-0.015-now)/(len(finalists)-index))
        low = np.min(a[:,:2]-a[:,2,None],axis=0)
        high = np.max(a[:,:2]+a[:,2,None],axis=0)
        p = a[:,:2]-low
        z = np.r_[p[:,0],p[:,1],a[:,2],high[0]-low[0]]
        saved = z.copy()
        try:
            result = minimize(objective,z,jac=True,method='SLSQP',
                              bounds=[(None,None)]*42+[(1e-7,None)]*21+[(0.2,1.8)],
                              constraints={'type':'ineq','fun':constraints,'jac':jacobian},
                              callback=callback,options={'maxiter':130,'ftol':2e-13})
            saved = result.x
        except Stop:
            pass
        if time.monotonic() < end:
            consider(repair(np.c_[saved[:21],saved[21:42],saved[42:63]]))
    return best
