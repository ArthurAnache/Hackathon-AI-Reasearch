import time
import numpy as np
from fractions import Fraction
from scipy.optimize import minimize

PARAMS = {
    'first_share': (0.32, 0.2, 0.45),
    'second_share': (0.30, 0.2, 0.4),
    'safety': (2e-14, 5e-15, 1e-10, 'log'),
    'perturb': (0.018, 0.0, 0.08)
}
I, J = np.triu_indices(21, 1)

class Timeout(Exception):
    pass

def valid(a):
    if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
        return False
    d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
    if np.any(d < a[I,2]+a[J,2]):
        return False
    x,y,r = [[Fraction(float(v)) for v in a[:,k]] for k in range(3)]
    return (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))+
            max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))) <= 2

def repair(a, eps):
    a = np.array(a, dtype=float, copy=True)
    if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
        return None
    d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
    rat = min(1., float(np.min(d/(a[I,2]+a[J,2]))))
    if rat <= 0:
        return None
    a[:,2] *= rat*(1-eps)
    lo = np.min(a[:,:2]-a[:,2,None],axis=0)
    hi = np.max(a[:,:2]+a[:,2,None],axis=0)
    scale = (2-8*eps)/float(np.sum(hi-lo))
    a[:,:2] = (a[:,:2]-(lo+hi)/2)*scale
    a[:,2] *= scale
    return a if valid(a) else None

def model(sym):
    n = 34 if sym else 64
    x = np.zeros((21,n)); y = x.copy(); r = x.copy()
    if sym:
        ids = np.r_[np.arange(12),np.arange(3,12)]
        x[np.arange(21),ids] = 1
        for i in range(3,12):
            y[i,9+i] = 1
            y[i+9,9+i] = -1
        r[np.arange(21),21+ids] = 1
        pairs = [(i,j) for i in range(12) for j in range(i+1,12)]
        pairs += [(i,j+9) for i in range(3,12) for j in range(i,12)]
        pi,pj = np.array(pairs).T
        wall = np.arange(12)
        rid = range(21,33)
    else:
        x[np.arange(21),np.arange(21)] = 1
        y[np.arange(21),21+np.arange(21)] = 1
        r[np.arange(21),42+np.arange(21)] = 1
        pi,pj = I,J
        wall = np.arange(21)
        rid = range(42,63)
    w = np.zeros(n); w[-1] = 1
    b = np.concatenate((x[wall]-r[wall],w-x[wall]-r[wall],
                        -.5*w-y[wall]-r[wall],-.5*w+y[wall]-r[wall]))
    off = np.r_[np.zeros(2*len(wall)),np.ones(2*len(wall))]
    bounds = [(None,None)]*n
    for i in rid:
        bounds[i] = (1e-5,.5)
    bounds[-1] = (.15,1.85)
    return x,y,r,x[pi]-x[pj],y[pi]-y[pj],r[pi]+r[pj],b,off,-r.sum(axis=0),bounds

def strip(counts):
    singles = []; pairs = []
    for j,c in enumerate(counts):
        xx = 1+j*np.sqrt(3.)
        if c % 2:
            singles.append((xx,0.,1.))
            for k in range(1,c//2+1):
                pairs.append((xx,2.*k,1.))
        else:
            for k in range(c//2):
                pairs.append((xx,2.*k+1,1.))
    a = np.array(singles+pairs+[(x,-y,r) for x,y,r in pairs])
    lo = np.min(a[:,:2]-a[:,2,None],axis=0)
    hi = np.max(a[:,:2]+a[:,2,None],axis=0)
    scale = 2/np.sum(hi-lo)
    a[:,:2] -= np.array([lo[0],(lo[1]+hi[1])/2])
    a *= scale
    return np.r_[a[:12,0],a[3:12,1],a[:12,2],(hi[0]-lo[0])*scale]

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start+.70*budget_s
    eps = float(params['safety'])
    rng = np.random.default_rng(seed)
    best = init
    bestscore = -np.inf
    bestarray = None
    if init is not None:
        try:
            a = np.asarray(init['circles'],float) if 'circles' in init else np.asarray(init['state'],float).reshape(3,21).T
            if valid(a):
                bestarray = a.copy()
                bestscore = float(np.sum(a[:,2]))
        except (ValueError,TypeError,KeyError):
            pass
    def offer(a):
        nonlocal best,bestscore,bestarray
        a = repair(a,eps)
        if a is not None:
            score = float(np.sum(a[:,2]))
            if score > bestscore:
                bestscore = score
                bestarray = a.copy()
                best = {'state':a.T.reshape(-1).tolist()}
    sm = model(True)
    fm = model(False)
    def unpack(z,m):
        return np.column_stack((m[0]@z,m[1]@z,m[2]@z))
    def fullencode(a):
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        return np.r_[a[:,0]-lo[0],a[:,1]-(lo[1]+hi[1])/2,a[:,2],hi[0]-lo[0]]
    def optimize(z,m,stop):
        x,y,r,dxm,dym,rsm,b,off,obj,bounds = m
        last = [z.copy()]
        cache = [None,None,None]
        def calc(v):
            if time.monotonic() >= stop:
                raise Timeout
            if cache[0] is not None and np.array_equal(v,cache[0]):
                return cache[1],cache[2]
            dx=dxm@v; dy=dym@v; rs=rsm@v
            c=np.r_[dx*dx+dy*dy-rs*rs,b@v+off]
            jac=np.concatenate((2*(dx[:,None]*dxm+dy[:,None]*dym-rs[:,None]*rsm),b))
            cache[:] = [v.copy(),c,jac]
            return c,jac
        def fun(v):
            if time.monotonic() >= stop:
                raise Timeout
            return float(obj@v),obj
        def callback(v):
            if time.monotonic() >= stop:
                raise Timeout
            last[0] = v.copy()
        try:
            res=minimize(fun,z,jac=True,method='SLSQP',bounds=bounds,
                         constraints={'type':'ineq','fun':lambda v:calc(v)[0],'jac':lambda v:calc(v)[1]},
                         callback=callback,options={'maxiter':160,'ftol':3e-13,'disp':False})
            z=res.x
        except Timeout:
            z=last[0]
        offer(unpack(z,m))
        return z
    seeds = [strip([3,4,3,4,3,4]),strip([5,4,3,4,5])]
    for z in seeds:
        if time.monotonic() >= deadline-.02:
            break
        offer(unpack(z,sm))
    cold = bestscore < 2.32
    if not cold and bestarray is not None:
        optimize(fullencode(bestarray),fm,min(deadline-.02,start+.32*budget_s))
    results=[]
    for k,z in enumerate(seeds):
        now=time.monotonic()
        if now >= deadline-.08:
            break
        share = float(params['first_share'] if k==0 else params['second_share'])
        stop = min(deadline-.08,now+share*.70*budget_s)
        z=optimize(z,sm,stop)
        results.append(z)
    if bestarray is not None and time.monotonic() < deadline-.06:
        optimize(fullencode(bestarray),fm,min(deadline-.025,time.monotonic()+.28*budget_s))
    for k in range(4):
        now=time.monotonic()
        if now >= deadline-.06 or not results:
            break
        z=results[k%len(results)].copy()
        z[:21] += rng.normal(0,float(params['perturb']),21)
        z[12:21] = np.abs(z[12:21])
        results[k%len(results)] = optimize(z,sm,min(deadline-.015,now+.12))
    if best is None:
        a=np.array([[(i%7)*.12,(i//7)*.12,.059] for i in range(21)])
        offer(a)
    return best
