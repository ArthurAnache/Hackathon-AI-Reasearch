import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {
    'full_fraction': (0.60, 0.35, 0.85),
    'iterations': (300, 100, 600),
    'safety': (1e-14, 3e-15, 1e-11, 'log'),
    'restart_size': (0.008, 0.001, 0.05)
}
I, J = np.triu_indices(21, 1)

class Timeout(Exception):
    pass

def valid(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return False
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return False
    x = list(map(Fraction, map(float, a[:, 0])))
    y = list(map(Fraction, map(float, a[:, 1])))
    r = list(map(Fraction, map(float, a[:, 2])))
    return (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r)) +
            max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))) <= 2

def repair(a, eps):
    a = np.array(a, dtype=float, copy=True)
    if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
        return None
    d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
    ratio = min(1., float(np.min(d/(a[I,2]+a[J,2]))))
    if ratio <= 0:
        return None
    a[:,2] *= ratio*(1-eps)
    lo = np.min(a[:,:2]-a[:,2,None], axis=0)
    hi = np.max(a[:,:2]+a[:,2,None], axis=0)
    scale = (2-8*eps)/np.sum(hi-lo)
    a[:,:2] = (a[:,:2]-(lo+hi)/2)*scale
    a[:,2] *= scale
    return a if valid(a) else None

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + .70*budget_s
    rng = np.random.default_rng(seed)
    eps = params['safety']
    best = None
    score = -np.inf
    original = None
    def read(obj):
        if 'circles' in obj:
            return np.asarray(obj['circles'], dtype=float)
        return np.asarray(obj['state'], dtype=float).reshape(3,21).T.copy()
    try:
        if init is not None:
            original = read(init)
            if valid(original):
                best = init
                score = float(np.sum(original[:,2]))
    except (ValueError, TypeError, KeyError):
        original = None
    def offer(a):
        nonlocal best, score
        b = repair(a, eps)
        if b is not None:
            s = float(np.sum(b[:,2]))
            if s > score:
                score = s
                best = {'state': b.T.reshape(-1).tolist()}
        return b
    if original is not None:
        offer(original)
    fallback = np.array([[2*j+(1 if n==4 else 0),np.sqrt(3)*i,1.]
                         for i,n in enumerate((4,5,4,5,3)) for j in range(n)])
    offer(fallback)
    if best is None:
        return {'circles': [[.2*(i%3), .2*(i//3), .09] for i in range(21)]}

    def centered(a):
        a = a.copy()
        lo = np.min(a[:,:2]-a[:,2,None], axis=0)
        hi = np.max(a[:,:2]+a[:,2,None], axis=0)
        a[:,:2] -= (hi+lo)/2
        return a, float(hi[0]-lo[0])

    def optimize(z, sym, stop):
        if sym:
            ids = np.r_[np.arange(12), np.arange(3,12)]
            ix = ids
            iy = np.where(ids>=3,12+ids-3,0)
            sy = np.r_[np.zeros(3),np.ones(9),-np.ones(9)]
            ir = 21+ids
            n = 34
            pairs = [(i,j) for i in range(12) for j in range(i+1,12)]
            pairs += [(i,12+j-3) for i in range(3,12) for j in range(i,12)]
            pi,pj = np.array(pairs).T
            wall = np.arange(12)
            ridx = np.arange(21,33)
        else:
            ix = np.arange(21)
            iy = 21+ix
            sy = np.ones(21)
            ir = 42+ix
            n = 64
            pi,pj = I,J
            wall = np.arange(21)
            ridx = np.arange(42,63)
        m = len(pi)
        q = len(wall)
        rows = np.arange(m)
        obj = np.zeros(n)
        np.add.at(obj, ir, -1.)
        def unpack(v):
            return np.column_stack((v[ix],v[iy]*sy,v[ir]))
        initial = unpack(z)
        weights = 1/np.maximum(2*(initial[pi,2]+initial[pj,2]),.08)
        walljac = np.zeros((4*q,n))
        wr = np.arange(q)
        walljac[wr,ix[wall]] = -1
        walljac[wr,ir[wall]] = -1
        walljac[wr,-1] = .5
        walljac[q+wr,ix[wall]] = 1
        walljac[q+wr,ir[wall]] = -1
        walljac[q+wr,-1] = .5
        walljac[2*q+wr,iy[wall]] += -sy[wall]
        walljac[2*q+wr,ir[wall]] -= 1
        walljac[2*q+wr,-1] = -.5
        walljac[3*q+wr,iy[wall]] += sy[wall]
        walljac[3*q+wr,ir[wall]] -= 1
        walljac[3*q+wr,-1] = -.5
        offset = np.r_[np.zeros(2*q), np.ones(2*q)]
        cache = [None,None,None]
        last = [z.copy()]
        count = [0]
        def calc(v):
            if time.monotonic() >= stop:
                raise Timeout()
            if cache[0] is not None and np.array_equal(cache[0],v):
                return cache[1],cache[2]
            x=v[ix]; y=v[iy]*sy; r=v[ir]
            dx=x[pi]-x[pj]; dy=y[pi]-y[pj]; rs=r[pi]+r[pj]
            c=np.r_[(dx*dx+dy*dy-rs*rs)*weights,walljac@v+offset]
            jac=np.zeros((m+4*q,n))
            tx=2*weights*dx; ty=2*weights*dy; tr=-2*weights*rs
            jac[rows,ix[pi]] += tx
            jac[rows,ix[pj]] -= tx
            jac[rows,iy[pi]] += ty*sy[pi]
            jac[rows,iy[pj]] -= ty*sy[pj]
            jac[rows,ir[pi]] += tr
            jac[rows,ir[pj]] += tr
            jac[m:] = walljac
            cache[:]=[v.copy(),c,jac]
            last[0]=v.copy()
            return c,jac
        def fun(v):
            if time.monotonic() >= stop:
                raise Timeout()
            return float(obj@v),obj
        def callback(v):
            if time.monotonic() >= stop:
                raise Timeout()
            count[0]+=1
            if count[0]%15==0:
                offer(unpack(v))
        bounds=[(None,None)]*n
        for k in ridx:
            if time.monotonic() >= stop:
                return z
            bounds[int(k)]=(1e-7,.5)
        bounds[-1]=(.15,1.85)
        try:
            res=minimize(fun,z,jac=True,method='SLSQP',bounds=bounds,
                         constraints={'type':'ineq','fun':lambda v:calc(v)[0],'jac':lambda v:calc(v)[1]},
                         callback=callback,options={'maxiter':int(params['iterations']),'ftol':8e-14,'disp':False})
            z=res.x
        except Timeout:
            z=last[0]
        offer(unpack(z))
        return z

    a = original if original is not None else read(best)
    if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2]<=0):
        a=read(best)
    a,w=centered(a)
    z=np.r_[a[:,0],a[:,1],a[:,2],w]
    z=optimize(z,False,min(deadline-.025,start+.70*budget_s*params['full_fraction']))
    if time.monotonic() >= deadline-.035:
        return best
    a=read(best)
    choices=[]
    for swap in (False,True):
        if time.monotonic() >= deadline-.025:
            return best
        b=a.copy()
        if swap:
            b[:,:2]=b[:,[1,0]]
        b,w=centered(b)
        reflected=b.copy(); reflected[:,1]*=-1
        cost=np.sum((b[:,None,:]-reflected[None,:,:])**2,axis=2)
        rows,cols=linear_sum_assignment(cost)
        choices.append((float(cost[rows,cols].sum()),b,w))
    _,a,w=min(choices,key=lambda p:p[0])
    singles=np.argsort(np.abs(a[:,1]))[:3]
    mask=np.ones(21,dtype=bool); mask[singles]=False
    rem=np.flatnonzero(mask)
    order=rem[np.argsort(a[rem,1])]
    neg,pos=order[:9],order[9:]
    cost=((a[pos,None,0]-a[None,neg,0])**2+
          (a[pos,None,1]+a[None,neg,1])**2+
          (a[pos,None,2]-a[None,neg,2])**2)
    _,cols=linear_sum_assignment(cost)
    neg=neg[cols]
    base=np.r_[a[singles,0],(a[pos,0]+a[neg,0])/2,
               (a[pos,1]-a[neg,1])/2,a[singles,2],(a[pos,2]+a[neg,2])/2,w]
    now=time.monotonic()
    base=optimize(base,True,min(deadline-.012,now+.88*(deadline-now)))
    for attempt in range(4):
        now=time.monotonic()
        if now>=deadline-.045:
            break
        trial=base.copy()
        trial[:21]+=rng.normal(0,params['restart_size'],21)
        trial[12:21]=np.abs(trial[12:21])
        base=optimize(trial,True,deadline-.012)
    return best
