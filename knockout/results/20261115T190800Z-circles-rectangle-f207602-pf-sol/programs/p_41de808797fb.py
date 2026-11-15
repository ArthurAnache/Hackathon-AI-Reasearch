import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment
from scipy.linalg import lstsq

PARAMS = {
    'recovery_fraction': (0.43, 0.2, 0.7),
    'quotient_fraction': (0.84, 0.65, 0.94),
    'iterations': (350, 100, 600),
    'safety': (1e-14, 3e-15, 1e-11, 'log')
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
    lo = np.min(a[:,:2]-a[:,2,None],axis=0)
    hi = np.max(a[:,:2]+a[:,2,None],axis=0)
    scale = (2-8*eps)/np.sum(hi-lo)
    a[:,:2] = (a[:,:2]-(lo+hi)/2)*scale
    a[:,2] *= scale
    return a if valid(a) else None

def model(sym):
    n = 34 if sym else 64
    x = np.zeros((21,n)); y=x.copy(); r=x.copy()
    if sym:
        ids=np.r_[np.arange(12),np.arange(3,12)]
        signs=np.r_[np.ones(12),-np.ones(9)]
        x[np.arange(21),ids]=1
        mask=ids>=3
        y[np.arange(21)[mask],12+ids[mask]-3]=signs[mask]
        r[np.arange(21),21+ids]=1
        pairs=[(i,j) for i in range(12) for j in range(i+1,12)]
        pairs += [(i,12+j-3) for i in range(3,12) for j in range(i,12)]
        pi,pj=np.array(pairs).T
        wall=np.arange(12)
        ridx=np.arange(21,33)
    else:
        x[np.arange(21),np.arange(21)]=1
        y[np.arange(21),21+np.arange(21)]=1
        r[np.arange(21),42+np.arange(21)]=1
        pi,pj=I,J
        wall=np.arange(21)
        ridx=np.arange(42,63)
    w=np.zeros(n); w[-1]=1
    b=np.concatenate((x[wall]-r[wall],w-x[wall]-r[wall],
                      -.5*w-y[wall]-r[wall],-.5*w+y[wall]-r[wall]))
    off=np.r_[np.zeros(2*len(wall)),np.ones(2*len(wall))]
    return x,y,r,x[pi]-x[pj],y[pi]-y[pj],r[pi]+r[pj],b,off,-r.sum(axis=0),ridx

def construct(params, seed, budget_s, init=None):
    start=time.monotonic()
    deadline=start+.70*budget_s
    eps=float(params['safety'])
    best=None; score=-np.inf; original=None
    def array(obj):
        if 'circles' in obj:
            return np.asarray(obj['circles'],dtype=float).copy()
        return np.asarray(obj['state'],dtype=float).reshape(3,21).T.copy()
    try:
        if init is not None:
            original=array(init)
            if valid(original):
                best=init
                score=float(np.sum(original[:,2]))
    except (ValueError,TypeError,KeyError):
        original=None
    def offer(a):
        nonlocal best,score
        b=repair(a,eps)
        if b is not None:
            s=float(np.sum(b[:,2]))
            if s>score:
                best={'state':b.T.reshape(-1).tolist()}
                score=s
        return b
    if original is not None:
        offer(original)
    fallback=np.array([[2*j+(1 if n==4 else 0),np.sqrt(3.)*k,1.]
                       for k,n in enumerate((4,5,4,5,3)) for j in range(n)])
    offer(fallback)
    if best is None:
        return {'circles': [[(i%3)*.2,(i//3)*.2,.099999999999] for i in range(21)]}
    full=model(False); quotient=model(True)
    def unpack(z,m):
        return np.column_stack((m[0]@z,m[1]@z,m[2]@z))
    def encode(a):
        lo=np.min(a[:,:2]-a[:,2,None],axis=0)
        hi=np.max(a[:,:2]+a[:,2,None],axis=0)
        return np.r_[a[:,0]-lo[0],a[:,1]-(lo[1]+hi[1])/2,a[:,2],hi[0]-lo[0]]
    def optimize(z,m,stop):
        dxm,dym,rsm,b,off,obj,ridx=m[3:]
        weights=1/np.maximum(2*(rsm@z),.08)
        cache=[None,None,None]
        last=[z.copy()]
        count=[0]
        feasible=[None,-np.inf]
        def calc(v):
            if time.monotonic()>=stop:
                raise Timeout()
            if cache[0] is not None and np.array_equal(v,cache[0]):
                return cache[1],cache[2]
            dx=dxm@v; dy=dym@v; rs=rsm@v
            c=np.r_[(dx*dx+dy*dy-rs*rs)*weights,b@v+off]
            jac=np.concatenate((2*weights[:,None]*(dx[:,None]*dxm+dy[:,None]*dym-rs[:,None]*rsm),b))
            cache[:]=[v.copy(),c,jac]
            return c,jac
        def fun(v):
            if time.monotonic()>=stop:
                raise Timeout()
            return float(obj@v),obj
        def callback(v):
            if time.monotonic()>=stop:
                raise Timeout()
            last[0]=v.copy()
            count[0]+=1
            if count[0]%15==0:
                offer(unpack(v,m))
        bounds=[(None,None)]*len(z)
        for k in ridx:
            if time.monotonic()>=stop:
                return z
            bounds[int(k)]=(1e-7,.6)
        bounds[-1]=(.12,1.88)
        try:
            res=minimize(fun,z,jac=True,method='SLSQP',bounds=bounds,
                         constraints={'type':'ineq','fun':lambda v:calc(v)[0],
                                      'jac':lambda v:calc(v)[1]},callback=callback,
                         options={'maxiter':int(params['iterations']),'ftol':8e-14,'disp':False})
            z=res.x
        except Timeout:
            z=last[0]
        offer(unpack(z,m))
        for k in range(3):
            if time.monotonic()>=stop-.004:
                break
            try:
                c,jac=calc(z)
            except Timeout:
                break
            active=np.flatnonzero(np.abs(c)<2e-7)
            if len(active)<len(z):
                break
            delta,_,rank,_=lstsq(jac[active],-c[active],lapack_driver='gelsy',check_finite=False)
            if rank<len(z) or np.max(np.abs(delta))>2e-4:
                break
            z=z+delta
            offer(unpack(z,m))
        return z
    a=original
    if a is None or a.shape!=(21,3) or not np.isfinite(a).all() or np.any(a[:,2]<=0):
        a=array(best)
    total=deadline-start
    z=optimize(encode(a),full,start+total*float(params['recovery_fraction']))
    if time.monotonic()>=deadline-.025:
        return best
    # Infer the quotient only after the unrestricted geometry has recovered.
    a=array(best)
    choices=[]
    for swap in (False,True):
        if time.monotonic()>=deadline-.02:
            return best
        b=a.copy()
        if swap:
            b[:,:2]=b[:,[1,0]]
        lo=np.min(b[:,:2]-b[:,2,None],axis=0)
        hi=np.max(b[:,:2]+b[:,2,None],axis=0)
        b[:,0]-=lo[0]
        b[:,1]-=(lo[1]+hi[1])/2
        reflected=b.copy(); reflected[:,1]*=-1
        cost=np.sum((b[:,None,:]-reflected[None,:,:])**2,axis=2)
        rows,cols=linear_sum_assignment(cost)
        choices.append((float(cost[rows,cols].sum()),b,hi[0]-lo[0]))
    _,a,width=min(choices,key=lambda t:t[0])
    singles=np.argsort(np.abs(a[:,1]))[:3]
    remaining=np.setdiff1d(np.arange(21),singles)
    order=remaining[np.argsort(a[remaining,1])]
    neg,pos=order[:9],order[9:]
    cost=((a[pos,None,0]-a[None,neg,0])**2+
          (a[pos,None,1]+a[None,neg,1])**2+
          (a[pos,None,2]-a[None,neg,2])**2)
    _,cols=linear_sum_assignment(cost)
    neg=neg[cols]
    q=np.r_[a[singles,0],(a[pos,0]+a[neg,0])/2,
            (a[pos,1]-a[neg,1])/2,a[singles,2],
            (a[pos,2]+a[neg,2])/2,width]
    optimize(q,quotient,start+total*float(params['quotient_fraction']))
    if time.monotonic()<deadline-.025:
        optimize(encode(array(best)),full,deadline-.008)
    return best
