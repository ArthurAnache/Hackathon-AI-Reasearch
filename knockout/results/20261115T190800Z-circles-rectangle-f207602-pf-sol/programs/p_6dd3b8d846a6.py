import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'first_fraction': (0.39, 0.2, 0.6),
    'second_fraction': (0.76, 0.6, 0.9),
    'iterations': (240, 100, 400),
    'safety': (2e-14, 5e-15, 1e-10, 'log')
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
    x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
    return (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r)) +
            max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))) <= 2

def repair(a, eps):
    a = np.asarray(a, float).copy()
    if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
        return None
    d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
    factor = min(1., float(np.min(d/(a[I,2]+a[J,2]))))
    if factor <= 0:
        return None
    a[:,2] *= factor*(1-eps)
    lo = np.min(a[:,:2]-a[:,2,None], axis=0)
    hi = np.max(a[:,:2]+a[:,2,None], axis=0)
    scale = (2-12*eps)/float(np.sum(hi-lo))
    a[:,:2] = (a[:,:2]-(hi+lo)/2)*scale
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
        boundary=np.arange(12)
        ridx=np.arange(21,33)
    else:
        x[np.arange(21),np.arange(21)]=1
        y[np.arange(21),21+np.arange(21)]=1
        r[np.arange(21),42+np.arange(21)]=1
        pi,pj=I,J
        boundary=np.arange(21)
        ridx=np.arange(42,63)
    w=np.zeros(n); w[-1]=1
    b=np.concatenate((x[boundary]-r[boundary],w-x[boundary]-r[boundary],
                      -.5*w-y[boundary]-r[boundary],-.5*w+y[boundary]-r[boundary]))
    off=np.r_[np.zeros(2*len(boundary)),np.ones(2*len(boundary))]
    return x,y,r,x[pi]-x[pj],y[pi]-y[pj],r[pi]+r[pj],b,off,-r.sum(axis=0),ridx

def construct(params, seed, budget_s, init=None):
    start=time.monotonic(); deadline=start+.70*budget_s
    eps=float(params['safety'])
    best=init; bestscore=-np.inf; besta=None
    try:
        a=np.asarray(init['circles'],float) if 'circles' in init else np.asarray(init['state'],float).reshape(3,21).T.copy()
        if valid(a):
            bestscore=float(np.sum(a[:,2])); besta=a.copy()
        else:
            best=None
    except (TypeError,KeyError,ValueError):
        best=None

    def offer(a):
        nonlocal best,bestscore,besta
        if time.monotonic()>=deadline:
            return
        b=repair(a,eps)
        if b is not None:
            score=float(np.sum(b[:,2]))
            if score>bestscore:
                bestscore=score; besta=b
                best={'state':b.T.reshape(-1).tolist()}

    # A permanently feasible fallback, independent of the supplied cold start.
    fallback=np.array([[.28*(i%5),.28*(i//5),.139999999999] for i in range(21)])
    offer(fallback)
    sym=model(True); full=model(False)

    def unpack(z,m):
        return np.column_stack((m[0]@z,m[1]@z,m[2]@z))

    def fullstate(a):
        lo=np.min(a[:,:2]-a[:,2,None],axis=0)
        hi=np.max(a[:,:2]+a[:,2,None],axis=0)
        return np.r_[a[:,0]-lo[0],a[:,1]-(hi[1]+lo[1])/2,a[:,2],hi[0]-lo[0]]

    def optimize(z,m,stop):
        x,y,r,dxm,dym,rsm,b,off,obj,ridx=m
        weights=1/np.maximum(2*(rsm@z),.06)
        cached=[None,None,None]; accepted=[z.copy()]; count=[0]
        def calc(v):
            if time.monotonic()>=stop:
                raise Timeout
            if cached[0] is not None and np.array_equal(cached[0],v):
                return cached[1],cached[2]
            dx=dxm@v; dy=dym@v; rs=rsm@v
            c=np.r_[(dx*dx+dy*dy-rs*rs)*weights,b@v+off]
            jac=np.concatenate((2*weights[:,None]*(dx[:,None]*dxm+dy[:,None]*dym-rs[:,None]*rsm),b))
            cached[:]=[v.copy(),c,jac]
            return c,jac
        def fun(v):
            if time.monotonic()>=stop:
                raise Timeout
            return float(obj@v),obj
        def callback(v):
            if time.monotonic()>=stop:
                raise Timeout
            accepted[0]=v.copy(); count[0]+=1
            if count[0]%16==0:
                offer(unpack(v,m))
        bounds=[(None,None)]*len(z)
        for k in ridx:
            if time.monotonic()>=stop:
                return z
            bounds[int(k)]=(1e-6,.5)
        bounds[-1]=(.2,1.8)
        try:
            result=minimize(fun,z,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda v:calc(v)[0],'jac':lambda v:calc(v)[1]},
                callback=callback,options={'maxiter':int(params['iterations']),'ftol':3e-13,'disp':False})
            z=result.x
        except Timeout:
            z=accepted[0]
        offer(unpack(z,m))
        return z

    def lattice(kind):
        if kind==0:
            singles=np.array([1.,4.,7.])
            px=np.r_[np.arange(5)*2.,np.arange(4)*2.+1.]
            py=np.r_[np.full(5,np.sqrt(3.)),np.full(4,2*np.sqrt(3.))]
        else:
            singles=np.array([0.,4.,8.])
            px=np.r_[np.arange(4)*2.+1.,np.arange(5)*2.]
            py=np.r_[np.full(4,np.sqrt(3.)),np.full(5,2*np.sqrt(3.))]
        # A scale-compatible quotient state with a modest initial overlap is intentional.
        scale=2/(10+2+4*np.sqrt(3.))
        return np.r_[(np.r_[singles,px]+1)*scale,py*scale,np.ones(12)*scale,10*scale]

    for k,frac in enumerate((float(params['first_fraction']),float(params['second_fraction']))):
        if time.monotonic()>=deadline-.04:
            break
        q=lattice(k)
        offer(unpack(q,sym))
        optimize(q,sym,min(start+frac*.70*budget_s,deadline-.035))

    if besta is not None and time.monotonic()<deadline-.035:
        optimize(fullstate(besta),full,deadline-.018)
    if best is None:
        return {'circles':[[.2*(i%3),.2*(i//3),.099999999999] for i in range(21)]}
    return best
