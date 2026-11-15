import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'iterations': (180, 80, 350),
    'search_fraction': (0.81, 0.60, 0.90),
    'safety': (2e-14, 5e-15, 1e-10, 'log')
}

I, J = np.triu_indices(21, 1)

class Stop(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    eps = float(params['safety'])
    best = None
    bestscore = -np.inf
    besta = None

    def score(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -np.inf
        d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
        if np.any(d < a[I, 2]+a[J, 2]):
            return -np.inf
        x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
        perimeter = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        perimeter += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        return float(np.sum(a[:, 2])) if perimeter <= 2 else -np.inf

    if init is not None:
        try:
            a = np.asarray(init['circles'], float) if 'circles' in init else np.asarray(init['state'], float).reshape(3,21).T
            s = score(a)
            if s > bestscore:
                best, bestscore, besta = init, s, a.copy()
        except (TypeError, KeyError, ValueError):
            pass

    def offer(a):
        nonlocal best, bestscore, besta
        a = np.array(a, dtype=float, copy=True)
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
        factor = min(1., float(np.min(d/(a[I,2]+a[J,2]))))
        if factor <= 0:
            return
        a[:,2] *= factor*(1-eps)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        scale = (2-12*eps)/float(np.sum(hi-lo))
        a[:,:2] = (a[:,:2]-(hi+lo)/2)*scale
        a[:,2] *= scale
        s = score(a)
        if s > bestscore:
            besta, bestscore = a, s
            best = {'state':a.T.ravel().tolist()}

    offer(np.array([[.2*(k%3),.2*(k//3),.099999999] for k in range(21)]))

    def model(symmetric):
        size = 34 if symmetric else 64
        X = np.zeros((21,size)); Y = X.copy(); R = X.copy()
        if symmetric:
            for k in range(9):
                if time.monotonic() >= deadline:
                    raise Stop
                X[3+k,k] = 1.; X[12+k,k] = -1.
                Y[3+k,12+k] = Y[12+k,12+k] = 1.
                R[3+k,24+k] = R[12+k,24+k] = 1.
            for k in range(3):
                if time.monotonic() >= deadline:
                    raise Stop
                Y[k,9+k] = 1.; R[k,21+k] = 1.
            reflection = np.r_[np.arange(3),np.arange(12,21),np.arange(3,12)]
            pairs = []
            for i,j in zip(I,J):
                if time.monotonic() >= deadline:
                    raise Stop
                if (int(i),int(j)) <= tuple(sorted((int(reflection[i]),int(reflection[j])))):
                    pairs.append((i,j))
            pi,pj = np.asarray(pairs).T
            reps = np.arange(12)
            bounds = [(0.,1.)]*9+[(-1.,1.)]*12+[(1e-7,.5)]*12+[(.1,1.9)]
        else:
            X[:,:21] = np.eye(21); Y[:,21:42] = np.eye(21); R[:,42:63] = np.eye(21)
            pi,pj = I,J
            reps = np.arange(21)
            bounds = [(-1.,1.)]*42+[(1e-7,.5)]*21+[(.1,1.9)]
        W = np.zeros(size); W[-1] = 1.
        if symmetric:
            walls = np.vstack((.5*W-X[reps]-R[reps],-.5*W-Y[reps]-R[reps],-.5*W+Y[reps]-R[reps]))
            offset = np.r_[np.zeros(12),np.ones(24)]
        else:
            walls = np.vstack((.5*W-X-R,.5*W+X-R,-.5*W-Y-R,-.5*W+Y-R))
            offset = np.r_[np.zeros(42),np.ones(42)]
        return X,Y,R,X[pi]-X[pj],Y[pi]-Y[pj],R[pi]+R[pj],walls,offset,-R.sum(axis=0),bounds

    try:
        sym = model(True)
        full = model(False)
    except Stop:
        return best

    def optimize(a, m, stop, symmetric):
        X,Y,R,DX,DY,RR,walls,offset,obj,bounds = m
        a = a.copy()
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        a[:,:2] -= (hi+lo)/2
        if symmetric:
            q = np.r_[a[3:12,0],a[:3,1],a[3:12,1],a[:3,2],a[3:12,2],hi[0]-lo[0]]
        else:
            q = np.r_[a.T.ravel(),hi[0]-lo[0]]
        q[-1] = np.clip(q[-1],.1,1.9)
        last = q.copy()
        cached = [None,None,None]
        count = 0

        def unpack(v):
            return np.column_stack((X@v,Y@v,R@v))

        def calc(v):
            if time.monotonic() >= stop:
                raise Stop
            if cached[0] is not None and np.array_equal(cached[0],v):
                return cached[1],cached[2]
            dx,dy = DX@v,DY@v
            dist = np.maximum(np.sqrt(dx*dx+dy*dy),1e-18)
            c = np.r_[dist-RR@v,walls@v+offset]
            jac = np.vstack(((dx/dist)[:,None]*DX+(dy/dist)[:,None]*DY-RR,walls))
            cached[:] = [v.copy(),c,jac]
            return c,jac

        def fun(v):
            if time.monotonic() >= stop:
                raise Stop
            return float(obj@v),obj

        def callback(v):
            nonlocal last,count
            last = v.copy(); count += 1
            if time.monotonic() >= stop:
                raise Stop
            if count % 18 == 0:
                offer(unpack(v))

        try:
            if time.monotonic() >= stop:
                return
            result = minimize(fun,q,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda v:calc(v)[0],'jac':lambda v:calc(v)[1]},
                callback=callback,options={'maxiter':int(params['iterations']),'ftol':2e-13,'disp':False})
            last = result.x
        except (Stop,ValueError,FloatingPointError):
            pass
        offer(unpack(last))

    def lattice(kind):
        if kind < 2:
            if kind == 0:
                singles = [0.,4.,8.]
                py = np.r_[np.arange(4)*2.+1.,np.arange(5)*2.]
                px = np.r_[np.full(4,np.sqrt(3.)),np.full(5,2*np.sqrt(3.))]
            else:
                singles = [1.,4.,7.]
                py = np.r_[np.arange(5)*2.,np.arange(4)*2.+1.]
                px = np.r_[np.full(5,np.sqrt(3.)),np.full(4,2*np.sqrt(3.))]
            a = np.array([[0.,v,1.] for v in singles]+[[float(x),float(y),1.] for x,y in zip(px,py)]+[[-float(x),float(y),1.] for x,y in zip(px,py)])
        else:
            pattern = (5,4,3,4,5) if kind == 2 else (4,3,4,3,4,3)
            singles=[]; pairs=[]
            for row,c in enumerate(pattern):
                if time.monotonic() >= deadline:
                    raise Stop
                y = row*np.sqrt(3.)
                if c%2:
                    singles.append([0.,y,1.])
                for k in range(c//2):
                    if time.monotonic() >= deadline:
                        raise Stop
                    pairs.append([float(2*(k+1) if c%2 else 2*k+1),y,1.])
            a = np.array(singles+pairs+[[-x,y,r] for x,y,r in pairs])
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        a[:,:2] -= (hi+lo)/2
        return a*(2./float(np.sum(hi-lo)))

    search_end = start+(deadline-start)*float(params['search_fraction'])
    for k in range(4):
        now = time.monotonic()
        if now >= search_end:
            break
        try:
            a = lattice(k)
        except Stop:
            break
        offer(a)
        end = min(search_end, now+(search_end-now)/(4-k))
        optimize(a,sym,end,True)
    if besta is not None and time.monotonic() < deadline-.025:
        optimize(besta,full,deadline-.012,False)
    return best
