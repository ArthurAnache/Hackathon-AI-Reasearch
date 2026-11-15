import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {'slice': (0.24, 0.08, 0.5), 'release': (0.012, 0.002, 0.05), 'contacts': (4, 3, 6)}


class StopSearch(Exception):
    pass


class StopSlice(Exception):
    pass


def canonical(obj):
    if 'circles' in obj:
        return obj
    out = []
    for x, y, r, m in obj['orbits']:
        out.append([x, y, r])
        if m == 2:
            out.append([x, -y, r])
    return {'circles': out}


def score(obj):
    try:
        a = np.asarray(canonical(obj)['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
        w = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        h = max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        if w+h > 2:
            return -float('inf')
        i,j = np.triu_indices(21, 1)
        dx = a[i,0]-a[j,0]
        dy = a[i,1]-a[j,1]
        if np.any(np.sqrt(dx*dx+dy*dy) < a[i,2]+a[j,2]):
            return -float('inf')
        return float(np.sum(a[:,2]))
    except (KeyError, ValueError, TypeError, OverflowError):
        return -float('inf')


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.69*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = score(init) if init is not None else -float('inf')
    anchor = None

    def tick():
        if time.monotonic() >= deadline:
            raise StopSearch()

    fallback = np.array([[float(i%7)*0.2, float(i//7)*0.2, 0.09999999999999] for i in range(21)])
    obj = {'circles': fallback.tolist()}
    val = score(obj)
    if val > best_score:
        best, best_score = obj, val

    try:
        tick()
        groups = np.repeat(np.arange(12), [1]*3+[2]*9)
        signs = np.array([0.]*3+[v for _ in range(9) for v in (1.,-1.)])
        X = np.zeros((21,34)); Y = X.copy(); R = X.copy()
        for k,g in enumerate(groups):
            tick()
            X[k,g] = 1.
            R[k,21+g] = 1.
            if g >= 3:
                Y[k,9+g] = signs[k]
        pairs = []
        seen = set()
        for i in range(21):
            tick()
            for j in range(i):
                tick()
                key = (min(groups[i],groups[j]),max(groups[i],groups[j]),int(signs[i]*signs[j]))
                if key not in seen:
                    seen.add(key)
                    pairs.append((i,j))
        pi,pj = np.array(pairs).T
        DX = X[pi]-X[pj]; DY = Y[pi]-Y[pj]; RS = R[pi]+R[pj]
        walls = np.zeros((36,34)); offsets = np.zeros(36)
        for g in range(12):
            tick()
            walls[g,g] = 1.; walls[g,21+g] = -1.; walls[g,33] = .5
            walls[12+g,g] = -1.; walls[12+g,21+g] = -1.; walls[12+g,33] = .5
            walls[24+g,21+g] = -1.; walls[24+g,33] = -.5; offsets[24+g] = 1.
            if g >= 3:
                walls[24+g,9+g] = -1.
        weights = np.r_[np.ones(3),np.full(9,2.)]
        grad = np.zeros(34); grad[21:33] = -weights
        bounds = [(-1.,1.)]*12+[(0.,1.)]*9+[(1e-6,.5)]*12+[(.12,1.88)]
        ii,jj = np.triu_indices(21,1)
        release = np.zeros(len(pi))

        def retain(v):
            nonlocal best, best_score, anchor
            tick()
            if not np.isfinite(v).all() or np.min(v[21:33]) <= 0:
                return
            a = np.column_stack((X@v,Y@v,R@v))
            dx = a[ii,0]-a[jj,0]; dy = a[ii,1]-a[jj,1]
            ratio = float(np.min(np.sqrt(dx*dx+dy*dy)/(a[ii,2]+a[jj,2])))
            a[:,2] *= max(0.,min(1.,ratio))*(1.-2e-14)
            if np.min(a[:,2]) <= 0:
                return
            lo = np.min(a[:,:2]-a[:,2,None],axis=0)
            hi = np.max(a[:,:2]+a[:,2,None],axis=0)
            scale = 2./float(np.sum(hi-lo))*(1.-2e-14)
            a *= scale
            if float(np.sum(a[:,2])) <= best_score:
                return
            candidate = {'circles':a.tolist()}
            value = score(candidate)
            if value > best_score:
                best,best_score = candidate,value
                anchor = v.copy()
                anchor[:33] *= scale
                anchor[21:33] *= max(0.,min(1.,ratio))*(1.-2e-14)
                anchor[33] = float(hi[0]-lo[0])*scale

        def fun(v):
            tick()
            return float(grad@v)

        def jac(v):
            tick()
            return grad

        def cons(v):
            tick()
            dx=DX@v; dy=DY@v
            return np.r_[np.sqrt(dx*dx+dy*dy)-RS@v-release,walls@v+offsets]

        def cjac(v):
            tick()
            dx=DX@v; dy=DY@v
            d=np.maximum(np.sqrt(dx*dx+dy*dy),1e-25)
            return np.vstack(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RS,walls))

        def solve(v, seconds, iterations=100):
            end = min(deadline,time.monotonic()+seconds)
            last = [v.copy()]
            count = [0]
            def cb(q):
                tick()
                last[0] = q.copy()
                count[0] += 1
                if count[0] % 3 == 0:
                    retain(q)
                if time.monotonic() >= end:
                    raise StopSlice()
            try:
                res=minimize(fun,v,jac=jac,method='SLSQP',bounds=bounds,constraints={'type':'ineq','fun':cons,'jac':cjac},callback=cb,options={'maxiter':iterations,'ftol':2e-13,'disp':False})
                last[0]=res.x
            except StopSlice:
                pass
            retain(last[0])
            return last[0]

        seeds=[]
        if init is not None and score(init) > 2.15:
            a=np.asarray(canonical(init)['circles'],float).copy()
            lo=np.min(a[:,:2]-a[:,2,None],axis=0); hi=np.max(a[:,:2]+a[:,2,None],axis=0)
            a[:,:2]-=(lo+hi)/2
            projections=[]
            for swap in (False,True):
                tick()
                b=a.copy()
                if swap:
                    b[:,[0,1]]=b[:,[1,0]]
                order=np.argsort(b[:,1]); low=order[:9]; axis=order[9:12]; up=order[12:]
                refl=b[low].copy(); refl[:,1]*=-1
                cost=np.sum((b[up,None,:]-refl[None,:,:])**2,axis=2)
                u,v=linear_sum_assignment(cost)
                g=np.vstack((np.column_stack((b[axis,0],np.zeros(3),b[axis,2])),(b[up[u]]+refl[v])/2))
                projections.append((float(cost[u,v].sum()+np.sum(b[axis,1]**2)),g))
            g=min(projections,key=lambda q:q[0])[1]
            z=np.zeros(34); z[:12]=g[:,0]; z[12:21]=np.maximum(0,g[3:,1]); z[21:33]=g[:,2]
            z[33]=np.ptp(g[:,0])+2*np.max(g[:,2]); seeds.append(z)
        for counts in ((4,5),(5,4),(3,3,3)):
            tick()
            g=[[x,0.,1.] for x in (-2.,0.,2.)]
            for row,n in enumerate(counts):
                tick()
                yy=(row+1)*np.sqrt(3.)
                for x in 2*np.arange(n)-(n-1):
                    tick()
                    g.append([float(x),yy,1.])
            g=np.array(g)
            w=float(np.ptp(g[:,0])+2); h=float(2*np.max(g[:,1]+1))
            g*=2/(w+h)
            z=np.zeros(34); z[:12]=g[:,0]; z[12:21]=g[3:,1]; z[21:33]=g[:,2]; z[33]=2*w/(w+h)
            seeds.append(z)
        for z in seeds:
            tick()
            release[:]=0
            solve(z,float(params['slice']))
        attempt=0
        while time.monotonic() < deadline:
            tick()
            if anchor is None:
                break
            z=anchor.copy(); release[:]=0
            if attempt % 3 != 0:
                dx=DX@z; dy=DY@z
                gaps=np.sqrt(dx*dx+dy*dy)-RS@z
                near=np.flatnonzero(gaps < .004)
                if len(near):
                    k=min(int(params['contacts']),len(near))
                    ids=rng.choice(near,k,replace=False)
                    release[ids]=float(params['release'])*rng.uniform(.7,1.3,k)
                    z=solve(z,.075,18)
                    release[:]=0
            solve(z,float(params['slice']))
            attempt+=1
    except (StopSearch,ValueError,FloatingPointError):
        pass
    return best
