import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'population': (7, 3, 12),
    'perturbation': (0.065, 0.005, 0.2),
    'iterations': (65, 20, 150),
    'memory': (6, 3, 15),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class StopSearch(Exception):
    pass

def exact_score(c):
    a = np.asarray(c['circles'], dtype=float)
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    p, r = a[:, :2], a[:, 2]
    d = np.sqrt((p[I, 0]-p[J, 0])**2 + (p[I, 1]-p[J, 1])**2)
    if np.any(d < r[I]+r[J]):
        return -np.inf
    x, y, q = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
    perimeter = max(u+v for u,v in zip(x,q))-min(u-v for u,v in zip(x,q))
    perimeter += max(u+v for u,v in zip(y,q))-min(u-v for u,v in zip(y,q))
    return float(np.sum(r)) if perimeter <= 2 else -np.inf

def repaired(z):
    p = np.stack((z[:N], z[N:2*N]), axis=1).copy()
    r = np.maximum(z[2*N:3*N], 1e-10).copy()
    d = np.sqrt((p[I,0]-p[J,0])**2 + (p[I,1]-p[J,1])**2)
    r *= min(1.0, float(np.min(d/(r[I]+r[J]))))*(1-2e-14)
    lo = np.min(p-r[:,None],axis=0)
    hi = np.max(p+r[:,None],axis=0)
    scale = 2.0 / np.sum(hi-lo) * (1-2e-14)
    p = (p-lo)*scale
    r *= scale
    return {'circles': np.column_stack((p,r)).tolist()}

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    batch_end = start + 0.40*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = exact_score(init) if init is not None else -np.inf
    if init is not None:
        a = np.array(init['circles'],dtype=float)
        p, r = a[:,:2].copy(), a[:,2].copy()
        lo = np.min(p-r[:,None],axis=0)
        sides = np.max(p+r[:,None],axis=0)-lo
        scale = 2/sides.sum()
        p = (p-lo)*scale
        r *= scale
        w = sides[0]*scale
    else:
        p = np.array([((i%5+.5)/5,(i//5+.5)/5) for i in range(N)])
        r = np.full(N, .085)
        w = 1.0
    base = np.r_[p[:,0],p[:,1],r,w]
    count = int(params['population'])
    states = np.tile(base,(count,1))
    for k in range(count):
        if time.monotonic() >= batch_end:
            break
        amp = float(params['perturbation'])*(0.45+1.4*k/max(1,count-1))
        # Spatially correlated translations move entire contact clusters together.
        pivot = p[rng.integers(N)]
        direction = rng.normal(size=2)
        direction /= np.linalg.norm(direction)
        projection = (p-pivot)@direction
        displacement = np.tanh(projection/0.14)[:,None]*rng.normal(size=(1,2))*amp
        displacement += rng.normal(size=(N,2))*amp*0.24
        states[k,:N] += displacement[:,0]
        states[k,N:2*N] += displacement[:,1]
        states[k,2*N:3*N] *= np.exp(rng.normal(0,.10,N))
        states[k,-1] += rng.normal(0,amp*.6)
    latest = states.copy()

    def penalty(flat, mu):
        nonlocal latest
        if time.monotonic() >= batch_end:
            raise StopSearch()
        z = flat.reshape(count,64)
        latest = z.copy()
        x,y,r,w = z[:,:N],z[:,N:2*N],z[:,2*N:3*N],z[:,-1,None]
        dx,dy = x[:,I]-x[:,J], y[:,I]-y[:,J]
        d = np.sqrt(dx*dx+dy*dy+1e-24)
        v = np.maximum(r[:,I]+r[:,J]-d,0)
        bs = [np.maximum(r-x,0),np.maximum(r+x-w,0),np.maximum(r-y,0),np.maximum(r+y-(2-w),0)]
        neg = np.maximum(1e-7-r,0)
        f = -r.sum()+mu*(np.sum(v*v)+sum(np.sum(b*b) for b in bs)+np.sum(neg*neg))
        gx,gy,gr = np.zeros_like(x),np.zeros_like(y),-np.ones_like(r)
        c = 2*mu*v
        rows = np.arange(count)[:,None]
        np.add.at(gr,(rows,I),c)
        np.add.at(gr,(rows,J),c)
        np.add.at(gx,(rows,I),-c*dx/d)
        np.add.at(gx,(rows,J),c*dx/d)
        np.add.at(gy,(rows,I),-c*dy/d)
        np.add.at(gy,(rows,J),c*dy/d)
        gr += 2*mu*(sum(bs)-neg)
        gx += 2*mu*(-bs[0]+bs[1])
        gy += 2*mu*(-bs[2]+bs[3])
        gw = 2*mu*(-bs[1].sum(axis=1)+bs[3].sum(axis=1))
        return f,np.concatenate((gx,gy,gr,gw[:,None]),axis=1).ravel()

    for mu in (12.,65.,350.,2500.,25000.):
        if time.monotonic() >= batch_end:
            break
        try:
            res = minimize(penalty,latest.ravel(),args=(mu,),jac=True,method='L-BFGS-B',options={'maxiter':int(params['iterations']),'maxcor':int(params['memory']),'ftol':1e-12,'maxls':15})
            latest = res.x.reshape(count,64).copy()
        except StopSearch:
            break

    def consider(z):
        nonlocal best,best_score
        c = repaired(z)
        score = exact_score(c)
        if score > best_score:
            best,best_score = c,score
        return score

    ranked = []
    for z in latest:
        if time.monotonic() >= deadline:
            break
        ranked.append((consider(z),z.copy()))
    ranked.sort(key=lambda item:item[0],reverse=True)
    objective_gradient = np.r_[np.zeros(42),-np.ones(21),0.]
    rr = np.arange(M)
    eye = np.eye(N)

    def constraints(z):
        if time.monotonic() >= deadline:
            raise StopSearch()
        x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
        dx,dy = x[I]-x[J],y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        val = np.r_[d-r[I]-r[J],x-r,w-x-r,y-r,2-w-y-r]
        jac = np.zeros((M+84,64))
        jac[rr,I]=dx/d
        jac[rr,J]=-dx/d
        jac[rr,N+I]=dy/d
        jac[rr,N+J]=-dy/d
        jac[rr,2*N+I]=-1
        jac[rr,2*N+J]=-1
        for k in range(4):
            if time.monotonic() >= deadline:
                raise StopSearch()
            sl = slice(M+k*N,M+(k+1)*N)
            jac[sl,2*N:3*N]=-eye
            jac[sl,(0 if k<2 else N):(N if k<2 else 2*N)] = eye*(1 if k%2==0 else -1)
        jac[M+N:M+2*N,-1]=1
        jac[M+3*N:,-1]=-1
        return val,jac

    cache_z = None
    cache_result = None
    def cached(z):
        nonlocal cache_z,cache_result
        if cache_z is None or not np.array_equal(cache_z,z):
            cache_result=constraints(z)
            cache_z=z.copy()
        return cache_result

    for _,z in ranked[:3]:
        if time.monotonic() >= deadline-.025:
            break
        last = [z.copy()]
        def callback(v):
            last[0] = v.copy()
            if time.monotonic() >= deadline:
                raise StopSearch()
        try:
            result=minimize(lambda v:-v[42:63].sum(),z,jac=lambda v:objective_gradient,method='SLSQP',bounds=[(None,None)]*42+[(1e-8,None)]*21+[(.15,1.85)],constraints={'type':'ineq','fun':lambda v:cached(v)[0],'jac':lambda v:cached(v)[1]},callback=callback,options={'maxiter':85,'ftol':2e-12})
            last[0]=result.x
        except StopSearch:
            pass
        consider(last[0])
    if best is None:
        best=repaired(base)
    return best
