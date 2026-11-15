import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linprog

PARAMS = {
    'aspect': (1.0, 0.5, 2.0, 'log'),
    'rows': (5, 3, 7),
    'jitter': (0.03, 0.0, 0.15),
    'init_jitter': (0.0, 0.0, 0.05),
    'mu0': (10.0, 1.0, 1000.0, 'log'),
    'mu_growth': (8.0, 2.0, 30.0),
    'stages': (6, 2, 12),
    'maxiter': (300, 30, 1000),
}
N = 21
I, J = np.triu_indices(N, 1)

class Deadline(Exception):
    pass

def perimeter(p, r):
    x = list(map(lambda v: Fraction(float(v)), p[:, 0]))
    y = list(map(lambda v: Fraction(float(v)), p[:, 1]))
    q = list(map(lambda v: Fraction(float(v)), r))
    return (max(a+b for a,b in zip(x,q))-min(a-b for a,b in zip(x,q))
            +max(a+b for a,b in zip(y,q))-min(a-b for a,b in zip(y,q)))

def valid(p, r):
    if p.shape != (21,2) or r.shape != (21,):
        return False
    if not np.isfinite(p).all() or not np.isfinite(r).all() or not (r>0).all():
        return False
    d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
    return not (d < r[I]+r[J]).any() and perimeter(p,r) <= 2

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    optimize_end = start + 0.60*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -float('inf')
    if init is not None:
        a = np.asarray(init['circles'], dtype=float)
        p0, r0 = a[:,:2].copy(), a[:,2].copy()
        if valid(p0,r0):
            best_score = float(np.sum(r0))
        lo = (p0-np.abs(r0)[:,None]).min(0)
        side = (p0+np.abs(r0)[:,None]).max(0)-lo
        scale = 2.0/side.sum()
        p = (p0-lo)*scale
        r = np.abs(r0)*scale
        w = side[0]*scale
        p += rng.normal(0,float(params['init_jitter']),p.shape)
        candidates = [p.copy()]
    else:
        w = 2*float(params['aspect'])/(1+float(params['aspect']))
        rows = int(params['rows'])
        cols = int(np.ceil(N/rows))
        points = []
        for k in range(rows):
            if time.monotonic() >= deadline:
                break
            count = min(cols,N-len(points))
            off = 0.25/cols if k%2 else 0
            points.extend([((j+0.5)/cols+off,(k+0.5)/rows) for j in range(count)])
        p = np.asarray(points[:N],dtype=float)
        if len(p) != N:
            p = np.column_stack((np.linspace(.05,.95,N),np.full(N,.5)))
        p = np.clip(p+rng.normal(0,float(params['jitter']),p.shape),.02,.98)*[w,2-w]
        r = np.full(N,.05)
        candidates = []
        fallback_p = np.column_stack((np.linspace(.025,1.975,N),np.full(N,.01)))
        best = {'circles': np.column_stack((fallback_p,np.full(N,.001))).tolist()}
        best_score = .021
    z = np.r_[p[:,0],p[:,1],r,w]
    last = z.copy()

    def penalty(z, mu):
        nonlocal last
        if time.monotonic() >= optimize_end:
            raise Deadline()
        last = z.copy()
        x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
        dx,dy = x[I]-x[J],y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-18)
        v = np.maximum(r[I]+r[J]-d,0)
        b = np.maximum(np.array([r-x,r+x-w,r-y,r+y-(2-w)]),0)
        neg = np.maximum(-r,0)
        f = -r.sum()+mu*((v*v).sum()+(b*b).sum()+(neg*neg).sum())
        gx,gy,gr = np.zeros(N),np.zeros(N),-np.ones(N)
        c = 2*mu*v
        np.add.at(gr,I,c)
        np.add.at(gr,J,c)
        np.add.at(gx,I,-c*dx/d)
        np.add.at(gx,J,c*dx/d)
        np.add.at(gy,I,-c*dy/d)
        np.add.at(gy,J,c*dy/d)
        gr += 2*mu*b.sum(axis=0)-2*mu*neg
        gx += 2*mu*(-b[0]+b[1])
        gy += 2*mu*(-b[2]+b[3])
        gw = 2*mu*(-b[1].sum()+b[3].sum())
        return f,np.r_[gx,gy,gr,gw]

    mu = float(params['mu0'])
    for stage in range(int(params['stages'])):
        if time.monotonic() >= optimize_end:
            break
        try:
            z = minimize(penalty,z,args=(mu,),jac=True,method='L-BFGS-B',
                         options={'maxiter':int(params['maxiter'])}).x
        except Deadline:
            z = last
            break
        mu *= float(params['mu_growth'])
    candidates.append(np.column_stack((z[:N],z[N:2*N])))
    for p in candidates:
        if time.monotonic() >= deadline:
            break
        m = len(I)
        A = np.zeros((m+4*N+1,N+4))
        b = np.zeros(m+4*N+1)
        A[np.arange(m),I] = 1
        A[np.arange(m),J] = 1
        distances = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
        b[:m] = distances
        for k,(col,sign,coord) in enumerate(((N,1,0),(N+1,-1,0),(N+2,1,1),(N+3,-1,1))):
            if time.monotonic() >= deadline:
                break
            rows = m+k*N+np.arange(N)
            A[rows,np.arange(N)] = 1
            A[rows,col] = sign
            b[rows] = sign*p[:,coord]
        if time.monotonic() >= deadline:
            break
        A[-1,N:] = [-1,1,-1,1]
        b[-1] = 2
        res = linprog(-np.r_[np.ones(N),np.zeros(4)],A_ub=A,b_ub=b,
                      bounds=[(0,None)]*N+[(None,None)]*4,method='highs',
                      options={'time_limit':max(.001,deadline-time.monotonic())})
        if not res.success:
            continue
        r = np.maximum(res.x[:N],1e-12)*(1-1e-12)
        for attempt in range(5):
            if time.monotonic() >= deadline:
                break
            sums = r[I]+r[J]
            if not (distances<sums).any():
                break
            for i,j,d in zip(I,J,distances):
                if time.monotonic() >= deadline:
                    break
                if r[i]+r[j] > d*(1-1e-12):
                    factor = d*(1-1e-12)/(r[i]+r[j])
                    r[i] *= factor
                    r[j] *= factor
        for attempt in range(60):
            if time.monotonic() >= deadline:
                break
            excess = perimeter(p,r)-2
            if excess <= 0:
                break
            r *= 1-max(2*float(excess)/float(r.sum()),1e-12)
        score = float(np.sum(r))
        if score > best_score and valid(p,r):
            best_score = score
            best = {'circles':np.column_stack((p,r)).tolist()}
    return best
