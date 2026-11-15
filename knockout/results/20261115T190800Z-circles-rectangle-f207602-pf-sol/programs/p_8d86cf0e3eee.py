import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'power': (3.6, 1.2, 6.0),
    'angle': (0.37, 0.0, 1.57),
    'excursion_fraction': (0.36, 0.15, 0.55),
    'iterations': (100, 20, 200),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class StopSearch(Exception):
    pass

def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return -np.inf
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    perimeter = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
    perimeter += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
    if perimeter > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf
    if init is not None:
        a = np.asarray(init['circles'], dtype=float).copy()
        best_score = exact_score(a)
    else:
        a = np.zeros((N,3))
        for k in range(N):
            if time.monotonic() >= deadline:
                break
            a[k] = ((k%5+0.5)/5, (k//5+0.5)/5, 0.075)
        best = {'circles': a.tolist()}
        best_score = exact_score(a)
    lo = np.min(a[:,:2]-a[:,2,None],axis=0)
    hi = np.max(a[:,:2]+a[:,2,None],axis=0)
    scale = 2.0/np.sum(hi-lo)
    p = (a[:,:2]-lo)*scale
    r = a[:,2]*scale
    z = np.r_[p[:,0],p[:,1],r,(hi[0]-lo[0])*scale]
    obj = np.r_[np.zeros(2*N),-np.ones(N),0.0]
    ids = np.arange(N)
    angle = float(params['angle'])
    c,s = np.cos(angle),np.sin(angle)
    rot = np.array([[c,-s],[s,c]])

    def consider(v):
        nonlocal best, best_score
        if time.monotonic() > deadline:
            return
        b = np.column_stack((v[:N],v[N:2*N],v[2*N:3*N])).copy()
        if not np.isfinite(b).all() or np.any(b[:,2] <= 0):
            return
        dist = np.sqrt((b[I,0]-b[J,0])**2+(b[I,1]-b[J,1])**2)
        ratio = np.min(dist/(b[I,2]+b[J,2]))
        if ratio < 1:
            b[:,2] *= max(0.0,ratio)*(1-2e-14)
        low = np.min(b[:,:2]-b[:,2,None],axis=0)
        high = np.max(b[:,:2]+b[:,2,None],axis=0)
        factor = min(1.0,2.0/np.sum(high-low))*(1-2e-14)
        b[:,:2] = (b[:,:2]-low)*factor
        b[:,2] *= factor
        val = exact_score(b)
        if val > best_score:
            best_score = val
            best = {'circles': b.tolist()}

    def solve(v, power, end, maxiter, track):
        last = [v.copy()]
        def guard():
            if time.monotonic() >= end:
                raise StopSearch()
        def fun(q):
            guard()
            return float(obj@q)
        def constraints(q):
            guard()
            x,y,r,w = q[:N],q[N:2*N],q[2*N:3*N],q[-1]
            delta = np.column_stack((x[I]-x[J],y[I]-y[J]))
            if power == 2.0:
                d = np.sqrt(np.sum(delta*delta,axis=1)+1e-30)
                g = delta/d[:,None]
            else:
                t = delta@rot
                ab = np.maximum(np.abs(t),1e-30)
                d = np.sum(ab**power,axis=1)**(1.0/power)
                gt = np.sign(t)*ab**(power-1)*d[:,None]**(1-power)
                g = gt@rot.T
            values = np.r_[d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r]
            jac = np.zeros((M+4*N,3*N+1))
            rows = np.arange(M)
            jac[rows,I] = g[:,0]
            jac[rows,J] = -g[:,0]
            jac[rows,N+I] = g[:,1]
            jac[rows,N+J] = -g[:,1]
            jac[rows,2*N+I] = -1
            jac[rows,2*N+J] = -1
            for block, col, sg in ((0,0,1),(1,0,-1),(2,N,1),(3,N,-1)):
                if time.monotonic() >= end:
                    raise StopSearch()
                rr = M+block*N+ids
                jac[rr,col+ids] = sg
                jac[rr,2*N+ids] = -1
            jac[M+N:M+2*N,-1] = 1
            jac[M+3*N:M+4*N,-1] = -1
            return values,jac
        cache = [None,None]
        def calc(q):
            guard()
            if cache[0] is None or not np.array_equal(q,cache[0]):
                cache[0] = q.copy()
                cache[1] = constraints(q)
            return cache[1]
        def cb(q):
            guard()
            last[0] = q.copy()
            if track:
                consider(q)
        try:
            result = minimize(fun,v,jac=lambda q:obj,method='SLSQP',
                bounds=[(None,None)]*(2*N)+[(1e-6,0.5)]*N+[(0.15,1.85)],
                constraints={'type':'ineq','fun':lambda q:calc(q)[0], 'jac':lambda q:calc(q)[1]},
                callback=cb,options={'maxiter':maxiter,'ftol':2e-12,'disp':False})
            last[0] = result.x
        except StopSearch:
            pass
        return last[0]

    duration = deadline-start
    excursion_end = start+float(params['excursion_fraction'])*duration
    power = float(params['power'])
    if time.monotonic() < excursion_end:
        z = solve(z,power,excursion_end,int(params['iterations']),False)
    bridge_end = start+0.55*duration
    if time.monotonic() < bridge_end:
        z = solve(z,0.5*(power+2),bridge_end,35,False)
    if time.monotonic() < deadline:
        z = solve(z,2.0,deadline,int(params['iterations']),True)
    if time.monotonic() < deadline:
        consider(z)
    return best
