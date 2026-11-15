import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment
from scipy.linalg import lstsq

PARAMS = {
    'kick': (0.028, 0.003, 0.12, 'log'),
    'slice': (0.155, 0.06, 0.35),
    'release': (4, 3, 8),
    'safety': (8e-15, 2e-15, 1e-11, 'log')
}
I, J = np.triu_indices(21, 1)
RR = np.arange(210)

class Stop(Exception):
    pass

def valid(a):
    if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
        return False
    d = np.sqrt((a[I,0]-a[J,0])**2 + (a[I,1]-a[J,1])**2)
    if np.any(d < a[I,2]+a[J,2]):
        return False
    x = [Fraction(float(v)) for v in a[:,0]]
    y = [Fraction(float(v)) for v in a[:,1]]
    r = [Fraction(float(v)) for v in a[:,2]]
    return (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r)) +
            max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))) <= 2

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    eps = float(params['safety'])
    best = init
    bestscore = -float('inf')
    a = None
    if init is not None:
        a = np.asarray(init['circles'],float).copy() if 'circles' in init else np.asarray(init['state'],float).reshape(3,21).T.copy()
        if valid(a):
            bestscore = float(np.sum(a[:,2]))
        else:
            best = None

    def offer(b):
        nonlocal best, bestscore
        if time.monotonic() >= end or not np.isfinite(b).all() or np.any(b[:,2] <= 0):
            return
        b = b.copy()
        d = np.sqrt((b[I,0]-b[J,0])**2+(b[I,1]-b[J,1])**2)
        f = min(1.,float(np.min(d/(b[I,2]+b[J,2]))))
        if f <= 0:
            return
        b[:,2] *= f*(1-eps)
        lo = np.min(b[:,:2]-b[:,2,None],axis=0)
        hi = np.max(b[:,:2]+b[:,2,None],axis=0)
        scale = (2-8*eps)/np.sum(hi-lo)
        b[:,:2] = (b[:,:2]-(hi+lo)/2)*scale
        b[:,2] *= scale
        score = float(np.sum(b[:,2]))
        if score > bestscore and valid(b):
            bestscore = score
            best = {'state': b.T.reshape(-1).tolist()}

    if a is None:
        a = np.array([[2*k+(row%2),np.sqrt(3)*row,1.] for row,n in enumerate((4,3,4,3,4,3)) for k in range(n)],float)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        a[:,:2] -= (hi+lo)/2
        a *= 2/np.sum(hi-lo)
        offer(a)
    if time.monotonic() >= end-.05:
        return best

    choices = []
    for swap in (False,True):
        if time.monotonic() >= end-.04:
            return best
        b = a.copy()
        if swap:
            b[:,:2] = b[:,[1,0]]
        lo = np.min(b[:,:2]-b[:,2,None],axis=0)
        hi = np.max(b[:,:2]+b[:,2,None],axis=0)
        b[:,:2] -= (hi+lo)/2
        ref = b.copy()
        ref[:,1] *= -1
        cost = np.sum((b[:,None,:]-ref[None,:,:])**2,axis=2)
        rr,cc = linear_sum_assignment(cost)
        choices.append((float(cost[rr,cc].sum()),b,hi[0]-lo[0]))
    _,a,width = min(choices,key=lambda v:v[0])
    singles = np.argsort(np.abs(a[:,1]))[:3]
    rem = np.setdiff1d(np.arange(21),singles)
    order = rem[np.argsort(a[rem,1])]
    neg,pos = order[:9],order[9:]
    cost = ((a[pos,None,0]-a[None,neg,0])**2 + (a[pos,None,1]+a[None,neg,1])**2 + (a[pos,None,2]-a[None,neg,2])**2)
    _,cc = linear_sum_assignment(cost)
    neg = neg[cc]
    q0 = np.r_[a[singles,0],.5*(a[pos,0]+a[neg,0]),.5*(a[pos,1]-a[neg,1]),a[singles,2],.5*(a[pos,2]+a[neg,2]),width]
    T = np.zeros((64,34))
    ids = np.r_[np.arange(12),np.arange(3,12)]
    signs = np.r_[np.ones(12),-np.ones(9)]
    T[np.arange(21),ids] = 1
    mask = ids >= 3
    T[21+np.arange(21)[mask],9+ids[mask]] = signs[mask]
    T[42+np.arange(21),21+ids] = 1
    T[63,33] = 1
    wall = np.zeros((84,64))
    k = np.arange(21)
    wall[k,k] = 1
    wall[21+k,k] = -1
    wall[42+k,21+k] = 1
    wall[63+k,21+k] = -1
    wall[:,42:63] = np.tile(-np.eye(21),(4,1))
    wall[:42,63] = .5
    wall[42:,63] = -.5
    W = wall@T
    offsets = np.r_[np.zeros(42),np.ones(42)]
    obj = np.r_[np.zeros(42),-np.ones(21),0.]@T
    bounds = [(None,None)]*21+[(1e-6,.6)]*12+[(.15,1.85)]
    weights = 1/np.maximum(2*((T@q0)[42+I]+(T@q0)[42+J]),.08)

    def geometry(q):
        z = T@q
        dx,dy = z[I]-z[J],z[21+I]-z[21+J]
        rs = z[42+I]+z[42+J]
        c = np.r_[(dx*dx+dy*dy-rs*rs)*weights,W@q+offsets]
        jac = np.zeros((210,64))
        vx,vy,vr = 2*weights*dx,2*weights*dy,-2*weights*rs
        jac[RR,I],jac[RR,J] = vx,-vx
        jac[RR,21+I],jac[RR,21+J] = vy,-vy
        jac[RR,42+I],jac[RR,42+J] = vr,vr
        return c,np.vstack((jac@T,W))

    def submit(q):
        offer((T@q)[:63].reshape(3,21).T)

    def solve(q, stop):
        cache = [None,None,None]
        last = [q.copy()]
        def calc(v):
            if time.monotonic() >= stop:
                raise Stop()
            if cache[0] is None or not np.array_equal(v,cache[0]):
                c,j = geometry(v)
                cache[:] = [v.copy(),c,j]
            return cache[1],cache[2]
        def fun(v):
            if time.monotonic() >= stop:
                raise Stop()
            return float(obj@v)
        def callback(v):
            last[0] = v.copy()
            if time.monotonic() >= stop:
                raise Stop()
        try:
            res = minimize(fun,q,jac=lambda v:obj,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda v:calc(v)[0],'jac':lambda v:calc(v)[1]},
                callback=callback,options={'ftol':2e-13,'maxiter':100,'disp':False})
            q = res.x
        except Stop:
            q = last[0]
        submit(q)
        for _ in range(3):
            if time.monotonic() >= stop-.004:
                break
            c,j = geometry(q)
            active = np.flatnonzero(np.abs(c)<1e-7)
            if len(active)<34:
                break
            delta,_,rank,_ = lstsq(j[active],-c[active],lapack_driver='gelsy',check_finite=False)
            if rank < 34 or np.max(np.abs(delta)) > .001:
                break
            q = q+delta
            submit(q)
        return q

    anchor = solve(q0,min(end-.02,time.monotonic()+.22))
    c,j = geometry(anchor)
    active = np.flatnonzero(np.abs(c)<2e-6)
    directions = None
    probabilities = None
    if len(active)>=34 and time.monotonic()<end-.035:
        A = j[active]
        multipliers = lstsq(A.T,obj,lapack_driver='gelsy',check_finite=False)[0]
        strength = np.abs(multipliers)
        probabilities = 1/np.maximum(strength,.015)**.65
        probabilities /= probabilities.sum()
        directions = lstsq(A,np.eye(len(active)),lapack_driver='gelsy',check_finite=False)[0]
    trial = 0
    while time.monotonic() < end-.045:
        trial += 1
        q = anchor.copy()
        amplitude = float(params['kick'])*(0.55,1.,1.8,2.8)[(trial-1)%4]
        if directions is not None and trial%4 != 0:
            count = min(int(params['release']),len(active))
            selected = rng.choice(len(active),count,replace=False,p=probabilities)
            coeff = rng.uniform(.4,1.3,count)
            if trial%3 == 0:
                coeff *= rng.choice([-1.,1.],count)
            d = directions[:,selected]@coeff
            norm = max(np.max(np.abs(d[:21])),2*np.max(np.abs(d[21:33])),abs(d[33]),1e-12)
            q += d*(amplitude/norm)
        else:
            q[:21] += rng.normal(0,amplitude,21)
            q[21:33] *= np.exp(rng.normal(0,amplitude*2,12))
            q[33] += rng.normal(0,amplitude)
        q[21:33] = np.clip(q[21:33],1e-5,.5)
        q[33] = np.clip(q[33],.2,1.8)
        stop = min(end-.012,time.monotonic()+float(params['slice']))
        candidate = solve(q,stop)
        if time.monotonic() >= end-.025:
            break
        cv,_ = geometry(candidate)
        if np.min(cv)>-1e-8 and obj@candidate < obj@anchor-1e-10:
            anchor = candidate
    return best
