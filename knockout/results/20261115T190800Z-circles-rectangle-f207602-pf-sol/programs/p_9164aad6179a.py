import time
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.linalg import lstsq

PARAMS = {
    'lp_steps': (16, 0, 60),
    'polish_steps': (90, 10, 200),
    'restart_noise': (0.012, 0.0, 0.06),
    'safety': (2e-13, 1e-14, 1e-10, 'log'),
    'contact_threshold': (3e-7, 1e-9, 1e-4, 'log')
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
D = 64
ROWS = np.arange(M)
C = np.zeros(D)
C[42:63] = -1
A = np.zeros((84, D))
B = np.zeros(84)
for k in range(N):
    A[k,k] = -1
    A[k,42+k] = 1
    A[21+k,k] = 1
    A[21+k,42+k] = 1
    A[21+k,63] = -1
    A[42+k,21+k] = -1
    A[42+k,42+k] = 1
    A[63+k,21+k] = 1
    A[63+k,42+k] = 1
    A[63+k,63] = 1
    B[63+k] = 2
BOUNDS = [(0.,2.)]*42 + [(1e-8,1.)]*21 + [(0.05,1.95)]

class Expired(Exception):
    pass

def perimeter(p,r):
    x = [Fraction(float(v)) for v in p[:,0]]
    y = [Fraction(float(v)) for v in p[:,1]]
    q = [Fraction(float(v)) for v in r]
    return max(a+b for a,b in zip(x,q))-min(a-b for a,b in zip(x,q))+max(a+b for a,b in zip(y,q))-min(a-b for a,b in zip(y,q))

def valid(p,r):
    if p.shape != (21,2) or r.shape != (21,) or not np.isfinite(p).all() or not np.isfinite(r).all() or np.min(r)<=0:
        return False
    d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
    return not np.any(d<r[I]+r[J]) and perimeter(p,r)<=2

def construct(params,seed,budget_s,init=None):
    started = time.monotonic()
    deadline = started+0.70*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf
    safety = float(params['safety'])
    def check():
        if time.monotonic()>=deadline:
            raise Expired()
    def save(p,r):
        nonlocal best,best_score
        check()
        p = np.asarray(p,dtype=float).copy()
        r = np.asarray(r,dtype=float).copy()
        if not np.isfinite(p).all() or not np.isfinite(r).all() or np.min(r)<=0:
            return
        d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
        ratio = min(1.,float(np.min(d/(r[I]+r[J]))))
        if ratio<=0:
            return
        r *= ratio*(1-safety)
        p -= np.min(p,axis=0)
        per = perimeter(p,r)
        if per>2:
            scale = (2./float(per))*(1-safety)
            p *= scale
            r *= scale
        score = float(np.sum(r))
        if score>best_score and valid(p,r):
            best_score = score
            best = {'circles':np.column_stack((p,r)).tolist()}
    if init is not None:
        arr = np.asarray(init['circles'],dtype=float)
        p,r = arr[:,:2].copy(),arr[:,2].copy()
        if valid(p,r):
            best_score = float(np.sum(r))
    else:
        p = np.array([((i%5)+0.5+0.35*((i//5)%2),(i//5)+0.5) for i in range(N)],dtype=float)*0.18
        r = np.full(N,0.075)
    r = np.maximum(np.abs(r),1e-7)
    lo = np.min(p-r[:,None],axis=0)
    side = np.max(p+r[:,None],axis=0)-lo
    scale = 2./np.sum(side)
    p = (p-lo)*scale
    r *= scale
    z = np.r_[p[:,0],p[:,1],r,side[0]*scale]
    def cons(v):
        check()
        dx,dy = v[I]-v[J],v[21+I]-v[21+J]
        dist = np.sqrt(dx*dx+dy*dy+1e-30)
        return np.r_[dist-v[42+I]-v[42+J],B-A@v]
    def jac(v):
        check()
        dx,dy = v[I]-v[J],v[21+I]-v[21+J]
        dist = np.sqrt(dx*dx+dy*dy+1e-30)
        out = np.zeros((M+84,D))
        out[ROWS,I] = dx/dist
        out[ROWS,J] = -dx/dist
        out[ROWS,21+I] = dy/dist
        out[ROWS,21+J] = -dy/dist
        out[ROWS,42+I] = -1
        out[ROWS,42+J] = -1
        out[M:] = -A
        return out
    def contact_polish(v):
        check()
        v = v.copy()
        residual = cons(v)
        if np.min(residual)<-2e-6:
            return v
        active = np.flatnonzero(residual<float(params['contact_threshold']))
        if len(active)<D:
            return v
        # Keep the inferred contact graph fixed: Newton corrects the geometry,
        # not the topology. Rank testing prevents movement along free modes.
        for iteration in range(6):
            check()
            residual = cons(v)
            g = jac(v)[active]
            step,_,rank,_ = lstsq(g,-residual[active],cond=1e-11,lapack_driver='gelsy',check_finite=False)
            if rank<D or not np.isfinite(step).all() or np.max(np.abs(step))>0.01:
                break
            trial = v+step
            trial_residual = cons(trial)
            if np.max(np.abs(trial_residual[active]))>np.max(np.abs(residual[active]))+2e-15:
                break
            v = trial
            save(np.column_stack((v[:21],v[21:42])),v[42:63])
            if np.max(np.abs(step))<2e-15:
                break
        return v
    def callback(v):
        check()
        if float(np.sum(v[42:63]))>best_score and np.min(cons(v))>-2e-5:
            save(np.column_stack((v[:21],v[21:42])),v[42:63])
    try:
        save(p,r)
        z = contact_polish(z)
        lp_end = started+0.22*budget_s
        for iteration in range(int(params['lp_steps'])):
            check()
            if time.monotonic()>=lp_end:
                break
            dx,dy = z[I]-z[J],z[21+I]-z[21+J]
            dist = np.maximum(np.hypot(dx,dy),1e-15)
            mat = np.zeros((M+84,D))
            mat[ROWS,I] = -dx/dist
            mat[ROWS,J] = dx/dist
            mat[ROWS,21+I] = -dy/dist
            mat[ROWS,21+J] = dy/dist
            mat[ROWS,42+I] = 1
            mat[ROWS,42+J] = 1
            mat[M:] = A
            remaining = max(0.01,min(lp_end,deadline)-time.monotonic())
            result = linprog(C,A_ub=mat,b_ub=np.r_[np.zeros(M),B],bounds=BOUNDS,method='highs',options={'time_limit':remaining})
            check()
            if not result.success:
                break
            change = np.max(np.abs(result.x-z))
            z = result.x
            callback(z)
            if change<2e-8:
                break
        z = contact_polish(z)
        for attempt in range(4):
            check()
            if attempt:
                arr = np.asarray(best['circles'],dtype=float)
                pp,rr = arr[:,:2].copy(),arr[:,2].copy()
                lower = np.min(pp-rr[:,None],axis=0)
                ss = np.max(pp+rr[:,None],axis=0)-lower
                pp -= lower
                z = np.r_[pp[:,0],pp[:,1],rr,ss[0]]
                noise = float(params['restart_noise'])*(1+0.5*(attempt-1))
                z[:42] += rng.normal(0,noise,42)
                z[:42] = np.clip(z[:42],0,2)
                z[42:63] *= 0.985
            result = minimize(lambda v:(float(C@v),C),z,jac=True,method='SLSQP',bounds=BOUNDS,constraints={'type':'ineq','fun':cons,'jac':jac},callback=callback,options={'maxiter':int(params['polish_steps']),'ftol':2e-12,'disp':False})
            z = result.x
            callback(z)
            z = contact_polish(z)
    except Expired:
        pass
    if best_score>-np.inf:
        return best
    return {'circles':[[0.1+0.2*(i%7),0.1+0.2*(i//7),0.01] for i in range(21)]}
