import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import lstsq

PARAMS = {
    'active_tol': (2e-6, 1e-9, 1e-4, 'log'),
    'perturb': (0.00015, 1e-7, 0.01, 'log'),
    'iterations': (160, 40, 400)
}
I, J = np.triu_indices(21, 1)
R = np.arange(210)
K = np.arange(21)
W = np.zeros((84, 64))
W[K,K] = 1
W[21+K,K] = -1
W[42+K,21+K] = 1
W[63+K,21+K] = -1
W[:,42:63] = np.tile(-np.eye(21),(4,1))
W[:42,63] = .5
W[42:,63] = -.5
OFF = np.r_[np.zeros(42),np.ones(42)]
OBJ = np.r_[np.zeros(42),-np.ones(21),0.]

class Stop(Exception):
    pass

def exact_score(a):
    if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
        return -float('inf')
    d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
    if np.any(d < a[I,2]+a[J,2]):
        return -float('inf')
    f = [[Fraction(float(v)) for v in row] for row in a]
    p = sum(max(row[k]+row[2] for row in f)-min(row[k]-row[2] for row in f) for k in (0,1))
    if p > 2:
        return -float('inf')
    return float(np.sum(a[:,2]))

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + .70*budget_s
    rng = np.random.default_rng(seed)
    best = init
    score = -float('inf')
    if init is not None:
        a = (np.asarray(init['circles'],float) if 'circles' in init else np.asarray(init['state'],float).reshape(3,21).T).copy()
        score = exact_score(a)
    else:
        a = np.array([[2*(k+.5*(row%2)),np.sqrt(3)*row,1.] for row,n in enumerate((3,4,3,4,3,4)) for k in range(n)],float)
    best_a = a.copy()

    def offer(b):
        nonlocal score,best,best_a
        if time.monotonic() >= end:
            return
        b = np.asarray(b,float).copy()
        if not np.isfinite(b).all() or np.any(b[:,2] <= 0):
            return
        d = np.sqrt((b[I,0]-b[J,0])**2+(b[I,1]-b[J,1])**2)
        ratio = min(1.,float(np.min(d/(b[I,2]+b[J,2]))))
        b[:,2] *= ratio*(1-5e-16)
        lo = np.min(b[:,:2]-b[:,2,None],axis=0)
        hi = np.max(b[:,:2]+b[:,2,None],axis=0)
        b[:,:2] -= (lo+hi)/2
        b *= (2-3e-15)/float(np.sum(hi-lo))
        for t in range(5):
            if time.monotonic() >= end:
                return
            s = exact_score(b)
            if s > score:
                score = s
                best_a = b.copy()
                best = {'state':b.T.ravel().tolist()}
            if np.isfinite(s):
                return
            b[:,2] *= 1-2e-15

    def pack(b):
        lo = np.min(b[:,:2]-b[:,2,None],axis=0)
        hi = np.max(b[:,:2]+b[:,2,None],axis=0)
        c = b.copy()
        c[:,:2] -= (lo+hi)/2
        return np.r_[c.T.ravel(),hi[0]-lo[0]]

    def geometry(z):
        if time.monotonic() >= end-.012:
            raise Stop()
        dx = z[I]-z[J]
        dy = z[21+I]-z[21+J]
        rr = z[42+I]+z[42+J]
        dd = np.sqrt(dx*dx+dy*dy)
        dd = np.maximum(dd,1e-100)
        c = np.r_[dd-rr,W@z+OFF]
        jac = np.zeros((294,64))
        ux,uy = dx/dd,dy/dd
        jac[R,I] = ux
        jac[R,J] = -ux
        jac[R,21+I] = uy
        jac[R,21+J] = -uy
        jac[R,42+I] = -1
        jac[R,42+J] = -1
        jac[210:] = W
        return c,jac

    def polish(z,tol):
        z = z.copy()
        c,j = geometry(z)
        active = np.flatnonzero(c < tol)
        if len(active) < 60:
            return z
        for it in range(12):
            if time.monotonic() >= end-.025:
                break
            c,j = geometry(z)
            ja = j[active]
            delta,_,rank,_ = lstsq(ja,-c[active],cond=2e-12,lapack_driver='gelsy',check_finite=False)
            if np.max(np.abs(delta)) > .003:
                break
            if rank < 64:
                # Complete the underdetermined contact system with stationarity.
                lam = lstsq(ja.T,OBJ,cond=1e-11,lapack_driver='gelsy',check_finite=False)[0]
                h = np.zeros((64,64))
                for q,weight in zip(active,lam):
                    if time.monotonic() >= end-.02:
                        raise Stop()
                    if q >= 210:
                        continue
                    i,k = int(I[q]),int(J[q])
                    dx,dy = z[i]-z[k],z[21+i]-z[21+k]
                    d = np.hypot(dx,dy)
                    v = np.array([-dy,dx])/d
                    hh = (-weight/d)*np.outer(v,v)
                    ix = [i,21+i]
                    iy = [k,21+k]
                    h[np.ix_(ix,ix)] += hh
                    h[np.ix_(iy,iy)] += hh
                    h[np.ix_(ix,iy)] -= hh
                    h[np.ix_(iy,ix)] -= hh
                mat = np.block([[h,-ja.T],[ja,np.zeros((len(active),len(active)))]])
                rhs = np.r_[-OBJ+ja.T@lam,-c[active]]
                sol = lstsq(mat,rhs,cond=1e-12,lapack_driver='gelsy',check_finite=False)[0]
                delta = sol[:64]
                if np.max(np.abs(delta)) > .003:
                    break
            z += delta
            offer(z[:63].reshape(3,21).T)
            if np.max(np.abs(delta)) < 3e-15:
                break
        return z

    offer(a)
    z = pack(best_a)
    try:
        z = polish(z,float(params['active_tol']))
        for trial in range(5):
            if time.monotonic() >= end-.06:
                break
            if trial:
                z = pack(best_a)
                amp = float(params['perturb'])*(1 if trial < 3 else 8)
                z[:42] += rng.normal(0,amp,42)
                z[42:63] *= 1-.5*amp
            cache = [None,None,None]
            last = [z.copy()]
            def calc(q):
                if time.monotonic() >= end-.035:
                    raise Stop()
                if cache[0] is None or not np.array_equal(q,cache[0]):
                    c,j = geometry(q)
                    cache[:] = [q.copy(),c,j]
                    last[0] = q.copy()
                return cache[1],cache[2]
            def fun(q):
                if time.monotonic() >= end-.035:
                    raise Stop()
                return float(OBJ@q)
            def callback(q):
                if time.monotonic() >= end-.035:
                    raise Stop()
            bounds = [(None,None)]*42+[(1e-6,.5)]*21+[(.15,1.85)]
            try:
                result = minimize(fun,z,jac=lambda q:OBJ,method='SLSQP',bounds=bounds,
                    constraints={'type':'ineq','fun':lambda q:calc(q)[0],'jac':lambda q:calc(q)[1]},
                    callback=callback,options={'ftol':3e-15,'maxiter':int(params['iterations']),'disp':False})
                z = result.x
            except Stop:
                z = last[0]
            offer(z[:63].reshape(3,21).T)
            if time.monotonic() < end-.025:
                z = polish(z,float(params['active_tol']))
    except Stop:
        pass
    if best is None:
        b = a.copy()
        lo = np.min(b[:,:2]-b[:,2,None],axis=0)
        hi = np.max(b[:,:2]+b[:,2,None],axis=0)
        b *= 1.999999/np.sum(hi-lo)
        return {'circles':b.tolist()}
    return best
