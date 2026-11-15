import time
import numpy as np
from fractions import Fraction
from scipy.optimize import minimize

PARAMS = {"iterations": (110, 40, 180), "perturbation": (0.025, 0.001, 0.08)}


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.68 * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    rows = np.arange(210)
    kk = np.arange(21)
    best = init
    best_score = -float('inf')
    best_v = None

    class Stop(Exception):
        pass

    def check():
        if time.monotonic() >= deadline:
            raise Stop()

    def canonical(obj):
        if 'circles' in obj:
            return obj
        return {'circles': [[p[0], p[1], r] for p, r in zip(obj['centers'], obj['radii'])]}

    def exact(obj):
        try:
            a = np.asarray(canonical(obj)['circles'], dtype=float)
            if a.shape != (21, 3) or not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
                return -float('inf')
            x = [Fraction(float(t)) for t in a[:, 0]]
            y = [Fraction(float(t)) for t in a[:, 1]]
            r = [Fraction(float(t)) for t in a[:, 2]]
            p = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
            p += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
            if p > 2:
                return -float('inf')
            d = a[ii, :2]-a[jj, :2]
            if np.any(np.sqrt(d[:,0]**2+d[:,1]**2) < a[ii,2]+a[jj,2]):
                return -float('inf')
            return float(np.sum(a[:,2]))
        except (ValueError, KeyError, TypeError):
            return -float('inf')

    if init is not None:
        best_score = exact(init)

    def retain(v):
        nonlocal best, best_score, best_v
        a = np.array(v[:63], dtype=float).reshape(21,3).copy()
        if not np.isfinite(a).all() or np.min(a[:,2]) <= 0:
            return
        a[:,:2] -= np.mean(a[:,:2], axis=0)
        d = a[ii,:2]-a[jj,:2]
        dist = np.sqrt(d[:,0]**2+d[:,1]**2)
        f = min(1.0, float(np.min(dist/(a[ii,2]+a[jj,2]))))
        if f <= 0:
            return
        a[:,2] *= f*(1-3e-14)
        x = [Fraction(float(t)) for t in a[:,0]]
        y = [Fraction(float(t)) for t in a[:,1]]
        r = [Fraction(float(t)) for t in a[:,2]]
        p = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        p += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        a *= float(Fraction(2)/p)*(1-3e-14)
        obj = {'centers': a[:,:2].tolist(), 'radii': a[:,2].tolist()}
        score = exact(obj)
        if score > best_score:
            best, best_score = obj, score
            lo = np.min(a[:,:2]-a[:,2,None],axis=0)
            hi = np.max(a[:,:2]+a[:,2,None],axis=0)
            a[:,0] -= lo[0]
            a[:,1] -= (lo[1]+hi[1])/2
            best_v = np.r_[a.ravel(),hi[0]-lo[0]]

    def values(v, jac=False):
        check()
        a = v[:63].reshape(21,3)
        x,y,r = a.T
        w = v[63]
        dx = x[ii]-x[jj]
        dy = y[ii]-y[jj]
        dist = np.sqrt(dx*dx+dy*dy)
        if not jac:
            return np.concatenate((dist-r[ii]-r[jj], x-r, w-x-r,
                                   1-w/2+y-r, 1-w/2-y-r, r-1e-7))
        den = np.maximum(dist,1e-14)
        J = np.zeros((315,64))
        J[rows,3*ii] = dx/den
        J[rows,3*jj] = -dx/den
        J[rows,3*ii+1] = dy/den
        J[rows,3*jj+1] = -dy/den
        J[rows,3*ii+2] = -1
        J[rows,3*jj+2] = -1
        J[210+kk,3*kk] = 1
        J[210+kk,3*kk+2] = -1
        J[231+kk,3*kk] = -1
        J[231+kk,3*kk+2] = -1
        J[231+kk,63] = 1
        J[252+kk,3*kk+1] = 1
        J[252+kk,3*kk+2] = -1
        J[252+kk,63] = -.5
        J[273+kk,3*kk+1] = -1
        J[273+kk,3*kk+2] = -1
        J[273+kk,63] = -.5
        J[294+kk,3*kk+2] = 1
        return J

    grad = np.zeros(64)
    grad[2:63:3] = -1

    def optimize(v, B=None, iterations=110):
        check()
        if B is None:
            B = np.eye(64)
        z = np.linalg.lstsq(B,v,rcond=None)[0]
        dg = grad@B
        latest = [B@z]
        count = [0]
        def fun(t):
            check()
            return float(dg@t)
        def cb(t):
            check()
            latest[0] = B@t
            count[0] += 1
            if count[0] % 12 == 0:
                retain(latest[0])
        try:
            res = minimize(fun,z,jac=lambda t: dg,method='SLSQP',
                           constraints={'type':'ineq','fun':lambda t: values(B@t),
                                        'jac':lambda t: values(B@t,True)@B},
                           callback=cb,
                           options={'maxiter':iterations,'ftol':3e-12,'disp':False})
            latest[0] = B@res.x
        except Stop:
            retain(latest[0])
            raise
        retain(latest[0])
        return latest[0]

    def layered(counts):
        # counts specify the axis row and successive upper-half rows.
        pts = []
        singles = []
        pairs = []
        for level, count in enumerate(counts):
            check()
            for col in range(count):
                check()
                x = (col-(count-1)/2)*.20
                if level == 0 and count == 3:
                    x *= 1.45
                if level == 0:
                    singles.append(len(pts))
                    pts.append([x,0.,.095])
                else:
                    i = len(pts)
                    pts.extend([[x,level*.177,.095],[x,-level*.177,.095]])
                    pairs.append((i,i+1))
        a = np.array(pts)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        s = 2/np.sum(hi-lo)
        a *= s
        a[:,0] -= lo[0]*s
        v = np.r_[a.ravel(),(hi[0]-lo[0])*s]
        B = np.zeros((64,2*len(singles)+3*len(pairs)+1))
        c = 0
        for i in singles:
            check()
            B[3*i,c] = 1
            B[3*i+2,c+1] = 1
            c += 2
        for i,j in pairs:
            check()
            B[3*i,c] = B[3*j,c] = 1
            B[3*i+1,c+1] = 1
            B[3*j+1,c+1] = -1
            B[3*i+2,c+2] = B[3*j+2,c+2] = 1
            c += 3
        B[63,c] = 1
        return v,B

    try:
        if init is not None and best_score > 1.8:
            a = np.asarray(canonical(init)['circles'],dtype=float).copy()
            lo = np.min(a[:,:2]-a[:,2,None],axis=0)
            hi = np.max(a[:,:2]+a[:,2,None],axis=0)
            a[:,0] -= lo[0]
            a[:,1] -= (lo[1]+hi[1])/2
            v = np.r_[a.ravel(),hi[0]-lo[0]]
            optimize(v,iterations=params['iterations'])
        for counts in ((3,5,4),(5,4,4),(5,5,3),(3,4,5)):
            check()
            v,B = layered(counts)
            retain(v)
            u = optimize(v,B,params['iterations'])
            if time.monotonic() < deadline-.12:
                optimize(u,iterations=35)
        attempt = 0
        while time.monotonic() < deadline and best_v is not None:
            check()
            v = best_v.copy()
            a = v[:63].reshape(21,3)
            selected = rng.choice(21,5,replace=False)
            a[selected,:2] += rng.normal(0,params['perturbation'],(5,2))
            optimize(v,iterations=45)
            attempt += 1
    except (Stop, ValueError, np.linalg.LinAlgError):
        pass
    if best is None:
        # This branch is only needed for exceptionally small budgets.
        a = [[(i%7)*.2,(i//7)*.2,.09999999999999] for i in range(21)]
        return {'circles':a}
    return best
