import time
import numpy as np
from fractions import Fraction
from scipy.optimize import minimize
from scipy.linalg import lstsq

PARAMS = {"perturbation": (0.035, 0.005, 0.15), "iterations": (65, 20, 120)}


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    best = init
    best_score = -float('inf')

    class Stop(Exception):
        pass

    def check():
        if time.monotonic() >= deadline:
            raise Stop()

    def exact(data):
        a = np.asarray(data['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return -float('inf')
        x = [Fraction(float(t)) for t in a[:, 0]]
        y = [Fraction(float(t)) for t in a[:, 1]]
        r = [Fraction(float(t)) for t in a[:, 2]]
        perimeter = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        perimeter += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        if perimeter > 2:
            return -float('inf')
        d = a[ii, :2] - a[jj, :2]
        if np.any(np.sqrt(d[:,0]**2+d[:,1]**2) < a[ii,2]+a[jj,2]):
            return -float('inf')
        return float(np.sum(a[:,2]))

    if init is not None:
        best_score = exact(init)

    def retain(v, swapped=False):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        a = np.asarray(v[:63]).reshape(21,3).copy()
        if not np.isfinite(a).all() or np.min(a[:,2]) <= 0:
            return
        a[:, :2] -= np.mean(a[:, :2], axis=0)
        d = a[ii,:2]-a[jj,:2]
        dist = np.sqrt(d[:,0]**2+d[:,1]**2)
        factor = min(1.0, float(np.min(dist/(a[ii,2]+a[jj,2]))))
        a[:,2] *= factor * (1.0-2e-15)
        if np.min(a[:,2]) <= 0:
            return
        qx = [Fraction(float(t)) for t in a[:,0]]
        qy = [Fraction(float(t)) for t in a[:,1]]
        qr = [Fraction(float(t)) for t in a[:,2]]
        p = max(x+r for x,r in zip(qx,qr))-min(x-r for x,r in zip(qx,qr))
        p += max(y+r for y,r in zip(qy,qr))-min(y-r for y,r in zip(qy,qr))
        scale = float(Fraction(2)/p) * (1.0-3e-15)
        a *= scale
        if swapped:
            a[:,[0,1]] = a[:,[1,0]]
        data = {'circles': a.tolist()}
        score = exact(data)
        if score > best_score:
            best, best_score = data, score

    # Coordinates use a horizontal symmetry axis at y=0, with width w
    # and height 2-w. All constraints are nonnegative.
    rows = np.arange(len(ii))
    def constraints(v, jac=False):
        check()
        a = v[:63].reshape(21,3)
        x,y,r = a.T
        w = v[63]
        dx = x[ii]-x[jj]
        dy = y[ii]-y[jj]
        rr = r[ii]+r[jj]
        g = np.concatenate((dx*dx+dy*dy-rr*rr,
                            x-r, w-x-r, 1-w/2+y-r, 1-w/2-y-r))
        if not jac:
            return g
        J = np.zeros((294,64))
        J[rows,3*ii] = 2*dx
        J[rows,3*jj] = -2*dx
        J[rows,3*ii+1] = 2*dy
        J[rows,3*jj+1] = -2*dy
        J[rows,3*ii+2] = -2*rr
        J[rows,3*jj+2] = -2*rr
        k = np.arange(21)
        J[210+k,3*k] = 1
        J[210+k,3*k+2] = -1
        J[231+k,3*k] = -1
        J[231+k,3*k+2] = -1
        J[231+k,63] = 1
        J[252+k,3*k+1] = 1
        J[252+k,3*k+2] = -1
        J[252+k,63] = -.5
        J[273+k,3*k+1] = -1
        J[273+k,3*k+2] = -1
        J[273+k,63] = -.5
        return J

    grad = np.zeros(64)
    grad[2:63:3] = -1

    def polish(v, B=None, iterations=65):
        if B is None:
            B = np.eye(64)
        z = np.linalg.lstsq(B, v, rcond=None)[0]
        dg = grad @ B
        def fun(z):
            check()
            return float(dg @ z)
        def cf(z):
            return constraints(B @ z)
        def cj(z):
            return constraints(B @ z, True) @ B
        result = minimize(fun, z, jac=lambda z: dg, method='SLSQP',
                          constraints={'type':'ineq','fun':cf,'jac':cj},
                          options={'maxiter':iterations,'ftol':2e-14,'disp':False})
        return B @ result.x

    def contact_polish(v):
        g = constraints(v)
        active = np.flatnonzero(g < 2e-7)
        if len(active) < 60:
            return v
        u = v.copy()
        for _ in range(7):
            check()
            g = constraints(u)[active]
            J = constraints(u, True)[active]
            step = lstsq(J, -g, lapack_driver='gelsy')[0]
            if np.max(np.abs(step)) > .01:
                break
            u += step
            retain(u, swapped)
            if np.max(np.abs(step)) < 2e-15:
                break
        return u

    if init is None:
        a = []
        for row in range(3):
            check()
            for col in range(7):
                check()
                a.append([.14+col*.26, (row-1)*.24, .12])
        a = np.array(a)
        a *= 2/(np.ptp(a[:,0])+np.ptp(a[:,1])+.48)
    else:
        a = np.asarray(init['circles'], dtype=float).copy()
    lo = np.min(a[:,:2]-a[:,2,None],axis=0)
    hi = np.max(a[:,:2]+a[:,2,None],axis=0)
    mid = (hi+lo)/2
    deviation = np.sort(np.abs(a[:,:2]-mid),axis=0)[:3].sum(axis=0)
    swapped = bool(deviation[0] < deviation[1])
    if swapped:
        a[:,[0,1]] = a[:,[1,0]]
    lo = np.min(a[:,:2]-a[:,2,None],axis=0)
    hi = np.max(a[:,:2]+a[:,2,None],axis=0)
    a[:,0] -= lo[0]
    a[:,1] -= (lo[1]+hi[1])/2
    v = np.r_[a.ravel(),hi[0]-lo[0]]
    try:
        retain(v, swapped)
        u = polish(v, iterations=45)
        retain(u, swapped)
        u = contact_polish(u)
        retain(u, swapped)
        # Identify three axis circles and nine reflected pairs.
        aa = u[:63].reshape(21,3)
        singles = list(np.argsort(np.abs(aa[:,1]))[:3])
        unused = set(range(21))-set(singles)
        pairs = []
        while unused:
            check()
            i = min(unused)
            unused.remove(i)
            target = aa[i].copy()
            target[1] *= -1
            j = min(unused, key=lambda j: float(np.sum((aa[j]-target)**2)))
            unused.remove(j)
            if aa[i,1] < aa[j,1]:
                i,j = j,i
            pairs.append((i,j))
        B = np.zeros((64,34))
        col = 0
        for i in singles:
            check()
            B[3*i,col] = 1
            B[3*i+2,col+1] = 1
            col += 2
        for i,j in pairs:
            check()
            B[3*i,col] = B[3*j,col] = 1
            B[3*i+1,col+1] = 1
            B[3*j+1,col+1] = -1
            B[3*i+2,col+2] = B[3*j+2,col+2] = 1
            col += 3
        B[63,33] = 1
        base = np.linalg.lstsq(B,u,rcond=None)[0]
        attempt = 0
        while time.monotonic() < deadline:
            check()
            z = base.copy()
            if attempt:
                amplitude = params['perturbation'] * (1 + .5*((attempt-1)%3))
                selected = rng.choice(9, size=3+(attempt%3), replace=False)
                for p in selected:
                    check()
                    k = 6+3*p
                    z[k:k+2] += rng.normal(0,amplitude,2)
                    z[k+2] *= np.exp(rng.normal(0,amplitude*2))
            candidate = polish(B@z, B, params['iterations'])
            retain(candidate, swapped)
            if float(grad@candidate) < float(grad@(B@base)) and np.min(constraints(candidate)) > -1e-8:
                base = np.linalg.lstsq(B,candidate,rcond=None)[0]
            if time.monotonic() < deadline-.08 and attempt % 2 == 1:
                candidate = polish(candidate, iterations=15)
                retain(candidate, swapped)
            attempt += 1
    except (Stop, ValueError, np.linalg.LinAlgError):
        pass
    return best if best is not None else {'circles': a.tolist()}
