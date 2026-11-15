import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize

PARAMS = {"lp_steps": (12, 5, 30), "polish_steps": (65, 20, 150), "safety": (2e-13, 1e-14, 1e-10, "log")}

class StopSearch(Exception):
    pass


def construct(params: dict, seed: int, budget_s: float, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    n = 21
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    ar = np.arange(n)
    best_obj = None
    best_score = -float('inf')
    best_z = None
    safety = float(params['safety'])

    def canonical(obj):
        if 'circles' in obj:
            return np.asarray(obj['circles'], dtype=float)
        return np.asarray(obj['coordinate_rows'], dtype=float).T

    def exact(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or not (a[:, 2] > 0).all():
            return -float('inf')
        xs = list(map(lambda v: Fraction(float(v)), a[:, 0]))
        ys = list(map(lambda v: Fraction(float(v)), a[:, 1]))
        rs = list(map(lambda v: Fraction(float(v)), a[:, 2]))
        width = max(x+r for x,r in zip(xs,rs)) - min(x-r for x,r in zip(xs,rs))
        height = max(y+r for y,r in zip(ys,rs)) - min(y-r for y,r in zip(ys,rs))
        if width + height > 2:
            return -float('inf')
        for i,j in zip(ii,jj):
            if time.monotonic() >= deadline:
                return -float('inf')
            d = np.sqrt((a[i,0]-a[j,0])**2 + (a[i,1]-a[j,1])**2)
            if d < a[i,2]+a[j,2]:
                return -float('inf')
        return float(np.sum(a[:,2]))

    def encode(a):
        a = a.copy()
        lo = np.min(a[:,:2]-a[:,2,None], axis=0)
        hi = np.max(a[:,:2]+a[:,2,None], axis=0)
        scale = 2.0 / max(float(np.sum(hi-lo)), 1e-12)
        a[:,:2] = (a[:,:2]-lo)*scale
        a[:,2] *= scale
        return np.r_[a[:,0], a[:,1], a[:,2], (hi[0]-lo[0])*scale]

    def record(z):
        nonlocal best_obj, best_score, best_z
        if time.monotonic() >= deadline:
            return
        a = np.column_stack((z[:21], z[21:42], z[42:63]))
        if not np.isfinite(a).all() or np.min(a[:,2]) <= 0:
            return
        d = np.sqrt((a[ii,0]-a[jj,0])**2+(a[ii,1]-a[jj,1])**2)
        ratio = min(1.0, float(np.min(d/(a[ii,2]+a[jj,2]))))
        if ratio <= 0:
            return
        a[:,2] *= ratio*(1.0-safety)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        a[:,:2] -= (hi+lo)*0.5
        a *= (2.0*(1.0-safety))/float(np.sum(hi-lo))
        if float(np.sum(a[:,2])) <= best_score:
            return
        score = exact(a)
        if score > best_score:
            best_score = score
            best_obj = {'coordinate_rows': a.T.tolist()}
            best_z = encode(a)

    incoming = None
    if init is not None:
        try:
            incoming = canonical(init)
            s = exact(incoming)
            if s > best_score:
                best_score, best_obj = s, init
                best_z = encode(incoming)
            if incoming.shape != (21,3) or not np.isfinite(incoming).all():
                incoming = None
        except (ValueError, TypeError, KeyError, OverflowError):
            incoming = None

    fallback = np.array([[0.1+0.2*(k%7),0.1+0.2*(k//7),0.1] for k in range(21)])
    record(encode(fallback))

    # Rectangle constraints B z <= b, with height = 2 - width.
    B = np.zeros((84,64))
    b = np.zeros(84)
    B[ar,ar] = -1
    B[ar,42+ar] = 1
    B[21+ar,ar] = 1
    B[21+ar,42+ar] = 1
    B[21+ar,63] = -1
    B[42+ar,21+ar] = -1
    B[42+ar,42+ar] = 1
    B[63+ar,21+ar] = 1
    B[63+ar,42+ar] = 1
    B[63+ar,63] = 1
    b[63:84] = 2
    A = np.zeros((m+84,64))
    A[m:] = B
    rows = np.arange(m)
    A[rows,42+ii] = 1
    A[rows,42+jj] = 1
    rhs = np.r_[np.zeros(m),b]
    objective = np.r_[np.zeros(42),-np.ones(21),0.0]
    bounds = [(0.0,2.0)]*42 + [(1e-7,0.5)]*21 + [(0.25,1.75)]
    J = np.zeros((m+84,64))
    J[:m,42+ii] = -1
    J[:m,42+jj] = -1
    J[m:] = -B

    def constraints(z):
        dx = z[ii]-z[jj]
        dy = z[21+ii]-z[21+jj]
        d = np.sqrt(dx*dx+dy*dy)
        return np.r_[d-z[42+ii]-z[42+jj], b-B@z]

    def jacobian(z):
        dx = z[ii]-z[jj]
        dy = z[21+ii]-z[21+jj]
        d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-15)
        out = J.copy()
        out[rows,ii] = dx/d
        out[rows,jj] = -dx/d
        out[rows,21+ii] = dy/d
        out[rows,21+jj] = -dy/d
        return out

    def polish(z, end, iterations):
        last = z.copy()
        count = 0
        def fun(v):
            if time.monotonic() >= end:
                raise StopSearch()
            return -float(np.sum(v[42:63]))
        def callback(v):
            nonlocal last, count
            if time.monotonic() >= end:
                raise StopSearch()
            last = v.copy()
            count += 1
            if count % 5 == 0:
                record(v)
        try:
            res = minimize(fun,z,jac=lambda v: objective,method='SLSQP',bounds=bounds,
                           constraints={'type':'ineq','fun':constraints,'jac':jacobian},
                           callback=callback,options={'maxiter':iterations,'ftol':2e-12,'disp':False})
            last = res.x
        except StopSearch:
            pass
        record(last)
        return last

    def trajectory(z, end):
        previous = -1.0
        for step in range(int(params['lp_steps'])):
            now = time.monotonic()
            if now >= end or now >= deadline:
                break
            dx,dy = z[ii]-z[jj], z[21+ii]-z[21+jj]
            d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-15)
            ux,uy = dx/d,dy/d
            A[rows,ii],A[rows,jj] = -ux,ux
            A[rows,21+ii],A[rows,21+jj] = -uy,uy
            result = linprog(objective,A_ub=A,b_ub=rhs,bounds=bounds,method='highs',
                             options={'time_limit':max(0.001,min(end,deadline)-now),
                                      'dual_feasibility_tolerance':1e-8,
                                      'primal_feasibility_tolerance':1e-8})
            if not result.success:
                break
            z = result.x
            current = float(np.sum(z[42:63]))
            if step % 3 == 0:
                record(z)
            if step >= 4 and current-previous < 8e-6:
                break
            previous = current
        record(z)
        if time.monotonic() < end-0.015:
            z = polish(z,end,int(params['polish_steps']))
        return z

    def staggered(counts):
        points = []
        for row,count in enumerate(counts):
            if time.monotonic() >= deadline:
                break
            for k in range(count):
                if time.monotonic() >= deadline:
                    break
                points.append([2*k-(count-1),row*np.sqrt(3.0),0.95])
        if len(points) != 21:
            return None
        return encode(np.asarray(points))

    seeds = []
    if incoming is not None:
        incoming = incoming.copy()
        incoming[:,2] = np.maximum(incoming[:,2],1e-6)
        seeds.append(encode(incoming))
    else:
        seeds.append(encode(fallback))
    for counts in ([4,3,4,3,4,3],[4,5,3,5,4]):
        if time.monotonic() >= deadline:
            break
        z = staggered(counts)
        if z is not None:
            seeds.append(z)
    for index,z in enumerate(seeds):
        if time.monotonic() >= deadline-0.025:
            break
        end = min(deadline-0.04*budget_s, time.monotonic()+0.185*budget_s)
        trajectory(z,end)
    if best_z is not None and time.monotonic() < deadline-0.025:
        polish(best_z,deadline-0.005,int(params['polish_steps']))
    if best_obj is not None:
        return best_obj
    a = fallback.copy()
    a *= 1.0-1e-10
    return {'coordinate_rows':a.T.tolist()}
