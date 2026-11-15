import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import lstsq

PARAMS = {
    'iterations': (95, 30, 200),
    'time_fraction': (0.69, 0.4, 0.7),
    'jitter': (0.014, 0.001, 0.08),
    'working_gap': (0.065, 0.02, 0.2)
}

II, JJ = np.triu_indices(21, 1)
IX = 3*np.arange(21)
IR = IX+2


def exact_score(data):
    try:
        a = np.asarray(data['circles'], dtype=np.float64)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        xs = [Fraction(float(v)) for v in a[:, 0]]
        ys = [Fraction(float(v)) for v in a[:, 1]]
        rs = [Fraction(float(v)) for v in a[:, 2]]
        w = max(x+r for x,r in zip(xs,rs))-min(x-r for x,r in zip(xs,rs))
        h = max(y+r for y,r in zip(ys,rs))-min(y-r for y,r in zip(ys,rs))
        if w+h > 2:
            return -float('inf')
        for p,q in itertools.combinations(a, 2):
            if np.sqrt((p[0]-q[0])**2+(p[1]-q[1])**2) < p[2]+q[2]:
                return -float('inf')
        return float(np.sum(a[:, 2]))
    except Exception:
        return -float('inf')


class Finished(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    deadline = time.monotonic()+float(params['time_fraction'])*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = exact_score(init)
    grad = np.zeros(64)
    grad[IR] = -1
    bounds = [(0,2),(0,2),(1e-8,0.7)]*21+[(0.15,1.85)]

    def check():
        if time.monotonic() >= deadline:
            raise Finished()

    def normalize(c):
        c = np.asarray(c, dtype=float).reshape(21,3).copy()
        lo = np.min(c[:,:2]-c[:,2,None], axis=0)
        hi = np.max(c[:,:2]+c[:,2,None], axis=0)
        c[:,:2] -= lo
        f = 2/float(np.sum(hi-lo))
        c *= f
        return np.r_[c.ravel(), (hi[0]-lo[0])*f]

    def consider(z):
        nonlocal best, best_score
        check()
        a = np.asarray(z[:63]).reshape(21,3).copy()
        if not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        dx = a[II,0]-a[JJ,0]
        dy = a[II,1]-a[JJ,1]
        ratio = float(np.min(np.sqrt(dx*dx+dy*dy)/(a[II,2]+a[JJ,2])))
        if ratio <= 0:
            return
        a[:,2] *= min(1.0,ratio)*(1-4e-15)
        lo = np.min(a[:,:2]-a[:,2,None], axis=0)
        a[:,:2] -= lo
        extent = np.max(a[:,:2]+a[:,2,None], axis=0)
        a *= (2-2e-14)/float(np.sum(extent))
        if float(np.sum(a[:,2])) > best_score:
            candidate = {'circles': a.tolist()}
            score = exact_score(candidate)
            if score > best_score:
                best, best_score = candidate, score

    def values(z, pi=II, pj=JJ):
        check()
        c = z[:63].reshape(21,3)
        x,y,r = c.T
        dx = x[pi]-x[pj]
        dy = y[pi]-y[pj]
        return np.r_[x-r,z[63]-x-r,y-r,2-z[63]-y-r,
                     np.sqrt(dx*dx+dy*dy)-r[pi]-r[pj]]

    def jac(z, pi=II, pj=JJ):
        check()
        c = z[:63].reshape(21,3)
        dx = c[pi,0]-c[pj,0]
        dy = c[pi,1]-c[pj,1]
        d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-25)
        j = np.zeros((84+len(pi),64))
        rr = np.arange(21)
        j[rr,IX] = 1
        j[rr,IR] = -1
        j[21+rr,IX] = -1
        j[21+rr,IR] = -1
        j[21+rr,63] = 1
        j[42+rr,IX+1] = 1
        j[42+rr,IR] = -1
        j[63+rr,IX+1] = -1
        j[63+rr,IR] = -1
        j[63+rr,63] = -1
        t = 84+np.arange(len(pi))
        j[t,3*pi] = dx/d
        j[t,3*pj] = -dx/d
        j[t,3*pi+1] = dy/d
        j[t,3*pj+1] = -dy/d
        j[t,3*pi+2] = -1
        j[t,3*pj+2] = -1
        return j

    def polish_contacts(z):
        active = np.flatnonzero(values(z)<2e-7)
        if len(active) < 60 or len(active)>72:
            return
        q = z.copy()
        for unused in range(7):
            check()
            v = values(q)[active]
            if np.max(np.abs(v)) < 2e-16:
                break
            step = lstsq(jac(q)[active],-v,cond=1e-11,lapack_driver='gelsy')[0]
            if np.max(np.abs(step)) > 0.003:
                break
            q += step
            consider(q)

    def solve(z):
        check()
        allv = values(z)
        selected = allv[84:] < float(params['working_gap'])
        pi,pj = II[selected],JJ[selected]
        def fun(q):
            check()
            return -float(np.sum(q[IR]))
        result = minimize(fun,z,jac=lambda q: grad,method='SLSQP',bounds=bounds,
            constraints={'type':'ineq','fun':lambda q: values(q,pi,pj),
                         'jac':lambda q: jac(q,pi,pj)},
            callback=consider,
            options={'maxiter':int(params['iterations']),'ftol':8e-15,'disp':False})
        consider(result.x)
        if np.min(values(result.x)) > -1e-6:
            polish_contacts(result.x)
        return result.x

    if best is None:
        r = 0.099999999999
        best = {'circles': [[0.1+0.2*(i%7),0.1+0.2*(i//7),r] for i in range(21)]}
        best_score = exact_score(best)
    anchor = normalize(np.asarray(best['circles']))
    attempt = 0
    while time.monotonic() < deadline:
        try:
            check()
            if attempt == 0:
                z = anchor.copy()
            elif attempt % 4 != 0:
                c = np.asarray(best['circles'],dtype=float).copy()
                size = float(params['jitter'])*(1+0.7*((attempt-1)//3))
                center = rng.integers(21)
                near = np.argsort(np.sum((c[:,:2]-c[center,:2])**2,axis=1))[:7]
                c[near,:2] += rng.normal(0,size,(len(near),2))
                c[near,2] *= np.exp(rng.normal(0,size*0.5,len(near)))
                z = normalize(c)
            elif attempt % 8 == 4:
                c = [[1+2*x,1+2*y,1.] for y in range(4) for x in range(4)]
                c += [[2+2*x,2+2*y,np.sqrt(2)-1] for x,y in [(0,0),(0,2),(1,1),(2,0),(2,2)]]
                z = normalize(c)
            else:
                c = []
                for row,count in enumerate([5,4,5,4,3]):
                    check()
                    offset = 1 if row%2 == 0 else 2
                    if count == 3:
                        offset += 2
                    for k in range(count):
                        check()
                        c.append([offset+2*k,1+np.sqrt(3)*row,1.])
                z = normalize(c)
            solve(z)
        except Finished:
            break
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            pass
        attempt += 1
    return best
