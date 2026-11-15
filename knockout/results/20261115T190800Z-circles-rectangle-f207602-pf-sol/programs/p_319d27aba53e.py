import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {"iterations": (170, 50, 300), "perturb": (0.022, 0.001, 0.08), "restart_iterations": (65, 20, 130)}


def canonical(obj):
    if 'circles' in obj:
        return obj
    a = []
    for x, y, r, m in obj['orbits']:
        a.append([x, y, r])
        if m == 2:
            a.append([x, -y, r])
    return {'circles': a}


def score(obj):
    try:
        a = np.asarray(canonical(obj)['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        x = [Fraction(float(t)) for t in a[:, 0]]
        y = [Fraction(float(t)) for t in a[:, 1]]
        r = [Fraction(float(t)) for t in a[:, 2]]
        if max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))+max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)) > 2:
            return -float('inf')
        for i in range(21):
            for j in range(i):
                dx = a[i,0]-a[j,0]
                dy = a[i,1]-a[j,1]
                if np.sqrt(dx*dx+dy*dy) < a[i,2]+a[j,2]:
                    return -float('inf')
        return float(np.sum(a[:,2]))
    except (ValueError, TypeError, KeyError, OverflowError):
        return -float('inf')


class StopSearch(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    end = time.monotonic() + 0.70*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = score(init) if init is not None else -float('inf')
    best_z = None

    def tick():
        if time.monotonic() >= end:
            raise StopSearch()

    group = np.array([0,1,2] + [g for g in range(3,12) for _ in range(2)])
    sign = np.array([0.,0.,0.] + [s for _ in range(9) for s in (1.,-1.)])
    X = np.zeros((21,34)); Y = np.zeros((21,34)); R = np.zeros((21,34))
    for k,g in enumerate(group):
        tick()
        X[k,g] = 1.
        R[k,21+g] = 1.
        if g >= 3:
            Y[k,9+g] = sign[k]
    pp = []; seen = set()
    for i in range(21):
        tick()
        for j in range(i):
            tick()
            key = (min(group[i],group[j]), max(group[i],group[j]), sign[i]*sign[j])
            if key not in seen:
                seen.add(key)
                pp.append((i,j))
    pi,pj = np.array(pp).T
    DX = X[pi]-X[pj]; DY = Y[pi]-Y[pj]; RS = R[pi]+R[pj]
    wall = np.zeros((36,34)); offset = np.zeros(36)
    for g in range(12):
        tick()
        wall[g,g] = 1.; wall[g,21+g] = -1.; wall[g,33] = .5
        wall[12+g,g] = -1.; wall[12+g,21+g] = -1.; wall[12+g,33] = .5
        wall[24+g,21+g] = -1.; wall[24+g,33] = -.5
        offset[24+g] = 1.
        if g >= 3:
            wall[24+g,9+g] = -1.
    weights = np.array([1.]*3+[2.]*9)
    grad = np.zeros(34); grad[21:33] = -weights
    bounds = [(-1.,1.)]*12 + [(0.,1.)]*9 + [(1e-7,.6)]*12 + [(.15,1.85)]
    ii,jj = np.triu_indices(21,1)

    def retain(v):
        nonlocal best, best_score, best_z
        tick()
        if not np.isfinite(v).all():
            return
        xx = v[:12].copy()
        yy = np.r_[np.zeros(3),np.abs(v[12:21])]
        rr = v[21:33].copy()
        if np.min(rr) <= 0:
            return
        cx = xx[group]; cy = yy[group]*sign; cr = rr[group]
        dist = np.sqrt((cx[ii]-cx[jj])**2+(cy[ii]-cy[jj])**2)
        shrink = min(1.,float(np.min(dist/(cr[ii]+cr[jj]))))*(1.-2e-14)
        if shrink <= 0:
            return
        rr *= shrink
        left = np.min(xx-rr); right = np.max(xx+rr)
        width = right-left
        height = 2.*np.max(yy+rr)
        scale = 2./(width+height)*(1.-2e-14)
        if float(weights@rr)*scale <= best_score:
            return
        xx = (xx-(left+right)*.5)*scale
        yy *= scale; rr *= scale
        obj = {'orbits': [[float(xx[g]),float(yy[g]),float(rr[g]),1 if g<3 else 2] for g in range(12)]}
        value = score(obj)
        if value > best_score:
            best = obj; best_score = value
            best_z = np.r_[xx,yy[3:],rr,width*scale]

    def fun(v):
        tick()
        return float(grad@v)

    def jac(v):
        tick()
        return grad

    def con(v):
        tick()
        dx = DX@v; dy = DY@v
        return np.r_[np.sqrt(dx*dx+dy*dy)-RS@v,wall@v+offset]

    def cjac(v):
        tick()
        dx = DX@v; dy = DY@v
        d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-24)
        return np.vstack(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RS,wall))

    def solve(v, iterations):
        tick()
        out = minimize(fun,v,jac=jac,bounds=bounds,method='SLSQP',constraints={'type':'ineq','fun':con,'jac':cjac},callback=retain,options={'maxiter':iterations,'ftol':2e-13,'disp':False})
        retain(out.x)
        return out.x

    # Three axis circles, four pairs in the inner row, five in the outer.
    r = 1./(6.+2.*np.sqrt(3.)) / 2.
    gx = np.array([-2.,0.,2.,-3.,-1.,1.,3.,-4.,-2.,0.,2.,4.])*r
    gy = np.array([np.sqrt(3.)]*4+[2.*np.sqrt(3.)]*5)*r
    z = np.r_[gx,gy,np.full(12,r),10.*r]
    try:
        retain(z)
        # Preserve useful existing geometry, but do not project tiny cold starts.
        if best_score > 2.25 and init is not None and score(init) > 2.25:
            a = np.asarray(canonical(init)['circles'],float).copy()
            lo = np.min(a[:,:2]-a[:,2,None],axis=0)
            hi = np.max(a[:,:2]+a[:,2,None],axis=0)
            a[:,:2] -= (lo+hi)/2
            choices = []
            for swap in (False,True):
                tick()
                b = a.copy()
                if swap:
                    b[:,[0,1]] = b[:,[1,0]]
                order = np.argsort(b[:,1]); lower = order[:9]; upper = order[12:]
                ref = b[lower].copy(); ref[:,1] *= -1
                cost = np.sum((b[upper,None,:]-ref[None,:,:])**2,axis=2)
                u,v = linear_sum_assignment(cost)
                gs = np.vstack((b[order[9:12]],(b[upper[u]]+ref[v])/2))
                error = np.sum(cost[u,v])+np.sum(gs[:3,1]**2)
                gs[:3,1] = 0.
                choices.append((error,gs))
            gs = min(choices,key=lambda t:t[0])[1]
            z = np.r_[gs[:,0],np.abs(gs[3:,1]),gs[:,2],np.max(gs[:,0]+gs[:,2])-np.min(gs[:,0]-gs[:,2])]
        z = solve(z,int(params['iterations']))
        attempt = 0
        while time.monotonic() < end:
            tick()
            anchor = best_z.copy() if best_z is not None else z.copy()
            amp = float(params['perturb'])*(1.+.6*(attempt%3))
            anchor[:12] += rng.normal(0.,amp,12)
            anchor[12:21] = np.maximum(0.,anchor[12:21]+rng.normal(0.,amp,9))
            anchor[21:33] *= rng.uniform(.86,1.08,12)
            z = solve(anchor,int(params['restart_iterations']))
            attempt += 1
    except (StopSearch, ValueError, FloatingPointError):
        pass
    return best
