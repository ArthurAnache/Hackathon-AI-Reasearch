import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {"release": (0.014, 0.002, 0.06), "contacts": (4, 3, 5), "polish": (80, 20, 180)}


def canonical(obj):
    if 'circles' in obj:
        return obj
    rows = []
    for x, y, r, multiplicity in obj['orbits']:
        rows.append([x, y, r])
        if multiplicity == 2:
            rows.append([x, -y, r])
    return {'circles': rows}


def score(obj):
    try:
        a = np.asarray(canonical(obj)['circles'], dtype=np.float64)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        xx = [Fraction(float(v)) for v in a[:, 0]]
        yy = [Fraction(float(v)) for v in a[:, 1]]
        rr = [Fraction(float(v)) for v in a[:, 2]]
        w = max(x+r for x,r in zip(xx,rr))-min(x-r for x,r in zip(xx,rr))
        h = max(y+r for y,r in zip(yy,rr))-min(y-r for y,r in zip(yy,rr))
        if w+h > 2:
            return -float('inf')
        for i in range(21):
            for j in range(i):
                dx = a[i,0]-a[j,0]
                dy = a[i,1]-a[j,1]
                if np.sqrt(dx*dx+dy*dy) < a[i,2]+a[j,2]:
                    return -float('inf')
        return float(np.sum(a[:,2]))
    except (TypeError, ValueError, KeyError, OverflowError):
        return -float('inf')


class Finished(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    deadline = time.monotonic()+0.69*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = score(init) if init is not None else -float('inf')

    def tick():
        if time.monotonic() >= deadline:
            raise Finished()

    if init is None:
        a = np.array([[2*(i%7), 2*(i//7), 1.] for i in range(21)], float)
        a /= 10.000000000001
        best = {'circles': a.tolist()}
        best_score = score(best)
    else:
        a = np.asarray(canonical(init)['circles'], float).copy()
    if a.shape != (21,3):
        return best
    lo = np.min(a[:,:2]-a[:,2,None],axis=0)
    hi = np.max(a[:,:2]+a[:,2,None],axis=0)
    a[:,:2] -= (lo+hi)/2

    # Infer the reflection axis and its twelve circle orbits.
    projections = []
    for swap in (False, True):
        if time.monotonic() >= deadline:
            return best
        b = a.copy()
        if swap:
            b[:,[0,1]] = b[:,[1,0]]
        order = np.argsort(b[:,1])
        lower, singles, upper = order[:9], order[9:12], order[12:]
        reflected = b[lower].copy()
        reflected[:,1] *= -1
        cost = np.sum((b[upper,None,:]-reflected[None,:,:])**2,axis=2)
        u,v = linear_sum_assignment(cost)
        groups = [[b[k,0],0.,b[k,2]] for k in singles]
        error = float(np.sum(b[singles,1]**2))
        for p,q in zip(u,v):
            tick()
            t = (b[upper[p]]+reflected[q])/2
            groups.append(t.tolist())
            error += float(cost[p,q])
        projections.append((error,np.array(groups)))
    g0 = min(projections,key=lambda t:t[0])[1]
    gx, signs = [], []
    for g in range(12):
        tick()
        gx.append(g)
        signs.append(0. if g < 3 else 1.)
        if g >= 3:
            gx.append(g)
            signs.append(-1.)
    gx = np.array(gx)
    signs = np.array(signs)
    X = np.zeros((21,34)); Y = X.copy(); R = X.copy()
    for k,g in enumerate(gx):
        tick()
        X[k,g] = 1
        R[k,21+g] = 1
        if g >= 3:
            Y[k,12+g-3] = signs[k]
    pairs = []
    seen = set()
    for i in range(21):
        tick()
        for j in range(i):
            tick()
            g,h = sorted((int(gx[i]),int(gx[j])))
            key = (g,h,int(signs[i]*signs[j]))
            if key not in seen:
                seen.add(key)
                pairs.append((i,j))
    pi,pj = np.array(pairs).T
    DX = X[pi]-X[pj]; DY = Y[pi]-Y[pj]; RS = R[pi]+R[pj]
    walls = np.zeros((36,34)); offsets = np.zeros(36)
    for g in range(12):
        tick()
        walls[g,g] = 1
        walls[g,21+g] = -1
        walls[g,33] = .5
        walls[12+g,g] = -1
        walls[12+g,21+g] = -1
        walls[12+g,33] = .5
        walls[24+g,21+g] = -1
        walls[24+g,33] = -.5
        offsets[24+g] = 1
        if g >= 3:
            walls[24+g,12+g-3] = -1
    weights = np.array([1.]*3+[2.]*9)
    grad = np.zeros(34); grad[21:33] = -weights
    release = np.zeros(len(pi))
    z = np.zeros(34)
    z[:12] = g0[:,0]
    z[12:21] = np.maximum(g0[3:,1],0)
    z[21:33] = g0[:,2]
    z[33] = np.max(g0[:,0]+g0[:,2])-np.min(g0[:,0]-g0[:,2])
    bounds = [(-1.,1.)]*12+[(0.,1.)]*9+[(1e-6,.5)]*12+[(.1,1.9)]
    ii,jj = np.triu_indices(21,1)

    def retain(v):
        nonlocal best,best_score
        tick()
        r = v[21:33].copy()
        if not np.isfinite(v).all() or np.any(r <= 0):
            return
        circles = np.column_stack((X@v,Y@v,R@v))
        dx = circles[ii,0]-circles[jj,0]
        dy = circles[ii,1]-circles[jj,1]
        ratio = np.min(np.sqrt(dx*dx+dy*dy)/(circles[ii,2]+circles[jj,2]))
        r *= min(1.,float(ratio))*(1.-5e-15)
        xx = v[:12].copy(); yy = np.r_[np.zeros(3),v[12:21]]
        width = np.max(xx+r)-np.min(xx-r)
        height = 2*np.max(yy+r)
        scale = min(1.,2./(width+height))*(1.-5e-15)
        if float(weights@r)*scale <= best_score:
            return
        obj = {'orbits': [[float(xx[g]*scale),float(yy[g]*scale),float(r[g]*scale),1 if g<3 else 2] for g in range(12)]}
        value = score(obj)
        if value > best_score:
            best,best_score = obj,value

    def fun(v):
        tick()
        return float(grad@v)

    def jac(v):
        tick()
        return grad

    def constraints(v):
        tick()
        dx,dy = DX@v,DY@v
        return np.r_[np.sqrt(dx*dx+dy*dy)-RS@v-release,walls@v+offsets]

    def derivatives(v):
        tick()
        dx,dy = DX@v,DY@v
        d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-25)
        return np.vstack(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RS,walls))

    def solve(v,iterations):
        tick()
        result = minimize(fun,v,jac=jac,bounds=bounds,method='SLSQP',constraints={'type':'ineq','fun':constraints,'jac':derivatives},callback=retain,options={'maxiter':iterations,'ftol':3e-14,'disp':False})
        retain(result.x)
        return result.x

    try:
        z = solve(z,int(params['polish']))
        anchor = z.copy()
        attempt = 0
        while time.monotonic() < deadline:
            tick()
            release[:] = 0
            dx,dy = DX@anchor,DY@anchor
            gaps = np.sqrt(dx*dx+dy*dy)-RS@anchor
            near = np.flatnonzero(gaps < .003)
            if len(near) < 3:
                break
            k = min(int(params['contacts']),len(near))
            chosen = rng.choice(near,k,replace=False)
            release[chosen] = float(params['release'])*rng.uniform(.65,1.35,k)*(1+.35*(attempt%3))
            displaced = solve(anchor.copy(),24)
            release[:] = 0
            recovered = solve(displaced,int(params['polish']))
            if float(grad@recovered) < float(grad@anchor)-1e-9 and np.min(constraints(recovered)) > -1e-8:
                anchor = recovered.copy()
            attempt += 1
    except (Finished, ValueError, FloatingPointError):
        pass
    return best
