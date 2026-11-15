import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {'perturbation': (0.065, 0.005, 0.2), 'iterations': (65, 15, 150), 'fraction': (0.69, 0.4, 0.75)}

class StopSearch(Exception):
    pass

def exact_score(data):
    try:
        a = np.asarray(data['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -np.inf
        xs = [Fraction(float(v)) for v in a[:, 0]]
        ys = [Fraction(float(v)) for v in a[:, 1]]
        rs = [Fraction(float(v)) for v in a[:, 2]]
        p = max(x+r for x,r in zip(xs,rs))-min(x-r for x,r in zip(xs,rs))
        p += max(y+r for y,r in zip(ys,rs))-min(y-r for y,r in zip(ys,rs))
        if p > 2:
            return -np.inf
        for i in range(21):
            for j in range(i):
                dx = a[i,0]-a[j,0]
                dy = a[i,1]-a[j,1]
                if np.sqrt(dx*dx+dy*dy) < a[i,2]+a[j,2]:
                    return -np.inf
        return float(np.sum(a[:,2]))
    except Exception:
        return -np.inf

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + float(params['fraction'])*budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        init = {'circles': [[(i%7-3)*0.2,(i//7-1)*0.2,0.09999999999999] for i in range(21)]}
    best = init
    bestscore = exact_score(init)
    a = np.array(init['circles'], dtype=float)
    lo = np.min(a[:,:2]-a[:,2,None],axis=0)
    hi = np.max(a[:,:2]+a[:,2,None],axis=0)
    a[:,:2] -= (lo+hi)/2
    def check():
        if time.monotonic() >= deadline:
            raise StopSearch()
    def reflection_error(axis):
        b = a.copy()
        b[:,axis] *= -1
        d = np.sum((a[:,None,:]-b[None,:,:])**2,axis=2)
        return float(np.sum(np.min(d,axis=1)))
    swapped = reflection_error(0) < reflection_error(1)
    if swapped:
        a[:,[0,1]] = a[:,[1,0]]
    b = a.copy()
    b[:,1] *= -1
    dd = np.sum((a[:,None,:]-b[None,:,:])**2,axis=2)
    mate = np.argmin(dd,axis=1)
    symmetric = all(mate[mate[i]] == i for i in range(21)) and float(np.max(np.min(dd,axis=1))) < 1e-10
    groups = []
    used = set()
    for i in range(21):
        check()
        if i in used:
            continue
        if symmetric and mate[i] != i:
            j = int(mate[i])
            if a[i,1] < a[j,1]:
                i,j = j,i
            groups.append((i,j))
            used.update((i,j))
        else:
            groups.append((i,))
            used.add(i)
    cols = []
    values = []
    bounds = []
    position_indices = []
    for group in groups:
        check()
        i = group[0]
        axes = (0,2) if symmetric and len(group)==1 else (0,1,2)
        for axis in axes:
            v = np.zeros((21,3))
            v[i,axis] = 1
            if len(group)==2:
                v[group[1],axis] = -1 if axis==1 else 1
            cols.append(v.reshape(-1))
            if len(group)==2:
                val = (a[i,axis] + (-a[group[1],axis] if axis==1 else a[group[1],axis]))/2
            else:
                val = a[i,axis]
            values.append(val)
            bounds.append((1e-7,None) if axis==2 else ((0,None) if axis==1 and len(group)==2 else (None,None)))
            if axis != 2:
                position_indices.append(len(values)-1)
    cols.append(np.zeros(63))
    width = float(np.max(a[:,0]+a[:,2])-np.min(a[:,0]-a[:,2]))
    values.append(width)
    bounds.append((0.2,1.8))
    B = np.stack(cols,axis=1).reshape(21,3,-1)
    q0 = np.array(values)
    nv = len(q0)
    mirror = mate if symmetric else np.arange(21)
    pairs = []
    for i in range(21):
        check()
        for j in range(i):
            p = (j,i)
            other = tuple(sorted((int(mirror[i]),int(mirror[j]))))
            if not symmetric or p <= other:
                pairs.append(p)
    ii = np.array([p[0] for p in pairs])
    jj = np.array([p[1] for p in pairs])
    DX = B[ii,0]-B[jj,0]
    DY = B[ii,1]-B[jj,1]
    RR = B[ii,2]+B[jj,2]
    walls = []
    offsets = []
    for i in range(21):
        check()
        for axis in (0,1):
            for sign in (-1,1):
                row = sign*B[i,axis]-B[i,2]
                row = row.copy()
                row[-1] += 0.5 if axis==0 else -0.5
                walls.append(row)
                offsets.append(0.0 if axis==0 else 1.0)
    aug = np.column_stack((np.array(walls),np.array(offsets)))
    aug = np.unique(aug,axis=0)
    W = aug[:,:-1]
    off = aug[:,-1]
    obj = -np.sum(B[:,2],axis=0)
    latest = q0.copy()
    def fun(q):
        check()
        return float(obj@q)
    def jac(q):
        check()
        return obj
    def cons(q):
        check()
        dx,dy,rr = DX@q,DY@q,RR@q
        return np.concatenate((dx*dx+dy*dy-rr*rr,W@q+off))
    def cjac(q):
        check()
        dx,dy,rr = DX@q,DY@q,RR@q
        return np.vstack((2*(dx[:,None]*DX+dy[:,None]*DY-rr[:,None]*RR),W))
    def callback(q):
        nonlocal latest
        latest = q.copy()
        check()
    def consider(q):
        nonlocal best,bestscore
        c = np.einsum('ijk,k->ij',B,q)
        if not np.isfinite(c).all() or np.any(c[:,2]<=0):
            return
        dx = c[ii,0]-c[jj,0]
        dy = c[ii,1]-c[jj,1]
        # Check all pairs, including symmetry duplicates, before exact verification.
        ids, jds = np.triu_indices(21,1)
        distances = np.sqrt(np.sum((c[ids,:2]-c[jds,:2])**2,axis=1))
        factor = min(1.0,float(np.min(distances/(c[ids,2]+c[jds,2]))))
        c[:,2] *= max(0.0,factor)*(1-4e-15)
        if np.any(c[:,2]<=0):
            return
        per = float(np.sum(np.max(c[:,:2]+c[:,2,None],axis=0)-np.min(c[:,:2]-c[:,2,None],axis=0)))
        c *= min(1.0,2.0/per)*(1-4e-15)
        if swapped:
            c[:,[0,1]] = c[:,[1,0]]
        candidate = {'circles':c.tolist()}
        score = exact_score(candidate)
        if score > bestscore:
            best,bestscore = candidate,score
    anchor = q0.copy()
    trial = 0
    try:
        while time.monotonic() < deadline:
            check()
            q = anchor.copy()
            if trial:
                scale = float(params['perturbation'])*[0.45,1.0,1.8,0.8][(trial-1)%4]
                count = min(len(groups),3+(trial%4))
                chosen = rng.choice(len(groups),count,replace=False)
                selected = set()
                for g in chosen:
                    check()
                    selected.update(groups[int(g)])
                for k in position_indices:
                    check()
                    if any(np.any(B[i,:,k]) for i in selected):
                        q[k] += rng.normal()*scale
                for k,(lower,upper) in enumerate(bounds):
                    check()
                    if lower is not None:
                        q[k] = max(q[k],lower+1e-9)
                    if upper is not None:
                        q[k] = min(q[k],upper-1e-9)
            latest = q.copy()
            result = minimize(fun,q,jac=jac,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':cons,'jac':cjac},callback=callback,
                options={'maxiter':int(params['iterations']),'ftol':2e-13,'disp':False})
            consider(result.x)
            if trial==0 and np.min(cons(result.x)) > -1e-8:
                anchor = result.x.copy()
            trial += 1
    except StopSearch:
        pass
    consider(latest)
    return best
