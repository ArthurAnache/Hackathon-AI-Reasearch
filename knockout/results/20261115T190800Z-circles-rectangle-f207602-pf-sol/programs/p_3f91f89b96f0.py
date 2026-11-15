import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {'projection_share': (0.43, 0.2, 0.65), 'lattice_share': (0.34, 0.15, 0.5), 'iterations': (130, 50, 250)}


def canonical(obj):
    if obj is None:
        return None
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
        xs = [Fraction(float(x)) for x in a[:, 0]]
        ys = [Fraction(float(x)) for x in a[:, 1]]
        rs = [Fraction(float(x)) for x in a[:, 2]]
        w = max(x+r for x,r in zip(xs,rs))-min(x-r for x,r in zip(xs,rs))
        h = max(y+r for y,r in zip(ys,rs))-min(y-r for y,r in zip(ys,rs))
        if w+h > 2:
            return -float('inf')
        i,j = np.triu_indices(21,1)
        d = np.sqrt((a[i,0]-a[j,0])**2+(a[i,1]-a[j,1])**2)
        if np.any(d < a[i,2]+a[j,2]):
            return -float('inf')
        return float(np.sum(a[:,2]))
    except Exception:
        return -float('inf')


class SliceEnd(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + .70*budget_s
    best = init
    best_score = score(init)
    best_z = None
    phase_end = deadline
    last = None
    rng = np.random.default_rng(seed)

    def tick():
        if time.monotonic() >= min(phase_end, deadline):
            raise SliceEnd()

    groups = np.r_[np.arange(3), np.repeat(np.arange(3,12),2)]
    signs = np.r_[np.zeros(3), np.tile([1.,-1.],9)]
    X = np.zeros((21,34)); Y = X.copy(); R = X.copy()
    X[np.arange(21),groups] = 1
    R[np.arange(21),21+groups] = 1
    mask = groups >= 3
    Y[np.flatnonzero(mask),groups[mask]+9] = signs[mask]
    ai,aj = np.triu_indices(21,1)
    keys = np.column_stack((np.minimum(groups[ai],groups[aj]),np.maximum(groups[ai],groups[aj]),signs[ai]*signs[aj]))
    _,keep = np.unique(keys,axis=0,return_index=True)
    pi,pj = ai[keep],aj[keep]
    DX,DY,RS = X[pi]-X[pj],Y[pi]-Y[pj],R[pi]+R[pj]
    W = np.zeros((36,34)); offset = np.zeros(36)
    q = np.arange(12)
    W[q,q] = 1; W[q,21+q] = -1; W[q,33] = .5
    W[12+q,q] = -1; W[12+q,21+q] = -1; W[12+q,33] = .5
    W[24+q,21+q] = -1; W[24+q,33] = -.5
    W[27+np.arange(9),12+np.arange(9)] = -1
    offset[24:] = 1
    objective = np.zeros(34)
    objective[21:33] = -np.r_[np.ones(3),np.full(9,2.)]
    bounds = [(-1.,1.)]*12+[(0.,1.)]*9+[(1e-8,.5)]*12+[(.1,1.9)]

    def retain(v):
        nonlocal best,best_score,best_z
        if v is None or not np.isfinite(v).all():
            return
        a = np.column_stack((X@v,Y@v,R@v))
        if np.any(a[:,2] <= 0):
            return
        d = np.sqrt((a[ai,0]-a[aj,0])**2+(a[ai,1]-a[aj,1])**2)
        f = min(1.,float(np.min(d/(a[ai,2]+a[aj,2]))))*(1.-4e-14)
        if f <= 0:
            return
        a[:,2] *= f
        low = np.min(a[:,:2]-a[:,2,None],axis=0)
        high = np.max(a[:,:2]+a[:,2,None],axis=0)
        s = 2./float(np.sum(high-low))*(1.-4e-14)
        a *= s
        if float(np.sum(a[:,2])) <= best_score:
            return
        obj = {'circles':a.tolist()}
        val = score(obj)
        if val > best_score:
            best,best_score = obj,val
            z = v.copy()
            z[:33] *= s
            z[21:33] *= f
            center = .5*(np.max(a[:,0]+a[:,2])+np.min(a[:,0]-a[:,2]))
            z[:12] -= center
            z[33] = float(np.max(a[:,0]+a[:,2])-np.min(a[:,0]-a[:,2]))
            best_z = z

    def fun(v):
        tick()
        return float(objective@v)
    def jac(v):
        tick()
        return objective
    def con(v):
        tick()
        dx,dy = DX@v,DY@v
        return np.r_[np.sqrt(dx*dx+dy*dy)-RS@v,W@v+offset]
    def cjac(v):
        tick()
        dx,dy = DX@v,DY@v
        d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-25)
        return np.vstack(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RS,W))
    count = 0
    def callback(v):
        nonlocal last,count
        last = v.copy()
        count += 1
        if count % 5 == 0:
            retain(v)
        tick()

    def solve(v,until):
        nonlocal phase_end,last,count
        phase_end = min(until,deadline)
        last = v.copy(); count = 0
        try:
            tick()
            out = minimize(fun,v,jac=jac,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':con,'jac':cjac},callback=callback,
                options={'maxiter':int(params['iterations']),'ftol':8e-13,'disp':False})
            last = out.x.copy()
        except (SliceEnd,ValueError,FloatingPointError):
            pass
        retain(last)
        return last.copy()

    def lattice(which):
        z = np.zeros(34)
        if which == 0:
            xx = np.r_[[-2.,0.,2.],[-3.,-1.,1.,3.],[-4.,-2.,0.,2.,4.]]
            yy = np.r_[np.full(4,np.sqrt(3.)),np.full(5,2*np.sqrt(3.))]
        else:
            xx = np.r_[[-2.,0.,2.],[-1.,1.,3.],[-2.,0.,2.],[-1.,1.,3.]]
            yy = np.repeat(np.arange(1,4)*np.sqrt(3.),3)
        w = np.max(xx+1)-np.min(xx-1)
        h = 2*(np.max(yy)+1)
        s = 2/(w+h)*(1-1e-13)
        xx -= (np.max(xx+1)+np.min(xx-1))*.5
        z[:12] = xx*s; z[12:21] = yy*s
        z[21:33] = s; z[33] = w*s
        return z

    structured = lattice(0)
    retain(structured)
    projections = []
    try:
        a = np.asarray(canonical(init)['circles'],float).copy()
        if a.shape == (21,3) and np.isfinite(a).all():
            low = np.min(a[:,:2]-a[:,2,None],axis=0)
            high = np.max(a[:,:2]+a[:,2,None],axis=0)
            a[:,:2] -= (low+high)*.5
            for swap in (False,True):
                if time.monotonic() >= deadline:
                    break
                b = a[:,[1,0,2]].copy() if swap else a.copy()
                order = np.argsort(b[:,1])
                lower,axis,upper = order[:9],order[9:12],order[12:]
                reflected = b[lower].copy(); reflected[:,1] *= -1
                cost = np.sum((b[upper,None,:2]-reflected[None,:,:2])**2,axis=2)
                u,v = linear_sum_assignment(cost)
                g = np.vstack((b[axis],.5*(b[upper[u]]+reflected[v])))
                z = np.zeros(34)
                z[:12] = g[:,0]; z[12:21] = np.abs(g[3:,1]); z[21:33] = np.maximum(g[:,2],1e-5)
                z[33] = np.clip(high[1 if swap else 0]-low[1 if swap else 0],.1,1.9)
                err = np.sum(cost[u,v])+np.sum(b[axis,1]**2)
                projections.append((err,z))
    except (ValueError,TypeError,KeyError,IndexError):
        pass

    duration = .70*budget_s
    if projections:
        projections.sort(key=lambda p:p[0])
        projected = projections[0][1]
        retain(projected)
        solve(projected, started+duration*params['projection_share'])
    else:
        solve(structured,started+duration*params['projection_share'])
    if time.monotonic() < deadline:
        solve(structured,started+duration*(params['projection_share']+params['lattice_share']))
    if time.monotonic() < deadline:
        anchor = best_z.copy() if best_z is not None else lattice(1)
        solve(anchor,deadline)
    return best
