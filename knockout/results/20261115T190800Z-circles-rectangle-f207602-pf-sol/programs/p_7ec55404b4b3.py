import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'fraction': (0.70, 0.5, 0.72),
    'first_share': (0.35, 0.2, 0.5),
    'second_share': (0.35, 0.2, 0.5),
    'ftol': (8e-13, 1e-14, 1e-8, 'log'),
    'safety': (2e-14, 5e-15, 1e-10, 'log')
}
I, J = np.triu_indices(21, 1)

class StopSearch(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -float('inf')
    dx = a[I, 0] - a[J, 0]
    dy = a[I, 1] - a[J, 1]
    if np.any(np.sqrt(dx*dx + dy*dy) < a[I, 2] + a[J, 2]):
        return -float('inf')
    xs = [Fraction(float(v)) for v in a[:, 0]]
    ys = [Fraction(float(v)) for v in a[:, 1]]
    rs = [Fraction(float(v)) for v in a[:, 2]]
    p = (max(x+r for x,r in zip(xs,rs)) - min(x-r for x,r in zip(xs,rs)) +
         max(y+r for y,r in zip(ys,rs)) - min(y-r for y,r in zip(ys,rs)))
    return float(np.sum(a[:, 2])) if p <= 2 else -float('inf')

def make_model(sym):
    n = 34 if sym else 64
    x = np.zeros((21,n)); y = np.zeros_like(x); r = np.zeros_like(x)
    if sym:
        ids = np.r_[np.arange(12),np.arange(3,12)]
        signs = np.r_[np.ones(12),-np.ones(9)]
        x[np.arange(21),ids] = 1
        mask = ids >= 3
        y[np.arange(21)[mask],12+ids[mask]-3] = signs[mask]
        r[np.arange(21),21+ids] = 1
        pi,pj = np.triu_indices(12,1)
        aa,bb = np.triu_indices(9,0)
        pi = np.r_[pi,aa+3]; pj = np.r_[pj,bb+12]
        walls = np.arange(12)
        radii = np.arange(21,33)
    else:
        q = np.arange(21)
        x[q,q] = 1; y[q,q+21] = 1; r[q,q+42] = 1
        pi,pj = I,J
        walls = q
        radii = np.arange(42,63)
    w = np.zeros(n); w[-1] = .5
    b = np.concatenate((w+x[walls]-r[walls],w-x[walls]-r[walls],
                        -w+y[walls]-r[walls],-w-y[walls]-r[walls]))
    off = np.r_[np.zeros(2*len(walls)),np.ones(2*len(walls))]
    return x,y,r,x[pi]-x[pj],y[pi]-y[pj],r[pi]+r[pj],b,off,-r.sum(axis=0),radii

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + float(params['fraction'])*budget_s
    eps = float(params['safety'])
    best = None
    best_value = -float('inf')
    best_a = None

    def offer(a, repair=True):
        nonlocal best,best_value,best_a
        a = np.array(a,dtype=float,copy=True)
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        if repair:
            ds = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
            factor = min(1.,float(np.min(ds/(a[I,2]+a[J,2]))))
            if factor <= 0:
                return
            a[:,2] *= factor*(1-eps)
            lo = np.min(a[:,:2]-a[:,2,None],axis=0)
            hi = np.max(a[:,:2]+a[:,2,None],axis=0)
            scale = (2-8*eps)/float(np.sum(hi-lo))
            a[:,:2] -= .5*(lo+hi)
            a *= scale
        value = exact_score(a)
        if value > best_value:
            best_value = value
            best_a = a.copy()
            best = {'state':a.T.reshape(-1).tolist()}

    original = None
    if init is not None:
        try:
            original = (np.array(init['circles'],dtype=float) if 'circles' in init else
                        np.array(init['state'],dtype=float).reshape(3,21).T.copy())
            value = exact_score(original)
            if np.isfinite(value):
                best = init
                best_value = value
                best_a = original.copy()
            offer(original)
        except (ValueError,KeyError,TypeError):
            original = None

    def unpack(z,m):
        return np.column_stack((m[0]@z,m[1]@z,m[2]@z))

    def encode(a,sym):
        a = a.copy()
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        a[:,:2] -= .5*(hi+lo)
        scale = 2/float(np.sum(hi-lo))
        a *= scale
        width = float(hi[0]-lo[0])*scale
        if sym:
            return np.r_[a[:12,0],a[3:12,1],a[:12,2],width]
        return np.r_[a[:,0],a[:,1],a[:,2],width]

    def solve(z,m,stop):
        x,y,r,dxm,dym,rsm,b,off,obj,ridx = m
        cached = [None,None,None]
        accepted = [z.copy()]
        count = [0]
        def check():
            if time.monotonic() >= stop:
                raise StopSearch()
        def calc(v):
            check()
            if cached[0] is not None and np.array_equal(cached[0],v):
                return cached[1],cached[2]
            dx = dxm@v; dy = dym@v
            d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-16)
            c = np.r_[d-rsm@v,b@v+off]
            jac = np.concatenate(((dx/d)[:,None]*dxm+(dy/d)[:,None]*dym-rsm,b))
            cached[:] = [v.copy(),c,jac]
            return c,jac
        def objective(v):
            check()
            return float(obj@v),obj
        def callback(v):
            check()
            accepted[0] = v.copy()
            count[0] += 1
            if count[0]%10 == 0:
                offer(unpack(v,m))
        bounds = [(None,None)]*len(z)
        for k in ridx:
            if time.monotonic() >= stop:
                return z
            bounds[int(k)] = (1e-7,.5)
        bounds[-1] = (.08,1.92)
        try:
            result = minimize(objective,z,jac=True,method='SLSQP',bounds=bounds,
                              constraints={'type':'ineq','fun':lambda v:calc(v)[0],
                                           'jac':lambda v:calc(v)[1]},
                              callback=callback,
                              options={'maxiter':240,'ftol':float(params['ftol']),'disp':False})
            accepted[0] = result.x
        except StopSearch:
            pass
        offer(unpack(accepted[0],m))
        return accepted[0]

    # Three odd columns give the three axial circles; the others form nine pairs.
    singles = []; upper = []
    for col,count in enumerate((3,4,3,4,3,4)):
        if time.monotonic() >= deadline:
            break
        xx = np.sqrt(3.)*col
        if count == 3:
            singles.append([xx,0.,1.])
            upper.append([xx,2.,1.])
        else:
            upper.extend([[xx,1.,1.],[xx,3.,1.]])
    seeds = []
    if len(singles)==3 and len(upper)==9:
        top = np.array(singles+upper)
        seeds.append(np.vstack((top,top[3:]*np.array([1.,-1.,1.]))))
    # A wider five-row hexagonal patch offers a different contact topology.
    top = np.array([[0.,0.,1.],[2.,0.,1.],[4.,0.,1.]]+
                   [[2*k-1.,np.sqrt(3.),1.] for k in range(4)]+
                   [[2*k-2.,2*np.sqrt(3.),1.] for k in range(5)])
    seeds.append(np.vstack((top,top[3:]*np.array([1.,-1.,1.]))))
    for a in seeds:
        if time.monotonic() >= deadline:
            break
        offer(a)
    if best is None:
        fallback = np.array([[.2*(k%3),.2*(k//3),.09] for k in range(21)])
        offer(fallback)

    full = make_model(False)
    if best_value > 2.35 and original is not None and exact_score(original) > 2.35:
        solve(encode(best_a,False),full,deadline-.015)
        return best

    sym = make_model(True)
    duration = deadline-started
    cuts = [started+duration*float(params['first_share']),
            started+duration*(float(params['first_share'])+float(params['second_share']))]
    for a,cut in zip(seeds,cuts):
        if time.monotonic() >= deadline-.05:
            break
        solve(encode(a,True),sym,min(cut,deadline-.08))
    if time.monotonic() < deadline-.025 and best_a is not None:
        solve(encode(best_a,False),full,deadline-.012)
    return best
