import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'first_fraction': (0.34, 0.2, 0.45),
    'second_fraction': (0.18, 0.1, 0.25),
    'finish_fraction': (0.71, 0.65, 0.75),
    'jitter': (0.008, 0.0, 0.04),
    'maxiter': (220, 50, 400)
}
N = 21
II, JJ = np.triu_indices(N, 1)

class StopSearch(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[II, 0] - a[JJ, 0])**2 + (a[II, 1] - a[JJ, 1])**2)
    if np.any(d < a[II, 2] + a[JJ, 2]):
        return -np.inf
    x = list(map(Fraction, map(float, a[:, 0])))
    y = list(map(Fraction, map(float, a[:, 1])))
    r = list(map(Fraction, map(float, a[:, 2])))
    p = (max(u+v for u,v in zip(x,r)) - min(u-v for u,v in zip(x,r))
         + max(u+v for u,v in zip(y,r)) - min(u-v for u,v in zip(y,r)))
    return float(np.sum(a[:, 2])) if p <= 2 else -np.inf

def model(symmetric):
    m = 34 if symmetric else 64
    X = np.zeros((21, m)); Y = X.copy(); R = X.copy()
    if symmetric:
        ids = np.r_[np.arange(12), np.arange(3, 12)]
        X[np.arange(21), ids] = 1
        R[np.arange(21), 21 + ids] = 1
        Y[np.arange(3, 12), np.arange(12, 21)] = 1
        Y[np.arange(12, 21), np.arange(12, 21)] = -1
        pairs = []
        for i, j in zip(II, JJ):
            if j < 12 or (3 <= i < 12 and j >= 12 and i <= j-9):
                pairs.append((i,j))
        pi, pj = np.array(pairs).T
        representatives = np.arange(12)
    else:
        X[np.arange(21), np.arange(21)] = 1
        Y[np.arange(21), 21+np.arange(21)] = 1
        R[np.arange(21), 42+np.arange(21)] = 1
        pi, pj = II, JJ
        representatives = np.arange(21)
    DX = X[pi]-X[pj]; DY = Y[pi]-Y[pj]; RS = R[pi]+R[pj]
    W = np.zeros(m); W[-1] = 0.5
    B = np.concatenate((W-X[representatives]-R[representatives],
                        W+X[representatives]-R[representatives],
                        -W-Y[representatives]-R[representatives],
                        -W+Y[representatives]-R[representatives]))
    c = np.r_[np.zeros(2*len(representatives)), np.ones(2*len(representatives))]
    unique = np.unique(np.column_stack((B,c)), axis=0)
    B, c = unique[:, :-1], unique[:, -1]
    objective = -R.sum(axis=0)
    bounds = [(None,None)]*m
    if symmetric:
        bounds[12:21] = [(1e-6, 1.0)]*9
        bounds[21:33] = [(1e-6, 0.5)]*12
    else:
        bounds[42:63] = [(1e-6, 0.5)]*21
    bounds[-1] = (0.25, 1.75)
    return X,Y,R,DX,DY,RS,B,c,objective,bounds

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + float(params['finish_fraction'])*budget_s
    rng = np.random.default_rng(seed)
    best = None; best_score = -np.inf; best_array = None
    if init is not None:
        try:
            a = np.asarray(init['circles'], dtype=float)
            value = exact_score(a)
            if np.isfinite(value):
                best, best_score, best_array = init, value, a.copy()
        except (ValueError, KeyError, TypeError):
            pass

    def consider(a):
        nonlocal best, best_score, best_array
        if time.monotonic() >= end:
            return
        if a.shape != (21,3) or not np.isfinite(a).all():
            return
        a = a.copy()
        a[:,2] = np.maximum(a[:,2], 1e-8)
        d = np.sqrt((a[II,0]-a[JJ,0])**2 + (a[II,1]-a[JJ,1])**2)
        factor = min(1.0, float(np.min(d/(a[II,2]+a[JJ,2]))))
        if factor <= 0:
            return
        a[:,2] *= factor*(1-2e-13)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        perimeter = float(np.sum(hi-lo))
        if perimeter <= 0:
            return
        a[:,:2] -= (hi+lo)*0.5
        a *= (2.0/perimeter)*(1-2e-13)
        if float(a[:,2].sum()) <= best_score:
            return
        value = exact_score(a)
        if value > best_score:
            best_score = value
            best_array = a.copy()
            best = {'circles': a.tolist()}

    fallback = np.array([[(j-1)*0.2,(i-3)*0.2,0.1] for i in range(7) for j in range(3)],float)
    consider(fallback)

    def optimize(z, symmetric, deadline):
        if time.monotonic() >= deadline:
            return
        X,Y,R,DX,DY,RS,B,c,obj,bounds = model(symmetric)
        cached_z = None; cached_values = None
        def check():
            if time.monotonic() >= deadline:
                raise StopSearch()
        def mv(A,v):
            return np.einsum('ij,j->i',A,v,optimize=False)
        def values(v):
            nonlocal cached_z, cached_values
            check()
            if cached_z is None or not np.array_equal(v,cached_z):
                dx,dy,sr = mv(DX,v),mv(DY,v),mv(RS,v)
                con = np.r_[dx*dx+dy*dy-sr*sr, mv(B,v)+c]
                jac = np.concatenate((2*(dx[:,None]*DX+dy[:,None]*DY-sr[:,None]*RS),B))
                cached_z = v.copy(); cached_values = (con,jac)
            return cached_values
        def fun(v):
            check()
            return float(np.sum(obj*v)),obj
        def callback(v):
            check()
            consider(np.column_stack((mv(X,v),mv(Y,v),mv(R,v))))
        try:
            callback(z)
            res = minimize(fun,z,jac=True,bounds=bounds,method='SLSQP',
                           constraints={'type':'ineq','fun':lambda v: values(v)[0],
                                        'jac':lambda v: values(v)[1]},
                           callback=callback,
                           options={'maxiter':int(params['maxiter']),'ftol':2e-11,'disp':False})
            callback(res.x)
        except StopSearch:
            pass

    # Three midline singles, then two upper rows of five and four circles.
    w = 1.08
    x = np.r_[[-0.34,0,0.34],np.linspace(-0.43,0.43,5),np.linspace(-0.39,0.39,4)]
    y = np.r_[np.full(5,0.19),np.full(4,0.365)]
    x += rng.normal(0,float(params['jitter']),12)
    z = np.r_[x,y,np.full(12,0.075),w]
    first_end = min(end,start+float(params['first_fraction'])*budget_s)
    optimize(z,True,first_end)

    # Independent three-column, seven-row quotient, with staggered rows.
    if time.monotonic() < end:
        w = 0.72
        x = np.r_[[-0.23,0,0.23],[-0.23,0,0.23],[-0.23,0,0.23],[-0.23,0,0.23]]
        x[3:6] += 0.025
        x[9:12] -= 0.025
        x += rng.normal(0,float(params['jitter']),12)
        y = np.repeat([0.17,0.34,0.51],3)
        z = np.r_[x,y,np.full(12,0.08),w]
        second_end = min(end,start+(float(params['first_fraction'])+float(params['second_fraction']))*budget_s)
        optimize(z,True,second_end)

    # Release all symmetry constraints for the most promising verified point.
    if best_array is not None and time.monotonic() < end:
        a = best_array.copy()
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        a[:,:2] -= (hi+lo)*0.5
        width = float(hi[0]-lo[0])
        z = np.r_[a[:,0],a[:,1],a[:,2],width]
        optimize(z,False,end)
    if best is None:
        fallback[:,2] *= 1-1e-12
        best = {'circles':fallback.tolist()}
    return best
