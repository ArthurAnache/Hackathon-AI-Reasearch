import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'iterations': (95, 30, 200),
    'band_fraction': (0.3333333333333333, 0.24, 0.42),
    'radius_reset': (0.93, 0.7, 1.0),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class StopSearch(Exception):
    pass

def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return -np.inf
    f = [[Fraction(float(v)) for v in row] for row in a]
    perimeter = (max(x+r for x,y,r in f)-min(x-r for x,y,r in f)
                 +max(y+r for x,y,r in f)-min(y-r for x,y,r in f))
    if perimeter > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))

def decode(z):
    return np.column_stack((z[:N], z[N:2*N], z[2*N:3*N]))

def repair(z):
    a = decode(z)
    if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return None
    d = np.sqrt((a[I, 0]-a[J, 0])**2+(a[I, 1]-a[J, 1])**2)
    factor = min(1.0, float(np.min(d/(a[I, 2]+a[J, 2]))))
    a[:, 2] *= factor*(1.0-3e-14)
    lo = np.min(a[:, :2]-a[:, 2,None], axis=0)
    hi = np.max(a[:, :2]+a[:, 2,None], axis=0)
    scale = min(1.0, 2.0/float(np.sum(hi-lo)))*(1.0-3e-14)
    a[:, :2] -= lo
    a *= scale
    return a

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf
    if init is not None:
        a0 = np.asarray(init['circles'], dtype=float).copy()
        best_score = exact_score(a0)
    else:
        a0 = np.empty((N,3))
        for k in range(N):
            if time.monotonic() >= deadline:
                break
            a0[k] = ((k%5+0.5)/5, (k//5+0.5)/5, 0.04)
        best = {'circles': a0.tolist()}
        best_score = exact_score(a0)
    lo = np.min(a0[:,:2]-a0[:,2,None], axis=0)
    extent = np.max(a0[:,:2]+a0[:,2,None],axis=0)-lo
    scale = 2.0/float(extent.sum())
    a0[:,:2] = (a0[:,:2]-lo)*scale
    a0[:,2] *= scale
    extent *= scale
    objective_grad = np.r_[np.zeros(2*N), -np.ones(N), 0.0]
    rows = np.arange(M)
    eye = np.eye(N)
    boundary_jac = np.zeros((4*N,3*N+1))
    boundary_jac[:N,:N] = eye
    boundary_jac[:N,2*N:3*N] = -eye
    boundary_jac[N:2*N,:N] = -eye
    boundary_jac[N:2*N,2*N:3*N] = -eye
    boundary_jac[N:2*N,-1] = 1
    boundary_jac[2*N:3*N,N:2*N] = eye
    boundary_jac[2*N:3*N,2*N:3*N] = -eye
    boundary_jac[3*N:,N:2*N] = -eye
    boundary_jac[3*N:,2*N:3*N] = -eye
    boundary_jac[3*N:,-1] = -1
    cache_z = None
    cache_c = None
    cache_j = None
    def constraints(z):
        nonlocal cache_z, cache_c, cache_j
        if time.monotonic() >= deadline:
            raise StopSearch()
        if cache_z is not None and np.array_equal(z,cache_z):
            return cache_c
        x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
        dx,dy = x[I]-x[J],y[I]-y[J]
        d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-15)
        c = np.r_[d-r[I]-r[J],x-r,w-x-r,y-r,2-w-y-r]
        jac = np.zeros((M+4*N,3*N+1))
        jac[rows,I] = dx/d
        jac[rows,J] = -dx/d
        jac[rows,N+I] = dy/d
        jac[rows,N+J] = -dy/d
        jac[rows,2*N+I] = -1
        jac[rows,2*N+J] = -1
        jac[M:] = boundary_jac
        cache_z,cache_c,cache_j = z.copy(),c,jac
        return c
    def jacobian(z):
        constraints(z)
        return cache_j
    def objective(z):
        if time.monotonic() >= deadline:
            raise StopSearch()
        return -float(z[2*N:3*N].sum())
    def consider(z):
        nonlocal best,best_score
        if time.monotonic() >= deadline:
            return
        a = repair(z)
        if a is not None:
            score = exact_score(a)
            if score > best_score:
                best_score = score
                best = {'circles':a.tolist()}
    bounds = [(0,2)]*(2*N)+[(1e-7,0.6)]*N+[(0.1,1.9)]
    trial = 0
    while time.monotonic() < deadline:
        p = a0.copy()
        axis = trial % 2
        side = extent[axis]
        f = float(params['band_fraction'])
        if trial >= 2:
            f += float(rng.uniform(-0.055,0.055))
        cuts = np.array([0.0,f*side,(1-f)*side,side])
        labels = np.searchsorted(cuts[1:3],p[:,axis])
        orders = ((1,0,2),(0,2,1),(2,0,1),(1,2,0))
        order = orders[(trial//2)%4]
        offset = 0.0
        for band in order:
            if time.monotonic() >= deadline:
                return best
            mask = labels == band
            p[mask,axis] += offset-cuts[band]
            if trial >= 4 and band == order[1]:
                p[mask,1-axis] = extent[1-axis]-p[mask,1-axis]
            offset += cuts[band+1]-cuts[band]
        p[:,2] *= float(params['radius_reset'])
        z = np.r_[p[:,0],p[:,1],p[:,2],extent[0]]
        cache_z = None
        last = [z]
        def callback(v):
            if time.monotonic() >= deadline:
                raise StopSearch()
            last[0] = v.copy()
        try:
            result = minimize(objective,z,jac=lambda v: objective_grad,
                constraints={'type':'ineq','fun':constraints,'jac':jacobian},
                bounds=bounds,method='SLSQP',callback=callback,
                options={'maxiter':int(params['iterations']),'ftol':2e-11,'disp':False})
            consider(result.x)
        except StopSearch:
            break
        trial += 1
    return best
