import time
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog

PARAMS = {
    'trust': (0.085, 0.02, 0.25),
    'iterations': (22, 8, 60),
    'merge_scale': (0.80, 0.55, 1.0),
    'split_fraction': (0.49, 0.3, 0.5),
}
N = 21
I, J = np.triu_indices(N, 1)


def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2] + a[J, 2]):
        return -np.inf
    x = list(map(lambda v: Fraction(float(v)), a[:, 0]))
    y = list(map(lambda v: Fraction(float(v)), a[:, 1]))
    r = list(map(lambda v: Fraction(float(v)), a[:, 2]))
    w = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
    h = max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
    if w+h > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a = np.array([[0.1+0.2*(k%7), 0.1+0.2*(k//7), 0.1*(1-1e-12)] for k in range(21)])
        init = {'circles': a.tolist()}
    original = np.asarray(init['circles'], dtype=float)
    best = init
    best_score = exact_score(original)
    lo = np.min(original[:, :2]-original[:, 2,None], axis=0)
    hi = np.max(original[:, :2]+original[:, 2,None], axis=0)
    scale = 2.0/np.sum(hi-lo)
    base = original.copy()
    base[:, :2] = (base[:, :2]-lo)*scale
    base[:, 2] *= scale
    width = (hi[0]-lo[0])*scale

    # x, y, radius, rectangle width; rectangle height is 2-width.
    M = len(I)
    A = np.zeros((M+84, 64))
    b = np.zeros(M+84)
    rows = np.arange(M)
    A[rows, 42+I] = 1.0
    A[rows, 42+J] = 1.0
    k = np.arange(21)
    A[M+k, 42+k] = 1
    A[M+k, k] = -1
    A[M+21+k, 42+k] = 1
    A[M+21+k, k] = 1
    A[M+21+k, 63] = -1
    A[M+42+k, 42+k] = 1
    A[M+42+k, 21+k] = -1
    A[M+63+k, 42+k] = 1
    A[M+63+k, 21+k] = 1
    A[M+63+k, 63] = 1
    b[M+63:] = 2
    objective = np.r_[np.zeros(42), -np.ones(21), 0.0]

    d0 = np.linalg.norm(base[I,:2]-base[J,:2], axis=1)
    near = d0 < 1.18*(base[I,2]+base[J,2])
    choices = np.flatnonzero(near)
    costs = base[I[choices],2]+base[J[choices],2]
    choices = choices[np.argsort(costs)]
    big = np.argsort(-base[:,2])

    def consider(z):
        nonlocal best, best_score
        if time.monotonic() >= end:
            return
        a = np.column_stack((z[:21], z[21:42], z[42:63]))
        rr = a[:,2]
        if np.any(rr <= 0):
            return
        dd = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
        factor = min(1.0, float(np.min(dd/(rr[I]+rr[J]))))
        a[:,2] *= factor*(1-3e-14)
        low = np.min(a[:,:2]-a[:,2,None],axis=0)
        high = np.max(a[:,:2]+a[:,2,None],axis=0)
        s = 2/np.sum(high-low)*(1-3e-14)
        a[:,:2] = (a[:,:2]-low)*s
        a[:,2] *= s
        if float(np.sum(a[:,2])) <= best_score:
            return
        val = exact_score(a)
        if val > best_score:
            best_score = val
            best = {'circles':a.tolist()}

    trial = 0
    while time.monotonic() < end - 0.025:
        a = base.copy()
        if trial:
            # The removed pair becomes one circle; its spare slot splits a large disk.
            pool = min(len(choices), 20)
            if not pool:
                break
            edge = choices[(trial-1) % pool] if trial <= pool else rng.choice(choices)
            i, j = int(I[edge]), int(J[edge])
            eligible = big[(big != i) & (big != j)]
            sidx = int(eligible[(trial//pool) % min(7,len(eligible))])
            ri, rj = a[i,2], a[j,2]
            a[i,:2] = (ri*a[i,:2]+rj*a[j,:2])/(ri+rj)
            a[i,2] = float(params['merge_scale'])*(ri+rj)
            center = a[sidx,:2].copy()
            rad = a[sidx,2]
            angle = (trial % 4)*np.pi/4 + rng.uniform(-0.12,0.12)
            direction = np.array([np.cos(angle),np.sin(angle)])
            f = float(params['split_fraction'])
            a[sidx,:2] = center+(1-f)*rad*direction
            a[j,:2] = center-(1-f)*rad*direction
            a[sidx,2] = a[j,2] = f*rad
        z = np.r_[a[:,0],a[:,1],a[:,2],width]
        last = -np.inf
        for iteration in range(int(params['iterations'])):
            now = time.monotonic()
            if now >= end-0.012:
                break
            dx = z[I]-z[J]
            dy = z[21+I]-z[21+J]
            dist = np.maximum(np.hypot(dx,dy),1e-12)
            ux, uy = dx/dist, dy/dist
            A[rows,I] = -ux
            A[rows,J] = ux
            A[rows,21+I] = -uy
            A[rows,21+J] = uy
            trust = float(params['trust'])
            bounds = [(max(-0.2,float(v)-trust),min(2.2,float(v)+trust)) for v in z[:42]]
            bounds += [(1e-7,0.6)]*21+[(0.15,1.85)]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result = linprog(objective,A_ub=A,b_ub=b,bounds=bounds,method='highs',options={'threads':1,'time_limit':max(0.001,end-now-0.006),'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
            if not result.success:
                break
            z = result.x
            value = float(np.sum(z[42:63]))
            if value > best_score-1e-10:
                consider(z)
            if abs(value-last) < 2e-11:
                break
            if iteration >= 9 and value < best_score-0.035:
                break
            last = value
        trial += 1
    return best
