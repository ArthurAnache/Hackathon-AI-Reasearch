import time
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog

PARAMS = {
    'angle': (1.22, 0.35, 2.5),
    'pairs': (3, 2, 5),
    'sweeps': (11, 4, 25),
    'trust': (0.16, 0.04, 0.5),
}
N = 21
I, J = np.triu_indices(N, 1)


def valid(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return False
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return False
    x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
    w = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
    h = max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
    return w+h <= 2


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.69*budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a = np.array([[(k%5+0.5)/5, (k//5+0.5)/5, 0.035] for k in range(N)])
        best = {'circles': a.tolist()}
    else:
        a = np.array(init['circles'], dtype=float)
        best = init
    best_score = float(a[:,2].sum()) if valid(a) else -np.inf
    lo = (a[:,:2]-a[:,2,None]).min(axis=0)
    sides = (a[:,:2]+a[:,2,None]).max(axis=0)-lo
    base = a.copy()
    base[:,:2] -= lo
    base *= 2.0/sides.sum()
    width = sides[0]*2.0/sides.sum()
    center = np.array([width/2, (2-width)/2])

    # Identify the best reflection involution in the supplied packing.
    choices = []
    for axis in range(2):
        if time.monotonic() >= end:
            return best
        q = base[:,:2].copy()
        q[:,axis] = 2*center[axis]-q[:,axis]
        dd = ((q[:,None,:]-base[None,:,:2])**2).sum(axis=2)
        mate = dd.argmin(axis=1)
        choices.append((float(dd[np.arange(N),mate].sum()), axis, mate))
    err, axis, mate = min(choices, key=lambda t:t[0])
    symmetric = err < 1e-10 and np.all(mate[mate] == np.arange(N))

    m = len(I)
    # Variables are x[21], y[21], r[21], width.
    A = np.zeros((m+4*N, 64))
    rows = np.arange(m)
    A[rows,42+I] = 1
    A[rows,42+J] = 1
    for k in range(N):
        b = m+4*k
        A[b,k] = -1; A[b,42+k] = 1
        A[b+1,k] = 1; A[b+1,42+k] = 1; A[b+1,63] = -1
        A[b+2,21+k] = -1; A[b+2,42+k] = 1
        A[b+3,21+k] = 1; A[b+3,42+k] = 1; A[b+3,63] = 1
    rhs = np.zeros(m+4*N)
    rhs[m+3::4] = 2
    obj = np.r_[np.zeros(42), -np.ones(N), 0.0]

    def certify(z):
        nonlocal best, best_score
        if time.monotonic() >= end:
            return
        out = np.column_stack((z[:21],z[21:42],z[42:63]))
        if float(out[:,2].sum()) <= best_score:
            return
        # Uniform contraction of radii clears solver and arithmetic residuals.
        p = out[:,:2]
        r = out[:,2]
        if np.any(r <= 0):
            return
        dist = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
        ratio = float(np.min(dist/(r[I]+r[J])))
        out[:,2] *= min(1.0, ratio)*(1-3e-13)
        for unused in range(5):
            if time.monotonic() >= end:
                return
            if valid(out):
                score = float(out[:,2].sum())
                if score > best_score:
                    best_score = score
                    best = {'circles': out.tolist()}
                return
            out[:,2] *= 1-2e-10

    gap = np.sqrt(((base[I,:2]-base[J,:2])**2).sum(axis=1))-base[I,2]-base[J,2]
    edges = np.where(gap < 0.018)[0]
    attempt = 0
    while time.monotonic() < end:
        p = base[:,:2].copy()
        if attempt:
            used = set()
            order = rng.permutation(edges)
            count = 0
            for e in order:
                if time.monotonic() >= end:
                    break
                i,j = int(I[e]),int(J[e])
                group = {i,j}
                if symmetric:
                    group |= {int(mate[i]),int(mate[j])}
                if used.intersection(group):
                    continue
                theta = float(params['angle'])*rng.choice([-1,1])*rng.uniform(.75,1.25)
                c,s = np.cos(theta),np.sin(theta)
                R = np.array([[c,-s],[s,c]])
                weights = base[[i,j],2]
                pivot = np.average(p[[i,j]],axis=0,weights=weights)
                moved = (p[[i,j]]-pivot)@R.T+pivot
                displacement = np.zeros_like(p)
                displacement[[i,j]] = moved-p[[i,j]]
                if symmetric:
                    reflection = displacement[mate].copy()
                    reflection[:,axis] *= -1
                    displacement = .5*(displacement+reflection)
                p += displacement
                used |= group
                count += 1
                if count >= int(params['pairs']):
                    break
            p = np.clip(p, 1e-5, np.array([width,2-width])-1e-5)
        z = np.r_[p[:,0],p[:,1],base[:,2],width]
        previous = -np.inf
        for sweep in range(int(params['sweeps'])):
            if time.monotonic() >= end:
                break
            dx = z[I]-z[J]
            dy = z[21+I]-z[21+J]
            d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-15)
            A[rows,I] = -dx/d
            A[rows,J] = dx/d
            A[rows,21+I] = -dy/d
            A[rows,21+J] = dy/d
            trust = float(params['trust']) if sweep else .3
            bounds = [(max(0,float(v)-trust),min(2,float(v)+trust)) for v in z[:42]]
            bounds += [(1e-8, .5)]*N+[(.1,1.9)]
            result = linprog(obj,A_ub=A,b_ub=rhs,bounds=bounds,method='highs',options={'time_limit':max(.001,end-time.monotonic()),'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
            if not result.success:
                break
            z = result.x
            score = float(z[42:63].sum())
            if score > best_score:
                certify(z)
            if abs(score-previous) < 2e-11:
                break
            previous = score
        attempt += 1
    return best
