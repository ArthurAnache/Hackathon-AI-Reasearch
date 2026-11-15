import time
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog

PARAMS = {
    'trust': (0.13, 0.03, 0.3),
    'steps': (18, 6, 40),
    'samples': (1300, 300, 4000),
    'transport': (0.17, 0.05, 0.4),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)


def exact(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return -np.inf
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    perimeter = (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
                 +max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)))
    return float(a[:, 2].sum()) if perimeter <= 2 else -np.inf


def repair(a):
    a = a.copy()
    if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return None
    d = np.sqrt((a[I, 0]-a[J, 0])**2+(a[I, 1]-a[J, 1])**2)
    factor = min(1.0, float(np.min(d/(a[I, 2]+a[J, 2]))))
    if factor <= 0:
        return None
    a[:, 2] *= factor * (1-8e-15)
    lo = (a[:, :2]-a[:, 2,None]).min(axis=0)
    hi = (a[:, :2]+a[:, 2,None]).max(axis=0)
    a[:, :2] -= lo
    a *= min(1.0, 2.0/float((hi-lo).sum()))*(1-8e-15)
    return a


def normalize(a):
    lo = (a[:, :2]-a[:, 2,None]).min(axis=0)
    hi = (a[:, :2]+a[:, 2,None]).max(axis=0)
    s = 2.0/(hi-lo).sum()
    p = (a[:, :2]-lo)*s
    return np.r_[p[:,0],p[:,1],a[:,2]*s,(hi[0]-lo[0])*s]


def unpack(z):
    return np.column_stack((z[:N], z[N:2*N], z[2*N:3*N]))


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.69*budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a = np.array([[(k%5+.5)*.2,(k//5+.5)*.2,.075] for k in range(N)])
        best = {'circles':a.tolist()}
    else:
        a = np.asarray(init['circles'], dtype=float).copy()
        best = init
    best_score = exact(a)
    base = normalize(a)
    best_z = base.copy()
    objective = np.r_[np.zeros(2*N),-np.ones(N),0.0]
    # Pair inequalities are supporting halfspaces of the exterior of each disk.
    A = np.zeros((M+4*N, 3*N+1))
    B = np.zeros(M+4*N)
    q = np.arange(M)
    A[q,2*N+I] = 1
    A[q,2*N+J] = 1
    k = np.arange(N)
    for group in range(4):
        rows = M+group*N+k
        A[rows,2*N+k] = 1
        if group == 0:
            A[rows,k] = -1
        elif group == 1:
            A[rows,k] = 1
            A[rows,-1] = -1
        elif group == 2:
            A[rows,N+k] = -1
        else:
            A[rows,N+k] = 1
            A[rows,-1] = 1
            B[rows] = 2

    def consider(z):
        nonlocal best, best_score, best_z
        if time.monotonic() >= deadline:
            return
        c = repair(unpack(z))
        if c is None or float(c[:,2].sum()) <= best_score:
            return
        value = exact(c)
        if value > best_score:
            best_score = value
            best = {'circles': c.tolist()}
            best_z = normalize(c)

    def improve(z, steps):
        trust = float(params['trust'])
        last = -1e30
        for it in range(steps):
            now = time.monotonic()
            if now >= deadline-0.015:
                break
            dx = z[I]-z[J]
            dy = z[N+I]-z[N+J]
            d = np.maximum(np.hypot(dx,dy),1e-15)
            ux, uy = dx/d, dy/d
            A[q,I] = -ux
            A[q,J] = ux
            A[q,N+I] = -uy
            A[q,N+J] = uy
            bounds = [(float(v-trust),float(v+trust)) for v in z[:2*N]]
            bounds += [(1e-8,0.6)]*N+[(0.25,1.75)]
            remaining = max(0.001, deadline-time.monotonic()-0.008)
            result = linprog(objective, A_ub=A, b_ub=B, bounds=bounds,
                             method='highs', options={'time_limit':remaining,
                             'dual_feasibility_tolerance':1e-8,
                             'primal_feasibility_tolerance':1e-8})
            if time.monotonic() >= deadline:
                break
            if not result.success:
                break
            z = result.x
            score = float(z[2*N:3*N].sum())
            consider(z)
            if score-last < 2e-10 and it >= 4:
                break
            last = score
        return z

    # Start with real vacancy transport rather than a small displacement.
    attempt = 0
    while time.monotonic() < deadline-0.035:
        z = base.copy() if attempt < 4 else best_z.copy()
        p = np.column_stack((z[:N],z[N:2*N]))
        r = z[2*N:3*N].copy()
        w, h = z[-1], 2-z[-1]
        order = np.argsort(r)
        if attempt == 0:
            removed = [int(order[0])]
        elif attempt == 1:
            removed = [int(order[1]),int(order[3])]
        elif attempt % 3 == 2:
            anchor = int(order[int(rng.integers(0,8))])
            nearby = np.argsort(np.linalg.norm(p-p[anchor],axis=1))
            removed = list(map(int,nearby[:3]))
        else:
            removed = list(map(int,rng.choice(order[:12],size=2,replace=False)))
        kept = [i for i in range(N) if i not in removed]
        old = p[removed].copy()
        success = True
        for slot, idx in enumerate(removed):
            if time.monotonic() >= deadline-0.025:
                success = False
                break
            count = int(params['samples'])
            samples = rng.random((count,2))*[w,h]
            # Include a coarse boundary grid to represent corner and edge voids.
            grid = np.linspace(0.025,0.975,23)
            edges = np.vstack((np.column_stack((grid,np.full(23,.025))),
                               np.column_stack((grid,np.full(23,.975))),
                               np.column_stack((np.full(23,.025),grid)),
                               np.column_stack((np.full(23,.975),grid))))*[w,h]
            samples = np.vstack((samples,edges))
            clearance = np.minimum.reduce((samples[:,0],w-samples[:,0],samples[:,1],h-samples[:,1]))
            if kept:
                dist = np.sqrt(np.sum((samples[:,None,:]-p[kept][None,:,:])**2,axis=2))-r[kept]
                clearance = np.minimum(clearance,dist.min(axis=1))
            # Forbid simply filling the same vacancy on the first insertion.
            if slot == 0:
                travel = np.linalg.norm(samples-old[0],axis=1)
                clearance[travel<float(params['transport'])] = -1
            j = int(np.argmax(clearance))
            if clearance[j] <= 1e-6:
                success = False
                break
            p[idx] = samples[j]
            r[idx] = clearance[j]*(1-1e-8)
            kept.append(idx)
        if success:
            z = np.r_[p[:,0],p[:,1],r,w]
            improve(z,int(params['steps']))
        attempt += 1
    return best
