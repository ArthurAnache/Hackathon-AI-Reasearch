import time
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog

PARAMS = {
    'cap_factor': (1.45, 1.05, 2.5),
    'trust': (0.10, 0.01, 0.4),
    'capped_steps': (12, 3, 30),
    'free_steps': (22, 5, 60),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)


def score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return -np.inf
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    perimeter = (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
                 +max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)))
    if perimeter > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))


def pack(z):
    p = np.column_stack((z[:N], z[N:2*N]))
    r = z[2*N:3*N].copy()
    if np.any(r <= 0) or not np.isfinite(z).all():
        return None
    lo = np.min(p-r[:, None], axis=0)
    hi = np.max(p+r[:, None], axis=0)
    scale = 2.0 / np.sum(hi-lo)
    p = (p-lo)*scale
    r *= scale
    d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
    factor = min(1.0, float(np.min(d/(r[I]+r[J]))))
    r *= factor*(1.0-3e-14)
    # Scaling the entire configuration fixes the exact perimeter without
    # changing its contact geometry. Exact validation remains authoritative.
    for unused in range(4):
        a = np.column_stack((p,r))
        if score(a) > -np.inf:
            return a
        p *= 1.0-2e-13
        r *= 1.0-3e-13
    return None


def normalized(a):
    p = a[:,:2].copy()
    r = a[:,2].copy()
    total = r.sum()
    p /= total
    r /= total
    lo = np.min(p-r[:,None],axis=0)
    p -= lo
    side = np.max(p+r[:,None],axis=0)
    return np.r_[p[:,0],p[:,1],r,side]


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a = np.array([[(k%5)*0.19+0.1,(k//5)*0.19+0.1,0.085] for k in range(N)])
        a[:,:2] *= 0.95
        best = {'circles': a.tolist()}
        best_score = score(a)
    else:
        best = init
        a = np.asarray(init['circles'],dtype=float)
        best_score = score(a)
    base = normalized(a)
    # Variables: 21 x, 21 y, 21 radii, width, height.
    nv = 3*N+2
    rows = M+4*N
    A = np.zeros((rows,nv))
    b = np.zeros(rows)
    ids = np.arange(M)
    A[ids,2*N+I] = 1
    A[ids,2*N+J] = 1
    k = np.arange(N)
    off = M
    A[off+k,k] = -1
    A[off+k,2*N+k] = 1
    off += N
    A[off+k,k] = 1
    A[off+k,2*N+k] = 1
    A[off+k,3*N] = -1
    off += N
    A[off+k,N+k] = -1
    A[off+k,2*N+k] = 1
    off += N
    A[off+k,N+k] = 1
    A[off+k,2*N+k] = 1
    A[off+k,3*N+1] = -1
    eq = np.zeros((1,nv))
    eq[0,2*N:3*N] = 1
    objective = np.zeros(nv)
    objective[-2:] = 1
    branch = 0
    while time.monotonic() < end:
        z = base.copy()
        # Alternate a size-capped branch with a randomly weighted size
        # redistribution. Both phases end with the exact perimeter objective.
        cap_factor = float(params['cap_factor']) * (1.0+0.17*(branch%3))
        cap = max(1.0001/N, cap_factor/N)
        if branch % 2:
            caps = cap*rng.uniform(0.80,1.20,N)
            caps *= max(1.0,1.08/caps.sum())
        else:
            caps = np.full(N,cap)
        steps = int(params['capped_steps'])+int(params['free_steps'])
        for step in range(steps):
            now = time.monotonic()
            if now >= end:
                break
            dx = z[I]-z[J]
            dy = z[N+I]-z[N+J]
            dist = np.maximum(np.sqrt(dx*dx+dy*dy),1e-14)
            ux,uy = dx/dist,dy/dist
            A[ids,I] = -ux
            A[ids,J] = ux
            A[ids,N+I] = -uy
            A[ids,N+J] = uy
            trust = float(params['trust'])
            bounds = [(float(v-trust),float(v+trust)) for v in z[:2*N]]
            capped = step < int(params['capped_steps'])
            bounds += [(1e-7,float(caps[i]) if capped else None) for i in range(N)]
            bounds += [(1e-5,None),(1e-5,None)]
            result = linprog(objective,A_ub=A,b_ub=b,A_eq=eq,b_eq=[1.0],
                             bounds=bounds,method='highs-ds',
                             options={'time_limit':max(0.001,end-now),
                                      'primal_feasibility_tolerance':1e-9,
                                      'dual_feasibility_tolerance':1e-9})
            if time.monotonic() >= end:
                break
            if not result.success:
                if capped:
                    caps = np.maximum(caps,z[2*N:3*N]*0.96)
                    continue
                break
            new = result.x
            movement = np.max(np.abs(new-z))
            z = new
            if not capped or step % 4 == 3:
                candidate = pack(z)
                if candidate is not None:
                    value = score(candidate)
                    if value > best_score:
                        best_score = value
                        best = {'circles':candidate.tolist()}
            if not capped and movement < 2e-9:
                break
        branch += 1
    return best
