import time
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog

PARAMS = {
    'release': (0.045, 0.005, 0.15),
    'cluster': (6, 3, 12),
    'trust': (0.085, 0.02, 0.25),
    'excursion_steps': (3, 1, 8),
    'recovery_steps': (18, 5, 40),
    'tilt': (0.09, 0.0, 0.4)
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)


def valid(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return False
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return False
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    return (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r)) +
            max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))) <= 2


def normalized(a):
    p, r = a[:, :2].copy(), a[:, 2].copy()
    lo = np.min(p-r[:, None], axis=0)
    sides = np.max(p+r[:, None], axis=0)-lo
    scale = 2.0/sides.sum()
    p = (p-lo)*scale
    return np.r_[p[:, 0], p[:, 1], r*scale, sides[0]*scale]


def repaired(z):
    a = np.column_stack((z[:N], z[N:2*N], z[2*N:3*N])).copy()
    r = a[:, 2]
    if not np.isfinite(a).all() or np.any(r <= 0):
        return None
    d = np.sqrt((a[I, 0]-a[J, 0])**2+(a[I, 1]-a[J, 1])**2)
    factor = min(1.0, float(np.min(d/(r[I]+r[J]))))
    r *= factor*(1-2e-14)
    lo = np.min(a[:, :2]-r[:, None], axis=0)
    hi = np.max(a[:, :2]+r[:, None], axis=0)
    a[:, :2] -= lo
    scale = min(1.0, 2.0/float(np.sum(hi-lo)))*(1-2e-14)
    a *= scale
    return a if valid(a) else None


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a = np.array([[(i%7)*0.13+0.06, (i//7)*0.13+0.06, 0.06] for i in range(N)])
        a *= 2.0/((a[:,0]+a[:,2]).max()-(a[:,0]-a[:,2]).min()+(a[:,1]+a[:,2]).max()-(a[:,1]-a[:,2]).min()+1e-12)
        init = {'circles': a.tolist()}
    original = np.asarray(init['circles'], dtype=float)
    best = init
    score = float(original[:,2].sum()) if valid(original) else -np.inf
    base = normalized(original)
    size = 3*N+1
    A = np.zeros((M+4*N, size))
    B = np.zeros(M+4*N)
    rows = np.arange(M)
    A[rows,2*N+I] = 1
    A[rows,2*N+J] = 1
    k = np.arange(N)
    for block in range(4):
        rr = M+block*N+k
        A[rr,2*N+k] = 1
        if block == 0:
            A[rr,k] = -1
        elif block == 1:
            A[rr,k] = 1
            A[rr,-1] = -1
        elif block == 2:
            A[rr,N+k] = -1
        else:
            A[rr,N+k] = 1
            A[rr,-1] = 1
            B[rr] = 2
    objective = np.zeros(size)
    objective[2*N:3*N] = -1
    attempt = 0
    while time.monotonic() < deadline:
        z = base.copy()
        dx,dy = z[I]-z[J],z[N+I]-z[N+J]
        dist = np.hypot(dx,dy)
        gap = dist-z[2*N+I]-z[2*N+J]
        active = np.flatnonzero(gap < 2e-6)
        if len(active) == 0:
            active = np.argsort(gap)[:30]
        release = np.zeros(M)
        tilt = objective.copy()
        if attempt:
            pivot = active[int(rng.integers(len(active)))]
            mx = (z[I]+z[J])*0.5
            my = (z[N+I]+z[N+J])*0.5
            near = (mx[active]-mx[pivot])**2+(my[active]-my[pivot])**2
            chosen = active[np.argsort(near)[:int(params['cluster'])]]
            release[chosen] = float(params['release'])*rng.uniform(0.7,1.3)
            nodes = np.unique(np.r_[I[chosen],J[chosen]])
            tilt[2*N+nodes] -= float(params['tilt'])*rng.normal(size=len(nodes))
            angle = rng.uniform(0,2*np.pi)
            tilt[nodes] = float(params['tilt'])*np.cos(angle)*rng.choice([-1,1],len(nodes))
            tilt[N+nodes] = float(params['tilt'])*np.sin(angle)*rng.choice([-1,1],len(nodes))
        excursion = int(params['excursion_steps']) if attempt else 0
        total = excursion+int(params['recovery_steps'])
        for step in range(total):
            now = time.monotonic()
            if now >= deadline:
                break
            dx,dy = z[I]-z[J],z[N+I]-z[N+J]
            dd = np.maximum(np.hypot(dx,dy),1e-12)
            ux,uy = dx/dd,dy/dd
            A[rows,I] = -ux
            A[rows,J] = ux
            A[rows,N+I] = -uy
            A[rows,N+J] = uy
            B[:M] = release if step < excursion else 0
            trust = float(params['trust'])
            bounds = [(max(0.0,float(v-trust)),min(2.0,float(v+trust))) for v in z[:2*N]]
            bounds += [(1e-7,0.5)]*N+[(0.15,1.85)]
            res = linprog(tilt if step < excursion else objective, A_ub=A, b_ub=B,
                          bounds=bounds, method='highs',
                          options={'time_limit':max(0.001,deadline-now),
                                   'dual_feasibility_tolerance':1e-9,
                                   'primal_feasibility_tolerance':1e-9})
            if not res.success:
                break
            change = np.max(np.abs(res.x-z))
            z = res.x
            if step >= excursion:
                candidate = repaired(z)
                if candidate is not None:
                    value = float(candidate[:,2].sum())
                    if value > score:
                        score = value
                        best = {'circles':candidate.tolist()}
                if change < 1e-8:
                    break
        attempt += 1
    return best
