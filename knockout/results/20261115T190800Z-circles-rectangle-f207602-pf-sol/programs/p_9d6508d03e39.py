import time
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog

PARAMS = {
    'tilt': (0.85, 0.1, 2.0),
    'trust': (0.065, 0.01, 0.2),
    'force': (0.18, 0.0, 0.6),
    'population': (5, 1, 12),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
K = np.arange(N)

class StopSearch(Exception):
    pass


def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    r = a[:, 2]
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < r[I]+r[J]):
        return -np.inf
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    q = [Fraction(float(v)) for v in r]
    wh = (max(u+v for u,v in zip(x,q))-min(u-v for u,v in zip(x,q))
          +max(u+v for u,v in zip(y,q))-min(u-v for u,v in zip(y,q)))
    if wh > 2:
        return -np.inf
    return float(np.sum(r))


def repair(z, deadline):
    a = np.column_stack((z[:N], z[N:2*N], z[2*N:3*N])).copy()
    if not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
        return None
    r = a[:, 2]
    d = np.sqrt((a[I, 0]-a[J, 0])**2+(a[I, 1]-a[J, 1])**2)
    factor = min(1.0, float(np.min(d/(r[I]+r[J]))))
    a[:, 2] *= factor*(1-3e-15)
    lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
    hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
    scale = 2.0/float(np.sum(hi-lo))
    a[:, :2] -= lo
    a *= scale*(1-3e-15)
    for _ in range(4):
        if time.monotonic() >= deadline:
            return None
        if np.isfinite(exact_score(a)):
            return a
        a[:, 2] *= 1-2e-13
    return None


def encode(a):
    lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
    hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
    p = a[:, :2]-lo
    return np.r_[p[:, 0], p[:, 1], a[:, 2], hi[0]-lo[0]]


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.69*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf
    if init is not None:
        arr = np.asarray(init['circles'], dtype=float)
        best_score = exact_score(arr)
    else:
        arr = np.array([[(i%5+0.5)*0.2, (i//5+0.5)*0.2, 0.085] for i in range(N)])
        best = {'circles': arr.tolist()}
        best_score = exact_score(arr)
    z0 = encode(arr)
    # A fixed linear boundary model; only the separating contact normals change.
    base = np.zeros((M+4*N, 3*N+1))
    rhs = np.zeros(M+4*N)
    base[np.arange(M), 2*N+I] = 1
    base[np.arange(M), 2*N+J] = 1
    for k in range(4):
        rows = M+k*N+K
        base[rows, 2*N+K] = 1
        if k == 0:
            base[rows, K] = -1
        elif k == 1:
            base[rows, K] = 1
            base[rows, -1] = -1
        elif k == 2:
            base[rows, N+K] = -1
        else:
            base[rows, N+K] = 1
            base[rows, -1] = 1
            rhs[rows] = 2

    def step(z, reward, force, trust):
        remaining = deadline-time.monotonic()
        if remaining < 0.018:
            raise StopSearch
        dx = z[I]-z[J]
        dy = z[N+I]-z[N+J]
        dist = np.maximum(np.hypot(dx,dy), 1e-14)
        ux, uy = dx/dist, dy/dist
        mat = base.copy()
        rows = np.arange(M)
        mat[rows, I] = -ux
        mat[rows, J] = ux
        mat[rows, N+I] = -uy
        mat[rows, N+J] = uy
        c = np.r_[-force[:,0], -force[:,1], -reward, 0.0]
        lower = np.maximum(z-trust, np.r_[np.zeros(2*N), np.full(N,1e-6), 0.15])
        upper = np.minimum(z+trust, np.r_[np.full(2*N,2.0), np.full(N,0.8), 1.85])
        result = linprog(c, A_ub=mat, b_ub=rhs,
                         bounds=np.column_stack((lower,upper)), method='highs-ds',
                         options={'primal_feasibility_tolerance':1e-9,
                                  'dual_feasibility_tolerance':1e-9,
                                  'time_limit':min(0.12,max(0.005,remaining-0.012))})
        if result.success:
            return result.x
        return z.copy()

    def consider(z):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            raise StopSearch
        if float(np.sum(z[2*N:3*N])) < best_score-2e-7:
            return
        a = repair(z, deadline)
        if a is not None:
            score = exact_score(a)
            if score > best_score:
                best_score = score
                best = {'circles': a.tolist()}

    pool = [z0.copy()]
    zero = np.zeros((N,2))
    try:
        # Establish the best feasible local reference before spending time on excursions.
        z = z0.copy()
        for _ in range(5):
            if time.monotonic() >= deadline:
                raise StopSearch
            zn = step(z,np.ones(N),zero,0.035)
            consider(zn)
            movement = np.max(np.abs(zn-z))
            z = zn
            if movement < 1e-9:
                break
        pool[0] = z.copy()
        cycle = 0
        while time.monotonic() < deadline-0.025:
            parent = pool[int(rng.integers(len(pool)))].copy()
            if cycle % 3 == 0:
                parent = pool[0].copy()
            z = parent
            points = np.column_stack((z[:N],z[N:2*N]))
            center = points[int(rng.integers(N))]
            width = rng.uniform(0.15,0.5)
            bump = np.exp(-np.sum((points-center)**2,axis=1)/(2*width*width))
            weights = 0.55*rng.normal(size=N)+1.5*(bump-bump.mean())
            weights /= max(0.5,float(np.std(weights)))
            # Smooth collective forcing avoids restricting the move to one released contact.
            direction = rng.normal(size=2)
            direction /= max(1e-12,float(np.linalg.norm(direction)))
            force = (bump-bump.mean())[:,None]*direction[None,:]
            force += 0.15*rng.normal(size=(N,2))
            force -= force.mean(axis=0)
            for amplitude in (1.0, 0.8, 0.35, 0.0, 0.0, 0.0, 0.0, 0.0):
                if time.monotonic() >= deadline-0.018:
                    raise StopSearch
                reward = 1+float(params['tilt'])*amplitude*weights
                f = float(params['force'])*amplitude*force
                zn = step(z,reward,f,float(params['trust']))
                consider(zn)
                movement = float(np.max(np.abs(zn-z)))
                z = zn
                if amplitude == 0 and movement < 2e-8:
                    break
            if time.monotonic() >= deadline:
                raise StopSearch
            if np.sum(z[2*N:3*N]) > np.sum(pool[0][2*N:3*N])+1e-11:
                pool.insert(0,z.copy())
            elif len(pool) < int(params['population']):
                pool.append(z.copy())
            elif rng.random() < 0.6 and len(pool)>1:
                pool[int(rng.integers(1,len(pool)))] = z.copy()
            pool = pool[:int(params['population'])]
            cycle += 1
    except StopSearch:
        pass
    return best
