import time
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog

PARAMS = {
    'drop': (0.000002, 0.0000001, 0.02, 'log'),
    'trust': (0.065, 0.005, 0.3, 'log'),
    'explore_steps': (4, 1, 12),
    'restore_steps': (11, 3, 30),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)


def score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return -np.inf
    x = list(map(lambda q: Fraction(float(q)), a[:, 0]))
    y = list(map(lambda q: Fraction(float(q)), a[:, 1]))
    r = list(map(lambda q: Fraction(float(q)), a[:, 2]))
    perimeter = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
    perimeter += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
    return float(a[:, 2].sum()) if perimeter <= 2 else -np.inf


def repaired(z):
    p = np.column_stack((z[:N], z[N:2*N]))
    r = np.maximum(z[2*N:3*N], 1e-15).copy()
    d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
    fac = min(1.0, float(np.min(d/(r[I]+r[J]))))
    r *= max(0.0, fac*(1-4e-15))
    lo = (p-r[:,None]).min(axis=0)
    p -= lo
    sides = (p+r[:,None]).max(axis=0)
    scale = min(1.0, 2.0/float(sides.sum()))*(1-4e-15)
    p *= scale
    r *= scale
    return np.column_stack((p, r))


def pack(a):
    p, r = a[:,:2].copy(), a[:,2].copy()
    lo = (p-r[:,None]).min(axis=0)
    p -= lo
    sides = (p+r[:,None]).max(axis=0)
    return np.r_[p[:,0],p[:,1],r,sides[0]]


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        xy = np.array([(i/5+.1,j/5+.1) for j in range(5) for i in range(5)])[:21]
        a0 = np.column_stack((xy,np.full(N,.095)))
        best = {'circles': a0.tolist()}
    else:
        best = init
        a0 = np.asarray(init['circles'],dtype=float).copy()
    best_score = score(a0)
    base = pack(a0)
    archive = base.copy()
    # Variables are all x coordinates, all y coordinates, radii, and width.
    A = np.zeros((M+4*N+1,3*N+1))
    b = np.zeros(M+4*N+1)
    rows = np.arange(M)
    A[rows,2*N+I] = 1
    A[rows,2*N+J] = 1
    k = np.arange(N)
    A[M+k,k] = -1
    A[M+k,2*N+k] = 1
    A[M+N+k,k] = 1
    A[M+N+k,2*N+k] = 1
    A[M+N+k,-1] = -1
    A[M+2*N+k,N+k] = -1
    A[M+2*N+k,2*N+k] = 1
    A[M+3*N+k,N+k] = 1
    A[M+3*N+k,2*N+k] = 1
    A[M+3*N+k,-1] = 1
    b[M+3*N:M+4*N] = 2
    A[-1,2*N:3*N] = -1
    objective = np.zeros(3*N+1)
    objective[2*N:3*N] = -1
    cycle = 0
    while time.monotonic() < end-0.035:
        z = (base if cycle % 3 != 2 else archive).copy()
        drop = float(params['drop']) * (10.0 ** ((cycle % 4)-1))
        floor = min(best_score, float(z[2*N:3*N].sum()))-drop
        # Smooth collective velocity field rather than independent centre jitter.
        xy = np.column_stack((z[:N],z[N:2*N]))
        q = xy-xy.mean(axis=0)
        coeff = rng.normal(size=(5,2))
        features = np.column_stack((q[:,0],q[:,1],q[:,0]*q[:,1],q[:,0]**2,q[:,1]**2))
        velocity = features @ coeff
        velocity -= velocity.mean(axis=0)
        velocity /= max(float(np.linalg.norm(velocity)),1e-12)
        wander = objective.copy()
        wander[:N] = -velocity[:,0]
        wander[N:2*N] = -velocity[:,1]
        wander[2*N:3*N] = -0.02
        steps = int(params['explore_steps'])+int(params['restore_steps'])
        previous_sum = -np.inf
        for step in range(steps):
            now = time.monotonic()
            if now >= end-0.025:
                break
            exploratory = step < int(params['explore_steps'])
            dx = z[I]-z[J]
            dy = z[N+I]-z[N+J]
            d = np.maximum(np.hypot(dx,dy),1e-15)
            ux,uy = dx/d,dy/d
            A[rows,I] = -ux
            A[rows,J] = ux
            A[rows,N+I] = -uy
            A[rows,N+J] = uy
            b[-1] = -floor if exploratory else 0.0
            trust = float(params['trust']) if exploratory else 0.16
            lower = z-trust
            upper = z+trust
            lower[2*N:3*N] = 1e-10
            upper[2*N:3*N] = 0.7
            lower[-1],upper[-1] = max(.05,z[-1]-trust),min(1.95,z[-1]+trust)
            bounds = np.column_stack((lower,upper))
            result = linprog(wander if exploratory else objective,A_ub=A,b_ub=b,
                             bounds=bounds,method='highs',
                             options={'time_limit':max(.001,end-now-.015),
                                      'dual_feasibility_tolerance':1e-9,
                                      'primal_feasibility_tolerance':1e-9})
            if not result.success:
                break
            z = result.x
            total = float(z[2*N:3*N].sum())
            if total > best_score:
                candidate = repaired(z)
                exact = score(candidate)
                if exact > best_score:
                    best_score = exact
                    best = {'circles':candidate.tolist()}
                    base = pack(candidate)
            if not exploratory and abs(total-previous_sum) < 2e-12:
                break
            previous_sum = total
        archive = z.copy()
        cycle += 1
    return best
