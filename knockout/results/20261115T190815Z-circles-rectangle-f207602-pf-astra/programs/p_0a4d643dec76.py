import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'trials': (5, 1, 12),
    'maxiter': (120, 30, 300),
    'merge_candidates': (18, 3, 50),
    'split_fraction': (0.5, 0.3, 0.7),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
K = np.arange(N)

class StopSearch(Exception):
    pass

def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return -np.inf
    x = list(map(lambda v: Fraction(float(v)), a[:, 0]))
    y = list(map(lambda v: Fraction(float(v)), a[:, 1]))
    r = list(map(lambda v: Fraction(float(v)), a[:, 2]))
    perimeter = (max(map(lambda t: t[0]+t[1], zip(x, r)))
                 - min(map(lambda t: t[0]-t[1], zip(x, r)))
                 + max(map(lambda t: t[0]+t[1], zip(y, r)))
                 - min(map(lambda t: t[0]-t[1], zip(y, r))))
    if perimeter > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + 0.70*budget_s
    search_end = deadline - min(0.07, 0.05*budget_s)
    rng = np.random.default_rng(seed)
    if init is None:
        a = np.column_stack(((np.arange(N) % 5 + 0.5)*0.2,
                             (np.arange(N) // 5 + 0.5)*0.2,
                             np.full(N, 0.099999999999)))
        best = {'circles': a.tolist()}
    else:
        best = init
        a = np.asarray(init['circles'], dtype=float).copy()
    best_score = exact_score(a)
    lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
    a[:, :2] -= lo
    sides = np.max(a[:, :2]+a[:, 2, None], axis=0)
    factor = 2.0/sides.sum()
    a *= factor
    width = float(sides[0]*factor)
    base = a.copy()

    def consider(z):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        c = np.column_stack((z[:N], z[N:2*N], z[2*N:3*N])).copy()
        if not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return
        d = np.sqrt((c[I, 0]-c[J, 0])**2+(c[I, 1]-c[J, 1])**2)
        f = min(1.0, float(np.min(d/(c[I, 2]+c[J, 2]))))
        c[:, 2] *= f*(1-3e-15)
        c[:, :2] -= np.min(c[:, :2]-c[:, 2, None], axis=0)
        span = np.max(c[:, :2]+c[:, 2, None], axis=0).sum()
        c *= min(1.0, 2.0/span)*(1-3e-15)
        score = exact_score(c)
        if score > best_score:
            best_score = score
            best = {'circles': c.tolist()}

    # Enumerate adjacent merges and estimate the disturbance to the other disks.
    choices = []
    for i, j in zip(I, J):
        if time.monotonic() >= search_end:
            break
        delta = base[j, :2]-base[i, :2]
        d = float(np.linalg.norm(delta))
        ri, rj = base[i, 2], base[j, 2]
        if d > ri+rj+0.008 or d < 1e-12:
            continue
        radius = (d+ri+rj)/2
        center = base[i, :2]+delta*((radius-ri)/d)
        others = (K != i) & (K != j)
        overlap = np.maximum(radius+base[:, 2]-np.linalg.norm(base[:, :2]-center, axis=1), 0)
        edge = np.maximum(radius-center, 0).sum()+np.maximum(center+radius-np.array([width, 2-width]), 0).sum()
        cost = float(np.sum(overlap[others]**2)+edge**2+0.008*radius**2)
        choices.append((cost, int(i), int(j), center, radius))
    choices.sort(key=lambda t: t[0])
    choices = choices[:int(params['merge_candidates'])]
    if not choices:
        return best

    jac0 = np.zeros(3*N+1)
    jac0[2*N:3*N] = -1
    bounds = [(None, None)]*(2*N)+[(1e-7, 0.7)]*N+[(0.15, 1.85)]
    trials = int(params['trials'])
    for trial in range(trials):
        now = time.monotonic()
        if now >= search_end:
            break
        local_end = min(search_end, now+(search_end-now)/max(1, trials-trial))
        choice_index = trial if trial < min(3, len(choices)) else int(rng.integers(len(choices)))
        _, i, j, center, radius = choices[choice_index % len(choices)]
        c = base.copy()
        eligible = (K != i) & (K != j)
        distance = np.linalg.norm(base[:, :2]-center, axis=1)
        merit = np.where(eligible, base[:, 2]*(0.7+np.minimum(distance, 0.7)), -1)
        donor_order = np.argsort(-merit)
        donor = int(donor_order[trial % min(3, N-2)])
        if donor in (i, j):
            donor = int(donor_order[0])
        c[i, :2], c[i, 2] = center, radius
        old = base[donor, :2]
        R = base[donor, 2]
        angle = (trial*0.6180339887498949 + rng.uniform(0, 0.25))*np.pi
        direction = np.array([np.cos(angle), np.sin(angle)])
        f = float(params['split_fraction'])
        c[donor, 2] = f*R
        c[j, 2] = (1-f)*R
        c[donor, :2] = old+(1-f)*R*direction
        c[j, :2] = old-f*R*direction
        z = np.r_[c[:, 0], c[:, 1], c[:, 2], width]
        last = [z.copy()]

        def check():
            if time.monotonic() >= local_end:
                raise StopSearch()

        def objective(v):
            check()
            return -float(np.sum(v[2*N:3*N]))

        def constraints(v):
            check()
            x, y, r, w = v[:N], v[N:2*N], v[2*N:3*N], v[-1]
            d = np.sqrt((x[I]-x[J])**2+(y[I]-y[J])**2)
            return np.r_[d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r]

        def derivative(v):
            check()
            x, y = v[:N], v[N:2*N]
            dx, dy = x[I]-x[J], y[I]-y[J]
            d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-20)
            g = np.zeros((M+4*N, 3*N+1))
            t = np.arange(M)
            g[t, I], g[t, J] = dx/d, -dx/d
            g[t, N+I], g[t, N+J] = dy/d, -dy/d
            g[t, 2*N+I], g[t, 2*N+J] = -1, -1
            g[M+K, K] = 1
            g[M+N+K, K] = -1
            g[M+N+K, -1] = 1
            g[M+2*N+K, N+K] = 1
            g[M+3*N+K, N+K] = -1
            g[M+3*N+K, -1] = -1
            g[M+K, 2*N+K] = -1
            g[M+N+K, 2*N+K] = -1
            g[M+2*N+K, 2*N+K] = -1
            g[M+3*N+K, 2*N+K] = -1
            return g

        def callback(v):
            last[0] = v.copy()
            check()

        try:
            result = minimize(objective, z, jac=lambda v: jac0, method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': constraints, 'jac': derivative},
                              callback=callback,
                              options={'maxiter': int(params['maxiter']), 'ftol': 2e-12, 'disp': False})
            last[0] = result.x
        except StopSearch:
            pass
        consider(last[0])
    return best
