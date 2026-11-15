import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'equalizing_power': (0.3, 0.1, 0.9),
    'polarizing_power': (1.8, 1.1, 3.0),
    'excursion_time': (0.22, 0.05, 0.4),
    'noise': (0.002, 0.0, 0.03),
}
N = 21
I, J = np.triu_indices(N, 1)

class Stop(Exception):
    pass

def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    r = a[:, 2]
    d = a[I, :2] - a[J, :2]
    if np.any(np.sqrt(d[:, 0]**2 + d[:, 1]**2) < r[I] + r[J]):
        return -np.inf
    f = [[Fraction(float(v)) for v in row] for row in a]
    perimeter = (max(v[0]+v[2] for v in f)-min(v[0]-v[2] for v in f)
                 + max(v[1]+v[2] for v in f)-min(v[1]-v[2] for v in f))
    if perimeter > 2:
        return -np.inf
    return float(np.sum(r))

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf
    if init is not None:
        a = np.array(init['circles'], dtype=float)
        best_score = exact_score(a)
    else:
        a = np.zeros((N, 3))
        for k in range(N):
            if time.monotonic() >= deadline:
                break
            a[k] = [(k % 5 + .5)/5, (k // 5 + .5)/5, .045]
        best = {'circles': a.tolist()}
        best_score = exact_score(a)
    lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
    hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
    scale = 2 / np.sum(hi-lo)
    p = (a[:, :2]-lo)*scale
    r = a[:, 2]*scale
    z0 = np.r_[p[:, 0], p[:, 1], r, (hi[0]-lo[0])*scale]
    # Variables are x, y, r, and width; height is 2-width.
    wall_jac = np.zeros((4*N, 3*N+1))
    q = np.arange(N)
    wall_jac[q, q] = 1
    wall_jac[q, 2*N+q] = -1
    wall_jac[N+q, q] = -1
    wall_jac[N+q, 2*N+q] = -1
    wall_jac[N+q, -1] = 1
    wall_jac[2*N+q, N+q] = 1
    wall_jac[2*N+q, 2*N+q] = -1
    wall_jac[3*N+q, N+q] = -1
    wall_jac[3*N+q, 2*N+q] = -1
    wall_jac[3*N+q, -1] = -1
    pair_jac = np.zeros((len(I), 3*N+1))
    rows = np.arange(len(I))
    pair_jac[rows, 2*N+I] = -1
    pair_jac[rows, 2*N+J] = -1
    local_end = deadline
    last = z0.copy()

    def check():
        if time.monotonic() >= local_end:
            raise Stop()

    def walls(z):
        check()
        x,y,rr,w = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
        return np.r_[x-rr, w-x-rr, y-rr, 2-w-y-rr]

    def contacts(z):
        check()
        dx = z[I]-z[J]
        dy = z[N+I]-z[N+J]
        return np.sqrt(dx*dx+dy*dy+1e-30)-z[2*N+I]-z[2*N+J]

    def contact_jac(z):
        check()
        dx = z[I]-z[J]
        dy = z[N+I]-z[N+J]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        g = pair_jac.copy()
        g[rows,I] = dx/d
        g[rows,J] = -dx/d
        g[rows,N+I] = dy/d
        g[rows,N+J] = -dy/d
        return g

    def save(z):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        pc = np.column_stack((z[:N],z[N:2*N]))
        rc = np.maximum(z[2*N:3*N],1e-10).copy()
        d = np.sqrt(np.sum((pc[I]-pc[J])**2,axis=1))
        factor = min(1.,float(np.min(d/(rc[I]+rc[J]))))
        rc *= max(0.,factor)*(1-4e-14)
        low = np.min(pc-rc[:,None],axis=0)
        high = np.max(pc+rc[:,None],axis=0)
        fac = min(1.,2/float(np.sum(high-low)))*(1-4e-14)
        candidate = np.column_stack(((pc-low)*fac,rc*fac))
        score = exact_score(candidate)
        if score > best_score:
            best_score = score
            best = {'circles':candidate.tolist()}

    bounds = [(None,None)]*(2*N)+[(1e-6,.6)]*N+[(.15,1.85)]
    constraints = [{'type':'ineq','fun':walls,'jac':lambda z:wall_jac},
                   {'type':'ineq','fun':contacts,'jac':contact_jac}]

    def solve(z, power, duration):
        nonlocal local_end,last
        local_end = min(deadline-0.025,time.monotonic()+duration)
        last = z.copy()
        def objective(v):
            nonlocal last
            check()
            last = v.copy()
            rr = np.maximum(v[2*N:3*N],1e-9)
            # Scaling keeps objective gradients comparable across exponents.
            t = rr/.11
            g = np.zeros(3*N+1)
            g[2*N:3*N] = -t**(power-1)
            return -.11*np.sum(t**power)/power,g
        def callback(v):
            nonlocal last
            check()
            last = v.copy()
        try:
            result = minimize(objective,z,jac=True,bounds=bounds,
                              constraints=constraints,method='SLSQP',callback=callback,
                              options={'maxiter':180,'ftol':2e-12,'disp':False})
            last = result.x
        except Stop:
            pass
        save(last)
        return last.copy()

    z = z0.copy()
    powers = [float(params['equalizing_power']),1.,float(params['polarizing_power']),1.]
    for k,power in enumerate(powers):
        if time.monotonic() >= deadline-.08:
            break
        if k == 2:
            # Restart the second excursion from the original geometry rather
            # than carry the near-equal-radius branch forward.
            z = z0.copy()
            z[:2*N] += rng.normal(0,float(params['noise']),2*N)
        remaining = deadline-time.monotonic()-.03
        duration = min(remaining,float(params['excursion_time'])*budget_s) if power != 1 else min(remaining,.20*budget_s)
        z = solve(z,power,duration)
    if time.monotonic() < deadline-.08:
        solve(z,1.,deadline-time.monotonic()-.035)
    return best
