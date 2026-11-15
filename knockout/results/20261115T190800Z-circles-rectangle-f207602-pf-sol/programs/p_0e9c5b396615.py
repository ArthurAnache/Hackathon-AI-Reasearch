import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {
    'kick': (0.028, 0.001, 0.15),
    'tilt': (0.18, 0.01, 0.8),
    'iterations': (65, 15, 150),
    'time_fraction': (0.69, 0.4, 0.72),
}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + float(params['time_fraction']) * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)

    def checktime():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def exact(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -np.inf
        d = np.sqrt((a[ii, 0]-a[jj, 0])**2 + (a[ii, 1]-a[jj, 1])**2)
        if np.any(d < a[ii, 2] + a[jj, 2]):
            return -np.inf
        x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
        perimeter = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        perimeter += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        if perimeter > 2:
            return -np.inf
        return float(np.sum(a[:, 2]))

    if init is None:
        a = np.array([[(i-3)*0.2, (j-1)*0.2, 0.09999999999999]
                      for j in range(3) for i in range(7)], dtype=float)
        init = {'circles': a.tolist()}
    original = np.array(init['circles'], dtype=float)
    best = init
    best_score = exact(original)
    p = original[:, :2].copy()
    r = original[:, 2].copy()
    lo = (p-r[:, None]).min(axis=0)
    hi = (p+r[:, None]).max(axis=0)
    p -= (lo+hi)/2

    # Detect which coordinate is the reflection coordinate.
    costs = []
    for ax in range(2):
        if time.monotonic() >= deadline:
            return best
        q = p.copy()
        q[:, ax] *= -1
        cost = ((q[:, None, :]-p[None, :, :])**2).sum(axis=2)
        rows, cols = linear_sum_assignment(cost)
        costs.append(float(cost[rows, cols].sum()))
    if costs[0] < costs[1]:
        p = p[:, ::-1].copy()
    singles = np.argsort(np.abs(p[:, 1]))[:3]
    rest = np.array([i for i in range(21) if i not in set(singles)])
    rest = rest[np.argsort(p[rest, 1])]
    neg, pos = rest[:9], rest[9:]
    reflected = p[pos].copy()
    reflected[:, 1] *= -1
    rows, cols = linear_sum_assignment(((reflected[:, None]-p[neg][None])**2).sum(axis=2))
    neg = neg[cols]
    pos = pos[rows]
    z0 = np.zeros(34)
    z0[:3] = p[singles, 0]
    z0[3:12] = (p[pos, 0]+p[neg, 0])/2
    z0[12:21] = (p[pos, 1]-p[neg, 1])/2
    z0[21:24] = r[singles]
    z0[24:33] = (r[pos]+r[neg])/2
    z0[33] = float(np.max(p[:, 0]+r)-np.min(p[:, 0]-r))

    groups = list(range(3)) + list(range(3,12)) + list(range(3,12))
    signs = [0]*3 + [1]*9 + [-1]*9
    X = np.zeros((21,34))
    Y = np.zeros((21,34))
    R = np.zeros((21,34))
    for i, (g, s) in enumerate(zip(groups, signs)):
        if time.monotonic() >= deadline:
            return best
        X[i,g] = 1
        R[i,21+g] = 1
        if s:
            Y[i,12+g-3] = s
    pairs = []
    seen = set()
    for i,j in zip(ii,jj):
        if time.monotonic() >= deadline:
            return best
        a,b = groups[i],groups[j]
        key = (min(a,b),max(a,b),signs[i]*signs[j])
        if key not in seen:
            seen.add(key)
            pairs.append((i,j))
    ai = np.array([v[0] for v in pairs])
    aj = np.array([v[1] for v in pairs])
    DX, DY, SR = X[ai]-X[aj], Y[ai]-Y[aj], R[ai]+R[aj]
    wall = []
    offsets = []
    for g in range(12):
        if time.monotonic() >= deadline:
            return best
        for sg in (-1,1):
            row = np.zeros(34)
            row[g] = sg
            row[21+g] = -1
            row[33] = 0.5
            wall.append(row)
            offsets.append(0.)
        row = np.zeros(34)
        row[21+g] = -1
        row[33] = -0.5
        if g >= 3:
            row[12+g-3] = -1
        wall.append(row)
        offsets.append(1.)
    wall = np.array(wall)
    offsets = np.array(offsets)
    weights = np.zeros(34)
    weights[21:24] = 1
    weights[24:33] = 2
    bounds = [(None,None)]*12 + [(0.,1.)]*9 + [(1e-7,0.5)]*12 + [(0.2,1.8)]
    cache_z = None
    cache_c = None
    cache_j = None

    def constraints(z):
        nonlocal cache_z, cache_c, cache_j
        checktime()
        if cache_z is None or not np.array_equal(z,cache_z):
            dx,dy = DX@z,DY@z
            d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-16)
            cache_c = np.concatenate((d-SR@z,wall@z+offsets))
            cache_j = np.vstack(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-SR,wall))
            cache_z = z.copy()
        return cache_c

    def jacobian(z):
        constraints(z)
        return cache_j

    def accept(z):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        a = np.column_stack((X@z,Y@z,R@z))
        if not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        d = np.sqrt((a[ii,0]-a[jj,0])**2+(a[ii,1]-a[jj,1])**2)
        sums = a[ii,2]+a[jj,2]
        factor = min(1.,float(np.min(d/sums)))
        a[:,2] *= factor*(1.-3e-15)
        extent = np.max(a[:,:2]+a[:,2,None],axis=0)-np.min(a[:,:2]-a[:,2,None],axis=0)
        scale = min(1.,2./float(extent.sum()))*(1.-3e-15)
        a *= scale
        score = exact(a)
        if score > best_score:
            best_score = score
            best = {'circles': a.tolist()}

    last = z0.copy()
    def solve(z, tilt, limit):
        nonlocal last
        def objective(v):
            checktime()
            last = v.copy()
            return -float((weights+tilt)@v), -(weights+tilt)
        result = minimize(objective,z,method='SLSQP',jac=True,bounds=bounds,
                          constraints={'type':'ineq','fun':constraints,'jac':jacobian},
                          options={'maxiter':limit,'ftol':3e-13,'disp':False})
        return result.x

    try:
        # First recover any accuracy lost by the warm start's certification slack.
        base = solve(z0,np.zeros(34),min(25,int(params['iterations'])))
        accept(base)
        attempt = 0
        while time.monotonic() < deadline:
            checktime()
            z = base.copy()
            amplitude = float(params['kick'])*(1.0+0.35*(attempt%3))
            # A simultaneous orbit kick breaks several contacts, not just one.
            chosen = rng.choice(12,size=5,replace=False)
            z[chosen] += rng.normal(0,amplitude,5)
            pairgroups = chosen[chosen>=3]
            z[12+pairgroups-3] += rng.normal(0,amplitude,len(pairgroups))
            z[12:21] = np.maximum(z[12:21],1e-4)
            z[33] = np.clip(z[33]+rng.normal(0,2*amplitude),0.25,1.75)
            z[21:33] *= 0.98
            tilt = np.zeros(34)
            tilt[:21] = rng.normal(0,float(params['tilt']),21)
            tilt[33] = rng.normal(0,float(params['tilt']))
            tilted = solve(z,tilt,16)
            checktime()
            relaxed = solve(tilted,np.zeros(34),int(params['iterations']))
            accept(relaxed)
            if float(weights@relaxed) > float(weights@base)+1e-9:
                base = relaxed.copy()
            attempt += 1
    except StopSearch:
        pass
    return best
