import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'first_iterations': (95, 30, 180),
    'restart_iterations': (45, 15, 100),
    'contact_margin': (0.095, 0.02, 0.25),
    'restart_noise': (0.014, 0.001, 0.08),
    'ftol': (3e-11, 1e-13, 1e-7, 'log')
}
N = 21
I, J = np.triu_indices(N, 1)
K = np.arange(N)

class StopSearch(Exception):
    pass

class ExpandContacts(Exception):
    pass

def exact_score(candidate):
    try:
        a = np.asarray(candidate['circles'], dtype=np.float64)
        if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -np.inf
        d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
        if np.any(d < a[I, 2]+a[J, 2]):
            return -np.inf
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        perimeter = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        perimeter += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        if perimeter > 2:
            return -np.inf
        return float(np.sum(a[:, 2]))
    except Exception:
        return -np.inf

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + 0.70 * budget_s
    search_end = started + 0.675 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = exact_score(init)

    def check():
        if time.monotonic() >= search_end:
            raise StopSearch()

    def consider(z):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        z = np.asarray(z, dtype=float)
        p = np.column_stack((z[:N], z[N:2*N]))
        r = np.maximum(z[2*N:3*N], 1e-10).copy()
        if not np.isfinite(p).all() or not np.isfinite(r).all():
            return
        d = np.sqrt(np.sum((p[I]-p[J])**2, axis=1))
        ratio = np.min(d/(r[I]+r[J]))
        if ratio <= 0:
            return
        r *= min(1.0, float(ratio)) * (1-8e-14)
        lo = np.min(p-r[:, None], axis=0)
        sides = np.max(p+r[:, None], axis=0)-lo
        scale = (2.0/sides.sum())*(1-8e-14)
        p = (p-lo)*scale
        r *= scale
        candidate = {'circles': np.column_stack((p,r)).tolist()}
        score = exact_score(candidate)
        if score > best_score:
            best, best_score = candidate, score

    def encode(a):
        a = np.asarray(a, dtype=float)
        p = a[:, :2].copy()
        r = np.maximum(a[:, 2], 1e-7).copy()
        lo = np.min(p-r[:, None], axis=0)
        sides = np.max(p+r[:, None], axis=0)-lo
        scale = 2.0/sides.sum()
        p = (p-lo)*scale
        return np.r_[p[:,0], p[:,1], r*scale, sides[0]*scale]

    points = []
    for row in range(5):
        if time.monotonic() >= deadline:
            break
        for col in range(5):
            if time.monotonic() >= deadline:
                break
            if row in (0,4) and col in (0,4):
                continue
            points.append((0.1+0.2*col, 0.1+0.2*row, 0.1))
    fallback = np.asarray(points)
    if len(points) == N:
        consider(encode(fallback))

    walls = np.zeros((4*N, 3*N+1))
    walls[K,K] = 1
    walls[K,2*N+K] = -1
    walls[N+K,K] = -1
    walls[N+K,2*N+K] = -1
    walls[N+K,-1] = 1
    walls[2*N+K,N+K] = 1
    walls[2*N+K,2*N+K] = -1
    walls[3*N+K,N+K] = -1
    walls[3*N+K,2*N+K] = -1
    walls[3*N+K,-1] = -1
    grad = np.r_[np.zeros(2*N), -np.ones(N), 0.0]
    bounds = [(-0.5,2.0)]*(2*N)+[(1e-7,0.65)]*N+[(0.2,1.8)]

    def objective(z):
        check()
        return -float(np.sum(z[2*N:3*N]))

    attempt = 0
    try:
        while time.monotonic() < search_end:
            if attempt == 0 and init is not None:
                try:
                    z = encode(init['circles'])
                    if not np.isfinite(z).all():
                        z = encode(best['circles'])
                except Exception:
                    z = encode(best['circles'])
            else:
                z = encode(best['circles'])
                amplitude = float(params['restart_noise']) * (1.0 + 0.55*((attempt-1)%4))
                chosen = rng.choice(N, size=6 if attempt % 3 else N, replace=False)
                z[chosen] += rng.normal(0, amplitude, len(chosen))
                z[N+chosen] += rng.normal(0, amplitude, len(chosen))
                if attempt % 3 == 0:
                    z[2*N:3*N] *= rng.uniform(0.94,1.06,N)
            dx = z[I]-z[J]
            dy = z[N+I]-z[N+J]
            distances = np.sqrt(dx*dx+dy*dy)
            active = distances < z[2*N+I]+z[2*N+J]+float(params['contact_margin'])
            # Add the nearest neighbours even when the initial radii are tiny.
            matrix = np.full((N,N), np.inf)
            matrix[I,J] = distances
            matrix[J,I] = distances
            near = np.argsort(matrix, axis=1)[:, :7]
            adjacency = np.zeros((N,N), dtype=bool)
            adjacency[np.arange(N)[:,None],near] = True
            active |= adjacency[I,J] | adjacency[J,I]
            maxiter = int(params['first_iterations'] if attempt == 0 else params['restart_iterations'])
            rebuilds = 0
            while time.monotonic() < search_end:
                selected = np.flatnonzero(active)
                ii,jj = I[selected],J[selected]
                rows = np.arange(len(ii))
                latest = [z.copy()]
                ticks = [0]

                def constraints(v):
                    check()
                    x,y,r,w = v[:N],v[N:2*N],v[2*N:3*N],v[-1]
                    d = np.sqrt((x[ii]-x[jj])**2+(y[ii]-y[jj])**2+1e-30)
                    return np.r_[d-r[ii]-r[jj],x-r,w-x-r,y-r,2-w-y-r]

                def jacobian(v):
                    check()
                    dx,dy = v[ii]-v[jj],v[N+ii]-v[N+jj]
                    d = np.sqrt(dx*dx+dy*dy+1e-30)
                    a = np.zeros((len(ii)+4*N,3*N+1))
                    a[rows,ii] = dx/d
                    a[rows,jj] = -dx/d
                    a[rows,N+ii] = dy/d
                    a[rows,N+jj] = -dy/d
                    a[rows,2*N+ii] = -1
                    a[rows,2*N+jj] = -1
                    a[len(ii):] = walls
                    return a

                def callback(v):
                    latest[0] = v.copy()
                    check()
                    ticks[0] += 1
                    if ticks[0] % 7 == 0:
                        consider(v)
                    dd = np.sqrt((v[I]-v[J])**2+(v[N+I]-v[N+J])**2)
                    missing = (~active) & (dd < v[2*N+I]+v[2*N+J]-1e-8)
                    if np.any(missing):
                        active[missing] = True
                        raise ExpandContacts()

                try:
                    result = minimize(objective,z,jac=lambda v: grad,method='SLSQP',bounds=bounds,
                                      constraints={'type':'ineq','fun':constraints,'jac':jacobian},
                                      callback=callback,options={'maxiter':maxiter,'ftol':float(params['ftol']),'disp':False})
                    z = result.x
                    consider(z)
                    dd = np.sqrt((z[I]-z[J])**2+(z[N+I]-z[N+J])**2)
                    missing = (~active) & (dd < z[2*N+I]+z[2*N+J]+1e-6)
                    if not np.any(missing):
                        break
                    active[missing] = True
                except ExpandContacts:
                    z = latest[0]
                    consider(z)
                except StopSearch:
                    consider(latest[0])
                    raise
                rebuilds += 1
                if rebuilds >= 5:
                    break
            attempt += 1
    except StopSearch:
        pass
    return best
