import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'rotation': (1.05, 0.35, 1.7),
    'cluster_size': (6, 3, 10),
    'beam_size': (7, 2, 12),
    'screen_iterations': (45, 15, 90),
    'penalty': (650.0, 100.0, 3000.0, 'log'),
    'stop_fraction': (0.70, 0.55, 0.75),
}


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + params['stop_fraction'] * budget_s
    screen_deadline = start + 0.32 * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    rows = np.arange(m)
    ids = np.arange(n)
    best = init
    best_score = -float('inf')

    class Stop(Exception):
        pass

    def check():
        if time.monotonic() >= deadline:
            raise Stop()

    def exact_score(data):
        check()
        a = np.asarray(data['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        f = [[Fraction(float(v)) for v in c] for c in a]
        wh = (max(x+r for x,y,r in f)-min(x-r for x,y,r in f)
              + max(y+r for x,y,r in f)-min(y-r for x,y,r in f))
        if wh > 2:
            return -float('inf')
        for u, v in itertools.combinations(a, 2):
            check()
            if np.sqrt((u[0]-v[0])**2+(u[1]-v[1])**2) < u[2]+v[2]:
                return -float('inf')
        return float(np.sum(a[:, 2]))

    def encode(data):
        a = np.asarray(data['circles'], dtype=float).copy()
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        scale = 2 / np.sum(hi-lo)
        a[:, :2] = (a[:, :2]-lo)*scale
        a[:, 2] *= scale
        return np.r_[a[:, 0], a[:, 1], a[:, 2], (hi[0]-lo[0])*scale]

    def values(z):
        check()
        x, y, r, w = z[:n], z[n:2*n], z[2*n:3*n], z[-1]
        dx, dy = x[ii]-x[jj], y[ii]-y[jj]
        d = np.sqrt(dx*dx+dy*dy)
        return np.r_[d-r[ii]-r[jj], x-r, w-x-r, y-r, 2-w-y-r]

    def jac(z):
        check()
        dx, dy = z[ii]-z[jj], z[n+ii]-z[n+jj]
        d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-30)
        j = np.zeros((m+4*n, 3*n+1))
        j[rows, ii], j[rows, jj] = dx/d, -dx/d
        j[rows, n+ii], j[rows, n+jj] = dy/d, -dy/d
        j[rows, 2*n+ii] = -1
        j[rows, 2*n+jj] = -1
        for k in range(4):
            check()
            rr = m+k*n+ids
            j[rr, 2*n+ids] = -1
            j[rr, (0 if k < 2 else n)+ids] = 1 if k % 2 == 0 else -1
        j[m+n:m+2*n, -1] = 1
        j[m+3*n:m+4*n, -1] = -1
        return j

    gradient = np.r_[np.zeros(2*n), -np.ones(n), 0.0]
    bounds = [(0.0, 2.0)]*(2*n)+[(1e-5, 0.5)]*n+[(0.4, 1.6)]

    def objective(z):
        check()
        return -float(np.sum(z[2*n:3*n]))

    def accept(z):
        nonlocal best, best_score
        check()
        a = np.column_stack((z[:n], z[n:2*n], z[2*n:3*n])).copy()
        if not np.isfinite(a).all():
            return
        dx, dy = a[ii, 0]-a[jj, 0], a[ii, 1]-a[jj, 1]
        d = np.sqrt(dx*dx+dy*dy)
        deficit = max(0.0, float(np.max(a[ii, 2]+a[jj, 2]-d)))
        a[:, 2] -= deficit/2 + 3e-15
        if np.any(a[:, 2] <= 0):
            return
        a[:, :2] -= np.min(a[:, :2]-a[:, 2, None], axis=0)
        f = [[Fraction(float(v)) for v in c] for c in a]
        wh = (max(x+r for x,y,r in f)-min(x-r for x,y,r in f)
              + max(y+r for x,y,r in f)-min(y-r for x,y,r in f))
        if wh > 2:
            a *= float(Fraction(2)/wh)*(1-5e-15)
        if np.sum(a[:, 2]) <= best_score:
            return
        data = {'circles': a.tolist()}
        score = exact_score(data)
        if score > best_score:
            best, best_score = data, score

    rho = float(params['penalty'])

    def penalty(z):
        check()
        v = np.minimum(values(z), 0.0)
        f = objective(z)+rho*np.dot(v, v)
        g = gradient+2*rho*(v @ jac(z))
        return f, g

    try:
        if init is None:
            a = [[0.1+0.2*(k % 5), 0.1+0.2*(k // 5), 0.099999999999] for k in range(n)]
            best = {'circles': a}
        best_score = exact_score(best)
        base = encode(best)
        base_contacts = values(base)[:m] < 1e-5
        adjacency = np.zeros((n, n))
        adjacency[ii, jj] = base_contacts
        adjacency += adjacency.T
        xy0 = np.column_stack((base[:n], base[n:2*n]))
        beam = []
        for attempt in range(int(params['beam_size'])):
            check()
            if time.monotonic() >= screen_deadline:
                break
            anchor = int(rng.integers(n))
            selected = [anchor]
            k = min(n-1, int(params['cluster_size'])+attempt % 3-1)
            while len(selected) < k:
                check()
                affinity = np.sum(adjacency[:, selected], axis=1)
                distance = np.min(np.sum((xy0[:, None, :]-xy0[selected][None, :, :])**2, axis=2), axis=1)
                priority = affinity-2*distance+rng.uniform(0, 0.6, n)
                priority[selected] = -np.inf
                selected.append(int(np.argmax(priority)))
            selected = np.asarray(selected)
            z = base.copy()
            xy = xy0.copy()
            centre = np.mean(xy[selected], axis=0)
            angle = params['rotation']*rng.uniform(0.75, 1.25)*(-1 if attempt % 2 else 1)
            c, s = np.cos(angle), np.sin(angle)
            xy[selected] = centre+(xy[selected]-centre) @ np.array([[c, s], [-s, c]])
            z[:n], z[n:2*n] = xy[:, 0], xy[:, 1]
            z[2*n+selected] *= 0.85
            result = minimize(penalty, z, jac=True, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': int(params['screen_iterations']), 'maxls': 12,
                                       'ftol': 1e-9, 'gtol': 1e-6})
            z = result.x
            v = values(z)
            changed = int(np.count_nonzero((v[:m] < 0.002) != base_contacts))
            merit = float(np.sum(z[2*n:3*n])-rho*np.sum(np.minimum(v, 0)**2))
            beam.append((merit, changed, z.copy()))
            accept(z)
        beam.sort(key=lambda item: item[0], reverse=True)
        if len(beam) > 1:
            cutoff = beam[0][0]-0.012
            competitive = [item for item in beam if item[0] >= cutoff]
            if competitive:
                first = max(competitive, key=lambda item: item[1])
                beam = [first]+[item for item in beam if item is not first]
        beam.append((0.0, 0, base))
        for _, _, z in beam:
            check()
            result = minimize(objective, z, jac=lambda q: gradient, method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': values, 'jac': jac},
                              callback=accept,
                              options={'maxiter': 85, 'ftol': 2e-13, 'disp': False})
            accept(result.x)
    except Stop:
        pass
    return best
