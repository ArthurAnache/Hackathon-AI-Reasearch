import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'polish_iterations': (24, 5, 70),
    'survivor_iterations': (9, 3, 25),
    'repair_iterations': (85, 25, 160),
    'void_samples': (480, 100, 1200),
    'survivor_growth': (1.12, 1.01, 1.4),
}

class StopSearch(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + 0.695 * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    best = init
    best_score = -float('inf')

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def exact_score(data):
        try:
            c = np.asarray(data['circles'], dtype=np.float64)
            if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
                return -float('inf')
            d = c[ii, :2] - c[jj, :2]
            if np.any(np.sqrt(d[:, 0]**2 + d[:, 1]**2) < c[ii, 2] + c[jj, 2]):
                return -float('inf')
            x = [Fraction(float(v)) for v in c[:, 0]]
            y = [Fraction(float(v)) for v in c[:, 1]]
            r = [Fraction(float(v)) for v in c[:, 2]]
            w = max(a+b for a,b in zip(x,r)) - min(a-b for a,b in zip(x,r))
            h = max(a+b for a,b in zip(y,r)) - min(a-b for a,b in zip(y,r))
            if w+h > 2:
                return -float('inf')
            return float(np.sum(c[:, 2]))
        except Exception:
            return -float('inf')

    if init is not None:
        best_score = exact_score(init)

    def retain(c):
        nonlocal best, best_score
        check()
        c = np.array(c, dtype=float, copy=True)
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return
        d = c[ii, :2]-c[jj, :2]
        dist = np.sqrt(d[:, 0]**2+d[:, 1]**2)
        f = min(1., float(np.min(dist/(c[ii, 2]+c[jj, 2]))))
        if f <= 0:
            return
        c[:, 2] *= f*(1.-6e-14)
        lo = np.min(c[:, :2]-c[:, 2, None], axis=0)
        hi = np.max(c[:, :2]+c[:, 2, None], axis=0)
        c[:, :2] -= (hi+lo)/2
        c *= (2./float(np.sum(hi-lo)))*(1.-6e-14)
        if float(np.sum(c[:, 2])) <= best_score:
            return
        candidate = {'circles': c.tolist()}
        score = exact_score(candidate)
        if score > best_score:
            best, best_score = candidate, score

    def canonical(c):
        c = np.array(c, dtype=float, copy=True)
        lo = np.min(c[:, :2]-c[:, 2, None], axis=0)
        hi = np.max(c[:, :2]+c[:, 2, None], axis=0)
        scale = 2./float(np.sum(hi-lo))
        c[:, :2] -= lo
        c *= scale
        return c, float((hi[0]-lo[0])*scale)

    def solve(c, width, iterations, until, caps=None):
        check()
        n = len(c)
        size = 3*n+1
        a, b = np.triu_indices(n, 1)
        dx = np.zeros((len(a), size))
        dy = dx.copy()
        sr = dx.copy()
        rows = np.arange(len(a))
        dx[rows, 3*a] = 1.; dx[rows, 3*b] = -1.
        dy[rows, 3*a+1] = 1.; dy[rows, 3*b+1] = -1.
        sr[rows, 3*a+2] = 1.; sr[rows, 3*b+2] = 1.
        wall = np.zeros((4*n, size))
        const = np.zeros(4*n)
        for k in range(n):
            check()
            wall[4*k, 3*k] = 1.
            wall[4*k, 3*k+2] = -1.
            wall[4*k+1, 3*k] = -1.
            wall[4*k+1, 3*k+2] = -1.
            wall[4*k+1, -1] = 1.
            wall[4*k+2, 3*k+1] = 1.
            wall[4*k+2, 3*k+2] = -1.
            wall[4*k+3, 3*k+1] = -1.
            wall[4*k+3, 3*k+2] = -1.
            wall[4*k+3, -1] = -1.
            const[4*k+3] = 2.
        obj = np.zeros(size)
        obj[2:3*n:3] = -1.
        z = np.r_[c.ravel(), width]
        bounds = [(None, None)]*size
        for k in range(n):
            check()
            bounds[3*k+2] = (1e-6, None if caps is None else float(caps[k]))
        bounds[-1] = (.12, 1.88)
        cache = [None, None, None]
        last = z.copy()
        calls = 0

        def local_check():
            if time.monotonic() >= min(until, deadline):
                raise StopSearch()

        def calc(v):
            local_check()
            if cache[0] is None or not np.array_equal(v, cache[0]):
                x, y = dx@v, dy@v
                dist = np.sqrt(x*x+y*y+1e-30)
                values = np.r_[dist-sr@v, wall@v+const]
                jac = np.vstack(((x/dist)[:, None]*dx+(y/dist)[:, None]*dy-sr, wall))
                cache[:] = [v.copy(), values, jac]
            return cache[1], cache[2]

        def callback(v):
            nonlocal last, calls
            local_check()
            last = v.copy()
            calls += 1
            if n == 21 and calls % 4 == 0 and -obj@v > best_score:
                retain(v[:-1].reshape(n, 3))

        try:
            result = minimize(lambda v: float(obj@v), z, jac=lambda v: obj,
                method='SLSQP', bounds=bounds,
                constraints={'type': 'ineq', 'fun': lambda v: calc(v)[0],
                             'jac': lambda v: calc(v)[1]}, callback=callback,
                options={'maxiter': int(iterations), 'ftol': 8e-13, 'disp': False})
            last = result.x
        except StopSearch:
            pass
        if n == 21 and time.monotonic() < deadline:
            retain(last[:-1].reshape(n, 3))
        return last[:-1].reshape(n, 3).copy(), float(last[-1])

    if init is None:
        base = np.array([[(k % 7)*.198, (k//7)*.198, .099] for k in range(21)])
        best = {'circles': base.tolist()}
        best_score = exact_score(best)
    else:
        base = np.asarray(init['circles'], dtype=float)
    base, width = canonical(base)
    archive = [(base.copy(), width)]
    graphs = set()

    try:
        polished, pw = solve(base, width, params['polish_iterations'],
                             min(deadline, started+.15*budget_s))
        archive.append((polished, pw))
        attempt = 0
        while time.monotonic() < deadline-.035:
            check()
            parent, w = archive[0 if attempt % 3 != 2 else int(rng.integers(len(archive)))]
            parent = parent.copy()
            distances = np.linalg.norm(parent[:, None, :2]-parent[None, :, :2], axis=2)
            gaps = distances-parent[:, None, 2]-parent[None, :, 2]
            np.fill_diagonal(gaps, np.inf)
            selected = [int(rng.integers(21))]
            count = 2 + attempt % 3
            while len(selected) < count:
                check()
                scores = np.min(gaps[selected], axis=0)
                scores[selected] = np.inf
                nearest = np.argsort(scores)[:3]
                selected.append(int(nearest[int(rng.integers(len(nearest)))]))
            keep = np.ones(21, dtype=bool)
            keep[selected] = False
            survivors = parent[keep].copy()
            caps = survivors[:, 2]*params['survivor_growth']
            survivors, w = solve(survivors, w, params['survivor_iterations'],
                                 min(deadline, time.monotonic()+.075*budget_s), caps)
            c = survivors.copy()
            for k in range(count):
                check()
                h = 2.-w
                points = rng.random((int(params['void_samples']), 2))*[w, h]
                clearance = np.minimum.reduce((points[:, 0], w-points[:, 0], points[:, 1], h-points[:, 1]))
                dist = np.linalg.norm(points[:, None, :]-c[None, :, :2], axis=2)-c[None, :, 2]
                clearance = np.minimum(clearance, dist.min(axis=1))
                order = np.argsort(clearance)[::-1]
                rank = 0 if attempt % 2 == 0 else int(rng.integers(4))
                pick = int(order[rank])
                radius = max(1e-5, float(clearance[pick])*(.7 if k % 2 else .98))
                c = np.vstack((c, [points[pick, 0], points[pick, 1], radius]))
            c, w = solve(c, w, params['repair_iterations'],
                         min(deadline, time.monotonic()+.27*budget_s))
            check()
            d = np.linalg.norm(c[ii, :2]-c[jj, :2], axis=1)
            gap = d-c[ii, 2]-c[jj, 2]
            signature = tuple(np.flatnonzero(gap < 1e-5).tolist())
            if np.min(gap) > -1e-6 and signature not in graphs:
                graphs.add(signature)
                if len(archive) < 9:
                    archive.append((c.copy(), w))
            attempt += 1
    except StopSearch:
        pass
    return best
