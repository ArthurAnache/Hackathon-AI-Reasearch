import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    "iterations": (100, 40, 180),
    "cluster_step": (0.028, 0.002, 0.12),
    "search_fraction": (0.69, 0.5, 0.72)
}


def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + float(params['search_fraction']) * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -float('inf')

    def exact(data):
        try:
            c = np.asarray(data['circles'], dtype=float)
            if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
                return -float('inf')
            xx = [Fraction(float(t)) for t in c[:, 0]]
            yy = [Fraction(float(t)) for t in c[:, 1]]
            rr = [Fraction(float(t)) for t in c[:, 2]]
            w = max(x+r for x, r in zip(xx, rr))-min(x-r for x, r in zip(xx, rr))
            h = max(y+r for y, r in zip(yy, rr))-min(y-r for y, r in zip(yy, rr))
            if w+h > 2:
                return -float('inf')
            i, j = np.triu_indices(21, 1)
            d = np.sqrt((c[i, 0]-c[j, 0])**2+(c[i, 1]-c[j, 1])**2)
            if np.any(d < c[i, 2]+c[j, 2]):
                return -float('inf')
            return float(np.sum(c[:, 2]))
        except Exception:
            return -float('inf')

    if init is not None:
        best_score = exact(init)

    ii, jj = np.triu_indices(21, 1)

    def retain(c):
        nonlocal best, best_score
        if not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return -float('inf')
        c = c.copy()
        d = np.sqrt(np.sum((c[ii, :2]-c[jj, :2])**2, axis=1))
        if np.min(d) < 1e-16:
            return -float('inf')
        shrink = min(1.0, float(np.min(d/(c[ii, 2]+c[jj, 2]))))
        c[:, 2] *= shrink * (1.0-4e-14)
        lo = np.min(c[:, :2]-c[:, 2, None], axis=0)
        hi = np.max(c[:, :2]+c[:, 2, None], axis=0)
        scale = 2.0 / float(np.sum(hi-lo)) * (1.0-4e-14)
        c[:, :2] -= (lo+hi)/2
        c *= scale
        score = float(c[:, 2].sum())
        if score > best_score:
            data = {'circles': c.tolist()}
            actual = exact(data)
            if actual > best_score:
                best_score, best = actual, data
        return score

    # Three circles lie on the reflection axis; nine groups are mirrored.
    gx = np.zeros((12, 34))
    gx[3:, :9] = np.eye(9)
    gy = np.zeros((12, 34))
    gy[:, 9:21] = np.eye(12)
    gr = np.zeros((12, 34))
    gr[:, 21:33] = np.eye(12)
    gh = np.zeros(34)
    gh[33] = 1
    a, b = np.triu_indices(12, 1)
    dx = gx[a]-gx[b]
    dy = gy[a]-gy[b]
    rs = gr[a]+gr[b]
    linear = np.vstack((gy-gr, gh-gy-gr, -gx-gr-gh/2, gx[3:]-gr[3:]))
    offset = np.r_[np.zeros(24), np.ones(12), np.zeros(9)]
    weights = np.r_[np.ones(3), np.full(9, 2.0)]
    objective_jac = -(weights @ gr)
    bounds = [(1e-6, 1.0)]*9 + [(0.0, 1.9)]*12 + [(1e-6, 0.5)]*12 + [(0.15, 1.85)]

    def expand(z):
        x, y, r = gx @ z, gy @ z, gr @ z
        return np.vstack((np.column_stack((x, y, r)), np.column_stack((-x[3:], y[3:], r[3:]))))

    cached_z = None
    cached_v = None
    cached_j = None

    def calculate(z):
        nonlocal cached_z, cached_v, cached_j
        if time.monotonic() >= deadline:
            raise TimeoutError
        if cached_z is not None and np.array_equal(z, cached_z):
            return
        x = dx @ z
        y = dy @ z
        distance = np.maximum(np.sqrt(x*x+y*y), 1e-15)
        cached_v = np.r_[distance-rs @ z, linear @ z+offset]
        cached_j = np.vstack(((x[:, None]*dx+y[:, None]*dy)/distance[:, None]-rs, linear))
        cached_z = z.copy()

    def con(z):
        calculate(z)
        return cached_v

    def jac(z):
        calculate(z)
        return cached_j

    constraint = {'type': 'ineq', 'fun': con, 'jac': jac}
    archive = []
    latest = None
    calls = 0

    def callback(z):
        nonlocal latest, calls
        if time.monotonic() >= deadline:
            raise TimeoutError
        latest = z.copy()
        calls += 1
        if calls % 3 == 0:
            retain(expand(z))

    def solve(z, count):
        nonlocal latest, cached_z
        if time.monotonic() >= deadline:
            return z
        latest = z.copy()
        cached_z = None
        try:
            result = minimize(lambda q: float(objective_jac @ q), z,
                              jac=lambda q: objective_jac,
                              bounds=bounds, constraints=constraint,
                              method='SLSQP', callback=callback,
                              options={'maxiter': int(count), 'ftol': 2e-11, 'disp': False})
            latest = result.x.copy()
        except TimeoutError:
            pass
        except (ValueError, FloatingPointError):
            pass
        score = retain(expand(latest))
        archive.append((score, latest.copy()))
        return latest.copy()

    def lattice(rows):
        singles = []
        pairs = []
        spacing = np.sqrt(3.0)/2.0
        for k, number in enumerate(rows):
            if time.monotonic() >= deadline:
                raise TimeoutError
            yy = 0.5+k*spacing
            if number % 2:
                singles.append((0.0, yy, 0.5))
                positive = np.arange(1, (number+1)//2, dtype=float)
            else:
                positive = np.arange(number//2, dtype=float)+0.5
            for xx in positive:
                if time.monotonic() >= deadline:
                    raise TimeoutError
                pairs.append((float(xx), yy, 0.5))
        groups = np.array(singles+pairs)
        h = 1+(len(rows)-1)*spacing
        w = float(max(rows))
        scale = 2/(w+h)
        groups *= scale
        z = np.zeros(34)
        z[:9] = groups[3:, 0]
        z[9:21] = groups[:, 1]
        z[21:33] = groups[:, 2]
        z[33] = h*scale
        return z

    # Guaranteed fallback, independent of the supplied start.
    fallback = np.array([[float(x), float(y), 0.499999999999] for y in range(3) for x in range(7)])
    retain(fallback)
    families = [(5, 4, 5, 4, 3), (5, 6, 5, 5), (3, 4, 5, 4, 5), (7, 6, 5, 3)]
    for family in families:
        if time.monotonic() >= deadline-0.18*budget_s:
            break
        try:
            z = lattice(family)
        except TimeoutError:
            break
        retain(expand(z))
        solve(z, params['iterations'])

    # Expand a connected cluster in the near-contact graph, moving several
    # contacts together rather than repeatedly releasing just one contact.
    attempt = 0
    while time.monotonic() < deadline-0.025:
        if not archive:
            break
        archive.sort(key=lambda item: item[0], reverse=True)
        z = archive[0][1].copy()
        if attempt:
            gaps = np.sqrt((dx @ z)**2+(dy @ z)**2)-rs @ z
            edges = [(int(a[k]), int(b[k])) for k in np.flatnonzero(gaps < 0.018)]
            cluster = {int(rng.integers(12))}
            target = int(rng.integers(3, 7))
            while len(cluster) < target:
                if time.monotonic() >= deadline-0.025:
                    break
                boundary = [(u, v) for u, v in edges if (u in cluster) != (v in cluster)]
                if not boundary:
                    break
                u, v = boundary[int(rng.integers(len(boundary)))]
                cluster.add(u)
                cluster.add(v)
            amplitude = float(params['cluster_step']) * (0.6+0.3*(attempt % 4))
            for g in sorted(cluster):
                if time.monotonic() >= deadline-0.025:
                    break
                if g >= 3:
                    z[g-3] += rng.normal(0, amplitude)
                z[9+g] += rng.normal(0, amplitude)
                z[21+g] *= float(rng.uniform(0.9, 1.05))
            z = np.clip(z, np.array([q[0] for q in bounds]), np.array([q[1] for q in bounds]))
        old = archive[0][0]
        solve(z, 110 if attempt == 0 else 70)
        attempt += 1
        if attempt > 18:
            break
    return best
