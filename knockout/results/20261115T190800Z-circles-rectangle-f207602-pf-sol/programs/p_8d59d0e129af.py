import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    "initial_iterations": (115, 40, 240),
    "restart_iterations": (85, 30, 180),
    "perturbation": (0.018, 0.001, 0.08, "log"),
}


def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + 0.695 * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    best = init
    best_score = -float('inf')
    best_z = None

    def exact_score(data):
        try:
            a = np.asarray(data['circles'], dtype=float)
            if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
                return -float('inf')
            d = a[ii, :2] - a[jj, :2]
            if np.any(np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2) < a[ii, 2] + a[jj, 2]):
                return -float('inf')
            x = [Fraction(float(v)) for v in a[:, 0]]
            y = [Fraction(float(v)) for v in a[:, 1]]
            r = [Fraction(float(v)) for v in a[:, 2]]
            w = max(u + v for u, v in zip(x, r)) - min(u - v for u, v in zip(x, r))
            h = max(u + v for u, v in zip(y, r)) - min(u - v for u, v in zip(y, r))
            if w + h > 2:
                return -float('inf')
            return float(np.sum(a[:, 2]))
        except Exception:
            return -float('inf')

    if init is not None:
        best_score = exact_score(init)

    # Variables: nine positive horizontal offsets, twelve heights,
    # twelve radii, and rectangle width. The height is 2-width.
    X = np.zeros((21, 34))
    Y = np.zeros((21, 34))
    R = np.zeros((21, 34))
    reps = list(range(3))
    mirror = np.arange(21)
    for k in range(3):
        Y[k, 9 + k] = 1
        R[k, 21 + k] = 1
    for k in range(9):
        a, b = 3 + 2*k, 4 + 2*k
        X[a, k], X[b, k] = 1, -1
        Y[a, 12+k] = Y[b, 12+k] = 1
        R[a, 24+k] = R[b, 24+k] = 1
        mirror[a], mirror[b] = b, a
        reps.append(a)
    pairlist = []
    for a, b in zip(ii, jj):
        if time.monotonic() >= deadline:
            break
        reflected = tuple(sorted((int(mirror[a]), int(mirror[b]))))
        if (int(a), int(b)) <= reflected:
            pairlist.append((a, b))
    if not pairlist:
        return best
    pi = np.array([p[0] for p in pairlist])
    pj = np.array([p[1] for p in pairlist])
    DX, DY, SR = X[pi]-X[pj], Y[pi]-Y[pj], R[pi]+R[pj]
    walls = []
    constants = []
    for a in reps:
        if time.monotonic() >= deadline:
            return best
        v = -X[a]-R[a]
        v = v.copy(); v[33] += 0.5
        walls.append(v); constants.append(0.)
        walls.append(Y[a]-R[a]); constants.append(0.)
        v = -Y[a]-R[a]
        v = v.copy(); v[33] -= 1.
        walls.append(v); constants.append(2.)
    W = np.asarray(walls)
    wc = np.asarray(constants)
    objective = -R.sum(axis=0)
    bounds = [(0., 1.)]*9 + [(0., 2.)]*12 + [(1e-6, 0.65)]*12 + [(0.15, 1.85)]

    class Stop(Exception):
        pass

    def check():
        if time.monotonic() >= deadline:
            raise Stop()

    def constraints(z):
        check()
        dx, dy, rr = DX @ z, DY @ z, SR @ z
        return np.concatenate((W @ z + wc, dx*dx + dy*dy - rr*rr))

    def jacobian(z):
        check()
        dx, dy, rr = DX @ z, DY @ z, SR @ z
        return np.vstack((W, 2*(dx[:, None]*DX + dy[:, None]*DY - rr[:, None]*SR)))

    def retain(z):
        nonlocal best, best_score, best_z
        if time.monotonic() >= deadline:
            return
        a = np.column_stack((X @ z, Y @ z, R @ z))
        if not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return
        d = a[ii, :2]-a[jj, :2]
        dist = np.sqrt(d[:, 0]**2+d[:, 1]**2)
        factor = min(1., float(np.min(dist/(a[ii, 2]+a[jj, 2]))))
        if factor <= 0:
            return
        a[:, 2] *= factor*(1.-2e-12)
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        perimeter = float(np.sum(hi-lo))
        if perimeter <= 0:
            return
        a[:, :2] -= (hi+lo)/2
        a *= (2./perimeter)*(1.-2e-12)
        tentative = float(a[:, 2].sum())
        if tentative <= best_score:
            return
        candidate = {'circles': a.tolist()}
        score = exact_score(candidate)
        if score > best_score:
            best_score, best, best_z = score, candidate, z.copy()

    archive = []

    def optimize(z, iterations):
        calls = 0
        last = z.copy()
        def callback(v):
            nonlocal calls, last
            check()
            calls += 1
            last = v.copy()
            if calls % 5 == 0:
                retain(v)
        try:
            check()
            result = minimize(lambda v: float(objective @ v), z,
                jac=lambda v: objective, method='SLSQP', bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                callback=callback,
                options={'maxiter': int(iterations), 'ftol': 3e-12, 'disp': False})
            last = result.x
            retain(last)
        except Stop:
            return last
        return last

    def lattice(rows):
        z = np.zeros(34)
        singles, pairs = [], []
        y = 1.
        previous = None
        for count in rows:
            check()
            if previous is not None:
                y += np.sqrt(3.) if count % 2 != previous % 2 else 2.
            if count % 2:
                singles.append((y, 1.))
                offsets = range(2, count, 2)
            else:
                offsets = range(1, count, 2)
            for x in offsets:
                check()
                pairs.append((float(x), y, 1.))
            previous = count
        if len(singles) != 3 or len(pairs) != 9:
            return None
        width = 2.*(max(v[0] for v in pairs)+1.)
        height = y+1.
        scale = 2./(width+height)
        for k, (yy, rr) in enumerate(singles):
            z[9+k], z[21+k] = yy*scale, rr*scale
        for k, (xx, yy, rr) in enumerate(pairs):
            z[k], z[12+k], z[24+k] = xx*scale, yy*scale, rr*scale
        z[33] = width*scale
        return z

    families = [(5,4,5,4,3), (5,4,3,4,5), (3,4,3,4,3,4), (7,6,5,3), (7,7,7)]
    try:
        for rows in families:
            check()
            z = lattice(rows)
            retain(z)
            out = optimize(z, params['initial_iterations'])
            archive.append(out.copy())
            if time.monotonic()-started > 0.48*budget_s:
                break
        attempt = 0
        while time.monotonic() < deadline:
            check()
            if best_z is None:
                break
            if attempt % 4 == 3 and archive:
                z = archive[int(rng.integers(len(archive)))].copy()
            else:
                z = best_z.copy()
            if attempt > 0:
                # Grow a connected group in the near-contact graph, then
                # displace its representatives together before restoration.
                xx, yy, rr = X @ z, Y @ z, R @ z
                dist = np.sqrt((xx[:,None]-xx[None,:])**2+(yy[:,None]-yy[None,:])**2)
                gaps = dist-rr[:,None]-rr[None,:]
                selected = {int(rng.integers(21))}
                target = int(rng.integers(3, 7))
                while len(selected) < target:
                    check()
                    edges = []
                    for a in sorted(selected):
                        check()
                        edges.extend((gaps[a,b], b) for b in range(21) if b not in selected)
                    edges.sort()
                    selected.add(int(edges[int(rng.integers(min(6,len(edges))))][1]))
                groups = sorted(set(a if a < 3 else 3+(a-3)//2 for a in selected))
                amplitude = params['perturbation'] * (0.4 + 1.8*rng.random())
                shift = rng.normal(size=2)*amplitude
                for g in groups:
                    check()
                    z[9+g] += shift[1] + rng.normal()*amplitude*0.3
                    z[21+g] *= 1.+rng.normal()*0.045
                    if g >= 3:
                        z[g-3] += shift[0] + rng.normal()*amplitude*0.3
                for k, (lo, hi) in enumerate(bounds):
                    check()
                    z[k] = np.clip(z[k], lo+1e-8, hi-1e-8)
            out = optimize(z, params['restart_iterations'])
            if len(archive) < 12:
                archive.append(out.copy())
            attempt += 1
    except Stop:
        pass
    if best is None:
        # Deterministic emergency packing, with generous geometric slack.
        a = [[(k % 7)*0.14, (k//7)*0.14, 0.069] for k in range(21)]
        best = {'circles': a}
    return best
