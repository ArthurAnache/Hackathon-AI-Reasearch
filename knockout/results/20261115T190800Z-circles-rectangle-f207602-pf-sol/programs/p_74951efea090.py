import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {'iterations': (180, 40, 300), 'safety': (0.69, 0.5, 0.72), 'jitter': (0.035, 0.0, 0.12)}


def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + float(params['safety']) * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -float('inf')
    ii, jj = np.triu_indices(21, 1)

    def canonical(obj):
        if obj is None:
            return None
        if 'circles' in obj:
            return np.asarray(obj['circles'], dtype=float)
        return np.column_stack((obj['centers'], obj['radii'])).astype(float)

    def verify(a):
        if a is None or a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        d = np.sqrt((a[ii, 0] - a[jj, 0]) ** 2 + (a[ii, 1] - a[jj, 1]) ** 2)
        if np.any(d < a[ii, 2] + a[jj, 2]):
            return -float('inf')
        xs = [Fraction(float(v)) for v in a[:, 0]]
        ys = [Fraction(float(v)) for v in a[:, 1]]
        rs = [Fraction(float(v)) for v in a[:, 2]]
        wh = max(x+r for x, r in zip(xs, rs)) - min(x-r for x, r in zip(xs, rs))
        wh += max(y+r for y, r in zip(ys, rs)) - min(y-r for y, r in zip(ys, rs))
        if wh > 2:
            return -float('inf')
        return float(np.sum(a[:, 2]))

    try:
        best_score = verify(canonical(init))
    except Exception:
        pass

    def retain(a):
        nonlocal best, best_score
        if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return
        a = a.copy()
        dist = np.sqrt((a[ii, 0]-a[jj, 0])**2 + (a[ii, 1]-a[jj, 1])**2)
        ratio = float(np.min(dist / (a[ii, 2]+a[jj, 2])))
        a[:, 2] *= min(1.0, ratio) * (1.0-8e-15)
        if np.any(a[:, 2] <= 0):
            return
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        a[:, :2] -= lo
        extent = np.max(a[:, :2]+a[:, 2, None], axis=0)-np.min(a[:, :2]-a[:, 2, None], axis=0)
        a *= (2.0 / float(np.sum(extent))) * (1.0-8e-15)
        if float(a[:, 2].sum()) <= best_score:
            return
        value = verify(a)
        if value > best_score:
            best_score = value
            best = {'centers': a[:, :2].tolist(), 'radii': a[:, 2].tolist()}

    # Three circles on y=0 and nine pairs reflected across that axis.
    X = np.zeros((21, 34))
    Y = np.zeros((21, 34))
    R = np.zeros((21, 34))
    X[np.arange(3), np.arange(3)] = 1
    R[np.arange(3), np.arange(3)+3] = 1
    for k in range(9):
        if time.monotonic() >= deadline:
            break
        for row, sign in ((3+k, 1), (12+k, -1)):
            X[row, 6+k] = 1
            Y[row, 15+k] = sign
            R[row, 24+k] = 1
    # Keep only one representative of each reflection-equivalent contact.
    pairs = []
    for i in range(3):
        for j in range(i+1, 3):
            pairs.append((i, j))
        for j in range(3, 12):
            pairs.append((i, j))
    for i in range(9):
        for j in range(i+1, 9):
            pairs.append((3+i, 3+j))
        for j in range(i, 9):
            pairs.append((3+i, 12+j))
    pi = np.array([p[0] for p in pairs])
    pj = np.array([p[1] for p in pairs])
    DX, DY, SR = X[pi]-X[pj], Y[pi]-Y[pj], R[pi]+R[pj]
    # Linear wall inequalities use W + 2H = 2.
    B = np.zeros((36, 34))
    b = np.zeros(36)
    B[:12] = X[:12]-R[:12]
    B[12:24] = -X[:12]-R[:12]
    B[12:24, 33] += 1
    B[24:36] = -Y[:12]-R[:12]
    B[24:36, 33] -= 0.5
    b[24:36] = 1
    weights = np.zeros(34)
    weights[3:6] = 1
    weights[24:33] = 2

    class Stop(Exception):
        pass

    def check():
        if time.monotonic() >= deadline:
            raise Stop()

    cache_z = None
    cache_f = None
    cache_j = None

    def evaluate(z):
        nonlocal cache_z, cache_f, cache_j
        check()
        if cache_z is not None and np.array_equal(z, cache_z):
            return
        dx, dy, sr = DX @ z, DY @ z, SR @ z
        cache_f = np.concatenate((dx*dx+dy*dy-sr*sr, B @ z+b))
        cache_j = np.vstack((2*(dx[:, None]*DX+dy[:, None]*DY-sr[:, None]*SR), B))
        cache_z = z.copy()

    def con(z):
        evaluate(z)
        return cache_f

    def jac(z):
        evaluate(z)
        return cache_j

    def objective(z):
        check()
        return -float(weights @ z)

    def unpack(z):
        return np.column_stack((X @ z, Y @ z, R @ z))

    counter = 0
    last_z = None

    def callback(z):
        nonlocal counter, last_z
        check()
        counter += 1
        last_z = z.copy()
        if counter % 4 == 0:
            retain(unpack(z))

    def layout(rows, variation=0.0):
        # Axis row is followed by positive rows, totaling nine circles.
        raw = []
        for row, count in enumerate([3]+list(rows)):
            if time.monotonic() >= deadline:
                raise Stop()
            spacing = 2.0
            xs = (np.arange(count)-(count-1)/2)*spacing
            y = row * 1.76
            for x in xs:
                raw.append([float(x), y, 0.98])
        raw = np.asarray(raw)
        if variation:
            raw[3:, :2] += rng.normal(0, variation, (9, 2))
            raw[:, 2] *= np.exp(rng.normal(0, variation, 12))
        w = float(np.max(raw[:, 0]+raw[:, 2])-np.min(raw[:, 0]-raw[:, 2]))
        h = float(np.max(raw[:, 1]+raw[:, 2]))
        scale = 2/(w+2*h)
        left = float(np.min(raw[:, 0]-raw[:, 2]))
        raw[:, 0] -= left
        raw *= scale
        z = np.zeros(34)
        z[:3] = raw[:3, 0]
        z[3:6] = raw[:3, 2]
        z[6:15] = raw[3:, 0]
        z[15:24] = raw[3:, 1]
        z[24:33] = raw[3:, 2]
        z[33] = w*scale
        return z

    bounds = [(0.0, 2.0)]*34
    for k in list(range(3, 6))+list(range(24, 33)):
        bounds[k] = (1e-7, 0.6)
    bounds[33] = (0.15, 1.85)
    families = [(4, 3, 2), (3, 3, 3), (5, 4), (4, 3, 2), (3, 4, 2), (4, 4, 1)]
    try:
        for attempt, rows in enumerate(families):
            check()
            z = layout(rows, 0.0 if attempt < 3 else float(params['jitter']))
            retain(unpack(z))
            last_z = z.copy()
            counter = 0
            cache_z = None
            result = minimize(objective, z, jac=lambda z: -weights,
                              method='SLSQP', bounds=bounds,
                              constraints=[{'type': 'ineq', 'fun': con, 'jac': jac}],
                              callback=callback,
                              options={'maxiter': int(params['iterations']), 'ftol': 2e-12, 'disp': False})
            retain(unpack(result.x))
    except Stop:
        if last_z is not None:
            retain(unpack(last_z))
    if best is None or best_score == -float('inf'):
        a = np.array([[0.1+0.2*(k % 7), 0.1+0.2*(k//7), 0.099] for k in range(21)], dtype=float)
        retain(a)
    return best
