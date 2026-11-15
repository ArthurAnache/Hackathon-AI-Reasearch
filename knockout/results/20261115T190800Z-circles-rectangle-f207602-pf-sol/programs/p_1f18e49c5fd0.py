import time
import math
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"iterations": (110, 40, 200), "time_fraction": (0.70, 0.5, 0.8), "jitter": (0.008, 0.0, 0.04)}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + float(params['time_fraction']) * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -float('inf')

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch

    def canonical(obj):
        if 'circles' in obj:
            return np.asarray(obj['circles'], dtype=float)
        return np.column_stack((obj['centers'], obj['radii'])).astype(float)

    def exact_score(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        dx = a[:, None, 0] - a[None, :, 0]
        dy = a[:, None, 1] - a[None, :, 1]
        ii, jj = np.triu_indices(21, 1)
        if np.any(np.sqrt(dx[ii, jj] ** 2 + dy[ii, jj] ** 2) < a[ii, 2] + a[jj, 2]):
            return -float('inf')
        xs = list(map(lambda v: Fraction(float(v)), a[:, 0]))
        ys = list(map(lambda v: Fraction(float(v)), a[:, 1]))
        rs = list(map(lambda v: Fraction(float(v)), a[:, 2]))
        w = max(x+r for x, r in zip(xs, rs)) - min(x-r for x, r in zip(xs, rs))
        h = max(y+r for y, r in zip(ys, rs)) - min(y-r for y, r in zip(ys, rs))
        if w+h > 2:
            return -float('inf')
        return float(np.sum(a[:, 2]))

    if init is not None:
        try:
            best_score = exact_score(canonical(init))
        except (ValueError, KeyError, TypeError):
            pass

    def consider(a):
        nonlocal best, best_score
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return
        a = a.copy()
        a[:, :2] -= np.mean(a[:, :2], axis=0)
        ii, jj = np.triu_indices(21, 1)
        d = np.sqrt(np.sum((a[ii, :2] - a[jj, :2]) ** 2, axis=1))
        ratio = float(np.min(d / (a[ii, 2] + a[jj, 2])))
        a[:, 2] *= min(1.0, ratio) * (1.0 - 3e-14)
        if np.any(a[:, 2] <= 0):
            return
        extent = np.max(a[:, :2] + a[:, 2, None], axis=0) - np.min(a[:, :2] - a[:, 2, None], axis=0)
        a *= (2.0 / float(np.sum(extent))) * (1.0 - 3e-14)
        score = exact_score(a)
        if score > best_score:
            best_score = score
            best = {'circles': a.tolist()}

    fallback = np.array([[float(j)*0.2, float(i)*0.2, 0.1] for i in range(5) for j in range(5) if i*5+j < 21])
    consider(fallback)

    def solve_layout(counts, perturb=0.0):
        check()
        singles = []
        pairs = []
        for col, count in enumerate(counts):
            check()
            x = math.sqrt(3.0) * col
            if count % 2:
                singles.append((x, 0.0))
                for j in range(1, (count+1)//2):
                    check()
                    pairs.append((x, 2.0*j))
            else:
                for j in range(count//2):
                    check()
                    pairs.append((x, 2.0*j+1.0))
        k = len(singles)
        p = len(pairs)
        m = k+p
        nvar = 2*m+p+1
        ridx = m+p
        wi = nvar-1
        groups = singles+pairs
        z = np.zeros(nvar)
        width = math.sqrt(3.0)*(len(counts)-1)+2.0
        height = 2.0*max(counts)
        scale = 2.0/(width+height)
        z[:m] = [(v[0]+1.0)*scale for v in groups]
        z[m:m+p] = [v[1]*scale for v in pairs]
        z[ridx:ridx+m] = scale
        z[wi] = width*scale
        if perturb:
            z[:m+p] += rng.normal(0.0, perturb, m+p)
        weights = np.r_[np.ones(k), np.full(p, 2.0)]
        grad = np.zeros(nvar)
        grad[ridx:ridx+m] = -weights
        A, B, R = [], [], []

        def add_pair(i, j, opposite=False):
            aa = np.zeros(nvar)
            bb = np.zeros(nvar)
            rr = np.zeros(nvar)
            aa[i] += 1.0
            aa[j] -= 1.0
            if i >= k:
                bb[m+i-k] += 1.0
            if j >= k:
                bb[m+j-k] += 1.0 if opposite else -1.0
            rr[ridx+i] += 1.0
            rr[ridx+j] += 1.0
            A.append(aa)
            B.append(bb)
            R.append(rr)

        for i in range(m):
            check()
            for j in range(i+1, m):
                check()
                add_pair(i, j)
                if i >= k:
                    add_pair(i, j, True)
        A = np.asarray(A)
        B = np.asarray(B)
        R = np.asarray(R)
        rows, constants = [], []
        for i in range(m):
            check()
            row = np.zeros(nvar)
            row[i], row[ridx+i] = 1.0, -1.0
            rows.append(row)
            constants.append(0.0)
            row = np.zeros(nvar)
            row[wi], row[i], row[ridx+i] = 1.0, -1.0, -1.0
            rows.append(row)
            constants.append(0.0)
            row = np.zeros(nvar)
            row[wi], row[ridx+i] = -0.5, -1.0
            if i >= k:
                row[m+i-k] = -1.0
            rows.append(row)
            constants.append(1.0)
            if i >= k:
                row = np.zeros(nvar)
                row[m+i-k], row[ridx+i] = 1.0, -1.0
                rows.append(row)
                constants.append(0.0)
        L = np.asarray(rows)
        c = np.asarray(constants)
        last = [z.copy()]
        cached = [None, None, None]

        def calc(v):
            check()
            if cached[0] is None or not np.array_equal(cached[0], v):
                dx, dy, rs = A@v, B@v, R@v
                cached[0] = v.copy()
                cached[1] = np.r_[dx*dx+dy*dy-rs*rs, L@v+c]
                cached[2] = np.vstack((2.0*(dx[:, None]*A+dy[:, None]*B-rs[:, None]*R), L))
            return cached

        def callback(v):
            last[0] = v.copy()
            check()

        def unpack(v):
            out = np.empty((21, 3))
            index = 0
            for i in range(m):
                if i < k:
                    out[index] = (v[i], 0.0, v[ridx+i])
                    index += 1
                else:
                    y = v[m+i-k]
                    out[index] = (v[i], y, v[ridx+i])
                    out[index+1] = (v[i], -y, v[ridx+i])
                    index += 2
            return out

        bounds = [(None, None)]*nvar
        for i in range(m):
            check()
            bounds[ridx+i] = (0.0001, 0.5)
        bounds[wi] = (0.1, 1.9)
        try:
            result = minimize(lambda v: float(grad@v), z, jac=lambda v: grad,
                              method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': lambda v: calc(v)[1], 'jac': lambda v: calc(v)[2]},
                              callback=callback,
                              options={'maxiter': int(params['iterations']), 'ftol': 2e-12, 'disp': False})
            consider(unpack(result.x))
        except StopSearch:
            consider(unpack(last[0]))
            raise
        return unpack(last[0])

    layouts = [
        [3,4,3,4,3,4],
        [4,5,4,5,3],
        [4,4,5,4,4],
        [2,3,4,3,4,3,2],
        [4,5,3,5,4],
        [1,2,3,4,5,6],
        [3,5,5,5,3],
        [3,3,3,3,3,3,3],
    ]
    try:
        for counts in layouts:
            check()
            solve_layout(counts)
        while time.monotonic() < deadline:
            check()
            counts = layouts[int(rng.integers(0, len(layouts)))]
            solve_layout(counts, float(params['jitter']))
    except StopSearch:
        pass
    return best
