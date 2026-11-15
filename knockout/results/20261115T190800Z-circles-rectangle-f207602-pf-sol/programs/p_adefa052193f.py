import time
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize

PARAMS = {
    'trust': (0.065, 0.015, 0.18),
    'elastic': (0.65, 0.2, 3.0),
    'steps': (8, 3, 20),
    'weight_noise': (0.16, 0.0, 0.5),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
D = 64

class StopSearch(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    deadline = time.monotonic() + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def exact(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -np.inf
        r = a[:, 2]
        ds = np.sqrt((a[I, 0] - a[J, 0]) ** 2 + (a[I, 1] - a[J, 1]) ** 2)
        if np.any(ds < r[I] + r[J]):
            return -np.inf
        x = list(map(lambda t: Fraction(float(t)), a[:, 0]))
        y = list(map(lambda t: Fraction(float(t)), a[:, 1]))
        q = list(map(lambda t: Fraction(float(t)), r))
        perimeter = (max(u + v for u, v in zip(x, q)) - min(u - v for u, v in zip(x, q))
                     + max(u + v for u, v in zip(y, q)) - min(u - v for u, v in zip(y, q)))
        return float(np.sum(r)) if perimeter <= 2 else -np.inf

    if init is not None:
        a0 = np.asarray(init['circles'], dtype=float)
        best_score = exact(a0)
    else:
        a0 = np.array([[(k % 7 + .5) / 3.5, (k // 7 + .5) / 3.5, .06] for k in range(N)])

    lo = np.min(a0[:, :2] - a0[:, 2, None], axis=0)
    sides = np.max(a0[:, :2] + a0[:, 2, None], axis=0) - lo
    scale = 2.0 / sides.sum()
    p = (a0[:, :2] - lo) * scale
    base = np.r_[p[:, 0], p[:, 1], a0[:, 2] * scale, sides[0] * scale]

    def constraints(z, jac=False):
        check()
        x, y, r, w = z[:21], z[21:42], z[42:63], z[63]
        dx, dy = x[I] - x[J], y[I] - y[J]
        dist = np.maximum(np.sqrt(dx * dx + dy * dy), 1e-15)
        g = np.r_[dist - r[I] - r[J], x-r, w-x-r, y-r, 2-w-y-r]
        if not jac:
            return g
        a = np.zeros((M + 84, D))
        row = np.arange(M)
        a[row, I] = dx / dist
        a[row, J] = -dx / dist
        a[row, 21+I] = dy / dist
        a[row, 21+J] = -dy / dist
        a[row, 42+I] = -1
        a[row, 42+J] = -1
        k = np.arange(N)
        a[M+k, k] = 1
        a[M+21+k, k] = -1
        a[M+21+k, 63] = 1
        a[M+42+k, 21+k] = 1
        a[M+63+k, 21+k] = -1
        a[M+63+k, 63] = -1
        a[M+k, 42+k] = -1
        a[M+21+k, 42+k] = -1
        a[M+42+k, 42+k] = -1
        a[M+63+k, 42+k] = -1
        return a

    def retain(z):
        nonlocal best, best_score
        check()
        a = np.column_stack((z[:21], z[21:42], np.maximum(z[42:63], 1e-12)))
        r = a[:, 2]
        dist = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
        factor = min(1.0, float(np.min(dist / (r[I]+r[J]))))
        a[:, 2] *= max(0.0, factor) * (1-3e-14)
        low = np.min(a[:, :2]-a[:, 2, None], axis=0)
        high = np.max(a[:, :2]+a[:, 2, None], axis=0)
        a[:, :2] -= low
        a *= min(1.0, 2.0 / np.sum(high-low)) * (1-3e-14)
        score = exact(a)
        if score > best_score:
            best_score = score
            best = {'circles': a.tolist()}

    objective = np.r_[np.zeros(42), -np.ones(21), 0.0]
    bounds = [(0., 2.)]*42 + [(1e-5, .6)]*21 + [(.15, 1.85)]
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    slack_matrix = np.zeros((M+84, M))
    slack_matrix[:M] = -np.eye(M)
    trial = 0
    try:
        retain(base)
        while time.monotonic() < deadline:
            check()
            z = base.copy()
            weights = np.ones(N)
            if trial:
                check()
                center = rng.integers(N)
                distances = (z[:21]-z[center])**2 + (z[21:42]-z[21+center])**2
                cluster = np.argsort(distances)[:int(rng.integers(4, 10))]
                weights[cluster] += rng.choice([-1., 1.]) * float(params['weight_noise'])
                z[cluster] += rng.normal(0, .012, len(cluster))
                z[21+cluster] += rng.normal(0, .012, len(cluster))
            for step in range(int(params['steps'])):
                check()
                phase = step / max(1, int(params['steps'])-1)
                mu = float(params['elastic']) * (25.0 ** phase)
                trust = float(params['trust']) * (1-.65*phase)
                g = constraints(z)
                jac = constraints(z, True)
                cost = np.r_[np.zeros(42), -(weights*(1-phase)+phase), 0., np.full(M, mu)]
                limlo = np.maximum(lower-z, -trust)
                limhi = np.minimum(upper-z, trust)
                lp_bounds = list(zip(limlo, limhi)) + [(0., None)]*M
                remaining = deadline-time.monotonic()
                if remaining < .035:
                    raise StopSearch()
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    res = linprog(cost, A_ub=np.hstack((-jac, slack_matrix)), b_ub=g,
                                  bounds=lp_bounds, method='highs',
                                  options={'threads': 1, 'time_limit': max(.001, remaining*.45),
                                           'primal_feasibility_tolerance': 1e-9,
                                           'dual_feasibility_tolerance': 1e-9})
                if res.x is None:
                    break
                z += res.x[:D]
            check()
            retain(z)
            result = minimize(lambda v: (check() or float(objective@v), objective), z,
                              jac=True, method='SLSQP', bounds=bounds,
                              constraints={'type':'ineq', 'fun':constraints,
                                           'jac':lambda v: constraints(v, True)},
                              options={'maxiter':90, 'ftol':2e-12, 'disp':False})
            retain(result.x)
            trial += 1
    except StopSearch:
        pass
    if best is None:
        best = {'circles': [[.06 + .13*(i % 7), .06 + .13*(i//7), .05] for i in range(N)]}
    return best
