import time
import warnings
from fractions import Fraction

import numpy as np
from scipy.optimize import linprog, minimize

PARAMS = {
    'relocation': (0.105, 0.02, 0.3),
    'lp_steps': (13, 5, 35),
    'tilted_steps': (3, 0, 8),
    'tilt': (0.22, 0.0, 0.6),
    'polish_iterations': (45, 10, 120),
}
N = 21
I, J = np.triu_indices(N, 1)
K = len(I)
D = 64

class Timeout(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf
    bounds = [(None, None)] * 42 + [(1e-8, None)] * 21 + [(0.1, 1.9)]
    objective = np.r_[np.zeros(42), -np.ones(21), 0.0]
    pair_rows = np.arange(K)
    idx = np.arange(N)

    def check():
        if time.monotonic() >= end:
            raise Timeout()

    def verify(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -np.inf
        r = a[:, 2]
        dx = a[I, 0] - a[J, 0]
        dy = a[I, 1] - a[J, 1]
        if np.any(np.sqrt(dx * dx + dy * dy) < r[I] + r[J]):
            return -np.inf
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        q = [Fraction(float(v)) for v in r]
        perimeter = max(u + v for u, v in zip(x, q)) - min(u - v for u, v in zip(x, q))
        perimeter += max(u + v for u, v in zip(y, q)) - min(u - v for u, v in zip(y, q))
        return float(np.sum(r)) if perimeter <= 2 else -np.inf

    if init is not None:
        initial = np.asarray(init['circles'], dtype=float)
        best_score = verify(initial)
        lo = np.min(initial[:, :2] - initial[:, 2, None], axis=0)
        sides = np.max(initial[:, :2] + initial[:, 2, None], axis=0) - lo
        scale = 2.0 / sides.sum()
        p = (initial[:, :2] - lo) * scale
        r = initial[:, 2] * scale
        root = np.r_[p[:, 0], p[:, 1], r, sides[0] * scale]
    else:
        p = np.array([((k % 5 + 0.5) / 5, (k // 5 + 0.5) / 5) for k in range(N)])
        root = np.r_[p[:, 0], p[:, 1], np.full(N, 0.07), 1.0]

    def consider(z):
        nonlocal best, best_score
        check()
        a = np.column_stack((z[:N], z[N:2*N], z[2*N:3*N])).copy()
        if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return
        a[:, :2] -= np.mean(a[:, :2], axis=0)
        r = a[:, 2]
        dx = a[I, 0] - a[J, 0]
        dy = a[I, 1] - a[J, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        factor = min(1.0, float(np.min(dist / (r[I] + r[J]))))
        r *= factor * (1.0 - 3e-15)
        sides = np.max(a[:, :2] + r[:, None], axis=0) - np.min(a[:, :2] - r[:, None], axis=0)
        scale = min(1.0, 2.0 / float(sides.sum())) * (1.0 - 3e-15)
        a *= scale
        for _ in range(5):
            check()
            score = verify(a)
            if score > best_score:
                best_score = score
                best = {'circles': a.tolist()}
            if np.isfinite(score):
                break
            a[:, 2] *= 1.0 - 2e-13

    # Static rectangle inequalities: A z <= b.
    box = np.zeros((4*N, D))
    rhs_box = np.zeros(4*N)
    box[idx, idx] = -1
    box[idx, 2*N+idx] = 1
    box[N+idx, idx] = 1
    box[N+idx, 2*N+idx] = 1
    box[N+idx, -1] = -1
    box[2*N+idx, N+idx] = -1
    box[2*N+idx, 2*N+idx] = 1
    box[3*N+idx, N+idx] = 1
    box[3*N+idx, 2*N+idx] = 1
    box[3*N+idx, -1] = 1
    rhs_box[3*N:] = 2
    A = np.zeros((K+4*N, D))
    A[K:] = box
    A[pair_rows, 2*N+I] = 1
    A[pair_rows, 2*N+J] = 1
    b = np.r_[np.zeros(K), rhs_box]

    def constraints(z):
        dx = z[I] - z[J]
        dy = z[N+I] - z[N+J]
        distances = np.sqrt(dx*dx + dy*dy + 1e-30)
        return np.r_[distances-z[2*N+I]-z[2*N+J], rhs_box-box@z]

    def jacobian(z):
        dx = z[I] - z[J]
        dy = z[N+I] - z[N+J]
        distances = np.sqrt(dx*dx + dy*dy + 1e-30)
        jac = np.zeros((K+4*N, D))
        jac[pair_rows, I] = dx/distances
        jac[pair_rows, J] = -dx/distances
        jac[pair_rows, N+I] = dy/distances
        jac[pair_rows, N+J] = -dy/distances
        jac[pair_rows, 2*N+I] = -1
        jac[pair_rows, 2*N+J] = -1
        jac[K:] = -box
        return jac

    def polish(z, iterations):
        check()
        def callback(q):
            check()
        out = minimize(lambda q: float(objective@q), z, jac=lambda q: objective,
                       method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                       callback=callback,
                       options={'maxiter': iterations, 'ftol': 8e-14, 'disp': False})
        consider(out.x)
        return out.x

    try:
        root = polish(root, 12)
        trial = 0
        while time.monotonic() < end:
            check()
            z = root.copy()
            p = np.column_stack((z[:N], z[N:2*N]))
            if trial % 3 == 0:
                # Change an entire local cluster, rather than releasing one contact.
                anchor = int(rng.integers(N))
                d = np.sum((p-p[anchor])**2, axis=1)
                selected = np.argsort(d)[:int(rng.integers(3, 7))]
                center = np.mean(p[selected], axis=0)
                angle = rng.uniform(-1.9, 1.9)
                rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                p[selected] = center + (p[selected]-center)@rotation
                p[selected] += rng.normal(0, float(params['relocation']), (len(selected), 2))
            elif trial % 3 == 1:
                # Reinsert several small circles into unrelated interstices.
                small = np.argsort(z[42:63])[:9]
                selected = rng.choice(small, size=3, replace=False)
                partners = rng.choice(N, size=3, replace=False)
                p[selected] = 0.5*(p[partners] + p[rng.integers(N, size=3)])
                p[selected] += rng.normal(0, 0.035, (3, 2))
            else:
                p += rng.normal(0, float(params['relocation'])*0.6, p.shape)
            z[:N], z[N:2*N] = p[:, 0], p[:, 1]
            weights = np.exp(rng.normal(0, float(params['tilt']), N))
            for step in range(int(params['lp_steps'])):
                check()
                dx = z[I]-z[J]
                dy = z[N+I]-z[N+J]
                distances = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-20)
                ux, uy = dx/distances, dy/distances
                A[pair_rows, I] = -ux
                A[pair_rows, J] = ux
                A[pair_rows, N+I] = -uy
                A[pair_rows, N+J] = uy
                c = objective.copy()
                if step < int(params['tilted_steps']):
                    c[42:63] = -weights
                remaining = end-time.monotonic()
                if remaining < 0.015:
                    raise Timeout()
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs',
                                     options={'threads': 1, 'time_limit': min(0.07, remaining),
                                              'dual_feasibility_tolerance': 1e-9,
                                              'primal_feasibility_tolerance': 1e-9})
                check()
                if not result.success:
                    break
                change = np.max(np.abs(result.x-z))
                z = result.x
                if step >= int(params['tilted_steps']) and change < 1e-7:
                    break
            consider(z)
            if time.monotonic() < end-0.025:
                z = polish(z, int(params['polish_iterations']))
            trial += 1
    except Timeout:
        pass
    if best is None:
        # Deterministic valid cold-start fallback.
        best = {'circles': [[(k % 5)*0.18, (k // 5)*0.18, 0.04] for k in range(N)]}
    return best
