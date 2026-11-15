import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    "perturbation": (0.035, 0.005, 0.15),
    "maxiter": (130, 40, 240),
    "time_fraction": (0.69, 0.5, 0.72)
}

class StopSearch(Exception):
    pass


def construct(params: dict, seed: int, budget_s: float, init=None):
    started = time.monotonic()
    deadline = started + float(params["time_fraction"]) * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -float("inf")
    best_q = None
    model_score = -float("inf")

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def exact_score(candidate):
        check()
        try:
            c = np.asarray(candidate["circles"], dtype=np.float64)
            if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
                return -float("inf")
            xs, ys, rs = [], [], []
            for x, y, r in c:
                check()
                xs.append(Fraction(float(x)))
                ys.append(Fraction(float(y)))
                rs.append(Fraction(float(r)))
            width = max(x+r for x, r in zip(xs, rs)) - min(x-r for x, r in zip(xs, rs))
            height = max(y+r for y, r in zip(ys, rs)) - min(y-r for y, r in zip(ys, rs))
            if width + height > Fraction(2):
                return -float("inf")
            for a, b in itertools.combinations(c, 2):
                check()
                d = np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
                if d < a[2] + b[2]:
                    return -float("inf")
            return float(np.sum(c[:, 2]))
        except (KeyError, TypeError, ValueError, OverflowError):
            return -float("inf")

    try:
        if init is not None:
            best_score = exact_score(init)

        # Linear expansion from nine mirrored pairs and three axis singles.
        T = np.zeros((64, 34))
        for k in range(9):
            check()
            for side in range(2):
                check()
                i = 2*k + side
                T[3*i, 3*k] = -1.0 if side == 0 else 1.0
                T[3*i, 33] = 0.5
                T[3*i+1, 3*k+1] = 1.0
                T[3*i+2, 3*k+2] = 1.0
        for k in range(3):
            check()
            i = 18+k
            T[3*i, 33] = 0.5
            T[3*i+1, 27+2*k] = 1.0
            T[3*i+2, 28+2*k] = 1.0
        T[63, 33] = 1.0
        objective_weights = T[2:63:3].sum(axis=0)

        pairs = []
        for i in range(21):
            check()
            for j in range(i+1, 21):
                check()
                ri = i ^ 1 if i < 18 else i
                rj = j ^ 1 if j < 18 else j
                reflected = (min(ri, rj), max(ri, rj))
                if (i, j) <= reflected:
                    pairs.append((i, j))
        ii = np.array([p[0] for p in pairs], dtype=int)
        jj = np.array([p[1] for p in pairs], dtype=int)
        DX = T[3*ii] - T[3*jj]
        DY = T[3*ii+1] - T[3*jj+1]
        RS = T[3*ii+2] + T[3*jj+2]

        wall_rows, wall_offsets = [], []
        representatives = list(range(1, 18, 2)) + [18, 19, 20]
        for i in representatives:
            check()
            wall_rows.extend((T[63]-T[3*i]-T[3*i+2],
                              T[3*i+1]-T[3*i+2],
                              -T[63]-T[3*i+1]-T[3*i+2]))
            wall_offsets.extend((0.0, 0.0, 2.0))
        A = np.asarray(wall_rows)
        b = np.asarray(wall_offsets)
        all_i, all_j = np.triu_indices(21, 1)

        def repaired(q):
            check()
            c = (T @ q)[:63].reshape(21, 3).copy()
            if not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
                return None
            delta = c[all_i, :2] - c[all_j, :2]
            distances = np.sqrt(np.sum(delta*delta, axis=1))
            sums = c[all_i, 2] + c[all_j, 2]
            shrink = min(1.0, float(np.min(distances/sums)))
            if shrink <= 0:
                return None
            c[:, 2] *= shrink * (1.0-3e-14)
            lo = np.min(c[:, :2]-c[:, 2, None], axis=0)
            hi = np.max(c[:, :2]+c[:, 2, None], axis=0)
            extent = hi-lo
            scale = (2.0-8e-14)/float(np.sum(extent))
            c[:, :2] -= lo
            c *= scale
            return c, float(extent[0]*scale)

        def encode(c, width):
            q = np.empty(34)
            for k in range(9):
                check()
                q[3*k] = 0.5*(c[2*k+1, 0]-c[2*k, 0])
                q[3*k+1] = c[2*k+1, 1]
                q[3*k+2] = c[2*k+1, 2]
            for k in range(3):
                check()
                q[27+2*k] = c[18+k, 1]
                q[28+2*k] = c[18+k, 2]
            q[33] = width
            return q

        def retain(q):
            nonlocal best, best_score, best_q, model_score
            check()
            result = repaired(q)
            if result is None:
                return
            c, width = result
            score = float(np.sum(c[:, 2]))
            if score <= model_score and score <= best_score:
                return
            candidate = {"circles": c.tolist()}
            verified = exact_score(candidate)
            if verified > model_score:
                model_score = verified
                best_q = encode(c, width)
            if verified > best_score:
                best_score = verified
                best = candidate

        radius = 2.0/(10.0+5.0*np.sqrt(3.0))
        q0 = np.zeros(34)
        q0[33] = 8.0*radius
        pair_index = 0
        single_index = 0
        for row in range(6):
            check()
            y = radius*(1.0+row*np.sqrt(3.0))
            offsets = (1.0, 3.0) if row % 2 == 0 else (2.0,)
            for offset in offsets:
                check()
                q0[3*pair_index:3*pair_index+3] = (offset*radius, y, radius)
                pair_index += 1
            if row % 2:
                q0[27+2*single_index:29+2*single_index] = (y, radius)
                single_index += 1
        retain(q0)

        bounds = [(0.0, 2.0)]*34
        for k in range(9):
            check()
            bounds[3*k+2] = (1e-7, 0.5)
        for k in range(3):
            check()
            bounds[28+2*k] = (1e-7, 0.5)
        bounds[33] = (0.15, 1.85)

        cached_q = None
        cached_values = None
        cached_jac = None
        def constraints(q):
            nonlocal cached_q, cached_values, cached_jac
            check()
            if cached_q is None or not np.array_equal(q, cached_q):
                dx = DX @ q
                dy = DY @ q
                distance = np.sqrt(dx*dx+dy*dy)
                safe = np.maximum(distance, 1e-15)
                cached_values = np.concatenate((distance-RS@q, A@q+b))
                cached_jac = np.vstack(((dx/safe)[:, None]*DX + (dy/safe)[:, None]*DY-RS, A))
                cached_q = q.copy()
            return cached_values

        def constraint_jac(q):
            constraints(q)
            return cached_jac

        def objective(q):
            check()
            return -float(objective_weights @ q)

        def objective_jac(q):
            check()
            return -objective_weights

        attempt = 0
        while time.monotonic() < deadline:
            check()
            if attempt == 0:
                initial = q0.copy()
            else:
                initial = best_q.copy() if best_q is not None else q0.copy()
                magnitude = float(params["perturbation"])*(1.0, 1.8, 0.55, 2.6)[(attempt-1) % 4]
                for k in range(9):
                    check()
                    initial[3*k] = max(0.001, initial[3*k]+rng.normal(0.0, magnitude))
                    initial[3*k+1] = np.clip(initial[3*k+1]+rng.normal(0.0, magnitude), 0.001, 1.999)
                    initial[3*k+2] *= rng.uniform(0.90, 1.02)
                for k in range(3):
                    check()
                    initial[27+2*k] = np.clip(initial[27+2*k]+rng.normal(0.0, magnitude), 0.001, 1.999)
                initial[33] = np.clip(initial[33]+rng.normal(0.0, magnitude*0.5), 0.16, 1.84)
            result = minimize(objective, initial, jac=objective_jac,
                              method="SLSQP", bounds=bounds,
                              constraints={"type": "ineq", "fun": constraints, "jac": constraint_jac},
                              callback=retain,
                              options={"maxiter": int(params["maxiter"]), "ftol": 2e-12, "disp": False})
            retain(result.x)
            attempt += 1
    except StopSearch:
        pass
    return best
