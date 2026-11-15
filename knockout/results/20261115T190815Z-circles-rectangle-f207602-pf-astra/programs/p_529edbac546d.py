import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    "tilt": (0.7, 0.1, 1.8),
    "trials": (4, 1, 10),
    "phase_fraction": (0.43, 0.2, 0.65),
    "maxiter": (65, 15, 150),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class StopSearch(Exception):
    pass


def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0] - a[J, 0]) ** 2 + (a[I, 1] - a[J, 1]) ** 2)
    if np.any(d < a[I, 2] + a[J, 2]):
        return -np.inf
    x = list(map(lambda v: Fraction(float(v)), a[:, 0]))
    y = list(map(lambda v: Fraction(float(v)), a[:, 1]))
    r = list(map(lambda v: Fraction(float(v)), a[:, 2]))
    perimeter = (max(map(lambda t: t[0] + t[1], zip(x, r)))
                 - min(map(lambda t: t[0] - t[1], zip(x, r)))
                 + max(map(lambda t: t[0] + t[1], zip(y, r)))
                 - min(map(lambda t: t[0] - t[1], zip(y, r))))
    if perimeter > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    fallback = {"circles": [[(i % 7 + 0.5) * 0.19999999999999,
                              (i // 7 + 0.5) * 0.19999999999999,
                              0.09999999999998] for i in range(N)]}
    best = init if init is not None else fallback
    a = np.asarray(best["circles"], dtype=float)
    best_score = exact_score(a)
    if not np.isfinite(best_score):
        best = fallback
        a = np.asarray(best["circles"], dtype=float)
        best_score = exact_score(a)
    lo = np.min(a[:, :2] - a[:, 2, None], axis=0)
    hi = np.max(a[:, :2] + a[:, 2, None], axis=0)
    total = a[:, 2].sum()
    p = (a[:, :2] - (lo + hi) / 2) / total
    base = np.r_[p[:, 0], p[:, 1], a[:, 2] / total, (hi - lo) / total]
    best_z = base.copy()
    phase_end = deadline
    last = base.copy()
    weights = np.zeros(N)
    objective_grad = np.zeros(65)
    objective_grad[63:] = 1
    equality_jac = np.zeros((1, 65))
    equality_jac[0, 42:63] = 1
    rows = np.arange(M)
    k = np.arange(N)
    wall_jac = np.zeros((4 * N, 65))
    wall_jac[k, k] = 1
    wall_jac[N + k, k] = -1
    wall_jac[2 * N + k, N + k] = 1
    wall_jac[3 * N + k, N + k] = -1
    wall_jac[np.arange(4 * N), 42 + np.tile(k, 4)] = -1
    wall_jac[:2 * N, 63] = 0.5
    wall_jac[2 * N:, 64] = 0.5

    def check_time():
        if time.monotonic() >= min(deadline, phase_end):
            raise StopSearch()

    def consider(z):
        nonlocal best, best_score, best_z
        if time.monotonic() >= deadline:
            return
        q = np.column_stack((z[:N], z[N:2*N]))
        r = z[2*N:3*N].copy()
        if not np.isfinite(z).all() or np.min(r) <= 0:
            return
        d = np.sqrt(np.sum((q[I] - q[J]) ** 2, axis=1))
        factor = min(1.0, float(np.min(d / (r[I] + r[J]))))
        if factor <= 0:
            return
        r *= factor * (1 - 8e-15)
        lower = np.min(q - r[:, None], axis=0)
        upper = np.max(q + r[:, None], axis=0)
        scale = 2 * (1 - 8e-15) / float(np.sum(upper - lower))
        if float(np.sum(r * scale)) <= best_score:
            return
        out = np.column_stack(((q - (lower + upper) / 2) * scale, r * scale))
        value = exact_score(out)
        if value > best_score:
            best_score = value
            best = {"circles": out.tolist()}
            rr = out[:, 2].sum()
            ll = np.min(out[:, :2] - out[:, 2, None], axis=0)
            hh = np.max(out[:, :2] + out[:, 2, None], axis=0)
            pp = (out[:, :2] - (ll + hh) / 2) / rr
            best_z = np.r_[pp[:, 0], pp[:, 1], out[:, 2] / rr, (hh - ll) / rr]

    def objective(z):
        check_time()
        return z[63] + z[64] - np.dot(weights, z[42:63]), objective_grad

    def constraints(z):
        check_time()
        x, y, r = z[:21], z[21:42], z[42:63]
        d = np.sqrt((x[I] - x[J]) ** 2 + (y[I] - y[J]) ** 2 + 1e-30)
        return np.r_[d - r[I] - r[J],
                     x + z[63] / 2 - r, z[63] / 2 - x - r,
                     y + z[64] / 2 - r, z[64] / 2 - y - r]

    def jacobian(z):
        check_time()
        dx, dy = z[I] - z[J], z[21 + I] - z[21 + J]
        d = np.sqrt(dx * dx + dy * dy + 1e-30)
        jac = np.zeros((M + 4*N, 65))
        jac[rows, I] = dx / d
        jac[rows, J] = -dx / d
        jac[rows, 21 + I] = dy / d
        jac[rows, 21 + J] = -dy / d
        jac[rows, 42 + I] = -1
        jac[rows, 42 + J] = -1
        jac[M:] = wall_jac
        return jac

    def equality(z):
        check_time()
        return np.array([z[42:63].sum() - 1])

    def callback(z):
        nonlocal last
        check_time()
        last = z.copy()
        consider(z)

    cons = [{"type": "ineq", "fun": constraints, "jac": jacobian},
            {"type": "eq", "fun": equality, "jac": lambda z: equality_jac}]
    bounds = [(None, None)] * 42 + [(1e-7, 0.35)] * 21 + [(0.03, 1.8)] * 2
    trials = int(params["trials"])
    for trial in range(trials):
        if time.monotonic() >= deadline:
            break
        remaining = deadline - time.monotonic()
        slot = remaining / (trials - trial)
        slot_end = time.monotonic() + slot
        last = best_z.copy()
        direction = rng.normal(size=N)
        # Correlate the tilt across reflection partners, when present.
        xy = np.column_stack((last[:21], last[21:42]))
        reflected = xy.copy()
        reflected[:, 1] *= -1
        partners = np.argmin(np.sum((reflected[:, None, :] - xy[None, :, :]) ** 2, axis=2), axis=1)
        direction = 0.5 * (direction + direction[partners])
        direction -= direction.mean()
        direction /= max(float(np.std(direction)), 1e-8)
        for phase in range(2):
            if time.monotonic() >= deadline:
                break
            if phase == 0:
                weights = float(params["tilt"]) * direction
                phase_end = min(deadline, time.monotonic() + slot * float(params["phase_fraction"]))
            else:
                weights = np.zeros(N)
                phase_end = min(deadline, slot_end)
            objective_grad[42:63] = -weights
            try:
                result = minimize(objective, last, jac=True, method="SLSQP",
                                  bounds=bounds, constraints=cons, callback=callback,
                                  options={"maxiter": int(params["maxiter"]), "ftol": 2e-12, "disp": False})
                last = result.x.copy()
            except StopSearch:
                pass
            consider(last)
    return best
