"""Seed program for circles-rectangle (21 circles in a rectangle of perimeter 4, maximize the sum of radii).

Contract of every candidate program:
    PARAMS = {"name": (default, low, high[, "log"])}
    construct(params: dict, seed: int, budget_s: float, init=None) -> {"circles": [[x, y, r], ...]}

Frozen verifier rules (no tolerance): every r > 0; bounding box width + height <= 2, computed in exact Fractions of
the floats; circles i, j are rejected when float64 sqrt(dx^2 + dy^2) < ri + rj (tangency passes).

Method (derived from the toy seed): the box is [0, W] x [0, 2 - W] with W a variable. Start from a jittered grid in
a box of aspect `aspect` (init=None), or from `init` rescaled so its bounding box has perimeter 4 (init can be an
invalid cold start). Minimize -sum(r) + mu * (squared violations) with L-BFGS while mu grows, then solve an LP for
the radii and the box at fixed centres, then shrink radii slightly until the exact rules above pass. The LP is also
run on `init` itself. The best valid candidate is returned, and a valid `init` is returned when nothing beats it.
"""
import time
from fractions import Fraction

import numpy as np
from scipy.optimize import linprog, minimize

N = 21
PARAMS = {
    "aspect": (1.0, 0.5, 2.0, "log"),       # width / height of the starting box (init=None only)
    "rows": (5, 3, 7),
    "jitter": (0.03, 0.0, 0.15),
    "init_jitter": (0.0, 0.0, 0.05),        # noise on the centres of init before the descent
    "mu0": (10.0, 1.0, 1000.0, "log"),
    "mu_growth": (8.0, 2.0, 30.0),
    "stages": (6, 2, 12),
    "maxiter": (150, 30, 1000),
}
_I, _J = np.triu_indices(N, 1)
_SHRINK = 1e-12


def _layout(rows, jitter, w, h, rng):
    per_row = int(np.ceil(N / rows))
    pts = []
    for k in range(rows):
        count = min(per_row, N - len(pts))
        if count <= 0:
            break
        off = 0.25 / per_row if k % 2 else 0.0
        pts += [((i + 0.5) / per_row + off, (k + 0.5) / rows) for i in range(count)]
    while len(pts) < N:
        pts.append(tuple(rng.uniform(0.1, 0.9, 2)))
    p = np.clip(np.array(pts[:N]) + rng.normal(0, jitter, (N, 2)), 0.02, 0.98)
    return p * np.array([w, h])


def _penalty(z, mu):
    x, y, r, w = z[:N], z[N:2 * N], z[2 * N:3 * N], z[3 * N]
    h = 2.0 - w
    dx, dy = x[_I] - x[_J], y[_I] - y[_J]
    d = np.sqrt(dx * dx + dy * dy + 1e-18)
    v = np.maximum(r[_I] + r[_J] - d, 0)
    b = [np.maximum(r - x, 0), np.maximum(r + x - w, 0), np.maximum(r - y, 0), np.maximum(r + y - h, 0)]
    neg = np.maximum(-r, 0)
    f = -r.sum() + mu * ((v * v).sum() + sum((bb * bb).sum() for bb in b) + (neg * neg).sum())
    gx, gy, gr = np.zeros(N), np.zeros(N), -np.ones(N)
    c = 2 * mu * v
    np.add.at(gr, _I, c)
    np.add.at(gr, _J, c)
    np.add.at(gx, _I, -c * dx / d)
    np.add.at(gx, _J, c * dx / d)
    np.add.at(gy, _I, -c * dy / d)
    np.add.at(gy, _J, c * dy / d)
    gr += 2 * mu * (b[0] + b[1] + b[2] + b[3]) - 2 * mu * neg
    gx += 2 * mu * (-b[0] + b[1])
    gy += 2 * mu * (-b[2] + b[3])
    gw = 2 * mu * (-b[1].sum() + b[3].sum())
    return f, np.concatenate([gx, gy, gr, [gw]])


def _lp_radii(p):
    """Max sum(r) at fixed centres; variables r (N) and the box X0, X1, Y0, Y1 with X1 - X0 + Y1 - Y0 <= 2."""
    m = len(_I)
    a = np.zeros((m + 4 * N + 1, N + 4))
    b = np.zeros(m + 4 * N + 1)
    a[np.arange(m), _I] = 1
    a[np.arange(m), _J] = 1
    b[:m] = np.sqrt(((p[_I] - p[_J]) ** 2).sum(1))
    for k, (col, sign, coord) in enumerate([(N, 1, 0), (N + 1, -1, 0), (N + 2, 1, 1), (N + 3, -1, 1)]):
        rows = m + k * N + np.arange(N)
        a[rows, np.arange(N)] = 1                # r_i + X0 <= x_i ; r_i - X1 <= -x_i ; same for y
        a[rows, col] = sign
        b[rows] = sign * p[:, coord]
    a[-1, N:] = [-1, 1, -1, 1]
    b[-1] = 2.0
    bounds = [(0, None)] * N + [(None, None)] * 4
    res = linprog(-np.r_[np.ones(N), np.zeros(4)], A_ub=a, b_ub=b, bounds=bounds, method="highs")
    return res.x[:N] if res.success else np.zeros(N)


def _half_perimeter(p, r):
    """Bounding-box width + height in exact Fractions (the verifier's rule)."""
    xs, ys, rs = ([Fraction(float(t)) for t in col] for col in (p[:, 0], p[:, 1], r))
    width = max(x + q for x, q in zip(xs, rs)) - min(x - q for x, q in zip(xs, rs))
    height = max(y + q for y, q in zip(ys, rs)) - min(y - q for y, q in zip(ys, rs))
    return width + height


def _overlap_free(p, r):
    d = np.sqrt((p[_I, 0] - p[_J, 0]) ** 2 + (p[_I, 1] - p[_J, 1]) ** 2)
    return not bool((d < r[_I] + r[_J]).any())


def _valid(p, r):
    return (np.isfinite(p).all() and np.isfinite(r).all() and bool((r > 0).all())
            and _half_perimeter(p, r) <= 2 and _overlap_free(p, r))


def _repair(p, r):
    """Shrink radii until the exact rules pass: overlapping pairs first, then the exact perimeter."""
    r = np.maximum(np.asarray(r, dtype=float), 1e-12) * (1 - _SHRINK)
    for _ in range(5):
        if _overlap_free(p, r):
            break
        d = np.sqrt(((p[_I] - p[_J]) ** 2).sum(1))
        for i, j, dd in zip(_I, _J, d):
            if r[i] + r[j] > dd * (1 - _SHRINK):
                s = dd * (1 - _SHRINK) / (r[i] + r[j])
                r[i] *= s
                r[j] *= s
    for _ in range(60):
        excess = _half_perimeter(p, r) - 2
        if excess <= 0:
            break
        r *= 1 - max(2.0 * float(excess) / float(r.sum()), _SHRINK)
    return np.maximum(r, 1e-12)


def _as_arrays(cand):
    a = np.asarray(cand["circles"], dtype=float)
    return a[:, :2].copy(), a[:, 2].copy()


def _out(p, r):
    return {"circles": [[float(x), float(y), float(q)] for (x, y), q in zip(p, r)]}


def construct(params, seed, budget_s, init=None):
    t_end = time.monotonic() + 0.8 * budget_s
    rng = np.random.default_rng(seed)
    best, best_sum = None, -np.inf
    if init is not None:
        p0, r0 = _as_arrays(init)
        if _valid(p0, r0):
            best, best_sum = init, float(r0.sum())
        # init rescaled so that its bounding box sits at the origin with width + height = 2
        lo = (p0 - np.abs(r0)[:, None]).min(0)
        side = (p0 + np.abs(r0)[:, None]).max(0) - lo
        s = 2.0 / side.sum()
        p, r, w = (p0 - lo) * s, np.abs(r0) * s, side[0] * s
        p = p + rng.normal(0, float(params["init_jitter"]), p.shape)
        cands = [p.copy()]
    else:
        a = float(params["aspect"])
        w = 2 * a / (1 + a)
        p = _layout(int(params["rows"]), float(params["jitter"]), w, 2 - w, rng)
        r = np.full(N, 0.05)
        cands = []
    z = np.concatenate([p[:, 0], p[:, 1], r, [w]])
    mu = float(params["mu0"])
    for _ in range(int(params["stages"])):
        if time.monotonic() > t_end:
            break
        z = minimize(_penalty, z, args=(mu,), jac=True, method="L-BFGS-B",
                     options={"maxiter": int(params["maxiter"])}).x
        mu *= float(params["mu_growth"])
    cands.append(np.stack([z[:N], z[N:2 * N]], 1))
    for pc in cands:
        rc = _repair(pc, _lp_radii(pc))
        if _valid(pc, rc) and float(rc.sum()) > best_sum:
            best, best_sum = _out(pc, rc), float(rc.sum())
    if best is None:                         # should not happen: return tiny valid circles at the last centres
        pc = cands[-1]
        best = _out(pc, _repair(pc, np.full(N, 1e-6)))
    return best
