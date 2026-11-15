"""Seed program for the toy problem (26 circles in the unit square, maximize the sum of radii).

Contract of every candidate program:
    PARAMS = {"name": (default, low, high[, "log"])}
    construct(params: dict, seed: int, budget_s: float, init=None) -> object
The returned object is mapped to the frozen candidate format by the island adapter (identity here).

Method: start from a jittered grid, minimize -sum(r) + mu * (squared violations) with L-BFGS while mu grows,
then solve an LP for the radii at fixed centres and shrink them slightly so the exact check passes.
"""
import time

import numpy as np
from scipy.optimize import linprog, minimize

N = 26

PARAMS = {
    "rows": (5, 4, 7),
    "jitter": (0.03, 0.0, 0.15),
    "mu0": (10.0, 1.0, 1000.0, "log"),
    "mu_growth": (8.0, 2.0, 30.0),
    "stages": (6, 2, 12),
    "maxiter": (150, 30, 1000),
}


def _layout(rows, jitter, rng):
    pts = []
    per_row = int(np.ceil(N / rows))
    for k in range(rows):
        count = min(per_row, N - len(pts))
        if count <= 0:
            break
        off = 0.5 / per_row if k % 2 else 0.0
        xs = (np.arange(count) + 0.5) / per_row + off * 0.5
        pts += [(x, (k + 0.5) / rows) for x in xs]
    while len(pts) < N:
        pts.append(tuple(rng.uniform(0.1, 0.9, 2)))
    p = np.clip(np.array(pts[:N]) + rng.normal(0, jitter, (N, 2)), 0.02, 0.98)
    return p


_I, _J = np.triu_indices(N, 1)


def _penalty(z, mu):
    x, y, r = z[:N], z[N:2 * N], z[2 * N:]
    dx, dy = x[_I] - x[_J], y[_I] - y[_J]
    d = np.sqrt(dx * dx + dy * dy + 1e-18)
    v = np.maximum(r[_I] + r[_J] - d, 0)
    b = [np.maximum(r - x, 0), np.maximum(r + x - 1, 0), np.maximum(r - y, 0), np.maximum(r + y - 1, 0)]
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
    return f, np.concatenate([gx, gy, gr])


def _lp_radii(p):
    d = np.sqrt(((p[_I] - p[_J]) ** 2).sum(1))
    a = np.zeros((len(_I), N))
    a[np.arange(len(_I)), _I] = 1
    a[np.arange(len(_I)), _J] = 1
    ub = np.minimum.reduce([p[:, 0], 1 - p[:, 0], p[:, 1], 1 - p[:, 1]])
    res = linprog(-np.ones(N), A_ub=a, b_ub=d, bounds=list(zip(np.zeros(N), ub)), method="highs")
    return res.x if res.success else np.zeros(N)


def _repair(p, r):
    r = np.minimum.reduce([r, p[:, 0], 1 - p[:, 0], p[:, 1], 1 - p[:, 1]]) * (1 - 1e-12)
    for _ in range(3):
        d = np.sqrt(((p[_I] - p[_J]) ** 2).sum(1))
        bad = r[_I] + r[_J] > d * (1 - 1e-12)
        if not bad.any():
            break
        for i, j, dd in zip(_I[bad], _J[bad], d[bad]):
            s = dd * (1 - 1e-12) / (r[i] + r[j])
            r[i] *= s
            r[j] *= s
    return np.maximum(r, 1e-12)


def construct(params, seed, budget_s, init=None):
    t_end = time.monotonic() + 0.8 * budget_s
    rng = np.random.default_rng(seed)
    p = _layout(int(params["rows"]), float(params["jitter"]), rng)
    z = np.concatenate([p[:, 0], p[:, 1], np.full(N, 0.05)])
    mu = float(params["mu0"])
    for _ in range(int(params["stages"])):
        if time.monotonic() > t_end:
            break
        z = minimize(_penalty, z, args=(mu,), jac=True, method="L-BFGS-B",
                     options={"maxiter": int(params["maxiter"])}).x
        mu *= float(params["mu_growth"])
    p = np.clip(np.stack([z[:N], z[N:2 * N]], 1), 1e-6, 1 - 1e-6)
    r = _repair(p, _lp_radii(p))
    return {"circles": [[float(a), float(b), float(c)] for (a, b), c in zip(p, r)]}
