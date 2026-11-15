"""Hypothesis math-20261115-002-active-envelope-packets: mass-normalized packet moves against the full max envelope.

Draft written by the auditor from the council's specification; provenance "untrusted" until Arthur reviews it.

Packets b_j = f on contiguous, equal-mass index ranges. A direction h = sum_j z_j b_j with sum_j m_j z_j = 0 keeps
the mass S and (for |z_j| < 1) non-negativity. With c = f*f, Q = Q(c), M = max c and r = Q / M, the first-order
model of Q(c') - r max(c') is the LP

    maximize  p.z - r s   subject to   c_k + (R z)_k <= M + s  for every knot k,   m.z = 0,   |z_j| <= zmax

with R_j = 2 (f * b_j) and p_j = <dQ/dc, R_j>. Every knot is enforced by constraint exchange: solve on a working
set, scan all knots with one matrix product, add the most violated, repeat. The step is then chosen by exact
scoring of f + alpha h over alpha = 1, 1/2, ..., 2^-20 (c_alpha = c + alpha (R z) + alpha^2 (h*h), no model
error). A model optimum <= 0 means no first-order ascent exists in this packet subspace; the packets are then
re-drawn with jitter.
"""
import time

import numpy as np
from scipy.optimize import linprog

from algolab import kit

PARAMS = {
    "N_PACKETS": 32,
    "ROW_BUDGET_FLOATS": 64_000_000,   # caps packets x (2n - 1) to ~512 MB of responses
    "Z_MAX": 0.05,
    "Z_MIN": 1e-10,
    "JITTER": 0.25,
    "INIT_ROWS": 256,
    "ADD_PER_ROUND": 64,
    "MAX_ROUNDS": 16,
    "VIOL_TOL": 1e-12,
    "N_ALPHAS": 21,
}


class Algorithm:
    def __init__(self, params, rng):
        self.p = params
        self.rng = rng
        self.stats = {"models": 0, "stationary": 0, "accepted": 0, "lp_fail": 0}

    def lp_direction(self, c, masses, R, zmax):
        P = len(masses)
        Q, M = kit.q_form(c), float(c.max())
        r = Q / M
        pz = R @ kit.q_gradient(c)
        cost = np.concatenate((-pz, [r]))
        k0 = min(int(self.p["INIT_ROWS"]), len(c))
        K = np.argpartition(c, -k0)[-k0:]
        bounds = [(-zmax, zmax)] * P + [(None, None)]
        a_eq = np.concatenate((masses, [0.0]))[None, :]
        z = s = None
        for _ in range(int(self.p["MAX_ROUNDS"])):
            a_ub = np.hstack((R[:, K].T, -np.ones((len(K), 1))))
            res = linprog(cost, A_ub=a_ub, b_ub=M - c[K], A_eq=a_eq, b_eq=[0.0], bounds=bounds, method="highs")
            if res.status != 0:
                self.stats["lp_fail"] += 1
                return None, 0.0
            z, s = res.x[:P], float(res.x[P])
            viol = c + z @ R - M - s
            vmax = float(viol.max())
            if vmax <= self.p["VIOL_TOL"] * M:
                return z, float(pz @ z - r * s)
            k = min(int(self.p["ADD_PER_ROUND"]), len(c))
            worst = np.argpartition(viol, -k)[-k:]
            K = np.union1d(K, worst[viol[worst] > self.p["VIOL_TOL"] * M])
        return None, 0.0

    def improve(self, f, deadline):
        f = np.maximum(np.asarray(f, dtype=np.float64), 0.0)
        f = f / f.max()
        c = kit.autoconv(f)
        cur = kit.score(f)
        n = len(f)
        n_packets = max(2, min(int(self.p["N_PACKETS"]), int(self.p["ROW_BUDGET_FLOATS"]) // (2 * n - 1)))
        zmax = float(self.p["Z_MAX"])
        alphas = [2.0 ** -i for i in range(int(self.p["N_ALPHAS"]))]
        while time.monotonic() < deadline - 1.0:
            edges = kit.equal_mass_packets(f, n_packets, self.rng, float(self.p["JITTER"]))
            masses, R = kit.packet_vectors(f, edges)
            if time.monotonic() >= deadline - 1.0:
                break
            z, model = self.lp_direction(c, masses, R, zmax)
            self.stats["models"] += 1
            if z is None or model <= 1e-15 * kit.q_form(c):
                self.stats["stationary"] += 1
                continue
            h = np.zeros(n)
            for j in range(len(masses)):
                h[edges[j]:edges[j + 1]] = z[j] * f[edges[j]:edges[j + 1]]
            trials = kit.exact_step_scores(f, c, h, alphas)
            best = max(trials, key=lambda t: t[1], default=None)
            if best is not None and best[1] > cur:
                f = np.maximum(f + best[0] * h, 0.0)
                f = f / f.max()
                c = kit.autoconv(f)
                cur = kit.score(f)
                self.stats["accepted"] += 1
                zmax = min(2.0 * zmax, float(self.p["Z_MAX"]))
            else:
                zmax *= 0.5
                if zmax < self.p["Z_MIN"]:
                    zmax = float(self.p["Z_MAX"])
        return f
