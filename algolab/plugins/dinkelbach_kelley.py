"""Hypothesis literature-dinkelbach-kelley-packet-exchange-v1: Dinkelbach-Kelley peak exchange in packet coordinates.

Draft written by the auditor from the council's specification; provenance "untrusted" until Arthur reviews it.
References named by the council: Dinkelbach (1967), doi:10.1287/mnsc.13.7.492; Kelley (1960), doi:10.1137/0108053.

Normalized model at the current point (N = Q(c), M = max c, mass-preserving packet multipliers a):

    maximize  (p.a) / N - z   subject to   c_k / M - 1 + (H a)_k / M <= z  (all knots),   m.a = 0,   |a_r| <= Delta

H_r = 2 (f * b_r), p_r = <dQ/dc, H_r>. The objective is the first-order change of log Q - log M (the Dinkelbach
residual at lambda = C). Kelley cutting planes: the working set starts with the largest knot of each of 16 equal
lag ranges; each round adds up to 8 most-violated knots, at most 16 rounds; an unresolved model is discarded.
Trust region: Delta starts at 1e-3, halves when no backtracking step improves the exact score, doubles (cap 0.1)
only when the full step realizes >= 75 % of the predicted improvement; below 1e-10 the packets are re-drawn.
"""
import math
import time

import numpy as np
from scipy.optimize import linprog

from algolab import kit

PARAMS = {
    "N_PACKETS": 16,
    "ROW_BUDGET_FLOATS": 64_000_000,
    "DELTA0": 1e-3,
    "DELTA_MAX": 0.1,
    "DELTA_MIN": 1e-10,
    "JITTER": 0.25,
    "SEED_RANGES": 16,
    "ADD_PER_ROUND": 8,
    "MAX_ROUNDS": 16,
    "VIOL_TOL": 1e-12,
    "N_ALPHAS": 21,
    "RATIO_DOUBLE": 0.75,
}


class Algorithm:
    def __init__(self, params, rng):
        self.p = params
        self.rng = rng
        self.stats = {"models": 0, "unresolved": 0, "accepted": 0}

    def model(self, c, masses, H, delta):
        P = len(masses)
        N, M = kit.q_form(c), float(c.max())
        p = H @ kit.q_gradient(c)
        cost = np.concatenate((-p / N, [1.0]))
        ranges = np.array_split(np.arange(len(c)), int(self.p["SEED_RANGES"]))
        K = np.unique([int(r[np.argmax(c[r])]) for r in ranges if len(r)])
        bounds = [(-delta, delta)] * P + [(None, None)]
        a_eq = np.concatenate((masses, [0.0]))[None, :]
        for _ in range(int(self.p["MAX_ROUNDS"])):
            a_ub = np.hstack((H[:, K].T / M, -np.ones((len(K), 1))))
            res = linprog(cost, A_ub=a_ub, b_ub=1.0 - c[K] / M, A_eq=a_eq, b_eq=[0.0], bounds=bounds,
                          method="highs")
            if res.status != 0:
                break
            a, z = res.x[:P], float(res.x[P])
            viol = c / M - 1.0 + (a @ H) / M - z
            if float(viol.max()) <= self.p["VIOL_TOL"]:
                return a, float(p @ a / N - z)
            k = min(int(self.p["ADD_PER_ROUND"]), len(c))
            worst = np.argpartition(viol, -k)[-k:]
            K = np.union1d(K, worst[viol[worst] > self.p["VIOL_TOL"]])
        self.stats["unresolved"] += 1
        return None, 0.0

    def improve(self, f, deadline):
        f = np.maximum(np.asarray(f, dtype=np.float64), 0.0)
        f = f / f.max()
        c = kit.autoconv(f)
        cur = kit.score(f)
        n = len(f)
        n_packets = max(2, min(int(self.p["N_PACKETS"]), int(self.p["ROW_BUDGET_FLOATS"]) // (2 * n - 1)))
        delta = float(self.p["DELTA0"])
        alphas = [2.0 ** -i for i in range(int(self.p["N_ALPHAS"]))]
        edges = masses = H = None
        while time.monotonic() < deadline - 1.0:
            if H is None:
                edges = kit.equal_mass_packets(f, n_packets, self.rng, float(self.p["JITTER"]))
                masses, H = kit.packet_vectors(f, edges)
            a, predicted = self.model(c, masses, H, delta)
            self.stats["models"] += 1
            if a is None or predicted <= 1e-15:
                H = None
                delta = float(self.p["DELTA0"])
                continue
            d = np.zeros(n)
            for r in range(len(masses)):
                d[edges[r]:edges[r + 1]] = a[r] * f[edges[r]:edges[r + 1]]
            trials = kit.exact_step_scores(f, c, d, alphas)
            best = max(trials, key=lambda t: t[1], default=None)
            if best is None or best[1] <= cur * (1.0 + 1e-15):
                delta *= 0.5
                if delta < self.p["DELTA_MIN"]:
                    H = None
                    delta = float(self.p["DELTA0"])
                continue
            realized = math.log(best[1] / cur)
            full_step = best[0] == 1.0
            f = np.maximum(f + best[0] * d, 0.0)
            f = f / f.max()
            c = kit.autoconv(f)
            cur = kit.score(f)
            H = None
            self.stats["accepted"] += 1
            if full_step and realized >= self.p["RATIO_DOUBLE"] * predicted:
                delta = min(2.0 * delta, float(self.p["DELTA_MAX"]))
        return f
