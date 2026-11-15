"""Hypothesis numerical-20261115-two-box-envelope-queue: amortized two-box transfers with peak-envelope line search.

Draft written by the auditor from the council's specification; provenance "untrusted" until Arthur reviews it.

A signed transfer h = 1_A - 1_B of equal length L = 2^k moves mass between two disjoint blocks while keeping the
sum S fixed. Then c(t) = c + t u + t^2 v with u = 2 (f * h) and v = h * h, and the exact score is
Q(c(t)) / (3 S^2 M(t)). The feasible interval is t in [-min_A(f), min_B(f)]. The line search evaluates a log grid
of t (and a golden refinement of the best) with the kit's exact quartic/quadratic forms, never a random D and never
a dense gradient.
"""
import math
import time

import numpy as np

from algolab import kit

PARAMS = {
    "BLOCK_LOG2_MAX": 16,
    "T_GRID": 16,
    "GOLDEN_ITERS": 8,
    "GUIDED": 0.5,
    "FIRST_IMPROVE_FRAC": 1.0,
}


PHI = (math.sqrt(5) - 1) / 2


class Algorithm:
    def __init__(self, params, rng):
        self.p = params
        self.rng = rng
        self.stats = {"proposals": 0, "accepted": 0}

    def _u_v(self, f, a, b, L):
        h = np.zeros_like(f)
        h[a:a + L] = 1.0
        h[b:b + L] = -1.0
        return 2.0 * kit.conv_full(f, h), kit.autoconv(h), h

    def _score_t(self, c, u, v, s, t):
        ct = c + t * u + (t * t) * v
        return kit.q_form(ct) / (3.0 * s * s * float(ct.max()))

    def propose(self, f, c):
        n = len(f)
        k = int(self.rng.integers(0, int(self.p["BLOCK_LOG2_MAX"]) + 1))
        L = min(2 ** k, n // 2)
        if L < 1:
            return None
        if self.rng.random() < self.p["GUIDED"]:
            top = np.flatnonzero(c >= c.max() * (1.0 - 1e-9))
            if len(top):
                mid = int(self.rng.choice(top))
                a = int(np.clip(mid - L // 2, 0, n - L))
            else:
                a = int(self.rng.integers(0, n - L + 1))
        else:
            a = int(self.rng.integers(0, n - L + 1))
        # second block: anywhere disjoint
        for _ in range(8):
            b = int(self.rng.integers(0, n - L + 1))
            if b + L <= a or a + L <= b:
                break
        else:
            return None
        lo = -float(f[a:a + L].min())
        hi = float(f[b:b + L].min())
        if not (lo < 0 < hi or lo < hi):
            return None
        u, v, h = self._u_v(f, a, b, L)
        s = float(f.sum())
        cur = kit.q_form(c) / (3.0 * s * s * float(c.max()))
        # log-spaced interior points in (lo, hi), both signs relative to 0 when feasible
        lo_e, hi_e = lo + 1e-15 * (hi - lo), hi - 1e-15 * (hi - lo)
        if lo_e >= hi_e:
            return None
        grid = np.unique(np.concatenate((
            np.linspace(lo_e, hi_e, max(3, int(self.p["T_GRID"]))),
            [0.0],
        )))
        grid = grid[(grid >= lo_e) & (grid <= hi_e)]
        vals = [self._score_t(c, u, v, s, float(t)) for t in grid]
        i = int(np.argmax(vals))
        best_t, best_v = float(grid[i]), vals[i]
        if best_v > cur and self.p["GOLDEN_ITERS"] > 0 and abs(best_t) > 0:
            span = (hi_e - lo_e) / max(1, len(grid) - 1)
            g_lo, g_hi = max(lo_e, best_t - span), min(hi_e, best_t + span)
            x1, x2 = g_hi - PHI * (g_hi - g_lo), g_lo + PHI * (g_hi - g_lo)
            v1, v2 = self._score_t(c, u, v, s, x1), self._score_t(c, u, v, s, x2)
            for _ in range(int(self.p["GOLDEN_ITERS"])):
                if v1 > v2:
                    g_hi, x2, v2 = x2, x1, v1
                    x1 = g_hi - PHI * (g_hi - g_lo)
                    v1 = self._score_t(c, u, v, s, x1)
                else:
                    g_lo, x1, v1 = x1, x2, v2
                    x2 = g_lo + PHI * (g_hi - g_lo)
                    v2 = self._score_t(c, u, v, s, x2)
            for x, vv in ((x1, v1), (x2, v2)):
                if vv > best_v:
                    best_t, best_v = float(x), vv
        if best_v <= cur:
            return None
        return best_t, h, best_v

    def improve(self, f, deadline):
        f = np.maximum(np.asarray(f, dtype=np.float64), 0.0)
        f = f / f.max()
        c = kit.autoconv(f)
        cur = kit.score(f)
        while time.monotonic() < deadline - 0.5:
            prop = self.propose(f, c)
            self.stats["proposals"] += 1
            if prop is None:
                continue
            t, h, s = prop
            f = np.maximum(f + t * h, 0.0)
            f = f / f.max()
            c = kit.autoconv(f)
            cur = s
            self.stats["accepted"] += 1
        return f
