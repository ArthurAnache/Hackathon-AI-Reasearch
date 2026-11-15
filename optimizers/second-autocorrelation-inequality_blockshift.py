"""Optimizer file (contract: PROBLEM, CONFIG, search(budget_s, seed, config)).

Problem: EinsteinArena second-autocorrelation-inequality (maximise C = ||f*f||_2^2 / (||f*f||_1 ||f*f||_inf)
over nonnegative step functions f with <= 2,000,000 values). Target-specific: it only works on this problem.

Why a generic method stalls here. The incumbent (#1, 1,999,999 values) is an equioscillating optimum: 384,530
points of f*f lie within 1e-7 of its max. A dense update (gradient ascent, Adam), however small, raises some of
those points and loses (measured: no gradient variant gained more than 5e-16 in 60 s). Sparse block moves can
still win (WS1's stage-3 probe gained 9e-9 in ~300 s at this resolution), but the probe pays one full 0.23 s
verifier evaluation per random try.

Method: exact incremental block moves with an optimal step.
  A move adds a constant D to f on a block [a, a+b). Then exactly
      f*f  ->  c + D*U + D^2*T,    U = 2*(box_b * f) (a sliding-window sum, O(n) with prefix sums),
                                   T = box_b * box_b (a triangle of width 2b-1),
  so the L2 term is an exact quartic in D (5 coefficients, computed once per block), the L1 term is
  (s + b D)^2 / (m+1), and the max only needs checking on the near-top points K = {c >= M - gap}: a point
  outside K rises by at most |D| max U + D^2 max T, and when that bound exceeds the gap the full max is used.
  Each proposal therefore scores a whole grid of D values (both signs, log-spaced) exactly and refines the
  best one by golden-section search, instead of trying one random D.
  Block sizes are 2^k (k uniform in 0..BLOCK_LOG2_MAX). Optionally (GUIDED) block positions are drawn
  near the points where f*f attains its max: moves there change the binding constraint.

Never worse than the start. Decisions use the incremental values; every REFRESH_S seconds and at the end, f*f
is recomputed from scratch and scored with exactly the pinned verifier's arithmetic. Only a checkpoint whose
pinned score is strictly better than the previous checkpoint is kept; otherwise the search reverts to it. The
returned candidate is that checkpoint (the parent re-verifies it anyway).

Warm start: CONFIG["INIT"] is set by the job runner from a re-verified certificate. Without it, the frozen #1
certificate of problems/second-autocorrelation-inequality/ is used.
"""
import json
import math
import time

import numpy as np
from scipy.signal import oaconvolve

from prepare import run

PROBLEM = "second-autocorrelation-inequality"

CONFIG = {
    "BLOCK_LOG2_MAX": 20,       # block sizes 1 .. 2^20 (capped at n); tuned on seed 1: larger blocks help
    "D_LOG10_LO": -12.0,        # step grid |D| in [10^lo, 10^hi] x (mean of f on its support)
    "D_LOG10_HI": -2.0,
    "D_GRID": 12,               # grid points per sign
    "GOLDEN_ITERS": 8,          # refinement of the best grid step
    "GAP_REL": 1e-6,            # near-top set K = {c >= M (1 - GAP_REL)}
    "GUIDED": 0.5,              # probability that a block is placed near a max point of f*f
    "REFRESH_S": 30.0,          # exact (pinned-arithmetic) checkpoint period
    "INIT": None,
}

SEARCH_SPACE = {
    "BLOCK_LOG2_MAX": (4, 21, "int"),
    "D_LOG10_LO": (-14.0, -8.0, "linear"),
    "D_LOG10_HI": (-6.0, -1.0, "linear"),
    "D_GRID": (6, 24, "int"),
    "GOLDEN_ITERS": (0, 16, "int"),
    "GAP_REL": (1e-8, 1e-4, "log"),
    "GUIDED": (0.0, 1.0, "linear"),
    "REFRESH_S": (5.0, 120.0, "log"),
}

PHI = (math.sqrt(5) - 1) / 2


def pinned_score(f: np.ndarray) -> float:
    """Exactly the arithmetic of problems/second-autocorrelation-inequality/verifier_pinned.py."""
    f_nonneg = np.maximum(f, 0.0)
    convolution = oaconvolve(f_nonneg, f_nonneg, mode="full")
    num_conv_points = len(convolution)
    x_points = np.linspace(-0.5, 0.5, num_conv_points + 2)
    x_intervals = np.diff(x_points)
    y_points = np.concatenate(([0], convolution, [0]))
    y1, y2 = y_points[:-1], y_points[1:]
    l2_norm_squared = float(np.sum((x_intervals / 3) * (y1**2 + y1 * y2 + y2**2)))
    norm_1 = np.sum(np.abs(convolution)) / (num_conv_points + 1)
    norm_inf = np.max(np.abs(convolution))
    return float(l2_norm_squared / (norm_1 * norm_inf))


def _q(u, v, h):
    """Bilinear form of the L2 term: L2(c) = _q(c, c, h) for the trapezoid-Simpson rule of the verifier."""
    return h / 3 * (2.0 * np.dot(u, v) + 0.5 * (np.dot(u[:-1], v[1:]) + np.dot(u[1:], v[:-1])))


def _load_start(config) -> np.ndarray:
    if config.get("INIT"):
        with open(config["INIT"]) as fh:
            values = json.load(fh)["solution"]["values"]
    else:
        from problems.catalog import load
        values = load(PROBLEM).certificate()["values"]
    f = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    return f / f.max()                       # the score is scale-invariant


class State:
    def __init__(self, f: np.ndarray, gap_rel: float):
        self.f, self.n, self.gap_rel = f, len(f), gap_rel
        self.m = 2 * self.n - 1
        self.h = 1.0 / (self.m + 1)
        self.rebuild()

    def rebuild(self):
        self.c = oaconvolve(self.f, self.f, mode="full")
        self.s = float(self.f.sum())
        self.l2 = float(_q(self.c, self.c, self.h))
        self.M = float(self.c.max())
        self.reindex()

    def reindex(self):
        self.gap = self.gap_rel * self.M
        self.K = np.flatnonzero(self.c >= self.M - self.gap)
        self.top = np.flatnonzero(self.c >= self.M * (1 - 1e-9))

    def block_terms(self, a: int, b: int):
        """U on its support [a, a+n+b-1) and the triangle T on [2a, 2a+2b-1)."""
        fp = np.concatenate(([0.0], np.cumsum(np.concatenate((np.zeros(b - 1), self.f, np.zeros(b - 1))))))
        u = 2.0 * (fp[b:] - fp[:-b])                              # 2 x window sums, length n + b - 1
        tri = np.concatenate((np.arange(1, b + 1), np.arange(b - 1, 0, -1))).astype(np.float64)
        return u, tri

    def _qc(self, lo: int, v: np.ndarray) -> float:
        """_q(c, V) for a vector V that is v on [lo, lo+len(v)) and zero elsewhere (only its support is read)."""
        c, h, hi = self.c, self.h, lo + len(v)
        left = c[lo - 1] * v[0] if lo > 0 else 0.0              # neighbours just outside the support
        right = c[hi] * v[-1] if hi < self.m else 0.0
        cs = c[lo:hi]
        return h / 3 * (2.0 * np.dot(cs, v) + 0.5 * (np.dot(cs[:-1], v[1:]) + np.dot(cs[1:], v[:-1]) + left + right))

    def propose(self, a: int, b: int, cfg, scale: float):
        n, h, c = self.n, self.h, self.c
        u, tri = self.block_terms(a, b)
        lo_u, lo_t = a, 2 * a
        ut = u[lo_t - lo_u:lo_t - lo_u + len(tri)]              # U restricted to T's support (inside U's)
        # L2(D) - L2(0) = k1 D + k2 D^2 + k3 D^3 + k4 D^4   (exact)
        k1 = 2 * self._qc(lo_u, u)
        k2 = _q(u, u, h) + 2 * self._qc(lo_t, tri)
        k3 = 2 * _q(ut, tri, h) + 2 * (h / 3) * 0.5 * ((u[lo_t - lo_u - 1] * tri[0] if lo_t > lo_u else 0.0)
                                                        + (u[lo_t - lo_u + len(tri)] * tri[-1]
                                                           if lo_t - lo_u + len(tri) < len(u) else 0.0))
        k4 = _q(tri, tri, h)
        K = self.K
        inU = (K >= lo_u) & (K < lo_u + len(u))
        UK = np.zeros(len(K))
        UK[inU] = u[K[inU] - lo_u]
        TK = np.zeros(len(K))
        inT = (K >= lo_t) & (K < lo_t + len(tri))
        TK[inT] = tri[K[inT] - lo_t]
        cK = c[K]
        umax, tmax = float(np.abs(u).max()), float(b)
        fmin = float(self.f[a:a + b].min())
        U = T = None

        def full_max(D):
            nonlocal U, T
            if U is None:
                U, T = np.zeros(self.m), np.zeros(self.m)
                U[lo_u:lo_u + len(u)] = u
                T[lo_t:lo_t + len(tri)] = tri
            return float(np.max(c + D * (U + D * T)))

        def dlog(D):
            dl2 = D * (k1 + D * (k2 + D * (k3 + D * k4)))
            if self.l2 + dl2 <= 0 or self.s + b * D <= 0:
                return -math.inf
            if abs(D) * umax + D * D * tmax < self.gap:
                Mn = float(np.max(cK + D * (UK + D * TK)))
            else:
                Mn = full_max(D)
            return math.log1p(dl2 / self.l2) - 2 * math.log1p(b * D / self.s) - math.log(Mn / self.M)

        mags = scale * np.logspace(cfg["D_LOG10_LO"], cfg["D_LOG10_HI"], int(cfg["D_GRID"]))
        grid = list(mags) + [-d for d in mags if d <= fmin]
        vals = [dlog(D) for D in grid]
        i = int(np.argmax(vals))
        bestD, bestV = grid[i], vals[i]
        if bestV > 0 and cfg["GOLDEN_ITERS"] > 0:                 # golden-section on [D/r, D*r]
            r = 10 ** ((cfg["D_LOG10_HI"] - cfg["D_LOG10_LO"]) / max(1, cfg["D_GRID"] - 1))
            lo, hi = math.log(abs(bestD) / r), math.log(abs(bestD) * r)
            sign = 1.0 if bestD > 0 else -1.0
            cap = math.log(fmin) if sign < 0 and fmin > 0 else math.inf
            hi = min(hi, cap)
            x1, x2 = hi - PHI * (hi - lo), lo + PHI * (hi - lo)
            v1, v2 = dlog(sign * math.exp(x1)), dlog(sign * math.exp(x2))
            for _ in range(int(cfg["GOLDEN_ITERS"])):
                if v1 > v2:
                    hi, x2, v2 = x2, x1, v1
                    x1 = hi - PHI * (hi - lo)
                    v1 = dlog(sign * math.exp(x1))
                else:
                    lo, x1, v1 = x1, x2, v2
                    x2 = lo + PHI * (hi - lo)
                    v2 = dlog(sign * math.exp(x2))
            for x, v in ((x1, v1), (x2, v2)):
                if v > bestV:
                    bestD, bestV = sign * math.exp(x), v
        return bestD, bestV, (u, tri, k1, k2, k3, k4)

    def apply(self, a: int, b: int, D: float, terms) -> None:
        u, tri, k1, k2, k3, k4 = terms
        self.f[a:a + b] = np.maximum(self.f[a:a + b] + D, 0.0)
        self.c[a:a + len(u)] += D * u
        self.c[2 * a:2 * a + len(tri)] += D * D * tri
        self.s += b * D
        self.l2 += D * (k1 + D * (k2 + D * (k3 + D * k4)))
        self.M = float(self.c.max())
        self.reindex()


def search(budget_s, seed, config=CONFIG):
    cfg = {**CONFIG, **(config or {})}
    t0 = time.monotonic()
    rng = np.random.default_rng(seed)
    f = _load_start(cfg)
    best_f, best = f.copy(), pinned_score(f)
    st = State(f, float(cfg["GAP_REL"]))
    scale = float(f[f > 0].mean())
    n = st.n
    last_check = time.monotonic()
    reserve = 3.0                                            # final exact check + output
    proposals = accepted = 0
    while time.monotonic() - t0 < budget_s - reserve:
        b = 2 ** int(rng.integers(0, int(cfg["BLOCK_LOG2_MAX"]) + 1))
        b = min(b, n)
        if rng.random() < cfg["GUIDED"] and len(st.top):
            k = int(rng.choice(st.top))                      # a max point of f*f: move mass that feeds it
            a = int(np.clip(rng.integers(max(0, k - n + 1), min(n, k + 1)) - b // 2, 0, n - b))
        else:
            a = int(rng.integers(0, n - b + 1))
        D, v, terms = st.propose(a, b, cfg, scale)
        proposals += 1
        if v > 1e-16:
            st.apply(a, b, D, terms)
            accepted += 1
        if time.monotonic() - last_check > cfg["REFRESH_S"]:
            last_check = time.monotonic()
            exact = pinned_score(st.f)
            if exact > best:
                best_f, best = st.f.copy(), exact
            else:                                            # incremental drift or no real gain: revert
                st.f[:] = best_f
            st.rebuild()
    exact = pinned_score(st.f)
    if exact > best:
        best_f, best = st.f.copy(), exact
    print(f"proposals {proposals} accepted {accepted} best {best!r}", flush=True)
    return {"values": best_f.tolist()}


if __name__ == "__main__":
    run(__file__)
