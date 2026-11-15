"""Seed program for third-autocorrelation-inequality (exploratory problem, minimize): melt and re-freeze.

Contract of every candidate program:
    PARAMS = {"name": (default, low, high[, "log"])}
    construct(params: dict, seed: int, budget_s: float, init=None) -> {"values": [...]}

Score (the pinned verifier's formula; f has n values, which may be negative, and c = f * f is the full
autoconvolution of length 2n - 1):
    C = |max_k c_k * dx| / (sum(f) * dx)^2 = 2n * max_k c_k / sum(f)^2        (dx = 0.5 / n)

Method (the problem scout's re-implementation of the method the MILO paper describes, arXiv:2609.38349 App. E.2;
selection/scout_third_autocorr.py):
- the max over lags is replaced by a log-sum-exp at temperature T (in score units), whose gradient costs three
  FFTs;
- L-BFGS stages on a geometric cooling ladder T: t_hot -> t_cold, the stages sharing the time equally;
- first a cold polish of `init` as it is, then re-melt hops: the best point so far gets additive Gaussian noise
  (`noise` times the mean |f|) and is cooled again.
Every objective evaluation also gives the exact score of its point, and the best exact point seen is kept. The
result is compared with `init` once more with the verifier's own direct convolution, and `init` itself is returned
unless the result is strictly better, so the program never returns something worse than `init`.

Measured (record margin delta = 1e-5; one core):
- by the scout (docs/problem_scout.md): from the live #1 (65,536 values) a 52 s cold polish and 12 hops of 300 s
  gained nothing; in another basin (25,600 values) one hop of 420 s with noise 1e-3, cooled from 1e-3 to 1e-8 in
  14 stages, gained 0.27 delta;
- here, on a machine shared with other jobs (2026-11-14): at 25,600 values a 40 s hop with noise 1e-3 is still
  0.32 delta above its start, with noise 1e-4 0.035 delta above, with noise 1e-5 0.004 delta above, and with noise
  1e-6 it lands 0.0003 delta below; a 60 s polish at T = 1e-9 gains 0.0009 delta (0.0002 delta from the live #1).
  So the noise a hop can absorb grows fast with the time it gets: the defaults below fit an evaluation of about
  two minutes, and a hot hop (noise 1e-3, t_hot 1e-3, t_cold 1e-8, 14 stages) needs several minutes;
- about half of the lags are within 1e-7 (relative) of the maximum at both starts (25,392 of 51,199 lags; 58,086
  of 131,071 for the live #1, of which 12,178 within 1e-9): the candidates are flat-topped minimax points;
- one objective + gradient takes about 1.5 ms at 25,600 values and 4 ms at 65,536 on a free core.
"""
import math
import time

import numpy as np
import scipy.fft as sf
from scipy.optimize import minimize

PARAMS = {
    "t_hot": (1.0e-5, 1.0e-9, 1.0e-1, "log"),       # first temperature of a re-melt hop
    "t_cold": (1.0e-10, 1.0e-12, 1.0e-6, "log"),    # last temperature of every cooling path
    "t_polish": (1.0e-8, 1.0e-11, 1.0e-3, "log"),   # first temperature of the polish of `init`
    "noise": (1.0e-5, 1.0e-8, 1.0e-1, "log"),       # std of the hop noise, relative to the mean |f|
    "stages": (6, 2, 24),                           # cooling stages per path
    "hops": (1, 0, 8),                              # re-melt hops after the polish
    "polish_fraction": (0.30, 0.0, 0.90),           # share of the time given to the polish of `init`
    "maxcor": (16, 4, 40),
    "time_fraction": (0.70, 0.30, 0.85),
}


class _Stop(Exception):
    pass


def exact_score(f):
    """The verifier's formula with its own direct convolution (O(n^2): 0.2 s at 25,600 values, 1.2 s at 65,536)."""
    f = np.asarray(f, dtype=np.float64)
    n = len(f)
    dx = 0.5 / n
    integral_sq = (np.sum(f) * dx) ** 2
    if not integral_sq >= 1e-9:
        return math.inf
    return float(abs(np.max(np.convolve(f, f, mode="full") * dx)) / integral_sq)


class _Problem:
    def __init__(self, n):
        self.n = n
        self.length = 2 * n - 1
        self.m = sf.next_fast_len(self.length, real=True)

    def score(self, f):
        """FFT version of the score (relative error about 1e-14); used during the search only."""
        s = float(f.sum())
        spec = sf.rfft(f, self.m)
        return 2.0 * self.n * float(sf.irfft(spec * spec, self.m)[: self.length].max()) / (s * s)

    def value_grad(self, f, temp):
        """Log-sum-exp surrogate at temperature `temp`, its gradient, and the exact score of f."""
        s = float(f.sum())
        if not math.isfinite(s) or s * s < 1e-9:
            return math.inf, None, math.inf
        spec = sf.rfft(f, self.m)
        k = 2.0 * self.n / (s * s)
        c = sf.irfft(spec * spec, self.m)[: self.length] * k
        cmax = float(c.max())
        e = np.exp((c - cmax) / temp)
        z = float(e.sum())
        w = e / z
        g = sf.irfft(sf.rfft(w, self.m) * np.conj(spec), self.m)[: self.n] * (2.0 * k)
        g -= 2.0 * float(w @ c) / s
        return cmax + temp * math.log(z), g, cmax


class _Best:
    def __init__(self, f, score):
        self.f, self.score = f.copy(), score

    def offer(self, f, score):
        if score < self.score:
            self.f, self.score = f.copy(), score


def _stage(prob, f, temp, maxcor, best, deadline):
    last = {"x": f}

    def fun(x):
        if time.monotonic() > deadline:
            raise _Stop
        value, grad, score = prob.value_grad(x, temp)
        if grad is None:
            raise _Stop
        best.offer(x, score)
        last["x"] = x
        return value, grad

    try:
        x = minimize(fun, f, jac=True, method="L-BFGS-B",
                     options={"maxiter": 10 ** 9, "maxfun": 10 ** 9, "maxcor": maxcor, "ftol": 0.0, "gtol": 0.0,
                              "maxls": 30}).x
    except _Stop:
        x = last["x"]
    s = float(x.sum())
    return x * (prob.n / s) if math.isfinite(s) and abs(s) > 1e-9 else f


def _cool(prob, f, t_first, t_last, stages, maxcor, best, deadline):
    """Geometric cooling t_first -> t_last; each stage gets an equal share of the time left."""
    stages = max(2, int(stages))
    ratio = (t_last / t_first) ** (1.0 / (stages - 1))
    temp = t_first
    for j in range(stages):
        now = time.monotonic()
        if now >= deadline:
            break
        f = _stage(prob, f, temp, maxcor, best, now + (deadline - now) / (stages - j))
        temp *= ratio
    return f


def construct(params, seed, budget_s, init=None):
    t0 = time.monotonic()
    if init is None:
        raise ValueError("this program needs a warm start (init)")
    f_init = np.asarray(init["values"], dtype=np.float64)
    n = len(f_init)
    prob = _Problem(n)
    # Two direct convolutions (the final check) take about 2.4 s at 65,536 values: keep that time aside.
    check_s = 2.5 * (n / 65536.0) ** 2
    deadline = t0 + max(0.0, float(params["time_fraction"]) * budget_s - check_s)
    f0 = f_init * (n / float(f_init.sum()))
    best = _Best(f0, prob.score(f0))
    rng = np.random.default_rng(seed)
    t_cold = float(params["t_cold"])
    maxcor = int(params["maxcor"])
    stages = int(params["stages"])
    hops = int(params["hops"])

    # 1. Cold polish of `init` as it is.
    share = float(params["polish_fraction"]) if hops > 0 else 1.0
    polish_end = time.monotonic() + share * (deadline - time.monotonic())
    if share > 0.0:
        _cool(prob, f0, max(float(params["t_polish"]), t_cold), t_cold, stages, maxcor, best, polish_end)

    # 2. Re-melt hops from the best point so far.
    for h in range(hops):
        now = time.monotonic()
        if now >= deadline:
            break
        hop_end = now + (deadline - now) / (hops - h)
        start = best.f + float(params["noise"]) * np.abs(best.f).mean() * rng.standard_normal(n)
        start *= n / float(start.sum())
        _cool(prob, start, max(float(params["t_hot"]), t_cold), t_cold, stages, maxcor, best, hop_end)

    # 3. Never worse than `init`, by the verifier's own formula.
    if best.score < prob.score(f0) and exact_score(best.f) < exact_score(f_init):
        return {"values": best.f.tolist()}
    return {"values": [float(v) for v in init["values"]]}
