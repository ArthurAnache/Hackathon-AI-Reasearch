"""Seed program for second-autocorrelation-inequality: warm-started ascent on a smoothed version of the score.

Contract of every candidate program:
    PARAMS = {"name": (default, low, high[, "log"])}
    construct(params: dict, seed: int, budget_s: float, init=None) -> {"values": [...]}

Score (same formula as the frozen verifier, with c = f * f the full autoconvolution of the non-negative f):
    C = (2 sum c_k^2 + sum c_k c_{k+1}) / (3 * sum c_k * max c_k)
Method: f = w^2 (keeps f >= 0 and keeps the support of `init`), max c replaced by a LogSumExp with sharpness
`beta` (c is scaled so that max c = 1), L-BFGS on -log C_beta, beta multiplied by `beta_growth` at each stage.
The stages share the time budget equally. Every stage result is re-scored with the exact formula, and the best
exact point seen is returned, so the program never returns something worse than `init`.

Measured on the live #1 of 2026-11-14 (solution 2783, score 0.9636615620620852, one core):
- n = 1,999,999 integer values, max 1e9; 482,716 are non-zero, in 20,962 runs (median run length 5) spread over
  the whole grid; one end block of 53,629 cells at indices 1,946,227.. holds 16 % of the mass;
- its autoconvolution is flat within 1e-7 of the max on ~374,000 lags, so beta below ~1e9 moves away from it
  (the exact score drops); beta = 1e9 gains ~5e-8 in 36 s, and later rounds gain only ~1e-9 to 1e-10;
- one rfft/irfft pair on 4,000,000 points takes ~0.06 s, one objective + gradient ~0.25 s.
"""
import time

import numpy as np
import scipy.fft as sf
from scipy.optimize import minimize

PARAMS = {
    "beta": (1.0e9, 1.0e7, 1.0e13, "log"),
    "beta_growth": (3.0, 1.0, 30.0),
    "stages": (2, 1, 6),
    "maxcor": (8, 3, 24),
    "time_fraction": (0.70, 0.30, 0.80),
}


def _conv(f, m):
    spec = sf.rfft(f, m)
    return sf.irfft(spec * spec, m)[: 2 * len(f) - 1], spec


def exact_score(f):
    """The verifier's formula (closed form), on a non-negative array."""
    n = len(f)
    c, _ = _conv(f, sf.next_fast_len(2 * n - 1, real=True))
    num = (2.0 * np.dot(c, c) + np.dot(c[:-1], c[1:])) / 3.0
    return float(num / (c.sum() * c.max()))


def _neg_log_c(w, beta, m, n):
    f = w * w
    c, spec = _conv(f, m)
    cmax = c.max()
    e = np.exp(beta * (c - cmax))
    z = e.sum()
    m_beta = cmax + np.log(z) / beta               # smooth max, >= max c
    num = (2.0 * np.dot(c, c) + np.dot(c[:-1], c[1:])) / 3.0
    s = f.sum()
    value = np.log(num) - 2.0 * np.log(s) - np.log(m_beta)
    g = 4.0 * c
    g[1:] += c[:-1]
    g[:-1] += c[1:]
    big = g / (3.0 * num) - (e / z) / m_beta       # d value / d c
    corr = sf.irfft(sf.rfft(big, m) * np.conj(spec), m)[:n]
    grad_f = 2.0 * corr - 2.0 / s
    return -value, -2.0 * w * grad_f


def construct(params, seed, budget_s, init=None):
    t0 = time.monotonic()
    deadline = t0 + float(params["time_fraction"]) * budget_s
    if init is None:
        raise ValueError("this program needs a warm start (init)")
    f0 = np.maximum(np.asarray(init["values"], dtype=np.float64), 0.0)
    n = len(f0)
    m = sf.next_fast_len(2 * n - 1, real=True)
    c0, _ = _conv(f0, m)
    f0 = f0 / np.sqrt(c0.max())                    # now max(f0 * f0) = 1, so beta is relative to the max
    best_f, best_s = f0, exact_score(f0)
    w = np.sqrt(f0)
    beta = float(params["beta"])

    class _Stop(Exception):
        pass

    stages = int(params["stages"])
    for k in range(stages):
        now = time.monotonic()
        if now >= deadline:
            break
        stage_end = now + (deadline - now) / (stages - k)

        def fun(x, beta=beta, stage_end=stage_end):
            if time.monotonic() >= stage_end:
                raise _Stop
            return _neg_log_c(x, beta, m, n)

        last = {"x": w}

        def callback(xk):
            last["x"] = xk.copy()

        try:
            res = minimize(fun, w, jac=True, method="L-BFGS-B", callback=callback,
                           options={"maxcor": int(params["maxcor"]), "maxiter": 100000, "maxfun": 100000,
                                    "ftol": 0.0, "gtol": 0.0})
            w = res.x
        except _Stop:
            w = last["x"]
        f = w * w
        s = exact_score(f)
        if s > best_s:
            best_f, best_s = f, s
        else:
            w = np.sqrt(best_f)
        beta *= float(params["beta_growth"])
    return {"values": best_f.tolist()}
