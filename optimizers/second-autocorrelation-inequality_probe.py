"""Optimizer file (contract: PROBLEM, CONFIG, search(budget_s, seed, config)).

BASELINE for second-autocorrelation-inequality: a faithful port of WS1's stage-3 probe
(selection/stage3_probe.py, ArrayAdapter + search), the method that is already measured on this problem
(+8.97e-9 over the #1 in ~300 s at the submittable resolution, experiment
ws1-stage3-second-autocorrelation-inequality-1001).

Same moves: a block of 2^k values (k in 0..10) at a uniform position gets either i.i.d. Gaussian noise or
one common Gaussian shift, sigma = 10^U(-12,-2) x mean|f| (10^U(-14,-6) in the final 30% "polish" phase),
clipped at 0; greedy strict acceptance on the pinned score. One difference, in the baseline's favour: the
pinned score is computed on the array directly instead of after a list conversion, so it gets more
evaluations per second than the original. Upsampling (phase B of the probe) is left out: x2 gives 4M values,
above the 2M the problem accepts.
"""
import json
import time

import numpy as np
from scipy.signal import oaconvolve

from prepare import run

PROBLEM = "second-autocorrelation-inequality"

CONFIG = {"POLISH_FRAC": 0.7, "BLOCK_LOG2_MAX": 10, "INIT": None}


def pinned_score(f):
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


def _load_start(config):
    if config.get("INIT"):
        with open(config["INIT"]) as fh:
            values = json.load(fh)["solution"]["values"]
    else:
        from problems.catalog import load
        values = load(PROBLEM).certificate()["values"]
    return np.asarray(values, dtype=np.float64)


def search(budget_s, seed, config=CONFIG):
    cfg = {**CONFIG, **(config or {})}
    t0 = time.monotonic()
    rng = np.random.default_rng(seed)
    x = _load_start(cfg)
    mag = float(np.mean(np.abs(x))) or 1.0
    best = pinned_score(x)
    n_evals = n_impr = 0
    reserve = 3.0
    while (elapsed := time.monotonic() - t0) < budget_s - reserve:
        polish = elapsed > cfg["POLISH_FRAC"] * budget_s
        lo, hi = (-14, -6) if polish else (-12, -2)
        sigma = 10 ** rng.uniform(lo, hi) * mag
        size = min(len(x), 2 ** int(rng.integers(0, int(cfg["BLOCK_LOG2_MAX"]) + 1)))
        start = int(rng.integers(0, len(x) - size + 1))
        idx = slice(start, start + size)
        old = x[idx].copy()
        noise = rng.normal(0, sigma, size) if rng.random() < 0.5 else rng.normal(0, sigma)
        x[idx] = np.clip(x[idx] + noise, 0.0, None)
        s = pinned_score(x)
        n_evals += 1
        if s > best:
            best, n_impr = s, n_impr + 1
        else:
            x[idx] = old
    print(f"evals {n_evals} improvements {n_impr} best {best!r}", flush=True)
    return {"values": x.tolist()}


if __name__ == "__main__":
    run(__file__)
