"""Optimizer file (contract: PROBLEM, CONFIG, search(budget_s, seed, config)).
Copy to search.py to let a coding agent edit it (program.md).

Problem 1d (minimise 2N*max|v*v|/(sum v)^2 over real v).
Adam on v, minimising a log-sum-exp smoothing of max|v*v| whose temperature
is annealed towards the true max. Run: python search.py "<hypothesis>"

Tuned (benchmark best, 1.467369 at 60 s): Adam on an annealed log-sum-exp surrogate of
max|v*v|, 800 cells, single resolution.

Warm start: with CONFIG["INIT"] = path of a saved solution, the search starts from it
(resampled to the final resolution) and refines it with a sharp annealing schedule.
"""
import json
import time
from pathlib import Path

import numpy as np

from prepare import run

PROBLEM = "1d"

# ---- hyperparameters: the optimizer's CONFIG (agents may override any key) ----
CONFIG = {
    "N_CELLS": 800,
    "LR": 0.01,
    "BETA_START": 10.0,
    "BETA_END": 30000.0,
    "REFINE_BETA_FRAC": 0.1,  # warm start: anneal from BETA_END*frac to BETA_END
    "REFINE_NOISE": 0.001,  # warm start: relative perturbation of the start (basin hopping)
    "INIT": None,  # path of a solution to start from (set by the research agent)
}

SEARCH_SPACE = {
    "N_CELLS": (100, 3000, "log"),
    "LR": (1e-3, 0.1, "log"),
    "BETA_START": (1.0, 100.0, "log"),
    "BETA_END": (1e3, 1e6, "log"),
    "REFINE_BETA_FRAC": (0.01, 1.0, "log"),
    "REFINE_NOISE": (1e-5, 0.1, "log"),
}


def score(v):
    return 2 * len(v) * np.abs(np.convolve(v, v)).max() / v.sum() ** 2


def grad(v, beta):
    n, s = len(v), v.sum()
    conv = np.convolve(v, v)
    c = 2 * n * np.concatenate([conv, -conv]) / s ** 2
    w = np.exp(beta * (c - c.max()))
    w /= w.sum()
    w = w[:len(conv)] - w[len(conv):]
    m = (w * conv).sum()
    r = np.correlate(w, v, "valid")
    return 4 * n * (r * s - m) / s ** 3


def load_init(path, n, noise=0.0, seed=0):
    """Warm start from a saved solution (solutions/*.json). The resolution is rounded to a
    multiple of the saved one and values are repeated, so the start scores exactly the same.
    `noise` multiplies every value by (1 + noise * N(0,1)) to hop to a nearby basin."""
    if not path:
        return None
    data = json.loads(Path(path).read_text())
    vals = np.asarray(data.get("solution", data)["values"], dtype=float)
    vals = np.repeat(vals, max(1, round(n / len(vals))))
    if noise:
        vals = vals * (1 + noise * np.random.default_rng(seed).normal(size=len(vals)))
    return vals

def search(budget_s, seed, config=CONFIG):
    rng = np.random.default_rng(seed)
    t0 = time.monotonic()
    deadline = t0 + 0.95 * budget_s
    init = load_init(config.get("INIT"), config["N_CELLS"], config["REFINE_NOISE"], seed)
    v = init.copy() if init is not None else 1 + 0.1 * rng.normal(size=config["N_CELLS"])
    beta_start = config["BETA_END"] * config["REFINE_BETA_FRAC"] if init is not None else config["BETA_START"]
    m = np.zeros(len(v))
    q = np.zeros(len(v))
    best_v, best = v.copy(), score(v)
    t = 0
    while (now := time.monotonic()) < deadline:
        frac = (now - t0) / (deadline - t0)
        beta = beta_start * (config["BETA_END"] / beta_start) ** frac
        g = grad(v, beta)
        t += 1
        m = 0.9 * m + 0.1 * g
        q = 0.999 * q + 0.001 * g * g
        v -= config["LR"] * (m / (1 - 0.9 ** t)) / (np.sqrt(q / (1 - 0.999 ** t)) + 1e-12)
        if t % 50 == 0:
            s = score(v)
            if s < best:
                best_v, best = v.copy(), s
    if score(v) < best:
        best_v = v
    if config.get("INIT"):   # warm start never returns worse than its (unperturbed) start
        start = load_init(config["INIT"], len(best_v))
        if score(start) < score(best_v):
            best_v = start
    return {"values": best_v.tolist()}


if __name__ == "__main__":
    run(__file__)
