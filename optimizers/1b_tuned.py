"""Optimizer file (contract: PROBLEM, CONFIG, search(budget_s, seed, config)).
Copy to search.py to let a coding agent edit it (program.md).

Problem 1b (minimise (2/N) max_k sum_i h_i(1-h_{i+k}), 0<=h<=1, sum h = N/2).
Projected Adam on a log-sum-exp smoothing of the max correlation, annealed;
projection onto {0<=h<=1, sum h = N/2} by bisection. Run: python search.py "<hypothesis>"

Tuned (benchmark best, 0.381599 at 60 s): projected Adam on an annealed log-sum-exp surrogate of
the max correlation, projection onto {0<=h<=1, sum h = N/2}, multi-resolution x3 refinements.
Warm start: with CONFIG["INIT"] = path of a saved solution, the search starts from it
(resampled to the final resolution) and refines it with a sharp annealing schedule.
"""
import json
import time
from pathlib import Path

import numpy as np

from prepare import run

PROBLEM = "1b"

# ---- hyperparameters: the optimizer's CONFIG (agents may override any key) ----
CONFIG = {
    "N_CELLS": 150,  # starting resolution
    "UPSAMPLE_AT": (0.3, 0.6),  # refine x3 at these budget fractions
    "LR": 0.005,
    "BETA_START": 20.0,
    "BETA_END": 5000000.0,
    "REFINE_BETA_FRAC": 0.1,  # warm start: anneal from BETA_END*frac to BETA_END
    "REFINE_NOISE": 0.001,  # warm start: relative perturbation of the start (basin hopping)
    "INIT": None,  # path of a solution to start from (set by the research agent)
}

SEARCH_SPACE = {
    "N_CELLS": (50, 600, "log"),  # starting resolution (final = x9)
    "LR": (1e-3, 0.1, "log"),
    "BETA_START": (1.0, 200.0, "log"),
    "BETA_END": (1e3, 1e7, "log"),
    "REFINE_BETA_FRAC": (0.01, 1.0, "log"),
    "REFINE_NOISE": (1e-5, 0.1, "log"),
}


def score(h):
    return 2 * np.convolve(h, (1 - h)[::-1]).max() / len(h)


def project(h, target):
    lo, hi = h.min() - 1, h.max()
    for _ in range(60):
        tau = (lo + hi) / 2
        if np.clip(h - tau, 0, 1).sum() > target:
            lo = tau
        else:
            hi = tau
    return np.clip(h - (lo + hi) / 2, 0, 1)


def grad(h, beta):
    n = len(h)
    g = (1 - h)[::-1]
    c = 2 * np.convolve(h, g) / n
    w = np.exp(beta * (c - c.max()))
    w /= w.sum()
    return 2 / n * (np.correlate(w, g, "valid") - np.correlate(w, h, "valid")[::-1])


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
    final_n = config["N_CELLS"] * 3 ** len(config["UPSAMPLE_AT"])
    init = load_init(config.get("INIT"), final_n, config["REFINE_NOISE"], seed)
    if init is not None:
        h = project(np.clip(init, 0, 1), final_n / 2)
    else:
        h = project(0.5 + 0.1 * rng.normal(size=config["N_CELLS"]), config["N_CELLS"] / 2)
    target = len(h) / 2
    beta_start = config["BETA_END"] * config["REFINE_BETA_FRAC"] if init is not None else config["BETA_START"]
    m = np.zeros(len(h))
    v = np.zeros(len(h))
    best_h, best = h.copy(), score(h)
    t = 0
    stage = len(config["UPSAMPLE_AT"]) if init is not None else 0
    while (now := time.monotonic()) < deadline:
        frac = (now - t0) / (deadline - t0)
        if stage < len(config["UPSAMPLE_AT"]) and frac >= config["UPSAMPLE_AT"][stage]:
            stage += 1
            h = np.repeat(h, 3)
            m, v = np.zeros(len(h)), np.zeros(len(h))
            t = 0
        beta = beta_start * (config["BETA_END"] / beta_start) ** frac
        g = grad(h, beta)
        t += 1
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        target = len(h) / 2
        h = project(h - config["LR"] * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-12), target)
        if t % 50 == 0:
            s = score(h)
            if s < best:
                best_h, best = h.copy(), s
    if score(h) < best:
        best_h = h
    if config.get("INIT"):   # warm start never returns worse than its (unperturbed) start
        start = load_init(config["INIT"], len(best_h))
        if score(start) < score(best_h):
            best_h = start
    return {"values": best_h.tolist()}


if __name__ == "__main__":
    run(__file__)
