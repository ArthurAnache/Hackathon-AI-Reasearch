"""Optimizer file (contract: PROBLEM, CONFIG, search(budget_s, seed, config)).
Copy to search.py to let a coding agent edit it (program.md).

Problem 1a (minimise 2N*max(a*a)/(sum a)^2 over a >= 0).
Adam on a = x^2, minimising a log-sum-exp smoothing of max(a*a) whose
temperature is annealed towards the true max. Run: python search.py "<hypothesis>"

Tuned (benchmark best, 1.514054 at 60 s): Adam on a = x^2 minimising an annealed log-sum-exp
surrogate of max(a*a), multi-resolution (cells x3 at 30% and 60% of the budget, LR/3 each time).
Warm start: with CONFIG["INIT"] = path of a saved solution, the search starts from it
(resampled to the final resolution) and refines it with a sharp annealing schedule.
"""
import json
import time
from pathlib import Path

import numpy as np

from prepare import run

PROBLEM = "1a"

# ---- hyperparameters: the optimizer's CONFIG (agents may override any key) ----
CONFIG = {
    "N_CELLS": 150,  # starting resolution
    "UPSAMPLE_AT": (0.3, 0.6),  # refine x3 at these budget fractions
    "LR": 0.01,
    "BETA_START": 10.0,
    "BETA_END": 3000.0,
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


def score(a):
    return 2 * len(a) * np.convolve(a, a).max() / a.sum() ** 2


def grad(a, beta):
    n, s = len(a), a.sum()
    c = 2 * n * np.convolve(a, a) / s ** 2
    w = np.exp(beta * (c - c.max()))
    w /= w.sum()
    m = (w * np.convolve(a, a)).sum()
    r = np.correlate(w, a, "valid")
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
    final_n = config["N_CELLS"] * 3 ** len(config["UPSAMPLE_AT"])
    init = load_init(config.get("INIT"), final_n, config["REFINE_NOISE"], seed)
    x = np.sqrt(np.maximum(init, 0)) if init is not None else 1 + 0.1 * rng.normal(size=config["N_CELLS"])
    beta_start = config["BETA_END"] * config["REFINE_BETA_FRAC"] if init is not None else config["BETA_START"]
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    best_a, best = x ** 2, score(x ** 2)
    t = 0
    stage = len(config["UPSAMPLE_AT"]) if init is not None else 0
    lr = config["LR"] / 3 ** stage
    while (now := time.monotonic()) < deadline:
        frac = (now - t0) / (deadline - t0)
        if stage < len(config["UPSAMPLE_AT"]) and frac >= config["UPSAMPLE_AT"][stage]:
            stage += 1
            x = np.repeat(x, 3)
            lr /= 3
            m, v = np.zeros(len(x)), np.zeros(len(x))
            t = 0
        beta = beta_start * (config["BETA_END"] / beta_start) ** frac
        a = x ** 2
        g = grad(a, beta) * 2 * x
        t += 1
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        x -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-12)
        if t % 50 == 0:
            s = score(x ** 2)
            if s < best:
                best_a, best = x ** 2, s
    s = score(x ** 2)
    if s < best:
        best_a = x ** 2
    if config.get("INIT"):   # warm start never returns worse than its (unperturbed) start
        start = load_init(config["INIT"], len(best_a))
        if score(start) < score(best_a):
            best_a = start
    return {"values": best_a.tolist()}


if __name__ == "__main__":
    run(__file__)
