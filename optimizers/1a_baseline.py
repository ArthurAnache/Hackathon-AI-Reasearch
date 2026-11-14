"""Optimizer file (contract: PROBLEM, CONFIG, search(budget_s, seed, config)).
Copy to search.py to let a coding agent edit it (program.md).

Problem 1a (Sidon autocorrelation, minimise 2N*max(a*a)/(sum a)^2 over a >= 0).
Baseline: hill climbing on N non-negative step heights. Run: python search.py "<hypothesis>"
"""
import time

import numpy as np

from prepare import run

PROBLEM = "1a"

# ---- hyperparameters: the optimizer's CONFIG (agents may override any key) ----
CONFIG = {
    "N_CELLS": 50,
    "STEP": 0.1,
}


def score(a):
    return 2 * len(a) * np.convolve(a, a).max() / a.sum() ** 2


def search(budget_s, seed, config=CONFIG):
    rng = np.random.default_rng(seed)
    deadline = time.monotonic() + 0.95 * budget_s
    a = np.ones(config["N_CELLS"])
    best = score(a)
    while time.monotonic() < deadline:
        cand = a.copy()
        i = rng.integers(config["N_CELLS"])
        cand[i] = max(0.0, cand[i] + rng.normal(0, config["STEP"]))
        s = score(cand)
        if s < best:
            a, best = cand, s
    return {"values": a.tolist()}


if __name__ == "__main__":
    run(__file__)
