"""Optimizer file (contract: PROBLEM, CONFIG, search(budget_s, seed, config)).
Copy to search.py to let a coding agent edit it (program.md).

Problem 1d (signed autocorrelation, minimise 2N*max|v*v|/(sum v)^2 over real v).
Baseline: hill climbing on N signed step heights. Run: python search.py "<hypothesis>"
"""
import time

import numpy as np

from prepare import run

PROBLEM = "1d"

# ---- hyperparameters: the optimizer's CONFIG (agents may override any key) ----
CONFIG = {
    "N_CELLS": 50,
    "STEP": 0.1,
}


def score(v):
    return 2 * len(v) * np.abs(np.convolve(v, v)).max() / v.sum() ** 2


def search(budget_s, seed, config=CONFIG):
    rng = np.random.default_rng(seed)
    deadline = time.monotonic() + 0.95 * budget_s
    v = np.ones(config["N_CELLS"])
    best = score(v)
    while time.monotonic() < deadline:
        cand = v.copy()
        i = rng.integers(config["N_CELLS"])
        cand[i] += rng.normal(0, config["STEP"])
        s = score(cand)
        if s < best:
            v, best = cand, s
    return {"values": v.tolist()}


if __name__ == "__main__":
    run(__file__)
