"""Optimizer file (contract: PROBLEM, CONFIG, search(budget_s, seed, config)).
Copy to search.py to let a coding agent edit it (program.md).

Problem 1b (Erdős minimum overlap, minimise (2/N)*max_k sum_i h_i(1-h_{i+k})
over 0 <= h <= 1 with sum h = N/2). Baseline: hill climbing that moves mass
between two cells so the total stays fixed. Run: python search.py "<hypothesis>"
"""
import time

import numpy as np

from prepare import run

PROBLEM = "1b"

# ---- hyperparameters: the optimizer's CONFIG (agents may override any key) ----
CONFIG = {
    "N_CELLS": 60,
    "STEP": 0.05,
}


def score(h):
    return 2 * np.convolve(h, (1 - h)[::-1]).max() / len(h)


def search(budget_s, seed, config=CONFIG):
    rng = np.random.default_rng(seed)
    deadline = time.monotonic() + 0.95 * budget_s
    h = np.full(config["N_CELLS"], 0.5)
    best = score(h)
    while time.monotonic() < deadline:
        i, j = rng.integers(config["N_CELLS"], size=2)
        d = rng.uniform(0, config["STEP"])
        d = min(d, 1 - h[i], h[j])          # keep both cells inside [0, 1]
        if i == j or d <= 0:
            continue
        cand = h.copy()
        cand[i] += d
        cand[j] -= d
        s = score(cand)
        if s < best:
            h, best = cand, s
    return {"values": h.tolist()}


if __name__ == "__main__":
    run(__file__)
