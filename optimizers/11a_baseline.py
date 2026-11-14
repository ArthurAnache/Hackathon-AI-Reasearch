"""Optimizer file (contract: PROBLEM, CONFIG, search(budget_s, seed, config)).
Copy to search.py to let a coding agent edit it (program.md).

Problem 11a (L^1 Poincaré on the Hamming cube, maximise E|f - Ef| / E|grad f|
over f: {-1,1}^n -> R). Baseline: hill climbing on the 2^n values of f.
Run: python search.py "<hypothesis>"
"""
import time

import numpy as np

from prepare import run

PROBLEM = "11a"

# ---- hyperparameters: the optimizer's CONFIG (agents may override any key) ----
CONFIG = {
    "DIM": 4,
    "STEP": 0.1,
}


def score(f, flips):
    num = np.abs(f - f.mean()).mean()
    grad = np.sqrt((((f[None, :] - f[flips]) / 2) ** 2).sum(axis=0)).mean()
    return num / grad


def search(budget_s, seed, config=CONFIG):
    rng = np.random.default_rng(seed)
    deadline = time.monotonic() + 0.95 * budget_s
    size = 1 << config["DIM"]
    flips = np.array([np.arange(size) ^ (1 << j) for j in range(config["DIM"])])
    f = rng.normal(size=size)
    best = score(f, flips)
    while time.monotonic() < deadline:
        cand = f.copy()
        cand[rng.integers(size)] += rng.normal(0, config["STEP"])
        s = score(cand, flips)
        if s > best:
            f, best = cand, s
    return {"n": config["DIM"], "values": f.tolist()}


if __name__ == "__main__":
    run(__file__)
