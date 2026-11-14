"""Optimizer file (contract: PROBLEM, CONFIG, search(budget_s, seed, config)).
Copy to search.py to let a coding agent edit it (program.md).

Problem 10c (Spencer discrepancy, maximise min_x ||Ax||_inf / sqrt(n) over
A in [-1,1]^{n x n}, x in {±1}^n). Baseline: hill climbing on a random ±1
matrix, flipping one entry at a time. Run: python search.py "<hypothesis>"
"""
import itertools
import time

import numpy as np

from prepare import run

PROBLEM = "10c"

# ---- hyperparameters: the optimizer's CONFIG (agents may override any key) ----
CONFIG = {
    "N": 6,
}


def all_signs(n):
    return np.array([(1,) + s for s in itertools.product((1, -1), repeat=n - 1)], dtype=float)


def score(a, signs):
    return np.abs(signs @ a.T).max(axis=1).min() / np.sqrt(len(a))


def search(budget_s, seed, config=CONFIG):
    rng = np.random.default_rng(seed)
    deadline = time.monotonic() + 0.95 * budget_s
    signs = all_signs(config["N"])
    a = rng.choice([-1.0, 1.0], size=(config["N"], config["N"]))
    best = score(a, signs)
    while time.monotonic() < deadline:
        cand = a.copy()
        i, j = rng.integers(config["N"], size=2)
        cand[i, j] = -cand[i, j]
        s = score(cand, signs)
        if s >= best:                        # accept sideways moves to drift on plateaus
            a, best = cand, s
    return {"matrix": a.tolist()}


if __name__ == "__main__":
    run(__file__)
