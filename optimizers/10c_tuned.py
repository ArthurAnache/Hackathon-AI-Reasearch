"""Optimizer file (contract: PROBLEM, CONFIG, search(budget_s, seed, config)).
Copy to search.py to let a coding agent edit it (program.md).

Problem 10c (maximise min_x ||Ax||_inf / sqrt(n)).
Local search over {-1,0,1} matrices with a lexicographic objective:
(discrepancy, -number of sign vectors attaining it). The tie-break gives the
search a gradient on the plateaus of the discrepancy. Run: python search.py "<hypothesis>"

Tuned (benchmark best, 5/sqrt(8) = proven optimum for n = 8): local search over {-1,0,1}
matrices, objective (discrepancy, -#minimising sign vectors), random restarts.
"""
import itertools
import time

import numpy as np

from prepare import run

PROBLEM = "10c"

# ---- hyperparameters: the optimizer's CONFIG (agents may override any key) ----
CONFIG = {
    "N": 8,
    "P_ZERO": 0.1,  # probability a mutated entry becomes 0
    "RESTART_AFTER": 3000,  # moves without improvement before a random restart
}


def all_signs(n):
    return np.array([(1,) + s for s in itertools.product((1, -1), repeat=n - 1)], dtype=float)


def objective(a, signs):
    norms = np.abs(signs @ a.T).max(axis=1)
    d = norms.min()
    return d, -int((norms == d).sum())


def search(budget_s, seed, config=CONFIG):
    rng = np.random.default_rng(seed)
    deadline = time.monotonic() + 0.95 * budget_s
    signs = all_signs(config["N"])
    best_a, best = None, (-1, 0)
    while time.monotonic() < deadline:
        a = rng.choice([-1.0, 1.0], size=(config["N"], config["N"]))
        cur = objective(a, signs)
        stale = 0
        while stale < config["RESTART_AFTER"] and time.monotonic() < deadline:
            cand = a.copy()
            i, j = rng.integers(config["N"], size=2)
            choices = [x for x in (-1.0, 0.0, 1.0) if x != cand[i, j]]
            cand[i, j] = 0.0 if (0.0 in choices and rng.random() < config["P_ZERO"]) else rng.choice([c for c in choices if c != 0.0] or choices)
            o = objective(cand, signs)
            if o >= cur:
                stale = 0 if o > cur else stale + 1
                a, cur = cand, o
            else:
                stale += 1
        if cur > best:
            best_a, best = a.copy(), cur
    return {"matrix": best_a.tolist()}


if __name__ == "__main__":
    run(__file__)
