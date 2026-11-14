"""Optimizer file (contract: PROBLEM, CONFIG, search(budget_s, seed, config)).
Copy to search.py to let a coding agent edit it (program.md).

Problem 11a (maximise E|f - Ef| / E|grad f| on {-1,1}^n).
Restrict to symmetric f (f depends only on k = number of -1 coordinates), so
only n+1 parameters, which allows n = 16. Hill climbing on the profile f(k).
Run: python search.py "<hypothesis>"

Tuned (benchmark best, 0.960518 at 60 s): symmetric f (depends only on the number of -1
coordinates), n = 20, hill climbing on the n+1 profile values.
"""
import time
from math import comb

import numpy as np

from prepare import run

PROBLEM = "11a"

# ---- hyperparameters: the optimizer's CONFIG (agents may override any key) ----
CONFIG = {
    "DIM": 20,
    "STEP": 0.05,
}


def ratio(f, p, n):
    k = np.arange(n + 1)
    mu = (p * f).sum()
    num = (p * np.abs(f - mu)).sum()
    up = np.append(f[1:] - f[:-1], 0.0)        # f(k+1) - f(k), k < n
    down = np.insert(f[1:] - f[:-1], 0, 0.0)   # f(k) - f(k-1), k > 0
    grad = 0.5 * np.sqrt((n - k) * up ** 2 + k * down ** 2)
    return num / (p * grad).sum()


def search(budget_s, seed, config=CONFIG):
    rng = np.random.default_rng(seed)
    deadline = time.monotonic() + 0.95 * budget_s
    n = config["DIM"]
    p = np.array([comb(n, k) for k in range(n + 1)], dtype=float) / 2 ** n
    f = np.arange(n + 1, dtype=float) - n / 2       # start from the linear function
    best = ratio(f, p, n)
    while time.monotonic() < deadline:
        cand = f + rng.normal(0, config["STEP"], size=n + 1) * (rng.random(n + 1) < 0.3)
        r = ratio(cand, p, n)
        if r > best:
            f, best = cand, r
    weights = np.array([bin(x).count("1") for x in range(1 << n)])
    return {"n": n, "values": f[weights].tolist()}


if __name__ == "__main__":
    run(__file__)
