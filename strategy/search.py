"""EDITABLE. The only code the agent is allowed to change (plus config.yaml).

Contract (see INTERFACES.md):
    search(config: dict, seed: int, budget_s: float) -> dict  # a solution for problem.verify

The runner calls this in a separate process with a hard timeout, then scores
the returned solution with problem.verify. Any score computed in here is only
for guiding the search; it is never trusted.
"""
from __future__ import annotations

import math
import random
import time

N_POINTS = 12  # must match problem.N_POINTS


def _min_dist(pts):
    best = math.inf
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
            if d < best:
                best = d
    return best


def _random_points(rng):
    return [[rng.random(), rng.random()] for _ in range(N_POINTS)]


def random_search(config, rng, deadline):
    best, best_score = None, -math.inf
    for _ in range(int(config.get("n_samples", 1000))):
        if time.monotonic() > deadline:
            break
        cand = _random_points(rng)
        s = _min_dist(cand)
        if s > best_score:
            best, best_score = cand, s
    return best


def hill_climb(config, rng, deadline):
    step = float(config.get("step_size", 0.05))
    cur = _random_points(rng)
    cur_score = _min_dist(cur)
    for _ in range(int(config.get("n_samples", 1000))):
        if time.monotonic() > deadline:
            break
        cand = [p[:] for p in cur]
        k = rng.randrange(N_POINTS)
        cand[k][0] = min(1.0, max(0.0, cand[k][0] + rng.gauss(0, step)))
        cand[k][1] = min(1.0, max(0.0, cand[k][1] + rng.gauss(0, step)))
        s = _min_dist(cand)
        if s >= cur_score:
            cur, cur_score = cand, s
    return cur


METHODS = {"random": random_search, "hill_climb": hill_climb}


def search(config: dict, seed: int, budget_s: float) -> dict:
    rng = random.Random(seed)
    deadline = time.monotonic() + budget_s * 0.9
    method = METHODS[config.get("method", "random")]
    return {"points": method(config, rng, deadline)}
