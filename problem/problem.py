"""FROZEN. The problem definition and its independent verifier.

The agent must never edit anything in problem/. The runner hash-checks this
folder before every experiment (see frozen.lock) and refuses to run if it changed.

Dummy problem (placeholder until we plug in a real record):
    Place N points in the unit square [0,1]^2 to MAXIMISE the minimum
    pairwise distance (point "spreading" / circle packing in a square).
    Best known value for N=12 is about 0.38873 (Packomania), so there
    is a real target to chase.

Contract:
    verify(solution) -> float      higher is better
    raises InvalidSolution         if the solution breaks a constraint
"""
from __future__ import annotations

import math
from typing import Any

N_POINTS = 12
DIRECTION = "maximize"          # the runner keeps a change only if score goes UP
KNOWN_BEST = 0.388730126      # best known value for N=12 (Packomania), reference only
TOL = 1e-12


class InvalidSolution(ValueError):
    pass


def verify(solution: Any) -> float:
    """Independently score a candidate. Never trusts anything the strategy reports."""
    if not isinstance(solution, dict) or "points" not in solution:
        raise InvalidSolution("solution must be a dict with key 'points'")
    pts = solution["points"]
    if not isinstance(pts, list) or len(pts) != N_POINTS:
        raise InvalidSolution(f"expected exactly {N_POINTS} points, got {len(pts) if isinstance(pts, list) else type(pts)}")
    clean = []
    for i, p in enumerate(pts):
        if not (isinstance(p, (list, tuple)) and len(p) == 2):
            raise InvalidSolution(f"point {i} is not a pair")
        x, y = float(p[0]), float(p[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise InvalidSolution(f"point {i} is not finite")
        if not (-TOL <= x <= 1 + TOL and -TOL <= y <= 1 + TOL):
            raise InvalidSolution(f"point {i}=({x},{y}) is outside the unit square")
        clean.append((x, y))
    best = math.inf
    for i in range(N_POINTS):
        xi, yi = clean[i]
        for j in range(i + 1, N_POINTS):
            d = math.hypot(xi - clean[j][0], yi - clean[j][1])
            if d < best:
                best = d
    return best
