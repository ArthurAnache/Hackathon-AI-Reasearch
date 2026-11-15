"""Toy problem (test fixture, NOT our target): pack n = 26 circles in the unit square, maximize the sum of radii.

Candidate format: {"circles": [[x, y, r], ...]} with exactly N_CIRCLES rows.
`score` runs an exact check in rational arithmetic (every float is converted exactly to a Fraction):
containment x - r >= 0, x + r <= 1 (same for y), r > 0, and no overlap (dx^2 + dy^2 >= (ri + rj)^2; tangency passes).
"""
from __future__ import annotations

import math
from fractions import Fraction

N_CIRCLES = 26
# Fixture reference only, not re-verified by us. AlphaEvolve (2025) reported about 2.635 for n = 26, and later
# reports gave about 2.63598; our first 60 s M0 run already reached 2.63592, so 2.635 was too low to be useful.
REFERENCE_SCORE = 2.63598
REFERENCE_SOURCE = "about 2.63598, values reported after AlphaEvolve for n=26 (fixture reference, not re-verified)"


class InvalidSolution(ValueError):
    """The candidate breaks a constraint. It gets no score."""


def _rows(candidate) -> list[tuple[float, float, float]]:
    if not isinstance(candidate, dict) or "circles" not in candidate:
        raise InvalidSolution('candidate must be {"circles": [[x, y, r], ...]}')
    rows = candidate["circles"]
    if not isinstance(rows, list) or len(rows) != N_CIRCLES:
        raise InvalidSolution(f"expected {N_CIRCLES} circles")
    out = []
    for i, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise InvalidSolution(f"circle {i} must be [x, y, r]")
        try:
            x, y, r = (float(v) for v in row)
        except (TypeError, ValueError) as exc:
            raise InvalidSolution(f"circle {i}: non-numeric value") from exc
        if not all(math.isfinite(v) for v in (x, y, r)):
            raise InvalidSolution(f"circle {i}: non-finite value")
        out.append((x, y, r))
    return out


def exact_check(candidate) -> Fraction:
    """Exact rational check. Returns the exact sum of radii; raises InvalidSolution on any violation."""
    rows = [(Fraction(x), Fraction(y), Fraction(r)) for x, y, r in _rows(candidate)]
    one = Fraction(1)
    for i, (x, y, r) in enumerate(rows):
        if r <= 0:
            raise InvalidSolution(f"circle {i}: radius must be positive")
        if x - r < 0 or x + r > one or y - r < 0 or y + r > one:
            raise InvalidSolution(f"circle {i}: not contained in the unit square")
    for i in range(N_CIRCLES):
        xi, yi, ri = rows[i]
        for j in range(i + 1, N_CIRCLES):
            xj, yj, rj = rows[j]
            if (xi - xj) ** 2 + (yi - yj) ** 2 < (ri + rj) ** 2:
                raise InvalidSolution(f"circles {i} and {j} overlap")
    return sum((r for _, _, r in rows), Fraction(0))


def score(candidate) -> float:
    """Sum of radii, after the exact check. Raises InvalidSolution."""
    return float(exact_check(candidate))


def verify(candidate) -> bool:
    try:
        exact_check(candidate)
    except InvalidSolution:
        return False
    return True


def guide_score(candidate) -> float:
    """Fast float score used inside the sandbox by the tuner (a guide only; the parent re-scores exactly)."""
    import numpy as np

    a = np.asarray(_rows(candidate), dtype=float)
    x, y, r = a[:, 0], a[:, 1], a[:, 2]
    if (r <= 0).any() or (x - r < 0).any() or (x + r > 1).any() or (y - r < 0).any() or (y + r > 1).any():
        raise InvalidSolution("containment")
    i, j = np.triu_indices(len(a), 1)
    if ((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 < (r[i] + r[j]) ** 2).any():
        raise InvalidSolution("overlap")
    return float(r.sum())
