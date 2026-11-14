"""1b: Erdős minimum overlap constant. DO NOT MODIFY.

https://teorth.github.io/optimizationproblems/constants/1b.html

For f: [-1,1] -> [0,1] with ∫f = 1 and g = 1 - f on [-1,1] (both zero outside),
    M(f) = sup_x ∫ f(t) g(x + t) dt  >=  C_1b.
Every admissible f gives an UPPER bound on C_1b, so a search MINIMISES M(f).

Candidate: a step function with N equal cells (width 2/N), heights h_i in [0,1].
The verifier rescales h to mass exactly 1 (h <- h * (N/2) / sum h). This is
rejected if any rescaled height exceeds 1. The correlation of two step functions
is piecewise linear in the shift with knots at multiples of the cell width, so

    score = (2/N) * max_k sum_i h_i (1 - h_{i+k})     over all shifts k (both signs)

which is the Atris exact scorer's formula.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from .common import InvalidSolution, convolve, exact_convolve, float_vector, to_integers

ID = "1b"
NAME = "Erdős minimum overlap constant (upper bound)"
URL = "https://teorth.github.io/optimizationproblems/constants/1b.html"
DIRECTION = "minimize"
BEST_KNOWN = 0.38085855031642856    # Atris, 114,688 steps (exact 0.38085855031642856010...)
BEST_KNOWN_SOURCE = "Atris records (114,688-step construction), listed on the teorth page"
SOLUTION_FORMAT = '{"values": [h_0, ..., h_{N-1}]}  0 <= h_i <= 1, 2 <= N <= 200000 (rescaled to mass 1)'
MAX_N = 200_000
TOL = 1e-12


def _normalised(solution) -> np.ndarray:
    h = float_vector(solution, "values", MAX_N, min_len=2)
    if np.any(h < 0) or np.any(h > 1):
        raise InvalidSolution("heights must lie in [0, 1]")
    if h.sum() <= 0:
        raise InvalidSolution("mass must be positive")
    h = h * (len(h) / 2) / h.sum()
    if h.max() > 1 + TOL:
        raise InvalidSolution(f"rescaling to mass 1 pushes a height to {h.max():.6g} > 1")
    return np.minimum(h, 1.0)


def verify(solution) -> float:
    h = _normalised(solution)
    corr = convolve(h, (1 - h)[::-1])
    return float(2 * corr.max() / len(h))


def verify_exact(solution) -> Fraction:
    _normalised(solution)  # same admissibility checks as verify()
    ints, _ = to_integers(float_vector(solution, "values", MAX_N, min_len=2))
    n, total = len(ints), sum(ints)
    hs = [Fraction(x * n, 2 * total) for x in ints]          # exact rescale to mass 1
    if max(hs) > 1:
        raise InvalidSolution("exact rescaling to mass 1 pushes a height above 1")
    hi, d = to_integers(hs)
    corr = exact_convolve(hi, [d - x for x in reversed(hi)])
    return Fraction(2 * max(corr), n * d * d)
