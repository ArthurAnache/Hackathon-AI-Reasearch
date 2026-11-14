"""1a: Sidon set (first) autocorrelation constant. DO NOT MODIFY.

https://teorth.github.io/optimizationproblems/constants/1a.html

C_1a is the largest C with  max_t (f*f)(t) >= C (∫f)^2  for all non-negative f
supported on [-1/4, 1/4]. Any admissible f gives the UPPER bound
max(f*f) / (∫f)^2 >= C_1a, so a search MINIMISES that ratio.

Candidate: a step function with N equal cells on [-1/4, 1/4] (width 1/(2N)),
given by its non-negative heights a_0..a_{N-1}. f*f is piecewise linear with
knots at multiples of the cell width, so its maximum is at a knot, and

    score = 2N * max_k (a*a)_k / (sum a)^2        (lower is better)

Same formula as the Atris / EinsteinArena exact scorers.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from .common import InvalidSolution, convolve, exact_convolve, float_vector, to_integers

ID = "1a"
NAME = "Sidon set autocorrelation constant (upper bound)"
URL = "https://teorth.github.io/optimizationproblems/constants/1a.html"
DIRECTION = "minimize"
BEST_KNOWN = 1.5027434867095834     # Atris, 65,536 steps (exact 1.50274348670958337974...)
BEST_KNOWN_SOURCE = "Atris records (65,536-step construction), listed on the teorth page"
SOLUTION_FORMAT = '{"values": [a_0, ..., a_{N-1}]}  with a_i >= 0, 1 <= N <= 200000, sum > 0'
MAX_N = 200_000


def _check(solution) -> np.ndarray:
    a = float_vector(solution, "values", MAX_N)
    if np.any(a < 0):
        raise InvalidSolution("heights must be non-negative")
    if a.sum() <= 0:
        raise InvalidSolution("integral of f must be positive")
    return a


def verify(solution) -> float:
    a = _check(solution)
    return float(2 * len(a) * convolve(a, a).max() / a.sum() ** 2)


def verify_exact(solution) -> Fraction:
    a = _check(solution)
    ints, _ = to_integers(a)
    conv = exact_convolve(ints, ints)
    return Fraction(2 * len(ints) * max(conv), sum(ints) ** 2)
