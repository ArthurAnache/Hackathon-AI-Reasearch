"""1d: signed third autocorrelation constant. DO NOT MODIFY.

https://teorth.github.io/optimizationproblems/constants/1d.html

C_1d is the largest C with  max_t |(f*f)(t)| >= C (∫f)^2  for all real-valued
(signed) f supported on [-1/4, 1/4]. Any admissible f gives the UPPER bound
max|f*f| / (∫f)^2 >= C_1d, so a search MINIMISES that ratio.

Candidate: a step function with N equal cells on [-1/4, 1/4], heights v_i of
any sign. |f*f| is the absolute value of a piecewise-linear function with knots
at multiples of the cell width, so its maximum is at a knot, and

    score = 2N * max_k |(v*v)_k| / (sum v)^2        (lower is better)

Same formula as the c1d-certificate verify.py linked from the teorth page.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from .common import InvalidSolution, convolve, exact_convolve, float_vector, to_integers

ID = "1d"
NAME = "Signed third autocorrelation constant (upper bound)"
URL = "https://teorth.github.io/optimizationproblems/constants/1d.html"
DIRECTION = "minimize"
BEST_KNOWN = 1.457854884921989      # SCiEL, 800 cells (asterisked on the page)
BEST_KNOWN_SOURCE = "SCiEL 800-cell construction (asterisked); best fully replayed: 1.4630715826075594 (300 cells)"
SOLUTION_FORMAT = '{"values": [v_0, ..., v_{N-1}]}  real, 1 <= N <= 200000, sum != 0'
MAX_N = 200_000


def _check(solution) -> np.ndarray:
    v = float_vector(solution, "values", MAX_N)
    if v.sum() == 0:
        raise InvalidSolution("integral of f must be non-zero")
    return v


def verify(solution) -> float:
    v = _check(solution)
    return float(2 * len(v) * np.abs(convolve(v, v)).max() / v.sum() ** 2)


def verify_exact(solution) -> Fraction:
    v = _check(solution)
    ints, _ = to_integers(v)
    conv = exact_convolve(ints, ints)
    return Fraction(2 * len(ints) * max(abs(c) for c in conv), sum(ints) ** 2)
