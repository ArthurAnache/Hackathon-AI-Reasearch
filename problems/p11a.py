"""11a: L^1 Poincaré constant on the Hamming cube (lower bound). DO NOT MODIFY.

https://teorth.github.io/optimizationproblems/constants/11a.html

C_11a is the smallest C with E|f - Ef| <= C E|∇f| for all n and f: {-1,1}^n -> R,
where |∇f|(x) = (sum_j D_j f(x)^2)^(1/2) and D_j f(x) = (f(x) - f(x^(j)))/2
(x^(j) = x with coordinate j flipped), x uniform. Every f gives the LOWER bound
E|f - Ef| / E|∇f| <= C_11a, so a search MAXIMISES that ratio.

Candidate: {"n": n, "values": [f(x) for x indexed 0..2^n-1]}, where bit j of the
index is 1 iff x_j = -1.  Known: C_11a >= sqrt(pi/2) (Pisier; ILvHV 2019); the
page says it is open whether C_11a = sqrt(pi/2). A finite (n, f) with ratio above
sqrt(pi/2) would improve the lower bound.
"""
from __future__ import annotations

import math

import numpy as np

from .common import InvalidSolution, float_vector

ID = "11a"
NAME = "L^1 Poincaré constant on the Hamming cube (lower bound)"
URL = "https://teorth.github.io/optimizationproblems/constants/11a.html"
DIRECTION = "maximize"
BEST_KNOWN = math.sqrt(math.pi / 2)  # 1.2533141373155003 (a limit as n -> infinity)
BEST_KNOWN_SOURCE = "sqrt(pi/2): Pisier 1986; Ivanisvili-Li-van Handel-Volberg 2019"
SOLUTION_FORMAT = '{"n": n, "values": [f_0, ..., f_{2^n - 1}]}  1 <= n <= 20, f not constant'
MAX_DIM = 20


def ratio(n: int, f: np.ndarray) -> float:
    idx = np.arange(1 << n)
    num = np.abs(f - f.mean()).mean()
    grad_sq = np.zeros(1 << n)
    for j in range(n):
        grad_sq += ((f - f[idx ^ (1 << j)]) / 2) ** 2
    den = np.sqrt(grad_sq).mean()
    if den <= 0:
        raise InvalidSolution("f is constant (E|grad f| = 0)")
    return float(num / den)


def verify(solution) -> float:
    if not isinstance(solution, dict) or "n" not in solution:
        raise InvalidSolution("solution must be a dict with keys 'n' and 'values'")
    n = solution["n"]
    if isinstance(n, bool) or not isinstance(n, int) or not (1 <= n <= MAX_DIM):
        raise InvalidSolution(f"n must be an integer in [1, {MAX_DIM}]")
    f = float_vector(solution, "values", 1 << n, min_len=1 << n)
    return ratio(n, f)
